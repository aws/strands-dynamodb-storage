// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { SearchVectorsCommand } from '@aws-sdk/client-dynamodb'
import { PutCommand, GetCommand, DeleteCommand, QueryCommand } from '@aws-sdk/lib-dynamodb'
import { marshall } from '@aws-sdk/util-dynamodb'
import { PutObjectCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3'
import { StorageError } from '@strands-agents/sdk'
import { DynamoDBStorage, type VectorSearchAdapter } from './dynamodb-storage.js'
import * as pkgIndex from './index.js'

/**
 * In-memory stand-in for a DynamoDBDocumentClient. Implements only `send`,
 * backed by a Map keyed by `pk\u0000sk`. Models Put/Get/Delete/Query including
 * ExclusiveStartKey pagination and ReturnValues=ALL_OLD for delete.
 */
class FakeDocumentClient {
  readonly items = new Map<string, Record<string, any>>()
  private _k(pk: string, sk: string): string {
    return `${pk}\u0000${sk}`
  }
  /** Captured input of the last SearchVectorsCommand, for request-shape assertions. */
  lastSearchVectorsInput: any = null
  /** Number of GetCommand point reads issued, for read-avoidance assertions. */
  getCalls = 0
  async send(command: any): Promise<any> {
    const input = command.input
    if (command instanceof SearchVectorsCommand) {
      this.lastSearchVectorsInput = input
      const queryVector: number[] = input.SearchVector.map((av: any) => Number(av.N))
      const pkFilter = input.ExpressionAttributeValues?.[':pk']?.S
      const scored = [...this.items.values()]
        .filter((item) => Array.isArray(item.vector))
        .filter((item) => pkFilter === undefined || item.pk === pkFilter)
        .map((item) => {
          const v: number[] = item.vector
          const dot = v.reduce((acc, x, i) => acc + x * (queryVector[i] ?? 0), 0)
          const norm =
            Math.sqrt(v.reduce((a, x) => a + x * x, 0)) * Math.sqrt(queryVector.reduce((a, x) => a + x * x, 0))
          const cosineDistance = 1 - (norm === 0 ? 0 : dot / norm)
          const { vector: _v, ...rest } = item
          // Honor ProjectionExpression the way the service does: return only the
          // projected attributes (all names are aliased through #placeholders).
          let projected = rest
          if (input.ProjectionExpression) {
            const wanted = new Set<string>(
              input.ProjectionExpression.split(',').map((alias: string) => input.ExpressionAttributeNames[alias.trim()])
            )
            projected = Object.fromEntries(Object.entries(rest).filter(([k]) => wanted.has(k)))
          }
          return { Item: marshall(projected), Score: cosineDistance }
        })
        .sort((a, b) => a.Score - b.Score)
        .slice(0, input.TopK)
      return { SearchResults: scored }
    }
    if (command instanceof PutCommand) {
      const key = this._k(input.Item.pk, input.Item.sk)
      const old = this.items.get(key)
      this.items.set(key, input.Item)
      return input.ReturnValues === 'ALL_OLD' && old ? { Attributes: old } : {}
    }
    if (command instanceof GetCommand) {
      this.getCalls += 1
      return { Item: this.items.get(this._k(input.Key.pk, input.Key.sk)) }
    }
    if (command instanceof DeleteCommand) {
      const key = this._k(input.Key.pk, input.Key.sk)
      const old = this.items.get(key)
      this.items.delete(key)
      return input.ReturnValues === 'ALL_OLD' ? { Attributes: old } : {}
    }
    if (command instanceof QueryCommand) {
      const v = input.ExpressionAttributeValues
      const pk: string = v[':pk']
      let matched = [...this.items.values()].filter((item) => item.pk === pk)
      if (v[':sk'] !== undefined) matched = matched.filter((i) => String(i.sk).startsWith(v[':sk']))
      if (v[':from'] !== undefined) matched = matched.filter((i) => i.sk >= v[':from'] && i.sk <= v[':to'])
      // Model the TTL FilterExpression: only present when the caller opted into TTL.
      if (v[':now'] !== undefined) {
        const ttlName = input.ExpressionAttributeNames['#ttl']
        matched = matched.filter((i) => i[ttlName] === undefined || i[ttlName] > v[':now'])
      }
      matched.sort((a, b) => (a.sk < b.sk ? -1 : a.sk > b.sk ? 1 : 0))
      // Model pagination: one item per page so ExclusiveStartKey is exercised.
      let start = 0
      if (input.ExclusiveStartKey) {
        const startSk = input.ExclusiveStartKey.sk
        start = matched.findIndex((i) => i.sk === startSk) + 1
      }
      const page = matched.slice(start, start + 1)
      const last = page[0]
      const more = start + 1 < matched.length
      return {
        Items: page.map((item) => ({ k: item.k })),
        LastEvaluatedKey: more && last ? { pk: last.pk, sk: last.sk } : undefined,
      }
    }
    throw new Error(`unexpected command ${command?.constructor?.name}`)
  }
}

/** In-memory stand-in for an S3Client covering Put/Get/Delete of object bytes. */
class FakeS3Client {
  readonly objects = new Map<string, Uint8Array>()
  async send(command: any): Promise<any> {
    const input = command.input
    if (command instanceof PutObjectCommand) {
      this.objects.set(input.Key, input.Body)
      return {}
    }
    if (command instanceof GetObjectCommand) {
      const body = this.objects.get(input.Key)
      if (!body) throw Object.assign(new Error('NoSuchKey'), { name: 'NoSuchKey' })
      return { Body: { transformToByteArray: async () => body } }
    }
    if (command instanceof DeleteObjectCommand) {
      this.objects.delete(input.Key)
      return {}
    }
    throw new Error(`unexpected S3 command ${command?.constructor?.name}`)
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!
    na += a[i]! * a[i]!
    nb += b[i]! * b[i]!
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1)
}

/** Brute-force cosine adapter over the fake client's stored vectors (stands in for SearchVectors). */
function cosineAdapter(client: FakeDocumentClient): VectorSearchAdapter {
  return async ({ pk, vector, topK }) => {
    return [...client.items.values()]
      .filter((i) => (pk ? i.pk === pk : true) && Array.isArray(i.vector))
      .map((i) => ({ key: i.k as string, score: cosine(vector, i.vector as number[]), metadata: i.meta }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
  }
}

function newStorage(opts?: {
  s3?: boolean
  prefix?: string
  vector?: boolean
  compression?: 'gzip' | 'none'
  ttlSeconds?: number
  ttlAttribute?: string
}) {
  const client = new FakeDocumentClient()
  const s3 = new FakeS3Client()
  const storage = new DynamoDBStorage('test-table', {
    client: client as any,
    ...(opts?.prefix ? { prefix: opts.prefix } : {}),
    ...(opts?.s3 ? { s3Bucket: 'test-bucket', s3Client: s3 as any } : {}),
    ...(opts?.vector ? { vectorSearch: cosineAdapter(client) } : {}),
    ...(opts?.compression ? { compression: opts.compression } : {}),
    ...(opts?.ttlSeconds !== undefined ? { ttlSeconds: opts.ttlSeconds } : {}),
    ...(opts?.ttlAttribute ? { ttlAttribute: opts.ttlAttribute } : {}),
  })
  return { client, s3, storage }
}

const bytes = (s: string) => new TextEncoder().encode(s)
const str = (b: Uint8Array | null) => (b ? new TextDecoder().decode(b) : null)

/** High-entropy bytes that gzip cannot meaningfully compress. */
function randomBytes(n: number): Uint8Array {
  const a = new Uint8Array(n)
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 256) | 0
  return a
}

describe('DynamoDBStorage — point operations', () => {
  it('round-trips a value through write and read', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/scopes/agent/a1/snapshots/snapshot_latest.json', bytes('hello'))
    expect(str(await storage.read('sessions/s1/scopes/agent/a1/snapshots/snapshot_latest.json'))).toBe('hello')
  })

  it('returns null for a missing key', async () => {
    const { storage } = newStorage()
    expect(await storage.read('sessions/s1/missing')).toBeNull()
  })

  it('overwrites an existing value', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/a', bytes('v1'))
    await storage.write('sessions/s1/a', bytes('v2'))
    expect(str(await storage.read('sessions/s1/a'))).toBe('v2')
  })

  it('delete removes the value', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/a', bytes('x'))
    await storage.delete('sessions/s1/a')
    expect(await storage.read('sessions/s1/a')).toBeNull()
  })

  it('delete is a no-op for a missing key', async () => {
    const { storage } = newStorage()
    await expect(storage.delete('sessions/s1/missing')).resolves.toBeUndefined()
  })

  it('round-trips a single-segment key via the sentinel sort key', async () => {
    const { storage } = newStorage()
    await storage.write('singleton', bytes('one'))
    expect(str(await storage.read('singleton'))).toBe('one')
    await storage.delete('singleton')
    expect(await storage.read('singleton')).toBeNull()
  })

  it('rejects empty and traversal keys', async () => {
    const { storage } = newStorage()
    await expect(storage.write('', bytes('x'))).rejects.toThrow(/must not be empty/)
    await expect(storage.read('sessions/../etc')).rejects.toThrow(/'\.\.'/)
  })
})

describe('DynamoDBStorage — listing', () => {
  it('lists keys under a string prefix, sorted, with pagination', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/scopes/agent/a1/immutable_history/snapshot_2.json', bytes('2'))
    await storage.write('sessions/s1/scopes/agent/a1/immutable_history/snapshot_1.json', bytes('1'))
    await storage.write('sessions/s1/scopes/agent/a1/immutable_history/snapshot_3.json', bytes('3'))
    await storage.write('sessions/s1/scopes/agent/a1/snapshots/snapshot_latest.json', bytes('L'))
    const keys = await storage.list('sessions/s1/scopes/agent/a1/immutable_history/')
    expect(keys).toEqual([
      'sessions/s1/scopes/agent/a1/immutable_history/snapshot_1.json',
      'sessions/s1/scopes/agent/a1/immutable_history/snapshot_2.json',
      'sessions/s1/scopes/agent/a1/immutable_history/snapshot_3.json',
    ])
  })

  it('lists via a structured DynamoDBListQuery naming the partition', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/a', bytes('a'))
    await storage.write('sessions/s1/b', bytes('b'))
    await storage.write('sessions/s2/c', bytes('c'))
    expect((await storage.list({ pk: 'sessions/s1' })).sort()).toEqual(['sessions/s1/a', 'sessions/s1/b'])
  })

  it('supports skBetween range queries', async () => {
    const { storage } = newStorage()
    for (const n of ['1', '2', '3', '4']) await storage.write(`sessions/s1/k/${n}`, bytes(n))
    const keys = await storage.list({ pk: 'sessions/s1', skBetween: ['k/2', 'k/3'] })
    expect(keys).toEqual(['sessions/s1/k/2', 'sessions/s1/k/3'])
  })

  it('honors startAfter as an exclusive cursor', async () => {
    const { storage } = newStorage()
    for (const n of ['a', 'b', 'c']) await storage.write(`sessions/s1/${n}`, bytes(n))
    const keys = await storage.list({ pk: 'sessions/s1', startAfter: 'sessions/s1/a' })
    expect(keys).toEqual(['sessions/s1/b', 'sessions/s1/c'])
  })

  it('fills a limit with post-cursor keys when startAfter is set (not truncated after the limit)', async () => {
    const { storage } = newStorage()
    for (const n of ['1', '2', '3', '4', '5']) await storage.write(`sessions/s1/k/${n}`, bytes(n))
    const keys = await storage.list({ pk: 'sessions/s1', startAfter: 'sessions/s1/k/1', limit: 2 })
    expect(keys).toEqual(['sessions/s1/k/2', 'sessions/s1/k/3'])
  })

  it('rejects a too-broad string prefix that cannot resolve a partition', async () => {
    const { storage } = newStorage()
    await expect(storage.list('sessions/')).rejects.toThrow(/too broad/)
  })

  it('rejects a query specifying both skPrefix and skBetween', async () => {
    const { storage } = newStorage()
    await expect(storage.list({ pk: 'x/y', skPrefix: 'a', skBetween: ['a', 'b'] })).rejects.toThrow(/not both/)
  })
})

describe('DynamoDBStorage — S3 offload', () => {
  it('offloads oversized values to S3 and reads them back transparently', async () => {
    const { storage, s3, client } = newStorage({ s3: true })
    const big = bytes('Z'.repeat(400_001))
    await storage.write('sessions/s1/big', big)
    // Pointer item in DynamoDB, bytes in S3.
    expect(s3.objects.size).toBe(1)
    const item = [...client.items.values()][0]!
    expect(item.s3).toBe(true)
    expect(item.data).toBeUndefined()
    expect(str(await storage.read('sessions/s1/big'))).toBe('Z'.repeat(400_001))
  })

  it('delete cleans up the offloaded S3 object', async () => {
    const { storage, s3 } = newStorage({ s3: true })
    await storage.write('sessions/s1/big', bytes('Z'.repeat(400_001)))
    await storage.delete('sessions/s1/big')
    expect(s3.objects.size).toBe(0)
    expect(await storage.read('sessions/s1/big')).toBeNull()
  })

  it('shrink-overwrite reclaims the offloaded S3 object', async () => {
    const { storage, s3, client } = newStorage({ s3: true })
    await storage.write('sessions/s1/doc', bytes('Z'.repeat(400_001)))
    expect(s3.objects.size).toBe(1)
    await storage.write('sessions/s1/doc', bytes('small now'))
    const item = [...client.items.values()][0]!
    expect(item.s3).toBeUndefined()
    expect(s3.objects.size).toBe(0) // reclaimed, not orphaned
    expect(str(await storage.read('sessions/s1/doc'))).toBe('small now')
  })

  it('inline-overwrite never touches S3 (negative control)', async () => {
    const { storage, s3 } = newStorage({ s3: true })
    const sends: unknown[] = []
    const realSend = s3.send.bind(s3)
    s3.send = async (c: any) => (sends.push(c), realSend(c))
    await storage.write('sessions/s1/doc', bytes('one'))
    await storage.write('sessions/s1/doc', bytes('two'))
    expect(sends).toHaveLength(0)
    expect(str(await storage.read('sessions/s1/doc'))).toBe('two')
  })

  it('offloaded-overwrite keeps the object readable (deterministic key reuse)', async () => {
    const { storage, s3 } = newStorage({ s3: true })
    await storage.write('sessions/s1/doc', bytes('A'.repeat(400_001)))
    await storage.write('sessions/s1/doc', bytes('B'.repeat(400_002)))
    expect(s3.objects.size).toBe(1)
    expect(str(await storage.read('sessions/s1/doc'))).toBe('B'.repeat(400_002))
  })

  it('a failed S3 reclamation does not fail the write (best-effort)', async () => {
    const { storage, s3 } = newStorage({ s3: true })
    await storage.write('sessions/s1/doc', bytes('Z'.repeat(400_001)))
    const realSend = s3.send.bind(s3)
    s3.send = async (c: any) => {
      if (c instanceof DeleteObjectCommand) throw new Error('s3 down')
      return realSend(c)
    }
    await storage.write('sessions/s1/doc', bytes('small now')) // must not throw
    expect(str(await storage.read('sessions/s1/doc'))).toBe('small now')
  })

  it('throws on an oversized value when no S3 bucket is configured', async () => {
    const { storage } = newStorage()
    await expect(storage.write('sessions/s1/big', bytes('Z'.repeat(400_001)))).rejects.toThrow(/s3Bucket/)
  })

  it('stores small values inline (no S3 write)', async () => {
    const { storage, s3 } = newStorage({ s3: true })
    await storage.write('sessions/s1/small', bytes('tiny'))
    expect(s3.objects.size).toBe(0)
    expect(str(await storage.read('sessions/s1/small'))).toBe('tiny')
  })
})

describe('DynamoDBStorage — namespace + prefix', () => {
  it('applies a constructor prefix transparently to callers', async () => {
    const { storage } = newStorage({ prefix: 'tenant-x' })
    await storage.write('sessions/s1/a', bytes('v'))
    expect(str(await storage.read('sessions/s1/a'))).toBe('v')
    expect((await storage.list('sessions/s1/')).sort()).toEqual(['sessions/s1/a'])
  })

  it('namespace() returns a prefixed view that round-trips and lists', async () => {
    const { storage } = newStorage()
    const view = storage.namespace!('sessions/s1')
    await view.write('scopes/agent/a1/x', bytes('v'))
    expect(str(await view.read('scopes/agent/a1/x'))).toBe('v')
    // Underlying key includes the namespace prefix.
    expect(str(await storage.read('sessions/s1/scopes/agent/a1/x'))).toBe('v')
  })
})

describe('DynamoDBStorage — vector search', () => {
  it('stores the embedding inline when a vector is written', async () => {
    const { storage, client } = newStorage({ vector: true })
    await storage.write('memory/u1/m1', bytes('likes window seats'), { vector: [1, 0, 0], metadata: { kind: 'pref' } })
    const item = [...client.items.values()][0]!
    expect(item.vector).toEqual([1, 0, 0])
    expect(item.meta).toEqual({ kind: 'pref' })
    expect(str(await storage.read('memory/u1/m1'))).toBe('likes window seats')
  })

  it('returns nearest neighbours ordered by similarity, capped at topK', async () => {
    const { storage } = newStorage({ vector: true })
    await storage.write('memory/u1/a', bytes('a'), { vector: [1, 0, 0] })
    await storage.write('memory/u1/b', bytes('b'), { vector: [0.9, 0.1, 0] })
    await storage.write('memory/u1/c', bytes('c'), { vector: [0, 1, 0] })
    const results = await storage.search({ vector: [1, 0, 0], topK: 2 })
    expect(results.map((r) => r.key)).toEqual(['memory/u1/a', 'memory/u1/b'])
    expect(results[0]!.score).toBeGreaterThanOrEqual(results[1]!.score)
  })

  it('projects the response floor: keys + metadata, payload attrs only with includeValues', async () => {
    const { storage, client } = newStorage()
    await storage.write('tenant/a/m1', bytes('v'), { vector: [1, 0, 0] })
    await storage.search({ vector: [1, 0, 0], topK: 1 })
    expect(client.lastSearchVectorsInput.ProjectionExpression).toBe('#pk, #sk, #m')
    expect(client.lastSearchVectorsInput.ExpressionAttributeNames).toMatchObject({
      '#pk': 'pk',
      '#sk': 'sk',
      '#m': 'meta',
    })
    await storage.search({ vector: [1, 0, 0], topK: 1, includeValues: true })
    expect(client.lastSearchVectorsInput.ProjectionExpression).toBe('#pk, #sk, #m, #d, #s3, #z')
  })

  it('includeValues reuses the projected payload without a point read', async () => {
    const { storage, client } = newStorage()
    await storage.write('tenant/a/m1', bytes('inline payload'), { vector: [1, 0, 0] })
    client.getCalls = 0
    const results = await storage.search({ vector: [1, 0, 0], topK: 1, includeValues: true })
    expect(new TextDecoder().decode(results[0]!.data!)).toBe('inline payload')
    expect(client.getCalls).toBe(0)
  })

  it('includeValues decodes a gzip-compressed projected payload', async () => {
    const { storage, client } = newStorage({ compression: 'gzip' })
    const compressible = 'a'.repeat(2048)
    await storage.write('tenant/a/m1', bytes(compressible), { vector: [1, 0, 0] })
    client.getCalls = 0
    const results = await storage.search({ vector: [1, 0, 0], topK: 1, includeValues: true })
    expect(new TextDecoder().decode(results[0]!.data!)).toBe(compressible)
    expect(client.getCalls).toBe(0)
  })

  it('includeValues falls back to a point read for an S3-offloaded payload', async () => {
    const { storage, client } = newStorage({ s3: true })
    const big = new Uint8Array(380_001).fill(65)
    await storage.write('tenant/a/big', big, { vector: [1, 0, 0] })
    client.getCalls = 0
    const results = await storage.search({ vector: [1, 0, 0], topK: 1, includeValues: true })
    expect(results[0]!.data!.byteLength).toBe(380_001)
    expect(client.getCalls).toBe(1)
  })

  it('hydrates stored bytes when includeValues is set', async () => {
    const { storage } = newStorage({ vector: true })
    await storage.write('memory/u1/a', bytes('remember me'), { vector: [1, 0, 0] })
    const results = await storage.search({ vector: [1, 0, 0], topK: 1, includeValues: true })
    expect(str(results[0]!.data ?? null)).toBe('remember me')
  })

  it('carries metadata through results', async () => {
    const { storage } = newStorage({ vector: true })
    await storage.write('memory/u1/a', bytes('a'), { vector: [1, 0, 0], metadata: { source: 'profile' } })
    const results = await storage.search({ vector: [1, 0, 0], topK: 1 })
    expect(results[0]!.metadata).toEqual({ source: 'profile' })
  })

  it('issues SearchVectors natively when no adapter is configured', async () => {
    const { storage, client } = newStorage({ prefix: 'tenant/a/' })
    await storage.write('memories/m1', bytes('a'), { vector: [1, 0, 0], metadata: { source: 'profile' } })
    await storage.write('memories/m2', bytes('b'), { vector: [0, 1, 0] })
    const results = await storage.search({ vector: [1, 0, 0], topK: 2, pk: 'tenant/a' })
    expect(results[0]?.key).toBe('memories/m1') // nearest first, prefix stripped
    expect(results[0]?.metadata).toEqual({ source: 'profile' })
    expect(client.lastSearchVectorsInput.SearchConditionExpression).toBe('#pk = :pk')
    expect(client.lastSearchVectorsInput.ExpressionAttributeValues).toEqual({ ':pk': { S: 'tenant/a' } })
    expect(client.lastSearchVectorsInput.TopK).toBe(2)
  })

  it('native path enforces the SearchVectors TopK bounds', async () => {
    const { storage } = newStorage()
    await expect(storage.search({ vector: [1, 0, 0], topK: 101 })).rejects.toThrow(/topK/)
    await expect(storage.search({ vector: [1, 0, 0], topK: 0 })).rejects.toThrow(/topK/)
  })

  it('write rejects non-finite vectors with the same message as search', async () => {
    const { storage, client } = newStorage()
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      await expect(storage.write('sessions/s1/m', bytes('v'), { vector: [bad, 0] })).rejects.toThrow(/non-finite/)
    }
    expect(client.items.size).toBe(0)
  })

  it('native path rejects non-finite query vectors', async () => {
    const { storage } = newStorage()
    await expect(storage.search({ vector: [Number.NaN, 0], topK: 1 })).rejects.toThrow(/non-finite/)
  })

  it('native path over-fetches for filters and post-filters to topK', async () => {
    const { storage, client } = newStorage({ prefix: 'tenant/a/' })
    await storage.write('m1', bytes('a'), { vector: [1, 0, 0], metadata: { source: 'web' } })
    await storage.write('m2', bytes('b'), { vector: [1, 0.1, 0], metadata: { source: 'profile' } })
    const results = await storage.search({ vector: [1, 0, 0], topK: 1, pk: 'tenant/a', filter: { source: 'profile' } })
    expect(client.lastSearchVectorsInput.TopK).toBe(10) // topK * over-fetch factor
    expect(results.map((r) => r.key)).toEqual(['m2'])
  })

  it('native path drops matches from outside the namespace', async () => {
    const { storage, client } = newStorage({ prefix: 'tenant/a/' })
    // Seed a foreign-tenant item directly into the fake table.
    client.items.set('tenant/b\u0000m1', { pk: 'tenant/b', sk: 'm1', vector: [1, 0, 0] })
    const results = await storage.search({ vector: [1, 0, 0], topK: 1 })
    expect(results).toEqual([])
  })
})

describe('DynamoDBStorage — compression', () => {
  it('round-trips a compressible value and stores it gzipped inline', async () => {
    const { storage, client } = newStorage({ compression: 'gzip' })
    const original = 'A'.repeat(5000)
    await storage.write('sessions/s1/big', bytes(original))
    const item = [...client.items.values()][0]!
    expect(item.z).toBe(true)
    expect((item.data as Uint8Array).byteLength).toBeLessThan(5000)
    expect(str(await storage.read('sessions/s1/big'))).toBe(original)
  })

  it('keeps a large-but-compressible value inline instead of offloading to S3', async () => {
    // 800 KB of repeated bytes: over the 380 KB inline limit raw, but gzips far under it.
    const { storage, s3, client } = newStorage({ compression: 'gzip', s3: true })
    const original = 'A'.repeat(800_000)
    await storage.write('sessions/s1/big', bytes(original))
    expect(s3.objects.size).toBe(0) // no offload — the cost win
    const item = [...client.items.values()][0]!
    expect(item.z).toBe(true)
    expect(item.s3).toBeUndefined()
    expect(str(await storage.read('sessions/s1/big'))).toBe(original)
  })

  it('stores uncompressed when compression does not shrink the value', async () => {
    const { storage, client } = newStorage({ compression: 'gzip' })
    await storage.write('sessions/s1/tiny', bytes('hi')) // gzip overhead exceeds 2 bytes
    const item = [...client.items.values()][0]!
    expect(item.z).toBeUndefined()
    expect(str(await storage.read('sessions/s1/tiny'))).toBe('hi')
  })

  it('reads a compressed item even when the reader has compression disabled', async () => {
    const { client } = newStorage({ compression: 'gzip' })
    const writer = new DynamoDBStorage('test-table', { client: client as any, compression: 'gzip' })
    await writer.write('sessions/s1/a', bytes('A'.repeat(2000)))
    const reader = new DynamoDBStorage('test-table', { client: client as any }) // compression off
    expect(str(await reader.read('sessions/s1/a'))).toBe('A'.repeat(2000))
  })

  it('offloads to S3 when even the compressed value exceeds the item limit', async () => {
    // High-entropy payload: gzip cannot shrink it below the threshold, so it offloads.
    const { storage, s3, client } = newStorage({ compression: 'gzip', s3: true })
    const original = randomBytes(400_001)
    await storage.write('sessions/s1/big', original)
    expect(s3.objects.size).toBe(1)
    const item = [...client.items.values()][0]!
    expect(item.s3).toBe(true)
    expect(item.data).toBeUndefined()
    expect(await storage.read('sessions/s1/big')).toEqual(original)
  })
})

describe('DynamoDBStorage — TTL', () => {
  it('stamps an epoch-seconds expiry when ttlSeconds is configured', async () => {
    const { storage, client } = newStorage({ ttlSeconds: 3600 })
    const before = Math.floor(Date.now() / 1000)
    await storage.write('sessions/s1/a', bytes('v'))
    const item = [...client.items.values()][0]!
    expect(typeof item.expireAt).toBe('number')
    expect(item.expireAt).toBeGreaterThanOrEqual(before + 3600)
    expect(item.expireAt).toBeLessThanOrEqual(Math.floor(Date.now() / 1000) + 3600 + 2)
  })

  it('writes no TTL attribute when ttlSeconds is not configured', async () => {
    const { storage, client } = newStorage()
    await storage.write('sessions/s1/a', bytes('v'))
    const item = [...client.items.values()][0]!
    expect(item.expireAt).toBeUndefined()
  })

  it('floors a fractional ttlSeconds to an integer stamp', async () => {
    const { storage, client } = newStorage({ ttlSeconds: 3600 })
    const before = Math.floor(Date.now() / 1000)
    await storage.write('sessions/s1/a', bytes('v'), { ttlSeconds: 90.5 })
    const item = [...client.items.values()][0]!
    expect(Number.isInteger(item.expireAt)).toBe(true)
    expect(item.expireAt).toBeGreaterThanOrEqual(before + 90)
  })

  it('lets a per-write ttlSeconds override the instance default', async () => {
    const { storage, client } = newStorage({ ttlSeconds: 60 })
    const before = Math.floor(Date.now() / 1000)
    await storage.write('sessions/s1/a', bytes('v'), { ttlSeconds: 7200 })
    const item = [...client.items.values()][0]!
    expect(item.expireAt).toBeGreaterThanOrEqual(before + 7200)
  })

  it('honors a custom ttlAttribute name', async () => {
    const { storage, client } = newStorage({ ttlSeconds: 60, ttlAttribute: 'ttl' })
    await storage.write('sessions/s1/a', bytes('v'))
    const item = [...client.items.values()][0]!
    expect(typeof item.ttl).toBe('number')
    expect(item.expireAt).toBeUndefined()
  })

  it('stamps TTL on the pointer item of an S3-offloaded value', async () => {
    const { storage, client } = newStorage({ ttlSeconds: 60, s3: true })
    await storage.write('sessions/s1/big', bytes('Z'.repeat(400_001)))
    const item = [...client.items.values()][0]!
    expect(item.s3).toBe(true)
    expect(typeof item.expireAt).toBe('number')
  })
})

describe('DynamoDBStorage — TTL read filtering (opt-in)', () => {
  it('read returns null for an item whose expiry has passed', async () => {
    const { storage } = newStorage({ ttlSeconds: 3600 })
    await storage.write('sessions/s1/a', bytes('v'), { ttlSeconds: -1 }) // already expired
    expect(await storage.read('sessions/s1/a')).toBeNull()
  })

  it('list omits expired items and keeps live ones', async () => {
    const { storage } = newStorage({ ttlSeconds: 3600 })
    await storage.write('sessions/s1/live', bytes('L'))
    await storage.write('sessions/s1/dead', bytes('D'), { ttlSeconds: -1 })
    expect(await storage.list({ pk: 'sessions/s1' })).toEqual(['sessions/s1/live'])
  })

  it('does NOT filter when TTL is not enabled (opt-in): a non-TTL reader returns the item', async () => {
    // Written by a TTL-enabled instance with a past expiry...
    const { client } = newStorage({ ttlSeconds: 3600 })
    const writer = new DynamoDBStorage('test-table', { client: client as any, ttlSeconds: 3600 })
    await writer.write('sessions/s1/a', bytes('v'), { ttlSeconds: -1 })
    // ...but a reader that did not opt into TTL injects no filter and returns it.
    const reader = new DynamoDBStorage('test-table', { client: client as any })
    expect(str(await reader.read('sessions/s1/a'))).toBe('v')
    expect(await reader.list({ pk: 'sessions/s1' })).toEqual(['sessions/s1/a'])
  })

  it('does not stamp an expiry when TTL is not enabled, even with a per-write ttlSeconds', async () => {
    const { storage, client } = newStorage() // TTL not enabled
    await storage.write('sessions/s1/a', bytes('v'), { ttlSeconds: 60 })
    const item = [...client.items.values()][0]!
    expect(item.expireAt).toBeUndefined()
  })
})

describe('DynamoDBStorage — namespace inherits compression + TTL', () => {
  it('applies compression and TTL to items written through a namespaced view', async () => {
    const { storage, client } = newStorage({ compression: 'gzip', ttlSeconds: 3600 })
    const view = storage.namespace('sessions/s1')
    await view.write('scopes/agent/a1/x', bytes('A'.repeat(2000)))
    const item = [...client.items.values()][0]!
    expect(item.z).toBe(true)
    expect(typeof item.expireAt).toBe('number')
    expect(item.k).toBe('sessions/s1/scopes/agent/a1/x')
    expect(str(await view.read('scopes/agent/a1/x'))).toBe('A'.repeat(2000))
  })
})

describe('DynamoDBStorage — construction + error wrapping', () => {
  // A Document client whose every send rejects, to drive the catch/wrap branches.
  const boom = {
    send: async () => {
      throw new Error('boom')
    },
  } as any

  it('rejects being configured with both client and region', () => {
    expect(() => new DynamoDBStorage('t', { client: boom, region: 'us-east-1' })).toThrow(/both client and region/)
  })

  it('wraps a write failure as StorageError preserving the cause', async () => {
    const s = new DynamoDBStorage('t', { client: boom })
    await expect(s.write('sessions/s1/a', bytes('x'))).rejects.toBeInstanceOf(StorageError)
    const err = await s.write('sessions/s1/a', bytes('x')).catch((e) => e)
    expect(err).toBeInstanceOf(StorageError)
    expect((err as Error).cause).toBeInstanceOf(Error)
  })

  it('wraps read, delete, and list failures as StorageError', async () => {
    const s = new DynamoDBStorage('t', { client: boom })
    await expect(s.read('sessions/s1/a')).rejects.toBeInstanceOf(StorageError)
    await expect(s.delete('sessions/s1/a')).rejects.toBeInstanceOf(StorageError)
    await expect(s.list({ pk: 'sessions/s1' })).rejects.toBeInstanceOf(StorageError)
  })

  it('wraps a vectorSearch adapter failure as StorageError', async () => {
    const s = new DynamoDBStorage('t', {
      client: boom,
      vectorSearch: async () => {
        throw new Error('adapter boom')
      },
    })
    await expect(s.search({ vector: [1, 0, 0], topK: 1 })).rejects.toBeInstanceOf(StorageError)
  })

  it('rejects a list prefix containing a ".." segment', async () => {
    const s = new DynamoDBStorage('t', { client: boom })
    await expect(s.list('sessions/../etc')).rejects.toThrow(/'\.\.'/)
  })

  it('namespace() on a region-configured instance returns a composable view', () => {
    const s = new DynamoDBStorage('t', { region: 'us-east-1' })
    const view = s.namespace('sessions/s1')
    expect(view).toBeInstanceOf(DynamoDBStorage)
    expect(view.namespace('scopes/agent/a1')).toBeInstanceOf(DynamoDBStorage)
  })

  it('is re-exported from the package index barrel', () => {
    expect(pkgIndex.DynamoDBStorage).toBe(DynamoDBStorage)
  })
})

describe('DynamoDBStorage — namespace isolation', () => {
  /** Two namespaced instances over one shared table. */
  function twoTenants(vectorSearch?: VectorSearchAdapter) {
    const client = new FakeDocumentClient()
    const opts = (prefix: string) => ({
      client: client as any,
      prefix,
      ...(vectorSearch ? { vectorSearch } : {}),
    })
    return {
      client,
      a: new DynamoDBStorage('test-table', opts('tenant-a')),
      b: new DynamoDBStorage('test-table', opts('tenant-b')),
    }
  }

  it('rejects a structured query naming another namespace partition', async () => {
    const { a, b } = twoTenants()
    await a.write('sessions/s1/secret', bytes('a-data'))
    await expect(b.list({ pk: 'tenant-a/sessions' })).rejects.toThrow(StorageError)
    await expect(b.list({ pk: 'tenant-a/sessions' })).rejects.toThrow(/outside this storage namespace/)
    // the owning instance can still query its own partition
    expect(await a.list({ pk: 'tenant-a/sessions' })).toEqual(['sessions/s1/secret'])
  })

  it('rejects a partition that only shares a string prefix with the namespace', async () => {
    const { b } = twoTenants()
    await expect(b.list({ pk: 'tenant' })).rejects.toThrow(/outside this storage namespace/)
  })

  it('leaves an unprefixed instance free to name any partition', async () => {
    const { storage } = newStorage()
    await storage.write('sessions/s1/a', bytes('1'))
    expect(await storage.list({ pk: 'sessions/s1' })).toEqual(['sessions/s1/a'])
  })

  it('allows a namespaced view to query the ancestor partition it lives in', async () => {
    const { storage } = newStorage()
    const view = storage.namespace!('sessions/s1')
    await view.write('scopes/agent/a1/x', bytes('v'))
    await storage.write('sessions/s2/other', bytes('sibling'))
    expect(await view.list({ pk: 'sessions/s1' })).toEqual(['scopes/agent/a1/x'])
  })

  it('rejects a search pk outside the namespace before calling the adapter', async () => {
    const adapter: VectorSearchAdapter = async () => {
      throw new Error('adapter must not be reached for an out-of-scope pk')
    }
    const { b } = twoTenants(adapter)
    await expect(b.search!({ vector: [1, 0], topK: 1, pk: 'tenant-a/sessions' })).rejects.toThrow(
      /outside this storage namespace/
    )
  })

  it('drops search matches that fall outside the namespace', async () => {
    const leaky: VectorSearchAdapter = async () => [
      { key: 'tenant-a/sessions/s1/secret', score: 0.99 },
      { key: 'tenant-b/sessions/s1/mine', score: 0.98 },
    ]
    const { b } = twoTenants(leaky)
    const results = await b.search!({ vector: [1, 0], topK: 2 })
    expect(results.map((r) => r.key)).toEqual(['sessions/s1/mine'])
  })

  it('drops listed rows whose stored key belongs to another namespace', async () => {
    const { client, b } = twoTenants()
    await b.write('sessions/s1/mine', bytes('ok'))
    // Plant a row inside our partition whose stored key attribute points elsewhere.
    client.items.set('tenant-b/sessions\u0000s1/planted', {
      pk: 'tenant-b/sessions',
      sk: 's1/planted',
      k: 'tenant-a/sessions/s1/secret',
      data: bytes('x'),
    })
    expect(await b.list({ pk: 'tenant-b/sessions' })).toEqual(['sessions/s1/mine'])
  })
})
