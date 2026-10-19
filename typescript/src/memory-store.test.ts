// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SearchVectorsCommand, type SearchVectorsCommandInput } from '@aws-sdk/client-dynamodb'
import { GetCommand, PutCommand, DeleteCommand, type DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb'
import { marshall } from '@aws-sdk/util-dynamodb'
import { MemoryManager, StorageError } from '@strands-agents/sdk'
import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { DynamoDBStorage } from './dynamodb-storage.js'
import { DynamoDBMemoryStore } from './memory-store.js'
import { BedrockEmbedder, type EmbedFunction, type Embedder, type EmbeddingPurpose } from './embedders.js'
import type { BedrockModelFamily, DynamoDBMemoryStoreConfig } from './index.js'
import * as pkgIndex from './index.js'

interface StoredItem {
  pk: string
  sk: string
  k?: string
  vector?: number[]
  [attribute: string]: unknown
}

/**
 * In-memory Document client covering Put/Get/Delete and SearchVectors with COSINE
 * distance, partition pinning, TopK, and ProjectionExpression the way the service applies them.
 */
class FakeDocumentClient {
  readonly items = new Map<string, StoredItem>()
  searches: SearchVectorsCommandInput[] = []
  getCalls = 0
  async send(command: unknown): Promise<unknown> {
    if (command instanceof SearchVectorsCommand) {
      const input = command.input
      this.searches.push(input)
      const query = input.SearchVector!.map((av) => Number(av.N))
      const pk = input.ExpressionAttributeValues?.[':pk']?.S
      const wanted = new Set(
        (input.ProjectionExpression ?? '').split(',').map((alias) => input.ExpressionAttributeNames![alias.trim()]!)
      )
      const results = [...this.items.values()]
        .filter((item): item is StoredItem & { vector: number[] } => Array.isArray(item.vector))
        .filter((item) => pk === undefined || item.pk === pk)
        .map((item) => {
          const dot = item.vector.reduce((acc, x, i) => acc + x * (query[i] ?? 0), 0)
          const norm =
            Math.sqrt(item.vector.reduce((a, x) => a + x * x, 0)) * Math.sqrt(query.reduce((a, x) => a + x * x, 0))
          const projected = Object.fromEntries(Object.entries(item).filter(([k]) => wanted.has(k)))
          return { Item: marshall(projected), Score: 1 - (norm === 0 ? 0 : dot / norm) }
        })
        .sort((a, b) => a.Score - b.Score)
        .slice(0, input.TopK)
      return { SearchResults: results }
    }
    if (command instanceof PutCommand) {
      const item = command.input.Item as StoredItem
      marshall(item) // throws where the real Document client would, before sending
      this.items.set(`${item.pk}\u0000${item.sk}`, item)
      return {}
    }
    if (command instanceof GetCommand) {
      this.getCalls += 1
      const k = command.input.Key as { pk: string; sk: string }
      return { Item: this.items.get(`${k.pk}\u0000${k.sk}`) }
    }
    if (command instanceof DeleteCommand) {
      const k = command.input.Key as { pk: string; sk: string }
      this.items.delete(`${k.pk}\u0000${k.sk}`)
      return {}
    }
    throw new Error(`unexpected command ${(command as object | undefined)?.constructor?.name}`)
  }
}

/** Deterministic embedder: one dimension per keyword, recording every call. */
class KeywordEmbedder implements Embedder {
  static readonly WORDS = ['window', 'seat', 'aisle', 'peanut', 'allergy', 'coffee']
  readonly calls: Array<[string, EmbeddingPurpose]> = []
  async embed(text: string, purpose: EmbeddingPurpose): Promise<number[]> {
    this.calls.push([text, purpose])
    const lowered = text.toLowerCase()
    return [...KeywordEmbedder.WORDS.map((w) => (lowered.includes(w) ? 1 : 0)), 0.01]
  }
}

const asDocClient = (fake: FakeDocumentClient) => fake as unknown as DynamoDBDocumentClient
const itemId = (content: string) => createHash('sha256').update(content, 'utf8').digest('hex').slice(0, 32)

function setup(opts?: { prefix?: string | null; config?: ConstructorParameters<typeof DynamoDBMemoryStore>[1] }) {
  const client = new FakeDocumentClient()
  const storage = new DynamoDBStorage('memory-table', { client: asDocClient(client) })
  const prefix = opts?.prefix === undefined ? 'user/u1' : opts.prefix
  const scoped = prefix ? storage.namespace(prefix) : storage
  const embedder = new KeywordEmbedder()
  const store = new DynamoDBMemoryStore(scoped, { embedder, ...opts?.config })
  return { client, storage: scoped, root: storage, store, embedder }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('DynamoDBMemoryStore: MemoryStore protocol', () => {
  it('exposes the protocol fields', () => {
    const { storage } = setup()
    const store = new DynamoDBMemoryStore(storage, {
      embedder: new KeywordEmbedder(),
      name: 'prefs',
      description: 'User preferences',
      maxSearchResults: 4,
    })
    expect(store.name).toBe('prefs')
    expect(store.description).toBe('User preferences')
    expect(store.maxSearchResults).toBe(4)
    expect(store.writable).toBe(true)
    expect(store.extraction).toBeUndefined()
  })

  it('keeps the MemoryManager from writing when writable is false', async () => {
    const { storage } = setup()
    const readOnlyEmbedder = new KeywordEmbedder()
    const writableEmbedder = new KeywordEmbedder()
    const readOnly = new DynamoDBMemoryStore(storage, {
      embedder: readOnlyEmbedder,
      name: 'read-only',
      writable: false,
    })
    const writable = new DynamoDBMemoryStore(storage, { embedder: writableEmbedder, name: 'notes' })
    expect(readOnly.writable).toBe(false)
    await new MemoryManager({ stores: [readOnly, writable], injection: false }).add('Prefers a window seat')
    expect(readOnlyEmbedder.calls).toEqual([])
    expect(writableEmbedder.calls.map(([, purpose]) => purpose)).toEqual(['document'])
  })

  it('takes a null description to mean no description', () => {
    const { storage } = setup()
    expect(new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), description: null }).description).toBe(
      undefined
    )
    expect(new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder() }).description).toBe(
      'Long-term memories stored in Amazon DynamoDB, searched by meaning'
    )
  })

  it.each([
    ['description', 5],
    ['writable', null],
    ['writable', 'yes'],
  ])('rejects a %s of %j', (option, value) => {
    const { storage } = setup()
    const config = { embedder: new KeywordEmbedder(), [option]: value } as unknown as DynamoDBMemoryStoreConfig
    expect(() => new DynamoDBMemoryStore(storage, config)).toThrow(`${option} must be`)
  })

  it('works inside a MemoryManager', async () => {
    const { storage } = setup()
    const store = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: 'prefs' })
    const manager = new MemoryManager({ stores: [store], injection: false })
    await manager.add('Prefers a window seat')
    await manager.add('Has a peanut allergy')
    const entries = await manager.search('which seat does the user like?')
    expect(entries[0]?.content).toBe('Prefers a window seat')
    expect(entries[0]?.storeName).toBe('prefs')
  })

  it('is accepted by a MemoryManager with extraction enabled', () => {
    const { storage } = setup()
    const store = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), extraction: true })
    expect(store.extraction).toBe(true) // the MemoryManager reads it to decide whether to extract
    expect(() => new MemoryManager({ stores: [store], injection: false })).not.toThrow()
  })

  it('is re-exported from the package index barrel', () => {
    expect(pkgIndex.DynamoDBMemoryStore).toBe(DynamoDBMemoryStore)
    expect(pkgIndex.BedrockEmbedder).toBe(BedrockEmbedder)
    expect(pkgIndex.DEFAULT_EMBEDDING_MODEL_ID).toBe('amazon.titan-embed-text-v2:0')
    // Type-only exports: this file fails to type-check if the index stops exporting them.
    const family: BedrockModelFamily = 'cohere.embed-v4'
    const embedder = new pkgIndex.BedrockEmbedder({ modelId: 'cohere.embed-v4:0', modelFamily: family })
    expect(embedder.modelId).toBe('cohere.embed-v4:0')
  })
})

describe('DynamoDBMemoryStore: add', () => {
  it('writes one item with the text, vector, and metadata', async () => {
    const { client, store, embedder } = setup({ config: { name: 'prefs' } })
    const key = await store.add('Prefers a window seat', { source: 'chat', turn: 3, confirmed: true })
    expect(key).toBe(`memory/prefs/${itemId('Prefers a window seat')}`)
    const item = client.items.get(`user/u1\u0000${key}`)!
    expect(item.k).toBe(`user/u1/${key}`)
    expect(new TextDecoder().decode(item.data as Uint8Array)).toBe('Prefers a window seat')
    expect(item.vector).toEqual(await new KeywordEmbedder().embed('Prefers a window seat', 'document'))
    expect(item.meta).toEqual({ source: 'chat', turn: 3, confirmed: true })
    expect(embedder.calls).toEqual([['Prefers a window seat', 'document']])
  })

  it('overwrites instead of duplicating when the same text is added again', async () => {
    const { client, store } = setup()
    const first = await store.add('Prefers a window seat', { source: 'chat' })
    const second = await store.add('Prefers a window seat', { source: 'profile' })
    expect(second).toBe(first)
    expect(client.items.size).toBe(1)
    expect([...client.items.values()][0]!.meta).toEqual({ source: 'profile' })
  })

  it('writes no metadata attribute and warns about nothing when there is none', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { client, store } = setup()
    await store.add('Prefers a window seat')
    await store.add('Likes coffee', {})
    await store.add('Has a peanut allergy', null as never)
    expect(client.items.size).toBe(3)
    for (const item of client.items.values()) expect(item.meta).toBeUndefined()
    expect(warn).not.toHaveBeenCalled()
  })

  it.each([
    [{ _key: 'x' }, 'is reserved for search results'],
    [{ '': 'x' }, 'is empty, which DynamoDB rejects'],
    [{ s: 'a\udcffb' }, 'contains a lone surrogate'],
    [{ k: '\udcffy' }, 'contains a lone surrogate'],
    [{ ['a\ud800']: 'v' }, 'contains a lone surrogate'],
    [{ ['k\udfff']: 'v' }, 'contains a lone surrogate'],
    [{ v: 'x\udfff' }, 'contains a lone surrogate'],
    [{ tags: ['a'] }, 'has type array'],
    [{ none: null }, 'has type null'],
    [{ n: 2 ** 53 }, 'is out of range'],
  ])('warns why it dropped the metadata entry %j', async (metadata, reason) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup()
    await store.add('Prefers a window seat', metadata as never)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining(reason))
  })

  it('names the store and the key in the drop warning, as Python does', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { _key: 'x' } as never)
    expect(warn).toHaveBeenCalledWith(
      'store=<prefs>, key=<_key> | dropping metadata entry: is reserved for search results'
    )
  })

  it('round-trips non-ASCII text', async () => {
    const { store } = setup()
    await store.add('Préfère la fenêtre: window seat 窓側')
    const [entry] = await store.search('window')
    expect(entry?.content).toBe('Préfère la fenêtre: window seat 窓側')
  })

  it('stamps TTL when the storage has TTL', async () => {
    const client = new FakeDocumentClient()
    const storage = new DynamoDBStorage('t', { client: asDocClient(client), ttlSeconds: 3600 }).namespace('user/u1')
    await new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder() }).add('Prefers a window seat')
    expect(typeof [...client.items.values()][0]!.expireAt).toBe('number')
  })

  it.each([
    ['', RangeError],
    ['   \n', RangeError],
    ['\ufeff', RangeError],
    ['\ufeff ', RangeError],
    [' \ufeff', RangeError],
    ['\u0085', RangeError],
    ['\u001c\u3000', RangeError],
    ['a\ud800', RangeError],
    ['a\udfffb', RangeError],
    [undefined, TypeError],
    [null, TypeError],
    [42, TypeError],
  ])('rejects content %j', async (content, error) => {
    const { store, embedder } = setup()
    const add = store.add(content as unknown as string)
    await expect(add).rejects.toBeInstanceOf(error)
    await expect(add).rejects.toThrow(
      error === TypeError
        ? 'content must be a string'
        : /^content must (not be empty|be valid Unicode text; it contains a lone surrogate)$/
    )
    expect(embedder.calls).toEqual([])
  })

  it.each([
    [{ _key: 'x' }],
    [{ _score: 1 }],
    [{ tags: ['a', 'b'] }],
    [{ nested: { a: 1 } }],
    [{ missing: null }],
    [{ bad: Number.NaN }],
    [{ bad: Number.POSITIVE_INFINITY }],
    [{ huge: 1e200 }],
    [{ unsafe: 2 ** 53 }],
    [{ negUnsafe: -(2 ** 53) }],
    [{ belowMin: 1e-131 }],
    [{ largeFloat: 1e20 }],
    [{ tiny: 1e-140 }],
    [{ '': 'empty key' }],
  ])('stores the memory and drops unsupported metadata %j with a warning', async (dropped) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { client, store, embedder } = setup()
    await store.add('Prefers a window seat', { source: 'chat', ...dropped } as never)
    const [item] = [...client.items.values()]
    expect(item!.meta).toEqual({ source: 'chat' })
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('dropping metadata entry'))
    expect(embedder.calls).toHaveLength(1)
  })

  it('keeps negative zero in metadata', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { client, store } = setup()
    await store.add('Prefers a window seat', { negZero: -0 })
    expect(Object.keys([...client.items.values()][0]!.meta as object)).toEqual(['negZero'])
    expect(warn).not.toHaveBeenCalled()
  })

  it('keeps numbers the AWS SDK and DynamoDB can store', async () => {
    const { client, store } = setup()
    const kept = {
      zero: 0,
      neg: -1.5,
      max: Number.MAX_SAFE_INTEGER,
      minNeg: -Number.MAX_SAFE_INTEGER,
      tiny: 1e-130,
      fraction: 0.25,
    }
    await store.add('Prefers a window seat', kept)
    expect([...client.items.values()][0]!.meta).toEqual(kept)
  })

  it.each([[['a']], ['source=chat'], [42], [new Map([['a', 1]])], [new Date(0)]])(
    'drops metadata %j that is not a plain object',
    async (metadata) => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const { client, store } = setup()
      await store.add('Prefers a window seat', metadata as never)
      expect([...client.items.values()][0]!.meta).toBeUndefined()
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('not a plain object'))
    }
  )

  it('accepts metadata created without a prototype', async () => {
    const { client, store } = setup()
    const metadata = Object.assign(Object.create(null) as Record<string, string>, { source: 'chat' })
    await store.add('Prefers a window seat', metadata)
    expect([...client.items.values()][0]!.meta).toEqual({ source: 'chat' })
  })

  it('stores an extracted memory whose model-emitted metadata is structured', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { client, storage } = setup()
    const store = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: 'prefs' })
    const manager = new MemoryManager({ stores: [store], injection: false })
    await manager.add('Prefers a window seat', { metadata: { tags: ['travel'], confidence: null, source: 'x' } })
    expect([...client.items.values()][0]!.meta).toEqual({ source: 'x' })
  })
})

describe('DynamoDBMemoryStore: search', () => {
  it('returns the closest memories first with key and score, embedding the query as a query', async () => {
    const { store, embedder } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { source: 'chat' })
    await store.add('Has a peanut allergy')
    await store.add('Drinks coffee black')
    const entries = await store.search('window seat please')
    expect(entries[0]?.content).toBe('Prefers a window seat')
    expect(entries[0]?.metadata?.source).toBe('chat')
    expect(entries[0]?.metadata?._key).toBe(`memory/prefs/${itemId('Prefers a window seat')}`)
    const scores = entries.map((e) => e.metadata?._score as number)
    expect(scores).toEqual([...scores].sort((a, b) => a - b))
    expect(embedder.calls.at(-1)).toEqual(['window seat please', 'query'])
  })

  it('pins the partition, projects the values, and makes no point reads', async () => {
    const { client, store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat')
    await store.search('window', { maxSearchResults: 7 })
    const request = client.searches.at(-1)!
    expect(request.TopK).toBe(7)
    expect(request.ExpressionAttributeValues).toEqual({ ':pk': { S: 'user/u1' } })
    expect(request.ProjectionExpression).toContain('#d')
    expect(client.getCalls).toBe(0)
  })

  it('returns the key and the index score in the metadata', async () => {
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { source: 'chat' })
    const [entry] = await store.search('window')
    // KeywordEmbedder: query [1,0,0,0,0,0,0.01], memory [1,1,0,0,0,0,0.01]; COSINE distance.
    const expected = 1 - 1.0001 / Math.sqrt(1.0001 * 2.0001)
    expect(entry!.metadata).toEqual({
      source: 'chat',
      _key: `memory/prefs/${itemId('Prefers a window seat')}`,
      _score: expect.any(Number),
    })
    expect(entry!.metadata!._score).toBeCloseTo(expected, 12)
  })

  it('skips an offloaded match whose item is gone and decodes invalid UTF-8', async () => {
    const { client, store } = setup({ config: { name: 'prefs' } })
    const vector = [1, 0, 0, 0, 0, 0, 0.01]
    // Offloaded to S3 (no inline payload) and deleted between SearchVectors and the point read.
    client.items.set('user/u1\u0000memory/prefs/gone', { pk: 'user/u1', sk: 'memory/prefs/gone', vector, s3: true })
    client.items.set('user/u1\u0000memory/prefs/bin', {
      pk: 'user/u1',
      sk: 'memory/prefs/bin',
      vector,
      data: new Uint8Array([0xff, 0xfe]),
    })
    const send = client.send.bind(client)
    client.send = async (command: unknown) => (command instanceof GetCommand ? {} : send(command))
    const entries = await store.search('window')
    expect(entries.map((e) => [e.content, e.metadata?._key])).toEqual([['\ufffd\ufffd', 'memory/prefs/bin']])
  })

  it('reads no value for a match outside the store; its own still get one', async () => {
    const { client, store } = setup({ config: { name: 'prefs' } })
    const vector = [1, 0, 0, 0, 0, 0, 0.01]
    const put = (sk: string, extra: Record<string, unknown>) =>
      client.items.set(`user/u1\u0000${sk}`, { pk: 'user/u1', sk, vector, ...extra })
    // Another store's offloaded memory, an application item whose payload the index does
    // not return, and a key that contains this store's prefix later on.
    put('memory/facts/big', { s3: true })
    put('notes/n1', { s3: true })
    put('x/memory/prefs/y', { s3: true })
    // A store whose name starts with this one's: "prefs2" is not under "memory/prefs/".
    put('memory/prefs2/sib', { data: new TextEncoder().encode('sibling') })
    put('memory/prefs2/off', { s3: true })
    // This store's own offloaded memory, deleted before the read.
    put('memory/prefs/gone', { s3: true })
    put('memory/prefs/abc', { data: new TextEncoder().encode('Prefers a window seat') })
    const reads: string[] = []
    const send = client.send.bind(client)
    client.send = async (command: unknown) => {
      if (!(command instanceof GetCommand)) return send(command)
      reads.push((command.input.Key as { sk: string }).sk)
      return {}
    }
    const entries = await store.search('window')
    expect(reads).toEqual(['memory/prefs/gone'])
    expect(entries.map((e) => [e.content, e.metadata?._key])).toEqual([['Prefers a window seat', 'memory/prefs/abc']])
  })

  it('reads back content with a byte order mark unchanged', async () => {
    const { store } = setup()
    const content = '\ufeffPrefers a window seat'
    await store.add(content)
    const [entry] = await store.search('window')
    expect(entry!.content).toBe(content)
  })

  it('rejects a lone surrogate before looking at metadata', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store, embedder } = setup()
    await expect(store.add('a\ud800b', { tags: ['dropped'] } as never)).rejects.toThrow(/lone surrogate/)
    expect(warn).not.toHaveBeenCalled()
    expect(embedder.calls).toEqual([])
  })

  it('drops the keys __proto__ and constructor, which marshall cannot write', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { client, store } = setup()
    await store.add('Prefers a window seat', JSON.parse('{"__proto__": "x", "constructor": "y", "ok": "z"}') as never)
    const [item] = [...client.items.values()]
    // What the Document client would send: marshall succeeds and keeps only "ok".
    expect(marshall({ meta: item!.meta })).toEqual({ meta: { M: { ok: { S: 'z' } } } })
    expect(warn).toHaveBeenCalledTimes(2)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('cannot be written by the AWS SDK for JavaScript'))
  })

  it('calls embed when a Proxy resolves it, as Python does for __getattr__', async () => {
    const target = () => {
      throw new Error('called as a function')
    }
    const proxy = new Proxy(target, {
      get: (object, property) =>
        property === 'embed'
          ? (text: string) => (text.includes('window') ? [1, 0] : [0, 1])
          : Reflect.get(object, property),
    })
    const { store } = setup({ config: { embedder: proxy as unknown as EmbedFunction } })
    await store.add('Prefers a window seat')
    await store.add('Has a peanut allergy')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
  })

  it('calls a function whose embed property is not a method, as Python does', async () => {
    const odd = Object.assign(async () => [1, 0], { embed: 5 })
    const { client, store } = setup({ config: { embedder: odd as unknown as EmbedFunction } })
    await store.add('Prefers a window seat')
    expect(client.items.size).toBe(1)
  })

  it('calls embed on a callable that also has an embed method, as Python does', async () => {
    const callable = Object.assign(
      () => {
        throw new Error('called as a function')
      },
      {
        embed: (text: string) => (text.includes('window') ? [1, 0] : [0, 1]),
      }
    )
    const { store } = setup({ config: { embedder: callable } })
    await store.add('Prefers a window seat')
    await store.add('Has a peanut allergy')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
  })

  it('accepts an embedder object whose embed method is typed as sync', async () => {
    const embedder: Embedder = { embed: (text) => (text.includes('window') ? [1, 0] : [0, 1]) }
    const { store } = setup({ config: { embedder } })
    await store.add('Prefers a window seat')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
  })

  it.each([['  Prefers a window seat\n'], ['Cafe\u0301 by the window'], ['\u{1f600} window seat']])(
    'stores and embeds %j exactly as given, with no trimming or Unicode normalization',
    async (content) => {
      const { client, store, embedder } = setup()
      await store.add(content)
      const [item] = [...client.items.values()]
      expect(embedder.calls).toEqual([[content, 'document']])
      expect(new TextDecoder().decode(item!.data as Uint8Array)).toBe(content)
      expect(item!.sk.endsWith(itemId(content))).toBe(true)
    }
  )

  it('shares one partition, and its TopK, across deep sibling namespaces', async () => {
    // tenant/a/user/u1 and tenant/a/user/u2 share partition tenant/a, so u2's nearer memories fill TopK.
    const client = new FakeDocumentClient()
    const root = new DynamoDBStorage('t', { client: asDocClient(client) })
    const store = (ns: string) => new DynamoDBMemoryStore(root.namespace(ns), { embedder: new KeywordEmbedder() })
    const near = ['Window seat', 'Window seat please', 'Always a window seat']
    const u1 = store('tenant/a/user/u1')
    const u2 = store('tenant/a/user/u2')
    await u1.add('Likes coffee by the window')
    for (const memory of near) await u2.add(memory)
    expect(u1['_partition']).toBe('tenant/a')
    expect(u2['_partition']).toBe('tenant/a')
    expect(await u1.search('window seat', { maxSearchResults: 3 })).toEqual([])
    expect((await u1.search('window seat', { maxSearchResults: 4 })).map((e) => e.content)).toEqual([
      'Likes coffee by the window',
    ])
    // Two-segment namespaces get partitions of their own.
    const v1 = store('user/v1')
    const v2 = store('user/v2')
    await v1.add('Likes coffee by the window')
    for (const memory of near) await v2.add(memory)
    expect((await v1.search('window seat', { maxSearchResults: 1 })).map((e) => e.content)).toEqual([
      'Likes coffee by the window',
    ])
  })

  it("keeps only this store's memories when the partition is shared", async () => {
    const client = new FakeDocumentClient()
    const storage = new DynamoDBStorage('t', { client: asDocClient(client) }).namespace('user/u1')
    const prefs = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: 'prefs' })
    const facts = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: 'facts' })
    await prefs.add('Prefers a window seat')
    await facts.add('Window seats are on the left')
    await storage.write('notes/n1', new TextEncoder().encode('app item'), { vector: [1, 1, 0, 0, 0, 0, 0.01] })
    // An application item whose key contains memory/prefs/ other than at the start.
    await storage.write('notes/memory/prefs/x', new TextEncoder().encode('app item'), {
      vector: [1, 1, 0, 0, 0, 0, 0.01],
    })
    expect((await prefs.search('window')).map((e) => e.content)).toEqual(['Prefers a window seat'])
    expect((await facts.search('window')).map((e) => e.content)).toEqual(['Window seats are on the left'])
  })

  it('is scoped to the storage namespace', async () => {
    const client = new FakeDocumentClient()
    const root = new DynamoDBStorage('t', { client: asDocClient(client) })
    const alice = new DynamoDBMemoryStore(root.namespace('user/alice'), { embedder: new KeywordEmbedder() })
    const bob = new DynamoDBMemoryStore(root.namespace('user/bob'), { embedder: new KeywordEmbedder() })
    await alice.add('Prefers a window seat')
    expect(await bob.search('window')).toEqual([])
  })

  it('returns keys that DynamoDBStorage.delete accepts', async () => {
    const { storage, store } = setup()
    await store.add('Prefers a window seat')
    const [entry] = await store.search('window')
    await storage.delete(entry!.metadata!._key as string)
    expect(await store.search('window')).toEqual([])
  })

  it.each([
    [undefined, undefined, 10],
    [{}, undefined, 10],
    [undefined, 4, 4],
    [{ maxSearchResults: null as unknown as number }, 4, 4],
    [{ maxSearchResults: 2 }, 4, 2],
    [{ maxSearchResults: 100 }, undefined, 100],
    [{ maxSearchResults: 500 }, undefined, 100],
  ])('asks SearchVectors for the right TopK (options %j, default %j)', async (options, storeDefault, expected) => {
    const client = new FakeDocumentClient()
    const storage = new DynamoDBStorage('t', { client: asDocClient(client) }).namespace('user/u1')
    const store = new DynamoDBMemoryStore(storage, {
      embedder: new KeywordEmbedder(),
      ...(storeDefault !== undefined ? { maxSearchResults: storeDefault } : {}),
    })
    await store.search('window', options)
    expect(client.searches.at(-1)!.TopK).toBe(expected)
  })

  it.each([[0], [-1], [2.5], [Number.NaN], [true], ['3']])('rejects maxSearchResults %j', async (bad) => {
    const { store, embedder } = setup()
    await expect(store.search('window', { maxSearchResults: bad as number })).rejects.toBeInstanceOf(RangeError)
    expect(embedder.calls).toEqual([])
  })

  it('checks the limit before returning nothing for a blank query', async () => {
    const { store, embedder } = setup()
    await expect(store.search('   ', { maxSearchResults: 0 })).rejects.toBeInstanceOf(RangeError)
    expect(embedder.calls).toEqual([])
  })

  it.each([[''], ['   '], ['\n\t'], ['\ufeff'], ['\u0085'], ['\u001f']])(
    'returns nothing for the empty query %j without calls',
    async (query) => {
      const { client, store, embedder } = setup()
      expect(await store.search(query)).toEqual([])
      expect(embedder.calls).toEqual([])
      expect(client.searches).toEqual([])
    }
  )

  it.each([[undefined], [null], [3]])('rejects the non-string query %j', async (query) => {
    const { store, embedder } = setup()
    const search = store.search(query as unknown as string)
    await expect(search).rejects.toBeInstanceOf(TypeError)
    await expect(search).rejects.toThrow('query must be a string')
    expect(embedder.calls).toEqual([])
  })

  it('raises StorageError when the search fails', async () => {
    const failing = {
      send: async () => {
        throw new Error('boom')
      },
    } as unknown as DynamoDBDocumentClient
    const storage = new DynamoDBStorage('t', { client: failing }).namespace('user/u1')
    const store = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder() })
    await expect(store.search('window')).rejects.toBeInstanceOf(StorageError)
  })
})

describe('DynamoDBMemoryStore: embedders', () => {
  it('reports the real key and score over stored reserved metadata', async () => {
    const { storage, store } = setup({ config: { name: 'prefs' } })
    const key = `memory/prefs/${itemId('Prefers a window seat')}`
    const vector = await new KeywordEmbedder().embed('Prefers a window seat', 'document')
    // Written directly, bypassing the store's metadata filter.
    await storage.write(key, new TextEncoder().encode('Prefers a window seat'), {
      vector,
      metadata: { _key: 'forged', _score: 9 },
    })
    const [entry] = await store.search('window')
    expect(entry!.metadata!._key).toBe(key)
    expect(entry!.metadata!._score).not.toBe(9)
  })

  it('accepts a sync embed function', async () => {
    const seen: string[] = []
    const embed = (text: string) => {
      seen.push(text)
      return text.includes('window') ? [1, 0] : [0, 1]
    }
    const { store } = setup({ config: { embedder: embed } })
    await store.add('Prefers a window seat')
    await store.add('Has a peanut allergy')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
    expect(seen).toEqual(['Prefers a window seat', 'Has a peanut allergy', 'window'])
  })

  it('accepts an async embed function', async () => {
    const embed = async (text: string) => (text.includes('window') ? [1, 0] : [0, 1])
    const { store } = setup({ config: { embedder: embed } })
    await store.add('Prefers a window seat')
    await store.add('Has a peanut allergy')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
  })

  // eslint-disable-next-line no-sparse-arrays
  const sparse = [1, , 3]
  it.each([
    [[]],
    [null],
    ['vector'],
    [[1, 'x']],
    [[1, Number.NaN]],
    [[true, false]],
    [{ a: 1 }],
    [sparse],
    [new Float32Array([1, 2])],
  ])('rejects an embedder that returns %j', async (bad) => {
    const { client, store } = setup({ config: { embedder: () => bad as unknown as number[] } })
    await expect(store.add('Prefers a window seat')).rejects.toThrow(/finite numbers/)
    await expect(store.search('window')).rejects.toThrow(/finite numbers/)
    expect(client.items.size).toBe(0)
  })

  it('names a thrown value that is not an Error in the wrapped message', async () => {
    const embed = () => {
      throw 'boom'
    }
    const { store } = setup({ config: { name: 'prefs', embedder: embed } })
    await expect(store.add('Prefers a window seat')).rejects.toThrow("Embedding failed in memory store 'prefs': boom")
  })

  it('wraps embedder failures in StorageError naming the cause', async () => {
    const failure = new Error('model unavailable')
    const { client, store } = setup({
      config: {
        name: 'prefs',
        embedder: () => {
          throw failure
        },
      },
    })
    const error = await store.add('Prefers a window seat').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(StorageError)
    expect((error as Error).cause).toBe(failure)
    expect((error as Error).message).toBe("Embedding failed in memory store 'prefs': Error: model unavailable")
    expect(client.items.size).toBe(0)
  })

  it('accepts an embedder object with a sync embed method', async () => {
    const embedder = { embed: (text: string) => (text.includes('window') ? [1, 0] : [0, 1]) }
    const { store } = setup({ config: { embedder: embedder as unknown as Embedder } })
    await store.add('Prefers a window seat')
    await store.add('Has a peanut allergy')
    expect((await store.search('window'))[0]?.content).toBe('Prefers a window seat')
  })

  it('passes a StorageError from the embedder through', async () => {
    const failure = new StorageError('bedrock throttled')
    const { store } = setup({
      config: {
        embedder: {
          embed: async () => {
            throw failure
          },
        },
      },
    })
    await expect(store.search('window')).rejects.toBe(failure)
  })

  it('defaults to Titan Text Embeddings V2 in the storage region', () => {
    const store = new DynamoDBMemoryStore(new DynamoDBStorage('t', { region: 'eu-west-1' }))
    const embedder = store['_embedder'] as BedrockEmbedder
    expect(embedder).toBeInstanceOf(BedrockEmbedder)
    expect(embedder.modelId).toBe('amazon.titan-embed-text-v2:0')
    expect(embedder.dimensions).toBe(1024)
    expect(embedder['_region']).toBe('eu-west-1')
  })

  it("follows an injected client's region", async () => {
    const { DynamoDBClient } = await import('@aws-sdk/client-dynamodb')
    const { DynamoDBDocumentClient } = await import('@aws-sdk/lib-dynamodb')
    const client = DynamoDBDocumentClient.from(new DynamoDBClient({ region: 'ap-southeast-2' }))
    const store = new DynamoDBMemoryStore(new DynamoDBStorage('t', { client }).namespace('user/u1'))
    const region = (store['_embedder'] as BedrockEmbedder)['_region']
    expect(typeof region).toBe('function')
    expect(await (region as () => Promise<string>)()).toBe('ap-southeast-2')
  })
})

describe('DynamoDBMemoryStore: construction', () => {
  it.each([
    [null, 'memory/dynamodb'],
    ['u1', 'u1/memory'],
    ['user/u1', 'user/u1'],
    ['tenant/a/user/u1', 'tenant/a'],
  ])('keeps all memories in one partition under namespace %j', (prefix, partition) => {
    const { storage, store } = setup({ prefix })
    expect(store['_partition']).toBe(partition)
    expect(storage.partitionKey(`memory/dynamodb/${itemId('a')}`)).toBe(partition)
    expect(storage.partitionKey(`memory/dynamodb/${itemId('b')}`)).toBe(partition)
  })

  it.each([
    ['', RangeError],
    ['  ', RangeError],
    ['a/b', RangeError],
    ['/', RangeError],
    ['.', RangeError],
    ['..', RangeError],
    [' prefs', RangeError],
    ['prefs ', RangeError],
    ['pre\nfs', RangeError],
    ['pre\u001ffs', RangeError],
    ['\u0001prefs', RangeError],
    ['prefs\u0001', RangeError],
    ['prefs\u007f', RangeError],
    ['pre\u007ffs', RangeError],
    ['pre\u0085fs', RangeError],
    ['\u0085prefs', RangeError],
    ['pre\u009ffs', RangeError],
    ['\ufeffprefs', RangeError],
    ['prefs\ufeff', RangeError],
    ['\u00a0prefs', RangeError],
    ['prefs\u2028', RangeError],
    [3, TypeError],
    ['pre\ud800fs', RangeError],
    ['pre\udffffs', RangeError],
    [null, TypeError],
  ])('rejects the name %j', (name, error) => {
    const { storage } = setup()
    const create = () => new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: name as string })
    expect(create).toThrow(error)
    expect(create).toThrow(/must be/)
  })

  it('lists the rules in the invalid-name message', () => {
    const { storage } = setup()
    const create = () => new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name: '..' })
    for (const rule of ['non-empty', "not '.' or '..'", "'/'", 'control characters', 'lone surrogates', 'whitespace']) {
      expect(create).toThrow(rule)
    }
  })

  it.each([
    ['prefs'],
    ['user prefs'],
    ['préférences'],
    ['prefs.v2'],
    ['a-b_c'],
    ['a\u200bb'],
    ['pre\ufefffs'],
    ['pre\u00a0fs'],
  ])('accepts the name %j', (name) => {
    const { storage } = setup()
    expect(new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), name }).name).toBe(name)
  })

  it.each([[1], [100]])('accepts maxSearchResults %j', (count) => {
    const { storage } = setup()
    const store = new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), maxSearchResults: count })
    expect(store.maxSearchResults).toBe(count)
  })

  it.each([[0], [101], [-5], [2.5], [true]])('rejects maxSearchResults %j', (bad) => {
    const { storage } = setup()
    const maxSearchResults = bad as number
    expect(() => new DynamoDBMemoryStore(storage, { embedder: new KeywordEmbedder(), maxSearchResults })).toThrow(
      RangeError
    )
  })

  it('treats null like undefined where the Python default is None', () => {
    const { storage } = setup()
    const config = { embedder: new KeywordEmbedder(), maxSearchResults: null, extraction: null }
    const store = new DynamoDBMemoryStore(storage, config as unknown as DynamoDBMemoryStoreConfig)
    expect(store.maxSearchResults).toBeUndefined()
    expect(store.extraction).toBeUndefined()
  })

  it('gives a null embedder the default BedrockEmbedder, as Python does for None', () => {
    const { storage } = setup()
    const store = new DynamoDBMemoryStore(storage, { embedder: null as never })
    expect(store['_embedder']).toBeInstanceOf(BedrockEmbedder)
  })

  it('leaves the default embedder Region to the SDK when the storage has none', () => {
    const store = new DynamoDBMemoryStore(new DynamoDBStorage('t'))
    expect((store['_embedder'] as BedrockEmbedder)['_region']).toBeUndefined()
  })

  it('treats an empty storage Region as unset for the default embedder, as the storage clients do', () => {
    const store = new DynamoDBMemoryStore(new DynamoDBStorage('t', { region: '' }))
    expect((store['_embedder'] as BedrockEmbedder)['_region']).toBeUndefined()
  })

  it("follows an injected client's region when the storage Region is empty", async () => {
    const { DynamoDBClient } = await import('@aws-sdk/client-dynamodb')
    const { DynamoDBDocumentClient } = await import('@aws-sdk/lib-dynamodb')
    const client = DynamoDBDocumentClient.from(new DynamoDBClient({ region: 'ap-southeast-2' }))
    const store = new DynamoDBMemoryStore(new DynamoDBStorage('t', { client, region: '' }))
    const region = (store['_embedder'] as BedrockEmbedder)['_region']
    expect(typeof region).toBe('function')
    expect(await (region as () => Promise<string>)()).toBe('ap-southeast-2')
  })

  it('requires a DynamoDBStorage', () => {
    expect(
      () =>
        new DynamoDBMemoryStore(new InMemoryStorage() as unknown as DynamoDBStorage, {
          embedder: new KeywordEmbedder(),
        })
    ).toThrow(TypeError)
    expect(() => new DynamoDBMemoryStore({} as DynamoDBStorage, { embedder: new KeywordEmbedder() })).toThrow(
      'DynamoDBMemoryStore requires a DynamoDBStorage'
    )
  })
})

describe('DynamoDBStorage: text queries and partition keys', () => {
  it('rejects a plain-string search and points at DynamoDBMemoryStore', async () => {
    const { storage, root } = setup()
    for (const s of [root, storage]) {
      await expect(s.search('window')).rejects.toThrow(/or use DynamoDBMemoryStore, which embeds text for you\.$/)
    }
  })

  it('reports the partition a key is stored under', () => {
    const { root } = setup()
    expect(root.partitionKey('sessions/s1/a/b')).toBe('sessions/s1')
    expect(root.namespace('user/u1').partitionKey('memory/x/y')).toBe('user/u1')
    expect(root.partitionKey('single')).toBe('single')
    expect(() => root.partitionKey('a/../b')).toThrow(StorageError)
  })
})

describe('DynamoDBMemoryStore: error messages', () => {
  const messageOf = async (run: () => unknown): Promise<string> => {
    try {
      await run()
    } catch (error: unknown) {
      return `${(error as Error).name}: ${(error as Error).message}`
    }
    throw new Error('expected an error')
  }
  const embed = () => [1]

  it.each<[string, (storage: DynamoDBStorage) => unknown, string]>([
    [
      'storage type',
      () => new DynamoDBMemoryStore({} as DynamoDBStorage),
      'TypeError: DynamoDBMemoryStore requires a DynamoDBStorage',
    ],
    [
      'name type',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed, name: 3 as never }),
      'TypeError: Memory store name must be a string',
    ],
    [
      'name ..',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed, name: '..' }),
      "RangeError: Memory store name must be non-empty, not '.' or '..', and without '/', control characters, " +
        'lone surrogates, or leading or trailing whitespace; got ".."',
    ],
    [
      'description type',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed, description: 5 as never }),
      'TypeError: description must be a string or null',
    ],
    [
      'writable type',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed, writable: null as never }),
      'TypeError: writable must be a boolean',
    ],
    [
      'constructor limit',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed, maxSearchResults: 0 }),
      'RangeError: maxSearchResults must be an integer from 1 to 100; got 0',
    ],
    [
      'query type',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).search(3 as never),
      'TypeError: query must be a string',
    ],
    [
      'search limit',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).search('x', { maxSearchResults: 0 }),
      'RangeError: maxSearchResults must be an integer >= 1; got 0',
    ],
    [
      'query type before limit',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).search(3 as never, { maxSearchResults: 0 }),
      'TypeError: query must be a string',
    ],
    [
      'embedder type',
      (s) => new DynamoDBMemoryStore(s, { embedder: 42 as never }),
      'TypeError: embedder must be a function or an object with an embed method',
    ],
    [
      'embedder with a non-function embed',
      (s) => new DynamoDBMemoryStore(s, { embedder: { embed: 5 } as never }),
      'TypeError: embedder must be a function or an object with an embed method',
    ],
    [
      'embedder type before limit',
      (s) => new DynamoDBMemoryStore(s, { embedder: 42 as never, maxSearchResults: 0 }),
      'TypeError: embedder must be a function or an object with an embed method',
    ],
    [
      'lone surrogate query',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).search('a\udcffb'),
      'RangeError: query must be valid Unicode text; it contains a lone surrogate',
    ],
    [
      'search limit before lone surrogate',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).search('a\udcffb', { maxSearchResults: 0 }),
      'RangeError: maxSearchResults must be an integer >= 1; got 0',
    ],
    [
      'content type',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).add(3 as never),
      'TypeError: content must be a string',
    ],
    [
      'blank content',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).add(' '),
      'RangeError: content must not be empty',
    ],
    [
      'lone surrogate content',
      (s) => new DynamoDBMemoryStore(s, { embedder: embed }).add('a\ud800'),
      'RangeError: content must be valid Unicode text; it contains a lone surrogate',
    ],
    [
      'embedder result',
      (s) => new DynamoDBMemoryStore(s, { embedder: () => [], name: 'prefs' }).search('x'),
      "StorageError: The embedder for memory store 'prefs' must return a non-empty array of finite numbers",
    ],
  ])('%s', async (_label, run, message) => {
    const { storage } = setup()
    expect(await messageOf(() => run(storage))).toBe(message)
  })

  it.each([
    [
      { n: 2 ** 53 },
      'key=<n> | dropping metadata entry: is out of range (finite, zero or 1e-130 to 2^53 - 1 in magnitude)',
    ],
    [{ s: 'a\udcffb' }, "key=<s> | dropping metadata entry: contains a lone surrogate, which DynamoDB stores as '?'"],
    [
      { tags: ['a'] },
      'key=<tags> | dropping metadata entry: has type array; values must be strings, numbers, or booleans',
    ],
    [
      { ['cl\u00e9']: null },
      'key=<cl\u00e9> | dropping metadata entry: has type null; values must be strings, numbers, or booleans',
    ],
  ])('writes the full drop warning for %j', async (metadata, line) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', metadata as never)
    expect(warn).toHaveBeenCalledWith(`store=<prefs>, ${line}`)
  })

  it.each([
    ['safe\nFORGED=admin', 'safe\\nFORGED=admin'],
    ['safe\rFORGED=admin', 'safe\\rFORGED=admin'],
    ['a\tb', 'a\\tb'],
    ['a\u2028b\u2029c', 'a\\u2028b\\u2029c'],
    ['a\u202eb\u2066c\u200fd', 'a\\u202eb\\u2066c\\u200fd'],
    ['a\u0085b\u001bc\u007fd\u0000e', 'a\\u0085b\\u001bc\\u007fd\\u0000e'],
    ['a\\nb', 'a\\\\nb'],
    ['a b\u001f\u009f\u00a0c', 'a b\\u001f\\u009f\u00a0c'],
    ['e\u{1f600}', 'e\u{1f600}'],
  ])('escapes the key %j in the drop warning, so it cannot forge a log record', async (key, shown) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { [key]: ['a'] } as never)
    expect(warn.mock.calls).toEqual([
      [
        `store=<prefs>, key=<${shown}> | dropping metadata entry: has type array; values must be strings, numbers, or booleans`,
      ],
    ])
  })

  it.each([
    ['a\ud800', 'a\\ud800'],
    ['k\udfff', 'k\\udfff'],
  ])('escapes the lone surrogate in %j at both ends of the range', async (key, shown) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { [key]: 'x' })
    expect(warn.mock.calls).toEqual([
      [
        `store=<prefs>, key=<${shown}> | dropping metadata entry: contains a lone surrogate, which DynamoDB stores as '?'`,
      ],
    ])
  })

  it('escapes the store name in both drop warnings', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'pr\u2028e\u202efs' } })
    await store.add('Prefers a window seat', { k: ['a'] } as never)
    await store.add('Prefers a window seat', ['a'] as never)
    expect(warn.mock.calls).toEqual([
      [
        'store=<pr\\u2028e\\u202efs>, key=<k> | dropping metadata entry: has type array; values must be strings, numbers, or booleans',
      ],
      ['store=<pr\\u2028e\\u202efs> | dropping metadata that is not a plain object'],
    ])
  })

  it('writes a lone surrogate in a dropped key as \\udcff, as the Python package does', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = setup({ config: { name: 'prefs' } })
    await store.add('Prefers a window seat', { 'n\udcff': 'x' })
    expect(warn.mock.calls).toEqual([
      [
        "store=<prefs>, key=<n\\udcff> | dropping metadata entry: contains a lone surrogate, which DynamoDB stores as '?'",
      ],
    ])
  })

  it('returns padded content unchanged from search', async () => {
    const { store } = setup()
    const content = '  Prefers a window seat\n'
    await store.add(content)
    const [entry] = await store.search('window')
    expect(entry!.content).toBe(content)
  })

  it('rejects a lone surrogate query before calling the embedder', async () => {
    const { store, embedder } = setup()
    await expect(store.search('window \udfff')).rejects.toThrow(/lone surrogate/)
    expect(embedder.calls).toEqual([])
  })

  it('embeds a query with a character outside the BMP', async () => {
    const { store, embedder } = setup()
    await store.search('\u{1f600} window')
    expect(embedder.calls).toEqual([['\u{1f600} window', 'query']])
  })

  it('embeds a one-character query', async () => {
    const { store, embedder } = setup()
    await store.search('?')
    expect(embedder.calls).toEqual([['?', 'query']])
  })
})
