// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration tests — run against REAL DynamoDB + S3 (never DynamoDB Local).
 *
 * Gated by RUN_INTEG=1 and AWS credentials in the environment. The suite provisions
 * its own table (pk/sk string schema, TTL enabled on `expireAt`) and — unless
 * INTEG_S3_BUCKET is supplied — its own S3 bucket, then tears everything down.
 *
 * Run:
 *   RUN_INTEG=1 AWS_REGION=us-east-1 npm run test:integ
 *
 * Optional env: INTEG_TABLE, INTEG_S3_BUCKET (reuse instead of create), AWS_PROFILE.
 *
 * Native vector search (`search()`) is covered in vector-search.integ.test.ts
 * (it provisions its own vector-indexed table).
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  CreateTableCommand,
  DeleteTableCommand,
  DynamoDBClient,
  UpdateTimeToLiveCommand,
  waitUntilTableExists,
} from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient, GetCommand } from '@aws-sdk/lib-dynamodb'
import {
  CreateBucketCommand,
  DeleteBucketCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
  S3Client,
  type BucketLocationConstraint,
} from '@aws-sdk/client-s3'
import { STSClient, GetCallerIdentityCommand } from '@aws-sdk/client-sts'
import { DynamoDBStorage } from '../../src/dynamodb-storage.js'

const RUN = process.env.RUN_INTEG === '1'
const REGION = process.env.AWS_REGION ?? process.env.INTEG_REGION ?? 'us-east-1'
const STAMP = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
const TABLE = process.env.INTEG_TABLE ?? `strands-ddb-storage-integ-${STAMP}`
const PROVIDED_BUCKET = process.env.INTEG_S3_BUCKET
const BUCKET = PROVIDED_BUCKET ?? `strands-ddb-storage-integ-${STAMP}`.toLowerCase()

const bytes = (s: string) => new TextEncoder().encode(s)
const str = (b: Uint8Array | null) => (b ? new TextDecoder().decode(b) : null)
function randomBytes(n: number): Uint8Array {
  const a = new Uint8Array(n)
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 256) | 0
  return a
}

let ddb: DynamoDBClient
let doc: DynamoDBDocumentClient
let s3: S3Client
let createdBucket = false

async function objectExists(key: string): Promise<boolean> {
  try {
    await s3.send(new HeadObjectCommand({ Bucket: BUCKET, Key: key }))
    return true
  } catch {
    return false
  }
}

beforeAll(async () => {
  if (!RUN) return
  ddb = new DynamoDBClient({ region: REGION })
  doc = DynamoDBDocumentClient.from(ddb)
  s3 = new S3Client({ region: REGION })

  try {
    const id = await new STSClient({ region: REGION }).send(new GetCallerIdentityCommand({}))
    console.log(`[integ] account=${id.Account} region=${REGION} table=${TABLE} bucket=${BUCKET}`)
  } catch (e) {
    console.warn('[integ] GetCallerIdentity failed — check credentials', e)
  }

  await ddb.send(
    new CreateTableCommand({
      TableName: TABLE,
      BillingMode: 'PAY_PER_REQUEST',
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
    })
  )
  await waitUntilTableExists({ client: ddb, maxWaitTime: 180 }, { TableName: TABLE })
  await ddb.send(
    new UpdateTimeToLiveCommand({
      TableName: TABLE,
      TimeToLiveSpecification: { AttributeName: 'expireAt', Enabled: true },
    })
  )

  if (!PROVIDED_BUCKET) {
    await s3.send(
      new CreateBucketCommand({
        Bucket: BUCKET,
        ...(REGION !== 'us-east-1'
          ? { CreateBucketConfiguration: { LocationConstraint: REGION as BucketLocationConstraint } }
          : {}),
      })
    )
    createdBucket = true
  }
})

afterAll(async () => {
  if (!RUN) return
  try {
    await ddb.send(new DeleteTableCommand({ TableName: TABLE }))
  } catch (e) {
    console.warn('[integ] table cleanup failed', e)
  }
  if (createdBucket) {
    try {
      let token: string | undefined
      do {
        const listed = await s3.send(new ListObjectsV2Command({ Bucket: BUCKET, ContinuationToken: token }))
        for (const obj of listed.Contents ?? []) {
          if (obj.Key) await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: obj.Key }))
        }
        token = listed.IsTruncated ? listed.NextContinuationToken : undefined
      } while (token)
      await s3.send(new DeleteBucketCommand({ Bucket: BUCKET }))
    } catch (e) {
      console.warn('[integ] bucket cleanup failed', e)
    }
  }
})

// One instance per capability, sharing the provisioned client/table/bucket.
function ddbStorage(opts?: { s3?: boolean; prefix?: string; compression?: 'gzip' | 'none'; ttlSeconds?: number }) {
  return new DynamoDBStorage(TABLE, {
    client: doc,
    ...(opts?.prefix ? { prefix: opts.prefix } : {}),
    ...(opts?.s3 ? { s3Bucket: BUCKET, s3Client: s3 } : {}),
    ...(opts?.compression ? { compression: opts.compression } : {}),
    ...(opts?.ttlSeconds !== undefined ? { ttlSeconds: opts.ttlSeconds } : {}),
  })
}

describe.skipIf(!RUN)('DynamoDBStorage integration — point operations', () => {
  const storage = () => ddbStorage()

  it('round-trips write/read and overwrites', async () => {
    const s = storage()
    const key = `sessions/pt-${STAMP}/a`
    await s.write(key, bytes('hello'))
    expect(str(await s.read(key))).toBe('hello')
    await s.write(key, bytes('world'))
    expect(str(await s.read(key))).toBe('world')
  })

  it('returns null for a missing key', async () => {
    expect(await storage().read(`sessions/pt-${STAMP}/missing`)).toBeNull()
  })

  it('deletes a value and is a no-op for a missing key', async () => {
    const s = storage()
    const key = `sessions/pt-${STAMP}/del`
    await s.write(key, bytes('x'))
    await s.delete(key)
    expect(await s.read(key)).toBeNull()
    await expect(s.delete(key)).resolves.toBeUndefined()
  })

  it('round-trips a single-segment key', async () => {
    const s = storage()
    const key = `singleton-${STAMP}`
    await s.write(key, bytes('one'))
    expect(str(await s.read(key))).toBe('one')
    await s.delete(key)
  })

  it('rejects empty and traversal keys', async () => {
    const s = storage()
    await expect(s.write('', bytes('x'))).rejects.toThrow(/must not be empty/)
    await expect(s.read('sessions/../etc')).rejects.toThrow(/'\.\.'/)
  })
})

describe.skipIf(!RUN)('DynamoDBStorage integration — listing', () => {
  it('lists a string prefix sorted', async () => {
    const s = ddbStorage()
    const base = `sessions/ls-${STAMP}/scopes/agent/a1/immutable_history`
    await s.write(`${base}/snapshot_2.json`, bytes('2'))
    await s.write(`${base}/snapshot_1.json`, bytes('1'))
    await s.write(`${base}/snapshot_3.json`, bytes('3'))
    expect(await s.list(`${base}/`)).toEqual([
      `${base}/snapshot_1.json`,
      `${base}/snapshot_2.json`,
      `${base}/snapshot_3.json`,
    ])
  })

  it('lists via a structured DynamoDBListQuery, skBetween, and startAfter', async () => {
    const s = ddbStorage()
    const pk = `sessions/lq-${STAMP}`
    for (const n of ['1', '2', '3', '4']) await s.write(`${pk}/k/${n}`, bytes(n))
    expect((await s.list({ pk })).sort()).toEqual([`${pk}/k/1`, `${pk}/k/2`, `${pk}/k/3`, `${pk}/k/4`])
    expect(await s.list({ pk, skBetween: ['k/2', 'k/3'] })).toEqual([`${pk}/k/2`, `${pk}/k/3`])
    expect(await s.list({ pk, startAfter: `${pk}/k/2` })).toEqual([`${pk}/k/3`, `${pk}/k/4`])
  })

  it('namespace() view round-trips and lists against the real table', async () => {
    const s = ddbStorage()
    const view = s.namespace(`sessions/ns-${STAMP}`)
    await view.write('scopes/agent/a1/x', bytes('v'))
    expect(str(await view.read('scopes/agent/a1/x'))).toBe('v')
    expect(str(await s.read(`sessions/ns-${STAMP}/scopes/agent/a1/x`))).toBe('v')
    expect(await view.list('scopes/agent/a1/')).toEqual(['scopes/agent/a1/x'])
  })
})

describe.skipIf(!RUN)('DynamoDBStorage integration — S3 offload', () => {
  it('offloads a >380KB value to S3 and reads it back, with a DDB pointer item', async () => {
    const s = ddbStorage({ s3: true })
    const key = `sessions/s3-${STAMP}/big`
    const payload = randomBytes(400_001) // incompressible-ish, above the inline limit
    await s.write(key, payload)
    // DDB item is a pointer, not inline bytes.
    const raw = await doc.send(new GetCommand({ TableName: TABLE, Key: { pk: `sessions/s3-${STAMP}`, sk: 'big' } }))
    expect(raw.Item?.s3).toBe(true)
    expect(raw.Item?.data).toBeUndefined()
    // Object physically present in S3.
    expect(await objectExists(key)).toBe(true)
    // Transparent read-back.
    expect(await s.read(key)).toEqual(payload)
    // Delete cleans up the S3 object too.
    await s.delete(key)
    expect(await objectExists(key)).toBe(false)
    expect(await s.read(key)).toBeNull()
  })

  it('throws on an oversized value when no S3 bucket is configured', async () => {
    const s = ddbStorage() // no s3
    await expect(s.write(`sessions/s3-${STAMP}/nobucket`, randomBytes(400_001))).rejects.toThrow(/s3Bucket/)
  })
})

describe.skipIf(!RUN)('DynamoDBStorage integration — compression', () => {
  it('gzips a compressible value inline and reads it back', async () => {
    const s = ddbStorage({ compression: 'gzip' })
    const key = `sessions/z-${STAMP}/a`
    const original = 'A'.repeat(5000)
    await s.write(key, bytes(original))
    const raw = await doc.send(new GetCommand({ TableName: TABLE, Key: { pk: `sessions/z-${STAMP}`, sk: 'a' } }))
    expect(raw.Item?.z).toBe(true)
    expect(str(await s.read(key))).toBe(original)
  })

  it('keeps a large-but-compressible value inline (no S3 object written)', async () => {
    const s = ddbStorage({ compression: 'gzip', s3: true })
    const key = `sessions/z-${STAMP}/inline-big`
    const original = 'A'.repeat(800_000) // >380KB raw, gzips far under it
    await s.write(key, bytes(original))
    expect(await objectExists(key)).toBe(false) // the cost win: stayed in DynamoDB
    expect(str(await s.read(key))).toBe(original)
  })

  it('offloads to S3 when even the compressed value exceeds the item limit', async () => {
    const s = ddbStorage({ compression: 'gzip', s3: true })
    const key = `sessions/z-${STAMP}/incompressible`
    const payload = randomBytes(400_001)
    await s.write(key, payload)
    expect(await objectExists(key)).toBe(true)
    expect(await s.read(key)).toEqual(payload)
    await s.delete(key)
  })
})

describe.skipIf(!RUN)('DynamoDBStorage integration — TTL', () => {
  it('stamps a future epoch-seconds expiry on the real item', async () => {
    const s = ddbStorage({ ttlSeconds: 3600 })
    const key = `sessions/ttl-${STAMP}/a`
    const before = Math.floor(Date.now() / 1000)
    await s.write(key, bytes('v'))
    const raw = await doc.send(new GetCommand({ TableName: TABLE, Key: { pk: `sessions/ttl-${STAMP}`, sk: 'a' } }))
    expect(typeof raw.Item?.expireAt).toBe('number')
    expect(raw.Item?.expireAt).toBeGreaterThanOrEqual(before + 3600)
  })

  it('read and list hide an already-expired item (filter, not physical deletion)', async () => {
    const s = ddbStorage({ ttlSeconds: 3600 })
    const pk = `sessions/ttlx-${STAMP}`
    await s.write(`${pk}/live`, bytes('L'))
    await s.write(`${pk}/dead`, bytes('D'), { ttlSeconds: -1 }) // expiry in the past
    expect(await s.read(`${pk}/dead`)).toBeNull()
    expect(await s.list({ pk })).toEqual([`${pk}/live`])
  })

  it('a reader that did not opt into TTL still returns the unswept item', async () => {
    const writer = ddbStorage({ ttlSeconds: 3600 })
    const key = `sessions/ttlopt-${STAMP}/a`
    await writer.write(key, bytes('v'), { ttlSeconds: -1 })
    const reader = ddbStorage() // no ttlSeconds -> no filter injected
    expect(str(await reader.read(key))).toBe('v')
  })
})

describe.skipIf(!RUN)('DynamoDBStorage integration — lazy client construction (no injected client)', () => {
  it('builds its own DynamoDB client from region and round-trips', async () => {
    // No `client` -> exercises the lazy dynamic-import + new DynamoDBClient path.
    const s = new DynamoDBStorage(TABLE, { region: REGION })
    const key = `sessions/reg-${STAMP}/a`
    await s.write(key, bytes('hello'))
    expect(str(await s.read(key))).toBe('hello')
    expect(await s.list({ pk: `sessions/reg-${STAMP}` })).toEqual([key])
    await s.delete(key)
    expect(await s.read(key)).toBeNull()
  })

  it('builds its own S3 client from region for offload', async () => {
    // No `s3Client` -> exercises the lazy new S3Client path.
    const s = new DynamoDBStorage(TABLE, { region: REGION, s3Bucket: BUCKET })
    const key = `sessions/reg-${STAMP}/big`
    const payload = randomBytes(400_001)
    await s.write(key, payload)
    expect(await objectExists(key)).toBe(true)
    expect(await s.read(key)).toEqual(payload)
    await s.delete(key)
    expect(await objectExists(key)).toBe(false)
  })
})
