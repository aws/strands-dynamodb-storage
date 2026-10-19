// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration test for DynamoDBMemoryStore: real Amazon DynamoDB and Amazon Bedrock.
 *
 * Gated by RUN_INTEG=1 and AWS credentials with `bedrock:InvokeModel` on Amazon Titan
 * Text Embeddings V2 and Cohere Embed v3 and v4 (cross-Region inference profile
 * `us.cohere.embed-v4:0`). Provisions its own table with a 1024-dimension vector index,
 * waits for the index to finish backfilling, exercises the memory store through the
 * Strands Memory Manager, and tears the table down afterwards.
 *
 * Run:
 *   RUN_INTEG=1 AWS_REGION=us-east-1 npm run test:integ
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  CreateTableCommand,
  DeleteTableCommand,
  DescribeTableCommand,
  DynamoDBClient,
  QueryCommand,
} from '@aws-sdk/client-dynamodb'
import { MemoryManager, StorageError, type MemoryEntry } from '@strands-agents/sdk'
import { BedrockEmbedder, DynamoDBMemoryStore, DynamoDBStorage } from '../../src/index.js'

const RUN = process.env.RUN_INTEG === '1'
const REGION = process.env.AWS_REGION ?? 'us-east-1'
const TABLE = `strands-integ-memory-ts-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
const MEMORIES = ['Prefers a window seat on flights', 'Has a severe peanut allergy', 'Drinks black coffee, no sugar']

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
const suffix = () => Math.random().toString(36).slice(2, 8)

/** The vector index is eventually consistent: poll until `count` memories are visible. */
async function searchUntil(store: DynamoDBMemoryStore, query: string, count: number): Promise<MemoryEntry[]> {
  const deadline = Date.now() + 300_000
  for (;;) {
    const entries = await store.search(query, { maxSearchResults: 10 })
    if (entries.length >= count) return entries
    if (Date.now() > deadline) throw new Error(`expected ${count} memories within 300s; got ${entries.length}`)
    await sleep(5000)
  }
}

describe.skipIf(!RUN)('DynamoDBMemoryStore (live)', () => {
  const client = new DynamoDBClient({ region: REGION })

  beforeAll(async () => {
    await client.send(
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
        VectorIndexes: [
          {
            IndexName: 'vector_index',
            VectorAttribute: { AttributeName: 'vector' },
            SearchSchema: [{ AttributeName: 'pk', SearchSchemaElementType: 'HASH' }],
            Projection: { ProjectionType: 'ALL' },
            Dimensions: 1024,
            DistanceFunction: 'COSINE',
          },
        ],
      })
    )
    const deadline = Date.now() + 600_000
    for (;;) {
      const table = (await client.send(new DescribeTableCommand({ TableName: TABLE }))).Table
      const idx = table?.VectorIndexes?.[0]
      if (table?.TableStatus === 'ACTIVE' && idx?.IndexStatus === 'ACTIVE' && !idx?.Backfilling) break
      if (Date.now() > deadline) throw new Error('vector index did not become ready within 600s')
      await sleep(5000)
    }
  }, 660_000)

  afterAll(async () => {
    await client.send(new DeleteTableCommand({ TableName: TABLE }))
  })

  it('works through the Memory Manager with the default embedder', async () => {
    const storage = new DynamoDBStorage(TABLE, { region: REGION }).namespace(`user/default-${suffix()}`)
    const store = new DynamoDBMemoryStore(storage, { name: 'prefs' })
    const manager = new MemoryManager({ stores: [store], injection: false })
    for (const memory of MEMORIES) await manager.add(memory)

    const entries = await searchUntil(store, 'Where does the user like to sit on a plane?', 3)
    expect(entries[0]?.content).toBe('Prefers a window seat on flights')
    expect(entries[0]!.metadata!._score as number).toBeLessThanOrEqual(entries[1]!.metadata!._score as number)

    const [top] = await manager.search("What can't the user eat?", { maxSearchResults: 1 })
    expect(top?.content).toBe('Has a severe peanut allergy')
    expect(top?.storeName).toBe('prefs')

    // Adding the same text again overwrites the item instead of duplicating it.
    const key = await store.add(MEMORIES[0]!, { source: 'profile' })
    const items = await client.send(
      new QueryCommand({
        TableName: TABLE,
        KeyConditionExpression: 'pk = :pk',
        ExpressionAttributeValues: { ':pk': { S: storage.partitionKey(key) } },
        ConsistentRead: true,
      })
    )
    expect(items.Items).toHaveLength(3)

    await storage.delete(key)
    expect(await storage.read(key)).toBeNull()
  }, 360_000)

  it.each([['us.cohere.embed-v4:0'], ['cohere.embed-english-v3']])(
    'works with %s',
    async (modelId) => {
      const storage = new DynamoDBStorage(TABLE, { region: REGION }).namespace(`user/cohere-${suffix()}`)
      const store = new DynamoDBMemoryStore(storage, { embedder: new BedrockEmbedder({ modelId, region: REGION }) })
      for (const memory of MEMORIES) await store.add(memory)
      const entries = await searchUntil(store, 'Where does the user like to sit on a plane?', 3)
      expect(entries[0]?.content).toBe('Prefers a window seat on flights')
    },
    360_000
  )

  it('fails without writing when the embedding dimensions do not match the index', async () => {
    const storage = new DynamoDBStorage(TABLE, { region: REGION }).namespace(`user/small-${suffix()}`)
    const store = new DynamoDBMemoryStore(storage, {
      embedder: new BedrockEmbedder({ dimensions: 512, region: REGION }),
    })
    const writeError = await store.add('Prefers a window seat').catch((e: unknown) => e)
    expect(writeError).toBeInstanceOf(StorageError)
    expect(String((writeError as Error).cause)).toContain('Expected: 1024, Actual: 512')
    const searchError = await store.search('window seat').catch((e: unknown) => e)
    expect(searchError).toBeInstanceOf(StorageError)
    expect(String((searchError as Error).cause)).toContain('does not match vector index dimension 1024')
    const items = await client.send(
      new QueryCommand({
        TableName: TABLE,
        KeyConditionExpression: 'pk = :pk',
        ExpressionAttributeValues: { ':pk': { S: storage.partitionKey('memory/dynamodb/x') } },
        ConsistentRead: true,
      })
    )
    expect(items.Items).toEqual([])
  }, 120_000)
})
