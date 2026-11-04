// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration test for native vector search — real Amazon DynamoDB.
 *
 * Gated by RUN_INTEG=1 and AWS credentials. Provisions its own table with a
 * vector index (SearchSchema HASH on pk so searches can be partition-pinned),
 * waits for the index to finish backfilling (ACTIVE alone is not ready), then
 * exercises the native `search()` path end to end. Tears the table down after.
 *
 * Run:
 *   RUN_INTEG=1 AWS_REGION=us-east-1 npm run test:integ
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { CreateTableCommand, DeleteTableCommand, DescribeTableCommand, DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { DynamoDBStorage, DynamoDBVectorSearchStrategy, type SearchResult } from '../../src/index.js'

const RUN = process.env.RUN_INTEG === '1'
const REGION = process.env.AWS_REGION ?? 'us-east-1'
const TABLE = `strands-integ-vector-ts-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
const INDEX = 'vector_index'

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

describe.skipIf(!RUN)('DynamoDBStorage — native vector search (live)', () => {
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
            IndexName: INDEX,
            VectorAttribute: { AttributeName: 'vector' },
            SearchSchema: [{ AttributeName: 'pk', SearchSchemaElementType: 'HASH' }],
            Projection: { ProjectionType: 'ALL' },
            Dimensions: 4,
            DistanceFunction: 'COSINE',
          },
        ],
      })
    )
    // Wait for table + index readiness (Backfilling must be false/absent).
    const deadline = Date.now() + 600_000
    for (;;) {
      const table = (await client.send(new DescribeTableCommand({ TableName: TABLE }))).Table
      const idx = table?.VectorIndexes?.find((i) => i.IndexName === INDEX)
      if (table?.TableStatus === 'ACTIVE' && idx?.IndexStatus === 'ACTIVE' && !idx?.Backfilling) break
      if (Date.now() > deadline) throw new Error('vector index did not become ready within 600s')
      await sleep(5000)
    }
  }, 660_000)

  afterAll(async () => {
    await client.send(new DeleteTableCommand({ TableName: TABLE }))
  })

  it('searches natively end to end', async () => {
    const storage = new DynamoDBStorage(TABLE, { region: REGION, prefix: 'tenant/a/', indexName: INDEX })
    await storage.write('memories/m1', new TextEncoder().encode('likes window seats'), {
      vector: [1, 0, 0, 0],
      metadata: { kind: 'pref' },
    })
    await storage.write('memories/m2', new TextEncoder().encode('allergic to peanuts'), {
      vector: [0, 1, 0, 0],
    })

    // Eventually consistent: poll until both items are visible to search.
    const deadline = Date.now() + 300_000
    let results: SearchResult[]
    for (;;) {
      results = await storage.search({ vector: [1, 0, 0, 0], topK: 2, pk: 'tenant/a' })
      if (results.length === 2) break
      if (Date.now() > deadline) throw new Error('expected both items in search results within 300s')
      await sleep(5000)
    }
    expect(results[0]?.key).toBe('memories/m1') // nearest first
    expect(results[0]?.metadata).toEqual({ kind: 'pref' })
    expect(results[0]!.score).toBeLessThanOrEqual(results[1]!.score) // COSINE distance

    const [top] = await storage.search({ vector: [1, 0, 0, 0], topK: 1, pk: 'tenant/a', includeValues: true })
    expect(new TextDecoder().decode(top?.data)).toBe('likes window seats')
  }, 360_000)

  it('recalls by meaning through DynamoDBVectorSearchStrategy, best first', async () => {
    const vocabulary = ['dark', 'mode', 'coffee', 'tea']
    const embed = (text: string) => {
      const tokens = text.toLowerCase().replace(/#/g, ' ').split(/\s+/)
      return vocabulary.map((word) => 0.01 + tokens.filter((token) => token === word).length)
    }
    const storage = new DynamoDBStorage(TABLE, {
      region: REGION,
      prefix: 'tenant/b/',
      indexName: INDEX,
      searchStrategy: new DynamoDBVectorSearchStrategy({ embedder: embed }),
    })
    const knowledge = new FileMemoryStore({ name: 'facts', storage, writable: true })
    await knowledge.add('# Display\nprefers dark mode')
    await knowledge.add('# Drinks\nlikes coffee and tea')

    // Eventually consistent: poll until both items are visible to search.
    const deadline = Date.now() + 300_000
    let found = await knowledge.search('dark')
    while (found.length < 2) {
      if (Date.now() > deadline) throw new Error('expected both memories in search results within 300s')
      await sleep(5000)
      found = await knowledge.search('dark')
    }
    expect(found.map((entry) => entry.content.split('\n')[0])).toEqual(['# Display', '# Drinks'])

    const raw = await storage.namespace('memory/facts').search('dark')
    expect(raw[0]!.score).toBeGreaterThan(raw[1]!.score) // converted COSINE scores rank the closest first
    expect(raw.every((r) => r.score > 0 && r.score <= 1)).toBe(true)
  }, 360_000)
})
