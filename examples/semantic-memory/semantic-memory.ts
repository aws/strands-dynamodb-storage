/**
 * Give an agent long-term memory it can search by meaning, on one DynamoDB table.
 *
 * Session persistence gets a conversation through a restart. This example covers
 * the other half: recalling something a user said weeks ago, in a new
 * conversation that shares no keys or words with the old one. Memories are
 * embedded on write, stored with a vector alongside the bytes, and recalled with
 * a DynamoDB vector index search scoped to the user's partition.
 *
 * The `DynamoDBMemoryStore` class here is application code: the package ships
 * the storage contract and vector search, and this bridge adapts them to the
 * Strands SDK's `MemoryStore` interface so the Memory Manager can pull relevant
 * memories into the model input before each call.
 *
 * Prerequisites:
 *   - A DynamoDB table with a vector index (see README.md; 1024 dims, COSINE).
 *   - Credentials for the table, dynamodb:SearchVectors, and Bedrock InvokeModel
 *     (agent model + amazon.titan-embed-text-v2:0 for embeddings).
 *
 * Usage:
 *   npx tsx semantic-memory.ts --table agent-storage seed
 *   npx tsx semantic-memory.ts --table agent-storage ask \
 *       "Book me a flight seat for my December trip. Which seat should I pick?"
 */

import { randomUUID } from 'node:crypto'
import { parseArgs } from 'node:util'

import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime'
import { Agent, MemoryManager } from '@strands-agents/sdk'
import type { MemoryEntry, MemoryStore, SearchOptions } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'
import type { SearchQuery, SearchResult } from 'strands-dynamodb-storage'

const USER_PARTITION = 'user/u1'

const SEED_MEMORIES = [
  'Prefers window seats on long flights',
  'Planning a trip to Tokyo in December',
  'Has a peanut allergy, flag it when booking meals',
]

type Embedder = (text: string) => Promise<number[]>

/** Return an embed(text) -> number[] callable backed by Titan V2 (1024 dims). */
function makeEmbedder(region: string): Embedder {
  const bedrock = new BedrockRuntimeClient({ region })

  return async (text: string): Promise<number[]> => {
    const response = await bedrock.send(
      new InvokeModelCommand({
        modelId: 'amazon.titan-embed-text-v2:0',
        body: JSON.stringify({ inputText: text, dimensions: 1024 }),
      }),
    )
    const parsed = JSON.parse(new TextDecoder().decode(response.body)) as { embedding: number[] }
    return parsed.embedding
  }
}

/** Adapts DynamoDBStorage vector search to the Strands MemoryStore interface. */
class DynamoDBMemoryStore implements MemoryStore {
  readonly name = 'dynamodb'
  readonly description = 'Long-term memories in DynamoDB, searched by meaning'
  readonly maxSearchResults = 3
  readonly writable = true

  constructor(
    private readonly storage: DynamoDBStorage,
    private readonly partition: string,
    private readonly embed: Embedder,
  ) {}

  async add(content: string, metadata?: Record<string, string | number | boolean>): Promise<void> {
    await this.storage.write(
      `memories/${randomUUID().replace(/-/g, '').slice(0, 8)}`,
      new TextEncoder().encode(content),
      { vector: await this.embed(content), metadata },
    )
  }

  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const request: SearchQuery = {
      vector: await this.embed(query),
      topK: this.maxSearchResults,
      pk: this.partition,
      includeValues: true,
    }
    const results = await this.storage.search(request)
    const decoder = new TextDecoder()
    return results
      .filter((r): r is SearchResult & { data: Uint8Array } => r.data !== undefined)
      .map((r) => ({
        content: decoder.decode(r.data),
        metadata: r.metadata as MemoryEntry['metadata'],
      }))
  }
}

function buildStore(table: string, region: string): DynamoDBMemoryStore {
  const storage = new DynamoDBStorage(table, { region, prefix: USER_PARTITION })
  return new DynamoDBMemoryStore(storage, USER_PARTITION, makeEmbedder(region))
}

async function seed(store: DynamoDBMemoryStore): Promise<void> {
  for (const memory of SEED_MEMORIES) {
    await store.add(memory, { kind: 'preference' })
    console.log(`stored: ${memory}`)
  }
}

async function ask(store: DynamoDBMemoryStore, question: string): Promise<void> {
  const memory = new MemoryManager({ stores: [store], addToolConfig: true })
  const agent = new Agent({ memoryManager: memory })
  await agent.invoke(question)
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    options: {
      table: { type: 'string', default: 'agent-storage' },
      region: { type: 'string', default: 'us-east-1' },
    },
    allowPositionals: true,
  })
  const verb = positionals[0]
  if (verb !== 'seed' && verb !== 'ask') {
    console.error('usage: npx tsx semantic-memory.ts [--table NAME] [--region REGION] {seed|ask} [question]')
    process.exit(2)
  }
  const question = positionals[1] ?? 'Which seat should I pick for my December trip?'

  const store = buildStore(values.table, values.region)
  if (verb === 'seed') {
    await seed(store)
  } else {
    await ask(store, question)
  }
}

await main()
