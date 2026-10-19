/**
 * Give an agent long-term memory it can search by meaning, on one DynamoDB table.
 *
 * Session persistence gets a conversation through a restart. This example covers
 * the other half: recalling something a user said weeks ago, in a new
 * conversation that shares no keys or words with the old one. DynamoDBMemoryStore
 * embeds each memory with Amazon Titan Text Embeddings V2, stores it with its
 * vector on the table, and recalls it with a vector index search scoped to the
 * user's partition, so the Memory Manager can pull relevant memories into the
 * model input before each call.
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

import { parseArgs } from 'node:util'

import { Agent, MemoryManager } from '@strands-agents/sdk'
import { DynamoDBMemoryStore, DynamoDBStorage } from 'strands-dynamodb-storage'

const USER_NAMESPACE = 'user/u1'

const SEED_MEMORIES = [
  'Prefers window seats on long flights',
  'Planning a trip to Tokyo in December',
  'Has a peanut allergy, flag it when booking meals',
]

/** One memory store per user: the namespace keeps every user's memories in their own partition. */
function buildStore(table: string, region: string): DynamoDBMemoryStore {
  const storage = new DynamoDBStorage(table, { region }).namespace(USER_NAMESPACE)
  return new DynamoDBMemoryStore(storage, { maxSearchResults: 3 })
}

async function seed(store: DynamoDBMemoryStore): Promise<void> {
  for (const memory of SEED_MEMORIES) {
    const key = await store.add(memory, { kind: 'preference' })
    console.log(`stored: ${memory} (${key})`)
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
