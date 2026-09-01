/**
 * A support assistant with per-customer memory, on one DynamoDB table.
 *
 * The capstone for this examples library: everything the other examples show,
 * combined the way a real assistant uses it. One table carries, per customer:
 *
 * - **Session state**, so a ticket conversation survives restarts and handoffs
 *   (`SessionManager`).
 * - **Long-term memories**, recalled by meaning across tickets
 *   (`MemoryManager` over a DynamoDB vector index).
 * - **Tenant isolation**, so one customer's history never reaches another's
 *   conversation (constructor prefix + partition-scoped search).
 *
 * The demo seeds account facts for one customer, then opens a support ticket
 * in a fresh process: the customer reports dropped connections, and the agent
 * connects that to a months-old memory about their proxy closing idle sockets,
 * which shares no words with the complaint. A follow-up run resumes the same
 * ticket from the persisted session.
 *
 * Prerequisites: vector-indexed table (see README.md), Bedrock access for the
 * agent model and Titan embeddings.
 *
 * Usage:
 *   npx tsx support-assistant.ts --table agent-storage seed
 *   npx tsx support-assistant.ts --table agent-storage ticket \
 *       "Our uploads keep failing after about a minute. Nothing changed on our side."
 *   npx tsx support-assistant.ts --table agent-storage ticket "Thanks. What did we conclude?"
 */

import { randomUUID } from 'node:crypto'
import { parseArgs } from 'node:util'

import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime'
import { Agent, MemoryManager, SessionManager } from '@strands-agents/sdk'
import type { JSONValue, MemoryEntry, MemoryStore, SearchOptions } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'
import type { SearchQuery } from 'strands-dynamodb-storage'

const CUSTOMER = 'customer/acme'
const TICKET_SESSION = 'ticket-7341'

const ACCOUNT_FACTS = [
  "Acme's network runs an outbound proxy that closes idle connections after 60 seconds",
  "Acme pinned their integration to a custom build of the SDK, version 2.14",
  "Acme's technical contact prefers email and asked not to be called by phone",
]

const SYSTEM_PROMPT =
  "You are a support engineer for a data-transfer service. Use the customer's " +
  'account memories when they are relevant to the reported symptom, and say ' +
  'which remembered fact you are basing your answer on.'

type Embedder = (text: string) => Promise<number[]>

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
  readonly name = 'account-memory'
  readonly description = "This customer's account facts and history, searched by meaning"
  readonly maxSearchResults = 3
  readonly writable = true

  constructor(
    private readonly storage: DynamoDBStorage,
    private readonly partition: string,
    private readonly embed: Embedder,
  ) {}

  async add(content: string, metadata?: Record<string, JSONValue>): Promise<void> {
    // The storage backend indexes flat primitive metadata; keep those entries.
    const flat: Record<string, string | number | boolean> = {}
    for (const [key, value] of Object.entries(metadata ?? {})) {
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        flat[key] = value
      }
    }
    await this.storage.write(
      `memories/${randomUUID().replaceAll('-', '').slice(0, 8)}`,
      new TextEncoder().encode(content),
      { vector: await this.embed(content), metadata: flat },
    )
  }

  async search(query: string, _options?: SearchOptions): Promise<MemoryEntry[]> {
    const searchQuery: SearchQuery = {
      vector: await this.embed(query),
      topK: this.maxSearchResults,
      pk: this.partition,
      includeValues: true,
    }
    const results = await this.storage.search(searchQuery)
    return results.flatMap((r) =>
      r.data === undefined
        ? []
        : [
            {
              content: new TextDecoder().decode(r.data),
              metadata: r.metadata as Record<string, JSONValue> | undefined,
            },
          ],
    )
  }
}

/** One storage instance carries the whole assistant for this customer. */
function buildAgent(table: string, region: string): Agent {
  const storage = new DynamoDBStorage(table, { region, prefix: CUSTOMER })
  const store = new DynamoDBMemoryStore(storage, CUSTOMER, makeEmbedder(region))
  const memory = new MemoryManager({ stores: [store], addToolConfig: true })
  const session = new SessionManager({ sessionId: TICKET_SESSION, storage })
  return new Agent({ systemPrompt: SYSTEM_PROMPT, sessionManager: session, memoryManager: memory })
}

async function seed(table: string, region: string): Promise<void> {
  const storage = new DynamoDBStorage(table, { region, prefix: CUSTOMER })
  const store = new DynamoDBMemoryStore(storage, CUSTOMER, makeEmbedder(region))
  for (const fact of ACCOUNT_FACTS) {
    await store.add(fact, { kind: 'account-fact' })
    console.log(`stored: ${fact}`)
  }
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    options: {
      table: { type: 'string', default: 'agent-storage' },
      region: { type: 'string', default: 'us-east-1' },
    },
    allowPositionals: true,
  })
  const [verb, message = 'Our uploads keep failing after about a minute.'] = positionals
  if (verb !== 'seed' && verb !== 'ticket') {
    console.error(
      'usage: npx tsx support-assistant.ts [--table NAME] [--region REGION] {seed|ticket} [message]',
    )
    process.exit(2)
  }

  if (verb === 'seed') {
    await seed(values.table, values.region)
  } else {
    const agent = buildAgent(values.table, values.region)
    await agent.invoke(message)
  }
}

await main()
