/**
 * Serve many tenants from one DynamoDB table, with per-tenant isolation.
 *
 * A memory store serving many users has to keep one user's memories out of
 * another user's results. With this package that is a schema decision made at
 * construction time, not a filter appended to every query:
 *
 * - The constructor `prefix` pins every write, read, and listing inside the
 *   tenant's own key space, so two tenants resolve the same logical key to
 *   physically distinct partitions.
 * - The vector index is partitioned by `pk` (its search schema HASH element),
 *   so a search scoped to one tenant's partition never ranges over another
 *   tenant's vectors, no matter how similar they are.
 *
 * This script seeds two users with near-identical memories, then shows that
 * listings and semantic searches for one user never surface the other's data,
 * even when the other user's memory is the closest match in the whole table.
 *
 * Prerequisites: vector-indexed table (see README.md), Bedrock access for
 * Titan embeddings.
 *
 * Usage:
 *   npx tsx multi-tenant.ts --table agent-storage
 */

import { parseArgs } from 'node:util'

import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime'
import { DynamoDBStorage } from 'strands-dynamodb-storage'
import type { SearchQuery } from 'strands-dynamodb-storage'

const ALICE = 'user/alice'
const BOB = 'user/bob'

function makeEmbedder(region: string): (text: string) => Promise<number[]> {
  const bedrock = new BedrockRuntimeClient({ region })

  return async (text: string): Promise<number[]> => {
    const response = await bedrock.send(
      new InvokeModelCommand({
        modelId: 'amazon.titan-embed-text-v2:0',
        body: JSON.stringify({ inputText: text, dimensions: 1024 }),
      }),
    )
    const { embedding } = JSON.parse(new TextDecoder().decode(response.body)) as {
      embedding: number[]
    }
    return embedding
  }
}

async function main(table: string, region: string): Promise<void> {
  const embed = makeEmbedder(region)
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()

  // One store per tenant. Same table, same code path; only the prefix differs.
  const alice = new DynamoDBStorage(table, { region, prefix: ALICE })
  const bob = new DynamoDBStorage(table, { region, prefix: BOB })

  // Near-identical content on both sides. If scoping leaked, Bob's espresso
  // memory would be the nearest neighbor for Alice's coffee question.
  await alice.write('memories/m1', encoder.encode('Alice drinks oat-milk lattes, decaf after noon'), {
    vector: await embed('Alice drinks oat-milk lattes, decaf after noon'),
  })
  await bob.write('memories/m1', encoder.encode('Bob drinks double espressos, no milk ever'), {
    vector: await embed('Bob drinks double espressos, no milk ever'),
  })
  console.log('seeded one coffee memory per user (same logical key memories/m1)\n')

  // 1. The byte contract is scoped: each tenant lists only its own keys.
  console.log(`alice.list('memories/') -> ${JSON.stringify(await alice.list('memories/'))}`)
  console.log(`bob.list('memories/')   -> ${JSON.stringify(await bob.list('memories/'))}\n`)

  // 2. Search is scoped by partition: each tenant's query is answered only
  //    from that tenant's vectors.
  const question = 'how does this user take their coffee?'
  const tenants: Array<[string, DynamoDBStorage, string]> = [
    ['alice', alice, ALICE],
    ['bob', bob, BOB],
  ]
  for (const [name, store, pk] of tenants) {
    const query: SearchQuery = { vector: await embed(question), topK: 5, pk, includeValues: true }
    const results = await store.search(query)
    const answers = results.filter((r) => r.data != null).map((r) => decoder.decode(r.data))
    console.log(`search as ${name}: ${JSON.stringify(answers)}`)
    const other = name === 'alice' ? 'Bob' : 'Alice'
    const leaked = answers.some((a) => a.includes(other))
    console.log(`  ${other}'s memory in ${name}'s results: ${leaked}`)
    if (leaked) throw new Error('cross-tenant leak')
  }

  // Clean up the demo items.
  await alice.delete('memories/m1')
  await bob.delete('memories/m1')
  console.log('\ndemo items deleted')
}

const { values } = parseArgs({
  options: {
    table: { type: 'string', default: 'agent-storage' },
    region: { type: 'string', default: 'us-east-1' },
  },
})

main(values.table, values.region).catch((err) => {
  console.error(err)
  process.exitCode = 1
})
