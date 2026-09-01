/**
 * Self-expiring agent state with DynamoDB-native TTL.
 *
 * Ephemeral state accumulates fast in agent systems: anonymous visitor
 * sessions, one-off tool scratch space, verification codes. Without a
 * retention mechanism, someone ends up writing a cleanup job. With this
 * package, retention is a constructor argument: every write stamps an
 * epoch-seconds `expireAt` attribute, reads and listings filter items whose
 * expiry has passed, and DynamoDB physically reaps them in the background at
 * no request cost.
 *
 * The script writes one item with a short TTL and one without, waits out the
 * expiry, and shows all three views of the aftermath: the expired item is
 * gone from `read` and `list`, the persistent item is untouched, and the raw
 * DynamoDB item may still be physically present for a while, which is why the
 * package filters on read rather than trusting the reaper's timing.
 *
 * Prerequisites:
 *   - A pk/sk table with TTL enabled on `expireAt` (see README.md).
 *   - Credentials with PutItem/GetItem/DeleteItem/Query on the table.
 *
 * Usage:
 *   npx tsx ttl-sessions.ts --table agent-storage
 */

import { setTimeout as sleep } from 'node:timers/promises'
import { parseArgs } from 'node:util'

import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const TTL_SECONDS = 8

const encoder = new TextEncoder()
const decoder = new TextDecoder()

function show(data: Uint8Array | null): string {
  return data === null ? 'null' : JSON.stringify(decoder.decode(data))
}

async function main(table: string, region: string): Promise<void> {
  // ttlSeconds opts the store in to TTL: stamp on write, filter on read/list.
  const ephemeral = new DynamoDBStorage(table, { region, prefix: 'guest/g1', ttlSeconds: TTL_SECONDS })
  const durable = new DynamoDBStorage(table, { region, prefix: 'guest/g1' })

  await ephemeral.write('session/cart', encoder.encode('3 items, checkout not started'))
  await durable.write('profile/consent', encoder.encode('cookie banner accepted v3'))
  console.log(`wrote session/cart with ttlSeconds=${TTL_SECONDS}, profile/consent with no TTL`)

  const data = await ephemeral.read('session/cart')
  console.log(`before expiry: read session/cart -> ${show(data)}`)
  const keys = [...(await ephemeral.list('session/')), ...(await durable.list('profile/'))].sort()
  console.log(`before expiry: list ->  ${JSON.stringify(keys)}`)

  console.log(`\nwaiting ${TTL_SECONDS + 2}s for expiry...`)
  await sleep((TTL_SECONDS + 2) * 1000)

  // The store filters the expired item immediately; DynamoDB reaps it later.
  console.log(`after expiry: read session/cart -> ${show(await ephemeral.read('session/cart'))}`)
  console.log(`after expiry: list session/ -> ${JSON.stringify(await ephemeral.list('session/'))}`)
  console.log(`after expiry: read profile/consent -> ${show(await durable.read('profile/consent'))}`)

  // Peek underneath: the raw item can still be physically present until the
  // background reaper removes it. This is why the package filters on read.
  const ddb = new DynamoDBClient({ region })
  const raw = await ddb.send(
    new GetItemCommand({
      TableName: table,
      Key: { pk: { S: 'guest/g1' }, sk: { S: 'session/cart' } },
    }),
  )
  console.log(`raw GetItem still returns an item: ${raw.Item !== undefined} (reaper removes it in the background)`)

  await durable.delete('profile/consent')
  console.log('\ndemo items cleaned up (expired item left for the reaper)')
}

const { values } = parseArgs({
  options: {
    table: { type: 'string', default: 'agent-storage' },
    region: { type: 'string', default: 'us-east-1' },
  },
})
await main(values.table, values.region)
