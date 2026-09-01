/**
 * Resume a conversation after a restart, with session state in Amazon DynamoDB.
 *
 * Run 1 tells the agent a fact and exits, taking the process (and all in-memory
 * state) with it. Run 2 constructs a brand-new agent with the same session id
 * and asks about the fact. The answer comes from the snapshot that
 * `SessionManager` persisted to DynamoDB, not from anything held in memory.
 *
 * Prerequisites:
 *   - A DynamoDB table with string keys `pk` / `sk` (see README.md).
 *   - AWS credentials with PutItem/GetItem/DeleteItem/Query on the table and
 *     InvokeModel on your Bedrock model.
 *
 * Usage:
 *   npx tsx session-resume.ts --table agent-storage tell "My rental car is a blue Nissan Micra"
 *   npx tsx session-resume.ts --table agent-storage ask "What car am I driving?"
 */

import { parseArgs } from 'node:util'

import { Agent, SessionManager } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const SESSION_ID = 'session-resume-demo'

/**
 * Construct an agent whose session state lives in DynamoDB.
 *
 * A new process starts with no in-memory state; if a snapshot for
 * SESSION_ID exists in the table, the session manager restores it here.
 */
function buildAgent(table: string, region: string): Agent {
  const storage = new DynamoDBStorage(table, { region })
  const session = new SessionManager({ sessionId: SESSION_ID, storage })
  return new Agent({ sessionManager: session })
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    options: {
      table: { type: 'string', default: 'agent-storage' },
      region: { type: 'string', default: 'us-east-1' },
    },
    allowPositionals: true,
  })

  const [verb, message] = positionals
  if ((verb !== 'tell' && verb !== 'ask') || message === undefined) {
    console.error('usage: npx tsx session-resume.ts [--table TABLE] [--region REGION] {tell,ask} message')
    process.exit(2)
  }

  const agent = buildAgent(values.table, values.region)
  await agent.invoke(message)
}

await main()
