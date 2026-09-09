/**
 * Host a Strands agent on Amazon Bedrock AgentCore Runtime with DynamoDB sessions.
 *
 * AgentCore Runtime runs each session in an isolated microVM that is recycled
 * when the session goes idle, so anything the agent holds in memory is gone on
 * the next invocation. Backing the session manager with DynamoDB moves the
 * conversation into your own table: the runtime supplies a session id on every
 * invocation, the session manager snapshots the conversation under that id,
 * and the same id arriving later restores it, whichever microVM serves the
 * request.
 *
 * There is no TypeScript runtime wrapper equivalent to the Python
 * bedrock-agentcore package, so this file implements the runtime's HTTP
 * contract directly with node:http: POST /invocations and GET /ping on port
 * 8080, with the session id arriving in the
 * X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header.
 *
 * Prerequisites:
 *   - A pk/sk table (see README.md).
 *   - The runtime execution role needs PutItem/GetItem/DeleteItem/Query on it.
 *
 * Usage (local):
 *   npx tsx agentcore-runtime.ts
 *   curl -X POST http://localhost:8080/invocations \
 *     -H 'Content-Type: application/json' \
 *     -d '{"prompt": "Remember this: my rental car is a blue Nissan Micra"}'
 */

import { createServer } from 'node:http'

import { Agent, SessionManager } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const TABLE = process.env.AGENT_STORAGE_TABLE ?? 'agent-storage'
const REGION = process.env.AWS_REGION ?? 'us-east-1'
const SESSION_HEADER = 'x-amzn-bedrock-agentcore-runtime-session-id'

// One storage client for the process; sessions differ only by key prefix,
// so every session this container serves lands in the same table.
const storage = new DynamoDBStorage(TABLE, { region: REGION })

const server = createServer(async (req, res) => {
  // Health check: the runtime polls this to decide the container is ready.
  if (req.method === 'GET' && req.url === '/ping') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ status: 'Healthy' }))
    return
  }

  if (req.method === 'POST' && req.url === '/invocations') {
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(chunk as Buffer)

    try {
      const payload = JSON.parse(Buffer.concat(chunks).toString() || '{}') as { prompt?: string }

      // The runtime passes the caller's runtimeSessionId in this header.
      // Locally (plain curl, no session header) there is none, so fall back.
      const sessionId = (req.headers[SESSION_HEADER] as string | undefined) ?? 'local-dev-session'

      // Snapshot on every turn, restore on the next one. The session manager
      // is cheap to construct; the durable state lives entirely in DynamoDB.
      const session = new SessionManager({ sessionId, storage })
      const agent = new Agent({ sessionManager: session })

      const result = await agent.invoke(payload.prompt ?? 'Hello')
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ result: result.lastMessage }))
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: String(err) }))
    }
    return
  }

  res.writeHead(404, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify({ error: 'not found' }))
})

server.listen(8080, '0.0.0.0', () => {
  console.log(`agent listening on :8080, sessions stored in table ${TABLE}`)
})
