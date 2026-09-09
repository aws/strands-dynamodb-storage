# DynamoDB sessions on Amazon Bedrock AgentCore Runtime

AgentCore Runtime runs each session in an isolated microVM that is recycled
when the session goes idle: anything the agent holds in memory is gone on the
next invocation. This example backs the Strands session manager with
DynamoDB, so the conversation snapshot lives in your own table. The runtime
supplies a session id on every invocation, the session manager persists the
conversation under that id, and the same id arriving later restores it,
whichever microVM serves the request.

## Where this fits next to the AgentCore harness

The AgentCore managed harness is configuration, not code: its state
persistence is AgentCore Memory, and its bring-your-own option attaches an
existing AgentCore Memory instance by ARN, not an arbitrary backend. There is
no route to DynamoDB-backed storage from harness configuration.

The moment you hold Strands code, this example applies. That happens two
ways: you write the agent yourself (this example), or you run
[harness export](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-export.html),
which generates your harness as editable Python on the Strands framework.
Adding DynamoDB storage to an exported agent is the same three lines this
example adds: construct the storage, hand it to a session manager keyed by
the runtime session id, hand the session manager to the agent.

## How it works

The Python variant uses the `bedrock-agentcore` runtime wrapper, which serves
the required HTTP contract for you and passes the caller's `runtimeSessionId`
in the request context:

```python
@app.entrypoint
def invoke(payload, context):
    session = SnapshotSessionManager(session_id=context.session_id, storage=storage)
    agent = Agent(session_manager=session)
    return {"result": agent(payload.get("prompt", "Hello")).message}
```

There is no TypeScript equivalent of that wrapper, so the TypeScript variant
implements the
[runtime HTTP contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
directly: `POST /invocations` and `GET /ping` on port 8080, with the session
id arriving in the `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header.

## Run it locally

Create a pk/sk table (once):

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

Start the agent server:

```bash
pip install bedrock-agentcore strands-agents strands-dynamodb-storage
python agentcore_runtime.py
```

Or the TypeScript equivalent:

```bash
npm install strands-dynamodb-storage @strands-agents/sdk tsx
npx tsx agentcore-runtime.ts
```

Then talk to it, and restart the server between the two calls to prove the
state lives in DynamoDB rather than in the process:

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: verify-session-001' \
  -d '{"prompt": "Remember this: my rental car is a blue Nissan Micra"}'

# restart the server, then:
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: verify-session-001' \
  -d '{"prompt": "What is my rental car?"}'
```

The second answer comes from a process that never saw the first message
(response metadata elided):

```text
{"result": {"role": "assistant", "content": [{"text": "Got it! Your rental car is a **blue Nissan Micra**. I'll remember that for our conversation."}]}}
{"result": {"role": "assistant", "content": [{"text": "Your rental car is a **blue Nissan Micra**!"}]}}
```

Locally the session id comes from the header on the curl call; on the
runtime, the platform sets it from the caller's `runtimeSessionId`.

## Deploy

Deployment to AgentCore Runtime (container build, ECR, role creation) is the
standard flow documented in
[deploying Strands agents to AgentCore Runtime](https://strandsagents.com/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/);
nothing about it changes for this example. Two additions:

- Set the `AGENT_STORAGE_TABLE` environment variable on the runtime if your
  table is not named `agent-storage`.
- Add the storage permissions to the runtime execution role:

```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:PutItem",
    "dynamodb:GetItem",
    "dynamodb:DeleteItem",
    "dynamodb:Query"
  ],
  "Resource": "arn:aws:dynamodb:us-east-1:111122223333:table/agent-storage"
}
```

The package's [README](../../README.md) carries the additional statements for
semantic memory, S3 offload, and embedding model invocation if you extend the
agent with those features.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
```
