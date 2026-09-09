"""Host a Strands agent on Amazon Bedrock AgentCore Runtime with DynamoDB sessions.

AgentCore Runtime runs each session in an isolated microVM that is recycled
when the session goes idle, so anything the agent holds in memory is gone on
the next invocation. Backing the session manager with DynamoDB moves the
conversation into your own table: the runtime supplies a session id on every
invocation, the session manager snapshots the conversation under that id, and
the same id arriving later restores it, whichever microVM serves the request.

The same wiring applies whether you wrote this agent by hand or generated it
with the AgentCore harness export (see README.md for the harness relationship).

Prerequisites:
  - A pk/sk table (see README.md).
  - The runtime execution role needs PutItem/GetItem/DeleteItem/Query on it.
  - pip install bedrock-agentcore strands-agents strands-dynamodb-storage

Usage (local):
  python agentcore_runtime.py
  curl -X POST http://localhost:8080/invocations \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "Remember this: my rental car is a blue Nissan Micra"}'
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBStorage

TABLE = os.environ.get("AGENT_STORAGE_TABLE", "agent-storage")
REGION = os.environ.get("AWS_REGION", "us-east-1")

app = BedrockAgentCoreApp()

# One storage client for the process; sessions differ only by key prefix,
# so every session this container serves lands in the same table.
storage = DynamoDBStorage(TABLE, region_name=REGION)


@app.entrypoint
def invoke(payload, context):
    # The runtime passes the caller's runtimeSessionId in the request context.
    # Locally (plain curl, no session header) there is none, so fall back.
    session_id = context.session_id or "local-dev-session"

    # Snapshot on every turn, restore on the next one. The session manager is
    # cheap to construct; the durable state lives entirely in DynamoDB.
    session = SnapshotSessionManager(session_id=session_id, storage=storage)
    agent = Agent(session_manager=session)

    result = agent(payload.get("prompt", "Hello"))
    return {"result": result.message}


if __name__ == "__main__":
    app.run()
