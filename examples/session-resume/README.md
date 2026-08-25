# Resume a session after a restart

An agent holds its conversation in process memory, so a deploy, a crash, or a
Lambda environment being recycled ends the conversation. This example persists
the agent's state to an Amazon DynamoDB table through
`SnapshotSessionManager`, so a brand-new process picks up exactly where the
last one stopped.

## How it works

`SnapshotSessionManager` snapshots the agent after each invocation and writes
the snapshot through the `Storage` contract. Handing it a `DynamoDBStorage`
makes DynamoDB the backend, and the manager namespaces everything it writes
under `session/<session_id>/`, so one table serves any number of sessions:

```python
storage = DynamoDBStorage("agent-storage", region_name="us-east-1")
session = SnapshotSessionManager("session-resume-demo", storage=storage)
agent = Agent(session_manager=session)
```

On construction, the manager looks for the latest snapshot under the session
id and restores it if present. That is the whole resume mechanism: there is no
separate restore call, and no difference between the code for the first run
and the hundredth.

## Run it

Create the table (once) and provide credentials that can reach it and invoke
your Bedrock model:

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

pip install strands-agents strands-dynamodb-storage
```

Tell the agent something, then exit the process:

```bash
python session_resume.py --table agent-storage tell \
  "My rental car is a blue Nissan Micra, parked in bay 42"
```

Ask about it from a completely new process:

```bash
python session_resume.py --table agent-storage ask \
  "What car am I driving and where is it parked?"
```

```text
Based on what you just told me, you are driving a blue Nissan Micra,
parked in bay 42.
```

The second process started with nothing in memory. The answer came from the
snapshot in DynamoDB.

## Why DynamoDB here

Session state sits on the hot path of every agent turn: it is loaded before
each model call. DynamoDB point reads keep that load at single-digit
milliseconds, on-demand capacity absorbs a fleet of sessions going from idle
to thousands of concurrent invocations, and the HTTP data plane needs no
connection pool, which matters on AWS Lambda and Amazon Bedrock AgentCore
Runtime where the process serving a session changes constantly.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
```
