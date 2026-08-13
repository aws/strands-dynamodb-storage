# strands-dynamodb-storage (Python)

Python implementation of the Amazon DynamoDB `Storage` backend for the Strands Agents SDK — at
parity with `../typescript/`. Implements the SDK's `strands.storage.Storage` protocol
(`write`/`read`/`delete`/`list`, plus `namespace`) so one DynamoDB-backed instance serves Session
Manager, Memory Manager, and any subsystem that persists bytes.

## Install

```bash
pip install strands-dynamodb-storage
```

## Usage

```python
from strands import Agent
from strands.session import SessionManager
from strands_dynamodb_storage import DynamoDBStorage

storage = DynamoDBStorage("agent-data", region_name="us-east-1")
agent = Agent(session_manager=SessionManager(storage=storage))
```

Direct byte usage (async):

```python
store = DynamoDBStorage("agent-data", region_name="us-east-1")
await store.write("sessions/s1/snapshot.json", b'{"turn": 1}')
data = await store.read("sessions/s1/snapshot.json")           # bytes | None
keys = await store.list("sessions/s1/")                         # native Query
scoped = await store.list(DynamoDBListQuery(pk="sessions/s1", sk_prefix="scopes/"))
await store.delete("sessions/s1/snapshot.json")
```

## Features (parity with the TypeScript package)

- Single-table design (`pk`/`sk`), with a structured `DynamoDBListQuery` extension point.
- Optional Amazon S3 offload for values above the item-size limit (`s3_bucket=...`).
- Optional gzip `compression="gzip"` (applied before the offload check).
- Optional per-item TTL (`ttl_seconds=...`) with read/list expiry filtering.
- Native vector `search()` via Amazon DynamoDB vector indexes (`SearchVectors`,
  requires boto3 >= 1.43.64); a `vector_search` adapter can override the call.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q            # unit tests (moto, offline)
.venv/bin/ruff check src tests && .venv/bin/mypy src
RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q   # real DynamoDB + S3
```
