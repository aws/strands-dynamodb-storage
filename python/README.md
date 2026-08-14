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
# Note: prefixes must cover at least a full scope and identifier ("scope/id/").
# list("") and single-segment prefixes are rejected as too broad -- they would
# require a cross-partition Scan. This deliberately narrows the SDK Storage
# contract (whose in-memory backends list everything on ""); SDK subsystems
# always pass namespaced prefixes and are unaffected.
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

## Semantic search

`search()` gives an agent semantic long-term memory over the same table: write each memory
with its embedding, then query by meaning. Scoring runs *in the database* against a DynamoDB
vector index (no second vector store, no ETL), and because the index is partitioned on `pk`,
every search is scoped to the caller's key space -- one tenant's memories can never surface
in another's results. Creating the table with a vector index (and the IAM permissions needed)
is covered in the repository README's [Provisioning and permissions](../#provisioning-and-permissions).

```python
from strands_dynamodb_storage import DynamoDBStorage, SearchQuery

store = DynamoDBStorage("agent-memory", region_name="us-east-1", prefix="user/u1")

# store a memory with its embedding (kept inline even when the payload offloads to S3)
await store.write(
    "memories/m1",
    b"likes window seats",
    vector=embed("likes window seats"),   # your embedding model, e.g. 1024 floats
    metadata={"kind": "preference"},
)

# recall by meaning, scoped to this store's partition
results = await store.search(SearchQuery(
    vector=embed("seating preferences?"),
    top_k=5,
    pk="user/u1/memories",                # required: the index declares a HASH element
    filter={"kind": "preference"},        # optional metadata equality filter
    include_values=True,                  # hydrate each match's stored bytes
))
for r in results:
    print(r.key, r.score, r.data)
# ordered most-similar-first; score direction follows the index's distance function
# (COSINE/EUCLIDEAN: lower = nearer; DOT_PRODUCT: higher = more similar)
```

Like a global secondary index, the vector index is eventually consistent, and a freshly
created index backfills before it is searchable. Requires `boto3 >= 1.43.64`; a
`vector_search` adapter, when configured, overrides the native call (testing, custom routing).

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q            # unit tests (moto, offline)
.venv/bin/ruff check src tests && .venv/bin/mypy src
RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q   # real DynamoDB + S3
```
