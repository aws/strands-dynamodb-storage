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
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBStorage

storage = DynamoDBStorage("agent-data", region_name="us-east-1")
agent = Agent(session_manager=SnapshotSessionManager(storage=storage))
```

The same instance backs any subsystem that accepts a `Storage`, for example offloading oversized
tool results with the context offloader:

```python
from strands import Agent
from strands.vended_plugins.context_offloader import ContextOffloader

agent = Agent(plugins=[ContextOffloader(storage=storage)])
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
- Optional per-item TTL (`ttl_seconds=...`) with read/list/search expiry filtering.
- Native vector `search()` via Amazon DynamoDB vector indexes (`SearchVectors`,
  requires boto3 >= 1.43.64); a `vector_search` adapter can override the call.
- `DynamoDBMemoryStore`, a Strands `MemoryStore` for the Memory Manager, with
  `BedrockEmbedder` (Amazon Titan Text Embeddings V2, Cohere Embed v3 and v4).

## Long-term memory

`DynamoDBMemoryStore` gives the Strands Memory Manager a long-term memory on the same table.
It embeds each memory through Amazon Bedrock, stores the text with its embedding, and finds
memories by meaning with one `SearchVectors` call. The table needs the vector index from the
repository README's [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions), with
`Dimensions` 1024 for the default embedder (Amazon Titan Text Embeddings V2).

```python
from strands import Agent
from strands.memory import MemoryManager
from strands_dynamodb_storage import DynamoDBMemoryStore, DynamoDBStorage

storage = DynamoDBStorage("agent-storage", region_name="us-east-1").namespace("user/u1")
memory = DynamoDBMemoryStore(storage)
agent = Agent(memory_manager=MemoryManager(stores=[memory], add_tool_config=True))

# The store also works on its own.
key = await memory.add("Prefers a window seat", {"source": "chat"})
entries = await memory.search("Where does the user like to sit?")
entries[0].content   # "Prefers a window seat"
entries[0].metadata  # {"source": "chat", "_key": "memory/dynamodb/8b29884b...", "_score": 0.59}
```

How it behaves:

- **One item per memory.** A memory is stored at `memory/<name>/<id>` inside the storage
  namespace, where `<name>` is the store name (default `dynamodb`) and `<id>` is a hash of the
  text. Adding the same text again overwrites that item, replacing its metadata, so retried
  writes do not create duplicates. `await storage.delete((entry.metadata or {})["_key"])` removes a memory.
- **Two calls per operation.** `add()` makes one Bedrock call and one `PutItem`; `search()`
  makes one Bedrock call and one `SearchVectors`, pinned to the partition that holds the
  namespace. With the index projection `ALL`, memories come back from the index, so a search
  reads no base-table items; a memory large enough to be offloaded to S3 costs one `GetItem`
  and one S3 read. Results come closest first; `_score` is the raw index score (lower is
  closer for COSINE).
- **One memory store per partition.** The partition is the first two segments of the full key:
  `namespace("user/u1")` gives the store partition `user/u1`, but `user/u1/agent-a` and
  `user/u1/agent-b` share partition `user/u1`. Every item with a vector in the partition competes
  for the same search results, and the store drops matches that are not its own memories before
  reading them, so a second memory store (or the application's own vector items) in the partition
  costs no extra reads but can leave results short of `max_search_results`. Give each memory
  store a two-segment namespace of its own, such as `user/<id>`.
- **Only memories are embedded.** Session snapshots and other values written through the same
  storage never call Bedrock.
- **TTL applies.** With `ttl_seconds` on the storage, memories expire like any other item, and
  search drops expired matches.
- **Metadata** values that are strings, booleans, or numbers that are zero or from 1e-130 to
  `2**53 - 1` in magnitude are kept, the same set the TypeScript package accepts. Other values
  (lists, nested dicts, `None`, numbers outside that range), keys or strings containing a lone
  surrogate (which DynamoDB stores as `?`), empty keys, the keys `__proto__` and `constructor`
  (which the AWS SDK for JavaScript cannot write), and the reserved keys `_key` and `_score` are
  dropped with a logged warning, so a memory whose metadata came from a model's extraction is
  still stored.

The default embedder runs in the storage's Region with its `boto_session` and
`boto_client_config`, and needs `bedrock:InvokeModel` on the model. When the storage was given a
`client`, the embedder takes that client's Region but resolves its own credentials through the
standard chain; pass `BedrockEmbedder(client=...)` to control them. To use another model, pass
an embedder. `dimensions` must match the index's `Dimensions`; a mismatch fails the write or
search with a `StorageError` whose cause names the expected and actual sizes:

```python
from strands_dynamodb_storage import BedrockEmbedder

# Cohere Embed v4 through a cross-Region inference profile; Cohere embeds memories and
# queries with separate input types.
memory = DynamoDBMemoryStore(storage, embedder=BedrockEmbedder("us.cohere.embed-v4:0", dimensions=1024))

# Any other model: a function from text to a list of floats (sync or async), or an object
# with an `embed(self, text, purpose)` method (sync or async) where purpose is "document" or
# "query".
memory = DynamoDBMemoryStore(storage, embedder=lambda text: my_model.encode(text).tolist())
```

`BedrockEmbedder` supports Titan Text Embeddings V2 (256, 512, or 1024 dimensions), Cohere Embed
v3 English and multilingual (1024), and Cohere Embed v4 (256, 512, 1024, or 1536), by model ID,
cross-Region inference profile ID, or ARN. An application inference profile or provisioned model
ARN does not name the model, so pass its request format as well:
`BedrockEmbedder(profile_arn, model_family="amazon.titan-embed-text-v2")`. Bedrock limits the
input size. Titan Text Embeddings V2 rejects a text over 8,192 tokens or 50,000 characters, and
Cohere Embed v3 a text over 2,048 characters; either fails with a `StorageError` that carries
Bedrock's message. Cohere Embed v3 also embeds only the first 512 tokens of a text, without an
error, so a long memory is found by its beginning. Cohere Embed v4 accepts up to about 128,000
tokens per text.

## Semantic search

`search()` is the lower-level API under the memory store: write each item with its embedding,
then query by vector. Scoring runs *in the database* against a DynamoDB
vector index (no second vector store, no ETL), and because the index is partitioned on `pk`,
every search is scoped to the caller's key space -- one tenant's memories can never surface
in another's results. Creating the table with a vector index (and the IAM permissions needed)
is covered in the repository README's [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions).

Two behaviours to know: `pk` is required whenever the index's `SearchSchema` declares a HASH
element (the provisioning guide's setup does) and must be omitted when it doesn't. And with TTL
enabled, `search()` drops matches whose expiry has passed, reading the TTL attribute from the vector
index at no extra read cost. That needs the index to project it (`ALL`, or `INCLUDE` listing the TTL
attribute); with a narrower projection matches are returned unfiltered. A `vector_search` adapter opts
in by returning `expire_at` (epoch seconds) on each match. Expired matches are not back-filled, so a
search can return fewer than `top_k` results, and an item whose TTL was just renewed can be briefly
hidden until the index catches up with the write.

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
    pk=store.partition_key("memories/m1"),  # the partition holding these keys ("user/u1")
    filter={"kind": "preference"},        # optional metadata equality filter
    include_values=True,                  # hydrate each match's stored bytes
))
for r in results:
    print(r.key, r.score, r.data)
# ordered most-similar-first; score direction follows the index's distance function
# (COSINE/EUCLIDEAN: lower = nearer; DOT_PRODUCT: higher = more similar)
```

`search()` takes a `SearchQuery` only; a plain-string query raises `StorageError`, because the
storage does not embed text. Like a global secondary index, the vector index is eventually
consistent, and a freshly created index backfills before it is searchable. Requires `boto3 >= 1.43.64`; a
`vector_search` adapter, when configured, overrides the native call (testing, custom routing).

## Examples

Runnable, live-verified examples for every capability, from session resume to a
customer-support capstone, live in the
[examples library](https://github.com/aws/strands-dynamodb-storage/tree/main/examples).

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q            # unit tests (moto, offline)
.venv/bin/ruff check src tests && .venv/bin/mypy src
RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q   # real DynamoDB + S3
```
