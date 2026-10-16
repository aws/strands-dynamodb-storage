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
- Opt-in `LexicalIndex` (preview) for term and exact-identifier retrieval, also usable by SDK
  consumers as a `search_strategy`; see below.

## Semantic search

`search()` gives an agent semantic long-term memory over the same table: write each memory
with its embedding, then query by meaning. Scoring runs *in the database* against a DynamoDB
vector index (no second vector store, no ETL), and because the index is partitioned on `pk`,
every search is scoped to the caller's key space -- one tenant's memories can never surface
in another's results. Creating the table with a vector index (and the IAM permissions needed)
is covered in the repository README's [Provisioning and permissions](../#provisioning-and-permissions).

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
    pk="user/u1",                         # the physical partition: the full key's first two segments
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

## Lexical document index (preview)

`LexicalIndex` is an opt-in index for finding stored documents by the words they contain
(`search`) or by an exact identifier such as an error code (`lookup`). It owns the write path of
indexed documents: `upsert` writes the document to the storage table and its index entries (a
manifest plus one posting per term and identifier) to a separate index table in one
`TransactWriteItems` call. The byte `Storage` contract is unchanged; `DynamoDBStorage.search()`
accepts a plain string only when a `search_strategy` is configured ([SDK search strategy](#sdk-search-strategy)).
*Preview* means the API and the on-table layout (`lexical-v1`, `LEXICAL_TOKENIZER_VERSION`) may
change before it is marked stable; the design is open for maintainer review.

```python
from strands_dynamodb_storage import DynamoDBStorage, IdentifierQuery, LexicalIndex, LexicalQuery, SearchableDocument

storage = DynamoDBStorage("agent-data", region_name="us-east-1")
# The storage prefix is the lexical scope: bind the authenticated tenant to its namespace.
index = LexicalIndex(storage.namespace("tenant-a"), index_table_name="agent-lexical-index")

revision = await index.upsert(SearchableDocument(
    key="tickets/42",
    data=b'{"title": "Login fails"}',                    # stored like write(); readable with read()
    text="Login fails with ERR_AUTH_403 on api-prod-7",  # tokenized, never stored
    identifiers=["ERR_AUTH_403"],                        # exact, case-sensitive
    metadata={"status": "open"},
))
response = await index.search(LexicalQuery(text="login api-prod-7", top_k=5, filter={"status": "open"}))
exact = await index.lookup(IdentifierQuery(identifier="ERR_AUTH_403", top_k=5, include_values=True))
current = await index.revision("tickets/42")                   # strongly consistent; None if absent
await index.delete("tickets/42", expected_revision=revision)   # RevisionConflictError if it moved on
```

`search` scores *matched terms ÷ distinct query terms*, ties broken by key in UTF-8 byte order;
`require_all_terms=True` keeps only full matches. `lookup` returns the documents indexed with
exactly that identifier (score 1.0), by key. Results carry `key`, `score`, `matched_terms`,
`metadata` and, with `include_values=True`, `data`; the response also reports
`truncated`, `truncation_reasons` and `candidates_examined`. `filter` is strict equality on current
metadata (a missing field never matches; a bool never equals a number). `upsert` also accepts
`expected_revision`. A key indexed in one scope cannot be overwritten or deleted through another.

Parts cause partial matches: `FC-123` and `FC-00123` are different terms but share the part `fc`,
so a text search for `FC-123` can also return FC-00123 documents with a partial score. For exact
matching use `lookup` or `require_all_terms=True`.

### Provisioning and IAM

The library never creates tables. Create the index table (string `pk`/`sk`) and wait for it. Only
if the storage uses TTL, enable TTL on the **same attribute name** it stamps (`ttl_attribute`,
default `expireAt`); postings copy their document's expiry, manifests never expire.

```bash
aws dynamodb create-table --table-name agent-lexical-index \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region us-east-1
aws dynamodb wait table-exists --table-name agent-lexical-index --region us-east-1
aws dynamodb update-time-to-live --table-name agent-lexical-index --region us-east-1 \
  --time-to-live-specification Enabled=true,AttributeName=expireAt   # optional, TTL only
```

Add these statements to the storage policy ([Provisioning and permissions](../#provisioning-and-permissions)).
Actions inside `TransactWriteItems` are authorized as `PutItem`, `DeleteItem` and
`ConditionCheckItem` (condition checks target only the index table). `s3:DeleteObject` (already in the
S3 offload statement) is used only to reclaim, best-effort, a value that `write()` previously offloaded
under the same key.

```json
[
  {
    "Effect": "Allow",
    "Action": ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem", "dynamodb:BatchGetItem"],
    "Resource": "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data"
  },
  {
    "Effect": "Allow",
    "Action": ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:ConditionCheckItem", "dynamodb:GetItem", "dynamodb:Query"],
    "Resource": "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-lexical-index"
  }
]
```

### Consistency and cost

- **Writes are atomic.** An `upsert` or `delete` never leaves a document and its index
  half-written. Conflicting or throttled transactions are retried (`max_conflict_retries`); an
  `expected_revision` mismatch raises `RevisionConflictError` without retrying.
- **Reads are not a point-in-time snapshot.** Postings come from eventually consistent `Query`
  calls; each candidate is then validated individually with a strongly consistent read of the
  document (revision, scope, key, TTL expiry, current metadata). Documents changed during a call
  may or may not appear, and a just-written document can be missing until its postings are
  visible. With `include_values`, kept results are re-read after ranking and dropped if they
  changed, so fewer than `top_k` may come back. Unprocessed `BatchGetItem` keys are retried; after
  `max_unprocessed_retries` rounds without progress the call raises rather than answer partially.

Transactional writes perform two underlying writes per item (prepare and commit) and consume
capacity even when cancelled, so conflict retries cost too. Strongly consistent reads cost twice
eventually consistent ones, and reads are charged on full item size regardless of projection.

| Operation | Reads | Writes |
| --- | --- | --- |
| `upsert` | 2 strongly consistent `GetItem` (manifest, document) | 1 transaction of 2 + postings(new ∪ old) items |
| `delete` | 2 strongly consistent `GetItem` | 1 transaction of up to 2 + old postings items (+1 manifest `ConditionCheck` when there is no manifest) |
| `revision` | 1 strongly consistent `GetItem` | — |
| `search` / `lookup` | ≤ `max_pages_per_term` eventually consistent `Query` pages per term; strongly consistent `BatchGetItem` of **all** validated candidates (not just `top_k`), about Σ ceil(item size / 4 KB) RCU; with `include_values`, a second one of the `top_k` winners | — |
| `repair` | 1 strongly consistent `Query` page of manifests; 1 strongly consistent `GetItem` per manifest | 1 transaction per removed or rebuilt document |

### Limits

`LexicalIndexLimits` is validated on construction. Limits are enforced before any write and text
is never silently truncated: an over-limit document, term or identifier raises `StorageError`.

| Limit | Default | Bound and why |
| --- | --- | --- |
| `max_postings_per_document` | 49 | Hard ceiling: an overwrite with disjoint postings needs 2 + old + new ≤ 100 `TransactWriteItems` actions, so (100 − 2) / 2 = 49 |
| `max_term_bytes` / `max_identifier_bytes` | 64 / 128 | Keeps posting keys small |
| `max_text_bytes` | 65,536 | Bounds tokenization work per document |
| `max_query_terms` | 16 | ≤ `max_postings_per_document`; bounds `Query` fan-out |
| `page_size` / `max_pages_per_term` | 100 / 5 | Page ≤ 1000; bounds posting reads per term |
| `max_candidates` | 300 | ≤ 1000; bounds validation reads per call |
| `max_concurrency` | 4 | ≤ 16 parallel requests per call |
| `max_conflict_retries` / `max_unprocessed_retries` | 3 / 5 | May be 0; the latter counts consecutive rounds without progress |

Also enforced: `top_k` 1–100; the full key (scope + key) and the scope ≤ 1024 UTF-8 bytes; posting
and manifest partition keys ≤ 2048 bytes; indexed values must fit inline. Retrieval truncation is
reported as `truncated=True` with `truncation_reasons` (`"max_candidates"`, `"max_pages_per_term"`);
postings are read in key order, so truncation biases results toward lower keys.

### Tokenizer and identifiers

- Text is NFC-normalized and split into runs of letters, marks, numbers and the ASCII joiners `-`
  and `_`; everything else (whitespace, punctuation, symbols, emoji) separates terms. Only ASCII
  `A`–`Z` is lowercased: `ÁRBOL` becomes `Árbol`, not `árbol`.
- A joined run is indexed whole and by its parts: `ERR_AUTH_403` → `err_auth_403`, `err`, `auth`,
  `403`. Leading zeroes are kept. No stemming, no stopwords; unspaced CJK text forms one term per
  run (`日本語テキスト` is a single term).
- Identifiers are NFC-normalized, then matched exactly and case-sensitively (`ERR_AUTH_403` ≠
  `err_auth_403`); they must be non-empty, free of control characters and not start or end with
  whitespace. Text, identifiers and queries that are not well-formed Unicode are rejected.
- The runtime's Unicode database is used: code points assigned after Unicode 15.0 may tokenize
  differently between runtimes.

### Direct writes, TTL expiry and `repair`

Indexed items carry the revision and scope that wrote them. A direct `storage.write()` over an
indexed key drops those markers; a direct `storage.delete()` or a TTL expiry leaves index entries
behind. Retrieval detects all three at read time and skips the document, and
`repair(max_documents=100, cursor=None, rebuild_postings=False)` cleans up: it walks this scope's
manifests with `Query` (never a Scan), removes the manifest and postings of every missing,
overwritten, foreign or expired document, and optionally re-puts the postings of valid ones;
documents changed concurrently are skipped. Pass each `RepairReport.cursor` to the next call until
it is `None`. Manifests never expire, so run `repair` periodically when documents use TTL.

### SDK search strategy

SDK consumers that only know the byte `Storage` API, such as the SDK `FileMemoryStore` (it calls
`write` and then `search("natural language")`), can use the index through `LexicalSearchStrategy`.
As with the SDK's own storage backends, `write()` passes each stored value to the strategy and a
plain-string `search()` is delegated to it; a `SearchQuery` still takes the vector path. Without a
strategy, a plain-string `search()` raises `StorageError`.

```python
from strands.vended_memory_stores import FileMemoryStore
from strands_dynamodb_storage import DynamoDBStorage, LexicalSearchStrategy, SearchableText

strategy = LexicalSearchStrategy(
    index_table_name="agent-lexical-index",
    extract=lambda key, data: SearchableText(text=data.decode("utf-8")),  # you choose what is searchable
)
storage = DynamoDBStorage("agent-data", region_name="us-east-1", search_strategy=strategy)
memory = FileMemoryStore(storage=storage.namespace("tenant-a"), name="agent-memory")
await memory.add("# User preferences\nPrefers dark mode")
entries = await memory.search("which mode does the user prefer?")
```

- **Explicit extractor.** Bytes are never decoded implicitly: `extract(key, data)` returns
  `SearchableText(text=..., identifiers=...)`, or `None` to leave the value unindexed (an earlier
  indexed version of that key is then hidden from search, and `repair` removes its entries).
  Extracted text is not truncated: its terms plus identifiers must fit `max_postings_per_document`
  (default 49), as for `upsert`.
- **One extra write, at-least-once.** `write()` stores the value, then the strategy stores it again
  with its index entries through `upsert` (two strongly consistent reads and one transaction),
  keeping `vector`, `metadata` and `ttl_seconds`. If that step fails (for example too many postings,
  or a value offloaded to S3), `write()` raises `StorageError("Wrote '<key>' but indexing failed")`;
  the value stays stored but not searchable until it is written again.
  `LexicalIndex.upsert` remains the single-write, atomic path.
- **First N query terms.** `search()` uses the first `max_query_terms` (default 16) distinct query
  terms, skipping terms above `max_term_bytes`; a query without terms returns `[]`. It returns up to
  `top_k` (default 10) results with `key`, `score` and, with `include_values=True` (the default),
  `data`; truncation reasons are not reported.
- Each namespaced view searches only its own scope. Searching through the strategy needs
  strands-agents 1.54 or later (it returns the SDK's `StorageSearchResult`).

### Not supported

S3 offload for indexed documents (a value that would offload is rejected); global tables or
multi-Region use (transactions are atomic only in the Region where they run); online backfill
(`repair` cannot re-tokenize because source text is not stored, so a tokenizer change requires
re-upserting); hybrid vector + lexical ranking (RRF, proposed as the next increment); BM25 or other
corpus-statistics scoring; phrase or substring search.

## Examples

Runnable, live-verified examples for every capability, from session resume to a
customer-support capstone, live in the
[examples library](https://github.com/aws/strands-dynamodb-storage/tree/main/examples).
The preview lexical document index is the exception: it has no example yet and has only been
tested offline, not against AWS.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q            # unit tests (moto, offline)
.venv/bin/ruff check src tests && .venv/bin/mypy src
RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q   # real DynamoDB + S3
```
