# strands-dynamodb-storage

An **Amazon DynamoDB `Storage` backend** for the [Strands Agents](https://github.com/strands-agents/harness-sdk) TypeScript SDK.

It implements the SDK's unified byte `Storage` interface (`write` / `read` / `delete` / `list` / `namespace`), so one
DynamoDB-backed instance can be passed to **Session Manager, Memory Manager**, the context offloader, transcripts, and
any other subsystem that persists bytes — no per-subsystem code. On top of the byte contract it adds S3 offload for large
values, optional gzip compression, TTL, optional **native vector search**, and an opt-in
[lexical document index](#lexical-document-index-preview) (preview).

## Install

```bash
npm install strands-dynamodb-storage @strands-agents/sdk @aws-sdk/client-dynamodb @aws-sdk/lib-dynamodb
# only if you enable S3 offload for values above the 400 KB item limit:
npm install @aws-sdk/client-s3
```

The AWS SDK packages are **peer dependencies** and are lazy-loaded — if you never construct a `DynamoDBStorage`, you
don't pay for them. `@aws-sdk/client-s3` is optional (needed only when `s3Bucket` is set).

## Table

A table with a string partition key `pk` and string sort key `sk` (`PAY_PER_REQUEST` recommended):

```bash
aws dynamodb create-table --table-name agent-data \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region us-east-1
```

You own the table: the package never creates infrastructure and holds no `CreateTable` permission at runtime. TTL
enablement and vector index creation are covered in [Provisioning and permissions](../#provisioning-and-permissions).

## Quick start — session persistence, zero custom code

```ts
import { Agent, SessionManager } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-data', { region: 'us-east-1' })
const agent = new Agent({ sessionManager: new SessionManager({ storage }) })
// sessions now persist to DynamoDB — nothing else to wire.
```

The same instance backs any subsystem that accepts a `Storage` — for example, offloading oversized
tool results with the context offloader:

```ts
import { Agent } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({ plugins: [new ContextOffloader({ storage })] })
```

## Direct byte usage

```ts
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const store = new DynamoDBStorage('agent-data', { region: 'us-east-1' })

await store.write('sessions/s1/snapshot.json', new TextEncoder().encode('{"turn":1}'))
const bytes = await store.read('sessions/s1/snapshot.json') // Uint8Array | null

// list by string prefix -> a native partition Query with begins_with
const keys = await store.list('sessions/s1/')

// Note: prefixes must cover at least a full scope and identifier ('scope/id/').
// list('') and single-segment prefixes are rejected as too broad -- they would
// require a cross-partition Scan. This deliberately narrows the SDK Storage
// contract (whose in-memory backends list everything on ''); SDK subsystems
// always pass namespaced prefixes and are unaffected.

// or a structured DynamoDB query (the intended pk/sk extension point) — no GSI
const scoped = await store.list({ pk: 'sessions/s1', skPrefix: 'scopes/agent/' })

await store.delete('sessions/s1/snapshot.json')

// namespaced view (keys transparently prefixed); nesting composes
const s1 = store.namespace('sessions/s1')
await s1.write('scopes/agent/a1/x', bytes ?? new Uint8Array())
```

Keys are opaque `/`-separated paths. The leading two segments become the partition key and the remainder the sort key,
so point operations are single-item `PutItem`/`GetItem`/`DeleteItem` and listing is a partition-scoped `Query`.

## Large values → S3 offload (optional)

```ts
const store = new DynamoDBStorage('agent-data', {
  region: 'us-east-1',
  s3Bucket: 'my-agent-offload-bucket', // values > ~380 KB go to S3; a pointer item stays in DynamoDB
})
```

Reads and deletes are transparent (the pointer is followed / the S3 object is cleaned up). Without `s3Bucket`, an
oversized write throws rather than silently truncating.

## Compression (optional)

```ts
new DynamoDBStorage('agent-data', { region: 'us-east-1', compression: 'gzip' })
```

Transparent gzip applied **before** the offload size check, so compressible values stay inline in DynamoDB (lower cost,
fewer S3 round-trips). Each item records whether it was compressed, so reads are correct regardless of the current
setting; values that don't shrink are stored uncompressed.

## TTL (optional)

```ts
new DynamoDBStorage('agent-data', { region: 'us-east-1', ttlSeconds: 86_400 }) // 1 day
// per-write override:
await store.write('sessions/tmp/x', data, { ttlSeconds: 3_600 })
```

Stamps a DynamoDB-native epoch-seconds `expireAt` attribute (enable TTL on that attribute at the table level for physical
cleanup). `read`/`list`/`search` also filter items whose expiry has passed, covering the lag before DynamoDB physically
deletes them. With S3 offload, add an S3 lifecycle rule to reclaim offloaded objects (TTL removes only the DynamoDB
pointer).

`search()` reads the TTL attribute from the vector index at no extra read cost, so the index must project it (`ALL`, or
`INCLUDE` listing the TTL attribute); with a narrower projection matches are returned unfiltered. A `vectorSearch`
adapter opts in by returning `expireAt` (epoch seconds) on each match. Expired matches are not back-filled, so a search
can return fewer than `topK` results, and an item whose TTL was just renewed can be briefly hidden until the index
catches up with the write.

## Semantic search — DynamoDB native vector index

`search()` gives an agent semantic long-term memory over the same table: write each memory with its embedding, then
query by meaning. It is an optional, feature-detected part of the `Storage` contract (`if (storage.search) { … }`).
This store searches pre-computed embedding vectors and does not embed text: pass a `SearchQuery` with a `vector`, as
every example does. A plain-string query is rejected with a `StorageError` at runtime unless a `searchStrategy` is
configured (see [Use with SDK consumers](#use-with-sdk-consumers-searchstrategy)), so a text-search consumer that
expects the backend to embed for it must wrap this store with an embedding bridge rather than wiring it in directly.
On DynamoDB the search runs against the **native vector
index**, so nearest-neighbour scoring happens _in the database_ -- no second vector store, no ETL -- and because the index
is partitioned on `pk`, every search is scoped to the caller's key space. Creating the table with a vector index (and the
IAM permissions needed) is covered in the repository README's [Provisioning and permissions](../#provisioning-and-permissions).

Write an embedding alongside the bytes, then query:

```ts
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const store = new DynamoDBStorage('agent-memory', {
  region: 'us-east-1',
  indexName: 'vector_index', // vector index on the table (the default)
  vectorAttribute: 'vector', // item attribute holding the embedding (the default)
})

// store a memory with its embedding (kept inline even when the payload offloads to S3)
await store.write('memory/u1/m1', new TextEncoder().encode('likes window seats'), {
  vector: [/* embedding, e.g. 1024 floats */],
  metadata: { kind: 'preference' },
})

// nearest-neighbour search, scoped to a partition
const results = await store.search({
  vector: queryEmbedding,
  topK: 5,
  pk: 'memory/u1', // required when the index declares a HASH element
  filter: { kind: 'preference' }, // optional metadata equality filter (applied client-side)
  includeValues: true, // hydrate each match's stored bytes
})
// results: Array<{ key: string; score: number; data?: Uint8Array; metadata?: Record<string, unknown> }>
// ordered nearest-first; score direction follows the index's distance function
// (COSINE/EUCLIDEAN: lower = nearer; DOT_PRODUCT: higher = more similar).
```

### The `vectorSearch` adapter (optional override)

`search()` issues DynamoDB `SearchVectors` natively (requires `@aws-sdk/client-dynamodb` >= 3.1103.0).
A `vectorSearch` adapter, when configured, **overrides** the native call — useful for testing or custom
routing. The adapter has the shape:

```ts
type VectorSearchAdapter = (params: {
  tableName: string
  indexName: string
  vectorAttribute: string
  pk?: string
  vector: number[]
  topK: number
  filter?: Record<string, string | number | boolean>
}) => Promise<Array<{ key: string; score: number; metadata?: Record<string, unknown> }>>
```

The adapter is purely an override: with none configured, `search()` issues the native `SearchVectorsCommand` itself.

## Lexical document index (preview)

`LexicalIndex` is an opt-in index for finding stored documents by the words they contain (`search`) or by an exact
identifier such as an error code (`lookup`). It owns the write path of indexed documents: `upsert` writes the document
to the storage table and its index entries (a manifest plus one posting per term and identifier) to a separate index
table in one `TransactWriteItems` call. The byte `Storage` contract is unchanged, and `DynamoDBStorage.search()` only
gains an opt-in text path for SDK consumers ([below](#use-with-sdk-consumers-searchstrategy)).
_Preview_ means the API and the on-table layout (`lexical-v1`, `LEXICAL_TOKENIZER_VERSION`) may change before it is
marked stable; the design is open for maintainer review.

```ts
import { DynamoDBStorage, LexicalIndex } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-data', { region: 'us-east-1' })
// The storage prefix is the lexical scope: bind the authenticated tenant to its namespace.
const index = new LexicalIndex(storage.namespace('tenant-a'), { indexTableName: 'agent-lexical-index' })

const revision = await index.upsert({
  key: 'tickets/42',
  data: new TextEncoder().encode('{"title":"Login fails"}'), // stored like write(); readable with read()
  text: 'Login fails with ERR_AUTH_403 on api-prod-7', // tokenized, never stored
  identifiers: ['ERR_AUTH_403'], // exact, case-sensitive
  metadata: { status: 'open' },
})
const response = await index.search({ text: 'login api-prod-7', topK: 5, filter: { status: 'open' } })
const exact = await index.lookup({ identifier: 'ERR_AUTH_403', topK: 5, includeValues: true })
const current = await index.revision('tickets/42') // strongly consistent; null if absent
await index.delete('tickets/42', { expectedRevision: revision }) // RevisionConflictError if it moved on
```

`search` scores _matched terms ÷ distinct query terms_, ties broken by key in UTF-8 byte order; `requireAllTerms: true`
keeps only full matches. `lookup` returns the documents indexed with exactly that identifier (score 1.0), by key.
Results carry `key`, `score`, `matchedTerms`, `metadata` and, with `includeValues: true`, `data`; the response also
reports `truncated`, `truncationReasons` and `candidatesExamined`. `filter` is strict equality on current metadata (a
missing field never matches; a boolean never equals a number). `upsert` also accepts `expectedRevision`. A key indexed
in one scope cannot be overwritten or deleted through another.

Parts cause partial matches: `FC-123` and `FC-00123` are different terms but share the part `fc`, so a text search for
`FC-123` can also return FC-00123 documents with a partial score. For exact matching use `lookup` or
`requireAllTerms: true`.

### Provisioning and IAM

The library never creates tables. Create the index table (string `pk`/`sk`) and wait for it. Only if the storage uses
TTL, enable TTL on the **same attribute name** it stamps (`ttlAttribute`, default `expireAt`); postings copy their
document's expiry, manifests never expire.

```bash
aws dynamodb create-table --table-name agent-lexical-index \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region us-east-1
aws dynamodb wait table-exists --table-name agent-lexical-index --region us-east-1
aws dynamodb update-time-to-live --table-name agent-lexical-index --region us-east-1 \
  --time-to-live-specification Enabled=true,AttributeName=expireAt   # optional, TTL only
```

Add these statements to the storage policy ([Minimal IAM](#minimal-iam)). Actions inside `TransactWriteItems` are
authorized as `PutItem`, `DeleteItem` and `ConditionCheckItem` (condition checks target only the index table).
`s3:DeleteObject` (already in the S3 statement) is used only to reclaim, best-effort, a value that `write()` previously
offloaded under the same key.

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

- **Writes are atomic.** An `upsert` or `delete` never leaves a document and its index half-written. Conflicting or
  throttled transactions are retried (`maxConflictRetries`); an `expectedRevision` mismatch throws
  `RevisionConflictError` without retrying.
- **Reads are not a point-in-time snapshot.** Postings come from eventually consistent `Query` calls; each candidate is
  then validated individually with a strongly consistent read of the document (revision, scope, key, TTL expiry, current
  metadata). Documents changed during a call may or may not appear, and a just-written document can be missing until its
  postings are visible. With `includeValues`, kept results are re-read after ranking and dropped if they changed, so
  fewer than `topK` may come back. Unprocessed `BatchGetItem` keys are retried; after `maxUnprocessedRetries` rounds
  without progress the call throws rather than answer partially.

Transactional writes perform two underlying writes per item (prepare and commit) and consume capacity even when
cancelled, so conflict retries cost too. Strongly consistent reads cost twice eventually consistent ones, and reads are
charged on full item size regardless of projection.

| Operation           | Reads                                                                                                                                                                                                                                                | Writes                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `upsert`            | 2 strongly consistent `GetItem` (manifest, document)                                                                                                                                                                                                 | 1 transaction of 2 + postings(new ∪ old) items                                                         |
| `delete`            | 2 strongly consistent `GetItem`                                                                                                                                                                                                                      | 1 transaction of up to 2 + old postings items (+1 manifest `ConditionCheck` when there is no manifest) |
| `revision`          | 1 strongly consistent `GetItem`                                                                                                                                                                                                                      | —                                                                                                      |
| `search` / `lookup` | ≤ `maxPagesPerTerm` eventually consistent `Query` pages per term; strongly consistent `BatchGetItem` of **all** validated candidates (not just `topK`), about Σ ceil(item size / 4 KB) RCU; with `includeValues`, a second one of the `topK` winners | —                                                                                                      |
| `repair`            | 1 strongly consistent `Query` page of manifests; 1 strongly consistent `GetItem` per manifest                                                                                                                                                        | 1 transaction per removed or rebuilt document                                                          |

### Limits

Override defaults (`DEFAULT_LEXICAL_INDEX_LIMITS`) with `limits`, validated on construction. Limits are enforced before
any write and text is never silently truncated: an over-limit document, term or identifier throws `StorageError`.

| Limit                                          | Default  | Bound and why                                                                                                                   |
| ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `maxPostingsPerDocument`                       | 49       | Hard ceiling: an overwrite with disjoint postings needs 2 + old + new ≤ 100 `TransactWriteItems` actions, so (100 − 2) / 2 = 49 |
| `maxTermBytes` / `maxIdentifierBytes`          | 64 / 128 | Keeps posting keys small                                                                                                        |
| `maxTextBytes`                                 | 65,536   | Bounds tokenization work per document                                                                                           |
| `maxQueryTerms`                                | 16       | ≤ `maxPostingsPerDocument`; bounds `Query` fan-out                                                                              |
| `pageSize` / `maxPagesPerTerm`                 | 100 / 5  | Page ≤ 1000; bounds posting reads per term                                                                                      |
| `maxCandidates`                                | 300      | ≤ 1000; bounds validation reads per call                                                                                        |
| `maxConcurrency`                               | 4        | ≤ 16 parallel requests per call                                                                                                 |
| `maxConflictRetries` / `maxUnprocessedRetries` | 3 / 5    | May be 0; the latter counts consecutive rounds without progress                                                                 |

Also enforced: `topK` 1–100; the full key (scope + key) and the scope ≤ 1024 UTF-8 bytes; posting and manifest partition
keys ≤ 2048 bytes; indexed values must fit inline. Retrieval truncation is reported as `truncated: true` with
`truncationReasons` (`'max_candidates'`, `'max_pages_per_term'`); postings are read in key order, so truncation biases
results toward lower keys.

### Tokenizer and identifiers

- Text is NFC-normalized and split into runs of letters, marks, numbers and the ASCII joiners `-` and `_`; everything
  else (whitespace, punctuation, symbols, emoji) separates terms. Only ASCII `A`–`Z` is lowercased: `ÁRBOL` becomes
  `Árbol`, not `árbol`.
- A joined run is indexed whole and by its parts: `ERR_AUTH_403` → `err_auth_403`, `err`, `auth`, `403`. Leading
  zeroes are kept. No stemming, no stopwords; unspaced CJK text forms one term per run (`日本語テキスト` is a single
  term).
- Identifiers are NFC-normalized, then matched exactly and case-sensitively (`ERR_AUTH_403` ≠ `err_auth_403`); they
  must be non-empty, free of control characters and not start or end with whitespace. Text, identifiers and queries
  that are not well-formed Unicode are rejected.
- The runtime's Unicode database is used: code points assigned after Unicode 15.0 may tokenize differently between
  runtimes.

### Direct writes, TTL expiry and `repair`

Indexed items carry the revision and scope that wrote them. A direct `storage.write()` over an indexed key drops those
markers; a direct `storage.delete()` or a TTL expiry leaves index entries behind. Retrieval detects all three at read
time and skips the document, and `repair({ maxDocuments: 100, cursor: null, rebuildPostings: false })` cleans up: it
walks this scope's manifests with `Query` (never a `Scan`), removes the manifest and postings of every missing,
overwritten, foreign or expired document, and optionally re-puts the postings of valid ones; documents changed
concurrently are skipped. Pass each `RepairReport.cursor` to the next call until it is `null`. Manifests never expire,
so run `repair` periodically when documents use TTL.

### Use with SDK consumers (`searchStrategy`)

SDK consumers that only know the byte `Storage` API, such as the SDK `FileMemoryStore` (it calls `write(key, bytes)`
and later `search('natural language')`), use the index through `LexicalSearchStrategy`, configured as the storage's
`searchStrategy` like the SDK's own backends:

```ts
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { DynamoDBStorage, LexicalSearchStrategy } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-data', {
  region: 'us-east-1',
  searchStrategy: new LexicalSearchStrategy({
    indexTableName: 'agent-lexical-index',
    extract: (_key, data) => ({ text: new TextDecoder().decode(data) }), // required; return null to skip a value
    topK: 10, // default; also includeValues (default true) and limits
  }),
})
const memory = new FileMemoryStore({ name: 'agent-memory', storage })
await memory.add('# User preferences\nPrefers dark mode')
const entries = await memory.search('dark mode')
```

- **The extractor is explicit.** Stored bytes are never decoded implicitly: `extract(key, data)` decides which text and
  `identifiers` a value carries. Returning `null` leaves the value unindexed; a previously indexed version of the key
  is then excluded at read time and cleaned up by `repair`.
- **One extra write, at-least-once.** After every successful `write()`, the strategy rewrites the item together with
  its postings through `upsert`, keeping the write's `vector`, `metadata` and TTL. If indexing fails, the value stays
  stored but unsearchable and `write()` throws `StorageError("Wrote '<key>' but indexing failed")` with the cause
  attached; retrying the write re-indexes it. `upsert` remains the single-write, atomic path.
- **Same limits as `upsert`.** Extracted text is never truncated: more than `maxPostingsPerDocument` (49) distinct terms
  and identifiers, or a value that would need S3 offload, fails indexing as above.
- **Tolerant queries.** A plain-string `search()` skips terms above `maxTermBytes`, uses only the first
  `maxQueryTerms` (16) distinct terms of a long query, and returns `[]` when no term remains. Results are
  `{ key, score, data? }`; truncation flags are dropped. A `SearchQuery` still runs the native vector search.
- Namespaced views (`FileMemoryStore` uses one) inherit the strategy, and each is searched in its own lexical scope.

### Not supported

S3 offload for indexed documents (a value that would offload is rejected); global tables or multi-Region use
(transactions are atomic only in the Region where they run); online backfill (`repair` cannot re-tokenize because
source text is not stored, so a tokenizer change requires re-upserting); hybrid vector + lexical ranking (RRF, proposed
as the next increment); BM25 or other corpus-statistics scoring; phrase or substring search.

## Configuration reference

| Option                               | Purpose                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| `region` / `client`                  | AWS region, or a pre-built `DynamoDBDocumentClient` (mutually exclusive)             |
| `prefix`                             | Key prefix prepended to every key (a namespace within the table)                     |
| `s3Bucket` / `s3Prefix` / `s3Client` | S3 offload target for large values                                                   |
| `compression`                        | `'gzip'` \| `'none'` (default `'none'`)                                              |
| `ttlSeconds` / `ttlAttribute`        | TTL duration + attribute name (default `expireAt`)                                   |
| `indexName` / `vectorAttribute`      | Vector index + embedding attribute (default `vector_index` / `vector`)               |
| `vectorSearch`                       | Adapter that performs the native vector search                                       |
| `searchStrategy`                     | Answers plain-string `search()` and indexes each write, e.g. `LexicalSearchStrategy` |

## Minimal IAM

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::my-agent-offload-bucket/*"
    }
  ]
}
```

(The S3 statement is only needed when `s3Bucket` is configured. Semantic `search()` additionally needs
`dynamodb:SearchVectors` on the table and its indexes:)

```json
{
  "Effect": "Allow",
  "Action": "dynamodb:SearchVectors",
  "Resource": [
    "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data",
    "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data/index/*"
  ]
}
```

The optional lexical index needs its own table and the extra statements in [Provisioning and IAM](#provisioning-and-iam).
The full provisioning story (TTL enablement, vector index creation, and the complete least-privilege policy) is in the
repository README's [Provisioning and permissions](../#provisioning-and-permissions).

## Examples

Runnable, live-verified examples for every capability, from session resume to a
customer-support capstone, live in the
[examples library](https://github.com/aws/strands-dynamodb-storage/tree/main/examples).
The scripts are Python; this package is a feature-parity mirror, so every pattern
translates directly. The preview lexical document index is the exception: it has no example yet and has only been
tested offline, not against AWS.

## License

Apache-2.0
