# strands-dynamodb-storage

An **Amazon DynamoDB `Storage` backend** for the [Strands Agents](https://github.com/strands-agents/harness-sdk) TypeScript SDK.

It implements the SDK's unified byte `Storage` interface (`write` / `read` / `delete` / `list` / `namespace`), so one
DynamoDB-backed instance can be passed to **Session Manager, Memory Manager**, the context offloader, transcripts, and
any other subsystem that persists bytes — no per-subsystem code. On top of the byte contract it adds S3 offload for large
values, optional gzip compression, TTL, and optional **native vector search**.

## Install

```bash
npm install strands-dynamodb-storage @strands-agents/sdk @aws-sdk/client-dynamodb @aws-sdk/lib-dynamodb
# only if you enable S3 offload for values above the 400 KB item limit:
npm install @aws-sdk/client-s3
# for BedrockEmbedder (the DynamoDBMemoryStore default embedder); @strands-agents/sdk already
# depends on it, so most installs have it, but strict installers need it declared:
npm install @aws-sdk/client-bedrock-runtime
```

The AWS SDK packages are **peer dependencies** and are lazy-loaded — if you never construct a `DynamoDBStorage`, you
don't pay for them. `@aws-sdk/client-s3` is optional (needed only when `s3Bucket` is set), and so is
`@aws-sdk/client-bedrock-runtime` (needed only for `BedrockEmbedder`).

## Table

A table with a string partition key `pk` and string sort key `sk` (`PAY_PER_REQUEST` recommended):

```bash
aws dynamodb create-table --table-name agent-data \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region us-east-1
```

You own the table: the package never creates infrastructure and holds no `CreateTable` permission at runtime. TTL
enablement and vector index creation are covered in [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions).

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

## Long-term memory

`DynamoDBMemoryStore` gives the Strands Memory Manager a long-term memory on the same table. It embeds each memory
through Amazon Bedrock, stores the text with its embedding, and finds memories by meaning with one `SearchVectors` call.
The table needs the vector index from [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions), with
`Dimensions` 1024 for the default embedder (Amazon Titan Text Embeddings V2).

```ts
import { Agent, MemoryManager } from '@strands-agents/sdk'
import { DynamoDBMemoryStore, DynamoDBStorage } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-storage', { region: 'us-east-1' }).namespace('user/u1')
const memory = new DynamoDBMemoryStore(storage)
const agent = new Agent({ memoryManager: new MemoryManager({ stores: [memory], addToolConfig: true }) })

// The store also works on its own.
const key = await memory.add('Prefers a window seat', { source: 'chat' })
const entries = await memory.search('Where does the user like to sit?')
entries[0]?.content // 'Prefers a window seat'
entries[0]?.metadata // { source: 'chat', _key: 'memory/dynamodb/8b29884b...', _score: 0.59 }
```

How it behaves:

- **One item per memory.** A memory is stored at `memory/<name>/<id>` inside the storage namespace, where `<name>` is
  the store name (default `dynamodb`) and `<id>` is a hash of the text. Adding the same text again overwrites that item,
  replacing its metadata, so retried writes do not create duplicates. `await storage.delete(entry.metadata?._key as string)`
  removes a memory.
- **Two calls per operation.** `add()` makes one Bedrock call and one `PutItem`; `search()` makes one Bedrock call and
  one `SearchVectors`, pinned to the partition that holds the namespace. With the index projection `ALL`, memories come
  back from the index, so a search reads no base-table items; a memory large enough to be offloaded to S3 costs one
  `GetItem` and one S3 read. Results come closest first; `_score` is the raw index score (lower is closer for COSINE).
- **One memory store per partition.** The partition is the first two segments of the full key: `namespace('user/u1')`
  gives the store partition `user/u1`, but `user/u1/agent-a` and `user/u1/agent-b` share partition `user/u1`. Every item
  with a vector in the partition competes for the same search results, and the store drops matches that are not its
  own memories before reading them, so a second memory store (or the application's own vector items) in the partition
  costs no extra reads but can leave results short of `maxSearchResults`. Give each memory store a two-segment
  namespace of its own, such as `user/<id>`.
- **Only memories are embedded.** Session snapshots and other values written through the same storage never call
  Bedrock.
- **TTL applies.** With `ttlSeconds` on the storage, memories expire like any other item, and search drops expired
  matches.
- **Metadata** values that are strings, booleans, or numbers that are zero or from 1e-130 to `Number.MAX_SAFE_INTEGER`
  in magnitude are kept, the same set the Python package accepts. Other values (arrays, nested objects, `null`, numbers
  outside that range; the AWS SDK refuses to marshall larger ones), keys or strings containing a lone surrogate (which
  DynamoDB stores as `?`), empty keys, the keys `__proto__` and `constructor` (which `marshall` cannot write), and the
  reserved keys `_key` and `_score` are dropped with a `console.warn`, so a memory whose metadata came from a model's
  extraction is still stored.

The default embedder runs in the storage's Region and needs `bedrock:InvokeModel` on the model. When the storage was
given a `client`, the embedder takes that client's Region but resolves its own credentials through the standard chain;
pass `new BedrockEmbedder({ client })` to control them. To use another model,
pass an embedder. `dimensions` must match the index's `Dimensions`; a mismatch fails the write or search with a
`StorageError` whose cause names the expected and actual sizes:

```ts
import { BedrockEmbedder } from 'strands-dynamodb-storage'

// Cohere Embed v4 through a cross-Region inference profile; Cohere embeds memories and queries with separate
// input types.
const cohere = new DynamoDBMemoryStore(storage, {
  embedder: new BedrockEmbedder({ modelId: 'us.cohere.embed-v4:0', dimensions: 1024 }),
})

// Any other model: a function from text to an array of numbers (sync or async), or an object
// with an `embed(text, purpose)` method (sync or async) where purpose is 'document' or 'query'.
// Convert a Float32Array with Array.from.
const custom = new DynamoDBMemoryStore(storage, { embedder: async (text) => Array.from(await myModel.embed(text)) })
```

`BedrockEmbedder` supports Titan Text Embeddings V2 (256, 512, or 1024 dimensions), Cohere Embed v3 English and
multilingual (1024), and Cohere Embed v4 (256, 512, 1024, or 1536), by model ID, cross-Region inference profile ID, or
ARN. An application inference profile or provisioned model ARN does not name the model, so pass its request format as
well: `new BedrockEmbedder({ modelId: profileArn, modelFamily: 'amazon.titan-embed-text-v2' })`. Bedrock limits the
input size. Titan Text Embeddings V2 rejects a text over 8,192 tokens or 50,000 characters, and Cohere Embed v3 a text
over 2,048 characters; either fails with a `StorageError` that carries Bedrock's message. Cohere Embed v3 also embeds
only the first 512 tokens of a text, without an error, so a long memory is found by its beginning. Cohere Embed v4
accepts up to about 128,000 tokens per text.

## Semantic search — DynamoDB native vector index

`search()` is the lower-level API under the memory store: write each item with its embedding, then query by vector.
It is an optional, feature-detected part of the `Storage` contract (`if (storage.search) { … }`). The storage searches
pre-computed embedding vectors and does not embed text: pass a `SearchQuery` with a `vector`, as every example does. A
plain-string query is rejected with a `StorageError` at runtime; for text search, use `DynamoDBMemoryStore`.
On DynamoDB the search runs against the **native vector
index**, so nearest-neighbour scoring happens _in the database_ -- no second vector store, no ETL -- and because the index
is partitioned on `pk`, every search is scoped to the caller's key space. Creating the table with a vector index (and the
IAM permissions needed) is covered in the repository README's [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions).

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
  pk: store.partitionKey('memory/u1/m1'), // the partition holding these keys; required when the index declares a HASH element
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

## Configuration reference

| Option                               | Purpose                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `region` / `client`                  | AWS region, or a pre-built `DynamoDBDocumentClient` (mutually exclusive) |
| `prefix`                             | Key prefix prepended to every key (a namespace within the table)         |
| `s3Bucket` / `s3Prefix` / `s3Client` | S3 offload target for large values                                       |
| `compression`                        | `'gzip'` \| `'none'` (default `'none'`)                                  |
| `ttlSeconds` / `ttlAttribute`        | TTL duration + attribute name (default `expireAt`)                       |
| `indexName` / `vectorAttribute`      | Vector index + embedding attribute (default `vector_index` / `vector`)   |
| `vectorSearch`                       | Adapter that performs the native vector search                           |

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

(The S3 statement is only needed when `s3Bucket` is configured. Semantic `search()` and `DynamoDBMemoryStore`
additionally need `dynamodb:SearchVectors` on the table and its indexes, and the memory store's default embedder needs
`bedrock:InvokeModel` on the Titan Text Embeddings V2 model:)

```json
[
  {
    "Effect": "Allow",
    "Action": "dynamodb:SearchVectors",
    "Resource": [
      "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data",
      "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data/index/*"
    ]
  },
  {
    "Effect": "Allow",
    "Action": "bedrock:InvokeModel",
    "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
  }
]
```

The full provisioning story (TTL enablement, vector index creation, and the complete least-privilege policy) is in the
repository README's [Provisioning and permissions](https://github.com/aws/strands-dynamodb-storage#provisioning-and-permissions).

## Examples

Runnable, live-verified examples for every capability, from session resume to a
customer-support capstone, live in the
[examples library](https://github.com/aws/strands-dynamodb-storage/tree/main/examples).
The scripts are Python; this package is a feature-parity mirror, so every pattern
translates directly.

## License

Apache-2.0
