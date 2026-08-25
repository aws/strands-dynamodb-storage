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
const bytes = await store.read('sessions/s1/snapshot.json')   // Uint8Array | null

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
  s3Bucket: 'my-agent-offload-bucket',   // values > ~380 KB go to S3; a pointer item stays in DynamoDB
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
new DynamoDBStorage('agent-data', { region: 'us-east-1', ttlSeconds: 86_400 })  // 1 day
// per-write override:
await store.write('sessions/tmp/x', data, { ttlSeconds: 3_600 })
```

Stamps a DynamoDB-native epoch-seconds `expireAt` attribute (enable TTL on that attribute at the table level for physical
cleanup). `read`/`list` also filter items whose expiry has passed, covering the lag before DynamoDB physically deletes
them. With S3 offload, add an S3 lifecycle rule to reclaim offloaded objects (TTL removes only the DynamoDB pointer).
Note that this filtering applies to `read`/`list` only: because TTL deletion is asynchronous, `search()` can briefly
return items whose expiry has passed but which DynamoDB has not yet physically deleted.

## Semantic search — DynamoDB native vector index

`search()` gives an agent semantic long-term memory over the same table: write each memory with its embedding, then
query by meaning. It is an optional, feature-detected part of the `Storage` contract: consumers do `if (storage.search) { … }`
and fall back to client-side KNN when a backend doesn't implement it. On DynamoDB it runs against the **native vector
index**, so nearest-neighbour scoring happens *in the database* -- no second vector store, no ETL -- and because the index
is partitioned on `pk`, every search is scoped to the caller's key space. Creating the table with a vector index (and the
IAM permissions needed) is covered in the repository README's [Provisioning and permissions](../#provisioning-and-permissions).

Write an embedding alongside the bytes, then query:

```ts
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const store = new DynamoDBStorage('agent-memory', {
  region: 'us-east-1',
  indexName: 'vector_index',        // vector index on the table (the default)
  vectorAttribute: 'vector',        // item attribute holding the embedding (the default)
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
  pk: 'memory/u1',          // required when the index declares a HASH element
  filter: { kind: 'preference' },   // optional metadata equality filter (applied client-side)
  includeValues: true,      // hydrate each match's stored bytes
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

| Option | Purpose |
|--------|---------|
| `region` / `client` | AWS region, or a pre-built `DynamoDBDocumentClient` (mutually exclusive) |
| `prefix` | Key prefix prepended to every key (a namespace within the table) |
| `s3Bucket` / `s3Prefix` / `s3Client` | S3 offload target for large values |
| `compression` | `'gzip'` \| `'none'` (default `'none'`) |
| `ttlSeconds` / `ttlAttribute` | TTL duration + attribute name (default `expireAt`) |
| `indexName` / `vectorAttribute` | Vector index + embedding attribute (default `vector_index` / `vector`) |
| `vectorSearch` | Adapter that performs the native vector search |

## Minimal IAM

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem","dynamodb:PutItem","dynamodb:DeleteItem","dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data" },
    { "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject"],
      "Resource": "arn:aws:s3:::my-agent-offload-bucket/*" }
  ]
}
```

(The S3 statement is only needed when `s3Bucket` is configured. Semantic `search()` additionally needs
`dynamodb:SearchVectors` on the table and its indexes:)

```json
{ "Effect": "Allow",
  "Action": "dynamodb:SearchVectors",
  "Resource": ["arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data",
               "arn:aws:dynamodb:us-east-1:ACCOUNT:table/agent-data/index/*"] }
```

The full provisioning story (TTL enablement, vector index creation, and the complete least-privilege policy) is in the
repository README's [Provisioning and permissions](../#provisioning-and-permissions).

## Examples

Runnable, live-verified examples for every capability, from session resume to a
customer-support capstone, live in the
[examples library](https://github.com/aws/strands-dynamodb-storage/tree/main/examples).
The scripts are Python; this package is a feature-parity mirror, so every pattern
translates directly.

## License

Apache-2.0
