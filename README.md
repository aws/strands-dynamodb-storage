# strands-dynamodb-storage

An **Amazon DynamoDB `Storage` backend** for the [Strands Agents](https://github.com/strands-agents/harness-sdk) SDK,
provided for **both TypeScript and Python** from one monorepo (following the Strands `extension-template` `typescript/` + `python/` layout).

It implements the SDK's unified byte `Storage` interface — `write` / `read` / `delete` / `list` / `namespace` — so a single
DynamoDB-backed instance can be passed to **Session Manager, Memory Manager, the context offloader, transcripts**, and any
other subsystem that persists bytes. No per-subsystem code.

## Layout

| Path | Package | Publishes to | Status |
|------|---------|--------------|--------|
| `typescript/` | `strands-dynamodb-storage` | npm | **complete** — unit + real-DynamoDB/S3 integ + live vector `search()` |
| `python/` | `strands-dynamodb-storage` | PyPI | **complete** — parity port; unit (moto) + real-DynamoDB/S3 integ |

## Capabilities (both languages)

- **Single-table design** — the `/`-separated key maps to a partition key (leading scope) + sort key (remainder); point ops
  are single-item `PutItem`/`GetItem`/`DeleteItem`.
- **`list(prefix)`** resolves to a native partition `Query` with `begins_with`; a structured **`DynamoDBListQuery`** (pk/sk)
  is the intended DynamoDB extension point of the generic `Storage` interface — no GSI required.
- **S3 offload** for values above the 400&nbsp;KB item limit (opt-in `s3Bucket`); a small pointer item stays in DynamoDB.
- **Optional gzip compression** — applied before the offload size check, so compressible values stay inline (lower cost);
  per-item marker so reads are correct regardless of the setting.
- **Optional TTL** — stamps a DynamoDB-native epoch-seconds attribute; `read`/`list` also filter already-expired items.
- **Optional `search()`** — nearest-neighbour vector search over DynamoDB's native vector index, feature-detected
  (`if (storage.search)`), with a client-side-KNN fallback for backends that don't implement it.

The AWS SDK clients are lazy-loaded / declared as peer (optional) dependencies, so consumers that never construct a
`DynamoDBStorage` are not forced to install them.

## Releases

Both packages are published from this repository under the name `strands-dynamodb-storage`: the npm package from
`typescript/` and the PyPI package from `python/`. Each is cut by publishing a GitHub Release whose tag carries the
language prefix (`typescript-v0.1.0`, `python-v0.1.0`); the workflow verifies the prefix, so the two languages version
independently.
