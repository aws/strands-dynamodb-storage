<div align="center">
  <h1>
    Strands DynamoDB Storage
  </h1>

  <h2>
    Durable, scalable agent storage on Amazon DynamoDB, in one line of configuration.
  </h2>

  <div align="center">
    <a href="https://github.com/aws/strands-dynamodb-storage/actions/workflows/ci-python.yml"><img alt="Python CI" src="https://github.com/aws/strands-dynamodb-storage/actions/workflows/ci-python.yml/badge.svg"/></a>
    <a href="https://github.com/aws/strands-dynamodb-storage/actions/workflows/ci-typescript.yml"><img alt="TypeScript CI" src="https://github.com/aws/strands-dynamodb-storage/actions/workflows/ci-typescript.yml/badge.svg"/></a>
    <a href="https://github.com/aws/strands-dynamodb-storage/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/aws/strands-dynamodb-storage"/></a>
    <a href="https://pypi.org/project/strands-dynamodb-storage/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-dynamodb-storage"/></a>
    <a href="https://www.npmjs.com/package/strands-dynamodb-storage"><img alt="npm version" src="https://img.shields.io/npm/v/strands-dynamodb-storage"/></a>
  </div>

  <p>
    <a href="https://strandsagents.com/">Strands Documentation</a>
    ◆ <a href="https://github.com/strands-agents/harness-sdk">Strands Agents SDK</a>
    ◆ <a href="python/">Python Package</a>
    ◆ <a href="typescript/">TypeScript Package</a>
  </p>
</div>

Strands DynamoDB Storage is an [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) backend for the [Strands Agents](https://github.com/strands-agents/harness-sdk) SDK `Storage` interface. It implements the SDK's unified byte contract (`write` / `read` / `delete` / `list` / `namespace`), so a single DynamoDB-backed instance serves Session Manager, Memory Manager, the context offloader, transcripts, and any other subsystem that persists bytes. No per-subsystem code, and your agent's state lives in your own AWS account, next to your operational data.

This repository ships the backend for **both TypeScript and Python**:

| Directory | Package | Publishes to | Description |
|-----------|---------|--------------|-------------|
| `typescript/` | `strands-dynamodb-storage` | npm | TypeScript implementation ([README](typescript/)) |
| `python/` | `strands-dynamodb-storage` | PyPI | Python implementation, full feature parity ([README](python/)) |

## Why DynamoDB for agent storage

Agents are bursty, stateful, and ephemeral, which is precisely the workload shape DynamoDB was built for.

- **Serverless to serverless.** Agents increasingly run on ephemeral compute such as AWS Lambda and Amazon Bedrock AgentCore Runtime, where anything that must survive teardown has to leave the process. DynamoDB's HTTP data plane needs no connection pools to warm or exhaust, and on-demand capacity absorbs a burst of concurrent agent sessions without pre-provisioning. Idle agents cost nothing.
- **State reads sit on the hot path of every turn.** An agent loads its session state before every model call. DynamoDB serves single-digit-millisecond point reads at any scale, so storage never becomes the reason your agent feels slow, whether you have one session or one hundred million.
- **The data model fits the contract.** The SDK's `/`-separated keys map directly onto DynamoDB's partition-plus-sort key model: point operations are single-item calls, and `list(prefix)` is a native `Query` with `begins_with` against one partition rather than a scan. Nothing is emulated, so cost and latency track what the interface promises.
- **Isolation is structural.** A constructor-bound `prefix` pins every operation inside its own key space. Two tenants sharing a table resolve the same logical key to physically distinct partitions, and neither can read or list into the other's data.
- **Memory that expires itself.** DynamoDB-native TTL reaps stale session state with no cleanup job to run, and reads filter items that have already expired.
- **Semantic memory without a second database.** Vector search runs over a DynamoDB vector index on the same table, so there is no separate vector store to provision, no ETL, and no reconciliation. Agent memories become searchable where they are written.
- **Fully managed durability.** Session state, long-term memories, and transcripts are replicated across multiple Availability Zones, with encryption at rest, point-in-time recovery, and backups available as table settings rather than infrastructure you operate.

## Quick Start

Create a table with a string partition key `pk` and a string sort key `sk`, then point the storage at it. AWS credentials are resolved through the standard SDK chain.

### Python

Requires Python 3.10+:

```bash
pip install strands-dynamodb-storage
```

```python
from strands_dynamodb_storage import DynamoDBStorage

storage = DynamoDBStorage("agent-storage", region_name="us-east-1")

await storage.write("sessions/s1/state", b"...")
await storage.read("sessions/s1/state")
await storage.list("sessions/s1/")
```

The same instance plugs into every SDK subsystem that accepts a `Storage`. The [Python README](python/) covers S3 offload, compression, TTL, multi-tenant prefixes, and vector search.

### TypeScript

Requires Node.js 20+:

```bash
npm install strands-dynamodb-storage
```

```typescript
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-storage', { region: 'us-east-1' })

await storage.write('sessions/s1/state', data)
await storage.read('sessions/s1/state')
await storage.list('sessions/s1/')
```

More in the [TypeScript README](typescript/), including the structured `DynamoDBListQuery` extension and typed search results.

## Features

Both languages ship the same capabilities with identical semantics.

- **Single-table design.** The `/`-separated key maps to a partition key (leading scope) plus a sort key (remainder), so point operations are single-item `PutItem` / `GetItem` / `DeleteItem` and listing is a partition `Query`. No GSI required.
- **S3 offload for large values.** Values above the DynamoDB item-size limit are transparently offloaded to [Amazon S3](https://aws.amazon.com/s3/) (opt-in `s3Bucket`), with a small pointer item remaining in DynamoDB. Callers see one byte contract regardless of size.
- **Optional gzip compression.** Applied before the offload size check, so compressible values stay inline at lower cost. Each item records whether it was compressed, so reads are correct regardless of the setting.
- **Optional TTL.** Writes stamp a DynamoDB-native epoch-seconds attribute, and `read` / `list` also filter already-expired items, covering the window before DynamoDB physically deletes them.
- **Vector search.** `search()` calls DynamoDB `SearchVectors` natively against a vector index on the table, with the search condition pinned to the caller's partition, bounded `TopK`, and results in most-similar-first order. Consumers feature-detect (`if (storage.search)`) and fall back to client-side KNN when absent. Like a global secondary index, the vector index is eventually consistent.
- **Multi-tenant prefixes.** A constructor-bound prefix namespaces every key, enforced on both structured and string-prefix queries.
- **Lazy, peer-declared AWS clients.** The AWS SDK clients are lazy-loaded and declared as peer (optional) dependencies, so consumers that never construct a `DynamoDBStorage` are not forced to install them.

## Releases

Both packages publish from this repository under the name `strands-dynamodb-storage`: the npm package from `typescript/` and the PyPI package from `python/`. Each release is cut by publishing a GitHub Release whose tag carries the language prefix (`typescript-v0.1.0`, `python-v0.1.0`). The workflow verifies the prefix, so the two languages version independently.

## Development

Each package has its own toolchain:

**Python** (`python/`):
```bash
cd python
pip install -e '.[dev]'
ruff format --check src tests && ruff check src tests
mypy src
pytest tests -q
```

**TypeScript** (`typescript/`):
```bash
cd typescript
npm ci
npm run check     # format + lint + type-check + unit tests
npm run build
```

Both packages carry live integration suites that run against real DynamoDB and S3, gated behind `RUN_INTEG=1` with AWS credentials configured. They provision a vector-indexed table, wait out index backfill, and clean up after themselves.

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
