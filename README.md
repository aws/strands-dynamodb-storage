<div align="center">
  <p>
    <a href="https://aws.amazon.com/dynamodb/">
      <img src=".github/assets/dynamodb.svg" alt="Amazon DynamoDB" width="78" height="78">
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset=".github/assets/strands-dark.svg">
        <img src=".github/assets/strands.svg" alt="Strands Agents" width="41" height="78">
      </picture>
    </a>
  </p>

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

### One store for the whole agent

Set `storage` once on the `Agent` and every subsystem that accepts a `Storage` — session persistence, the context offloader — uses it, each namespaced under its own prefix (`session/`, `offloader/`). Storage passed directly to a subsystem still takes precedence. Agent-level storage is a shared backend, not a switch: session state persists only once you add a session manager, which then inherits the store.

```python
from strands import Agent
from strands.session import SnapshotSessionManager
from strands.vended_plugins.context_offloader import ContextOffloader
from strands_dynamodb_storage import DynamoDBStorage

storage = DynamoDBStorage("agent-storage", region_name="us-east-1")

# One backend for the whole agent — subsystems without their own storage inherit it.
agent = Agent(
    storage=storage,
    session_manager=SnapshotSessionManager(),  # persists under session/
    plugins=[ContextOffloader()],              # offloads oversized tool results under offloader/
)
```

```typescript
import { Agent, SessionManager } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-storage', { region: 'us-east-1' })

// One backend for the whole agent — subsystems without their own storage inherit it.
const agent = new Agent({
  storage,
  sessionManager: new SessionManager({}),   // persists under session/
  plugins: [new ContextOffloader({})],      // offloads oversized tool results under offloader/
})
```

To scope a single subsystem to a different store instead, pass `storage` on that subsystem directly, as the per-language Quick Starts below show.

### Python

Requires Python 3.10+:

```bash
pip install strands-dynamodb-storage
```

```python
from strands import Agent
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBStorage

storage = DynamoDBStorage("agent-storage", region_name="us-east-1")

# Persist the agent's session to DynamoDB — nothing else to wire.
agent = Agent(session_manager=SnapshotSessionManager("s1", storage=storage))
agent("Remember that my favorite color is blue.")
```

The same instance plugs into every SDK subsystem that accepts a `Storage`. Offload oversized tool
results to DynamoDB via the context offloader:

```python
from strands import Agent
from strands.vended_plugins.context_offloader import ContextOffloader

agent = Agent(plugins=[ContextOffloader(storage=storage)])
```

Or use the byte contract directly:

```python
await storage.write("sessions/s1/state", b"...")
await storage.read("sessions/s1/state")
await storage.list("sessions/s1/")
```

The [Python README](python/) covers S3 offload, compression, TTL, multi-tenant prefixes, and vector search.

### TypeScript

Requires Node.js 20+:

```bash
npm install strands-dynamodb-storage
```

```typescript
import { Agent, SessionManager } from '@strands-agents/sdk'
import { DynamoDBStorage } from 'strands-dynamodb-storage'

const storage = new DynamoDBStorage('agent-storage', { region: 'us-east-1' })

// Persist the agent's session to DynamoDB — nothing else to wire.
const agent = new Agent({ sessionManager: new SessionManager({ sessionId: 's1', storage }) })
await agent.invoke('Remember that my favorite color is blue.')
```

The same instance plugs into every SDK subsystem that accepts a `Storage`. Offload oversized tool
results to DynamoDB via the context offloader:

```typescript
import { Agent } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({ plugins: [new ContextOffloader({ storage })] })
```

Or use the byte contract directly:

```typescript
await storage.write('sessions/s1/state', data)
await storage.read('sessions/s1/state')
await storage.list('sessions/s1/')
```

More in the [TypeScript README](typescript/), including the structured `DynamoDBListQuery` extension and typed search results.

## Provisioning and permissions

**You own the table.** The package never creates infrastructure: it holds no `CreateTable` or `UpdateTable` permission at runtime, so the table lives in your infrastructure as code next to your other resources, with your tagging, backup, and encryption posture applied. Everything it needs is a table with a string partition key `pk` and a string sort key `sk`:

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

On-demand billing suits agent traffic, which is bursty and idles at zero, but provisioned capacity works identically.

**TTL (optional).** Constructing the storage with a TTL stamps an epoch-seconds `expireAt` attribute on every item and filters already-expired items on `read` / `list`. For DynamoDB to physically reap expired items, enable TTL on the table once:

```bash
aws dynamodb update-time-to-live \
  --table-name agent-storage \
  --time-to-live-specification "Enabled=true, AttributeName=expireAt"
```

**Semantic search (optional).** `search()` runs against a DynamoDB vector index on the same table. The index is created with the table (its name, dimensions, and distance function are immutable afterwards), so size `Dimensions` to your embedding model:

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --vector-indexes '[{
    "IndexName": "vector_index",
    "VectorAttribute": {"AttributeName": "vector"},
    "SearchSchema": [{"AttributeName": "pk", "SearchSchemaElementType": "HASH"}],
    "Projection": {"ProjectionType": "ALL"},
    "Dimensions": 1024,
    "DistanceFunction": "COSINE"
  }]'
```

The `SearchSchema` HASH element on `pk` partitions the index the same way the table is partitioned, so every search is scoped to one key space and one tenant's memories can never surface in another's results. A newly created index backfills before it is searchable; wait for `IndexStatus: ACTIVE` with `Backfilling` false in `describe-table` before the first search. The index names above (`vector_index`, `vector`) are the package defaults; both are configurable. Requires an up-to-date AWS CLI, and SDK floors of `boto3 >= 1.43.64` / `@aws-sdk/client-dynamodb >= 3.1103.0`.

**S3 offload (optional).** Values above the DynamoDB item-size limit offload to a bucket you provide (`s3Bucket` / `s3_bucket`); objects are keyed by the item's full storage key under an optional prefix. Overwrites and deletes reclaim the object, and if you enable TTL, add an S3 lifecycle rule as the backstop: DynamoDB reaping an expired pointer item does not delete its S3 object.

**Runtime permissions.** The package issues exactly these operations, so least privilege is:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AgentStorageTable",
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:DeleteItem", "dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/agent-storage"
    },
    {
      "Sid": "SemanticSearchOptional",
      "Effect": "Allow",
      "Action": "dynamodb:SearchVectors",
      "Resource": [
        "arn:aws:dynamodb:us-east-1:123456789012:table/agent-storage",
        "arn:aws:dynamodb:us-east-1:123456789012:table/agent-storage/index/*"
      ]
    },
    {
      "Sid": "S3OffloadOptional",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::my-offload-bucket/*"
    }
  ]
}
```

Drop the optional statements for features you don't use. Nothing else is required: no `DescribeTable`, no `Scan`, no table-level wildcards, and credentials resolve through the standard SDK chain (environment, instance profile, or an injected client).

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
