# Multi-tenant isolation on one table

An agent platform serving many users has to keep one user's memories out of
another user's results. This example shows that with
strands-dynamodb-storage that is a schema decision made once, at construction
time, rather than a filter you have to remember on every query.

## How it works

Two mechanisms, both set once:

**The constructor prefix scopes the byte contract.** A store built with
`prefix="user/alice"` pins every `write`, `read`, `delete`, and `list` inside
Alice's key space. Two tenants using the same logical key `memories/m1`
resolve to physically distinct partitions:

```python
alice = DynamoDBStorage(table, prefix="user/alice")
bob = DynamoDBStorage(table, prefix="user/bob")
```

**The search schema scopes the vector index.** The index is partitioned by
`pk` (its search schema HASH element), and every search supplies one
partition. A search scoped to Alice's partition never ranges over Bob's
vectors, no matter how similar the content is:

```python
results = await alice.search(SearchQuery(vector=embed(q), top_k=5, pk="user/alice"))
```

The script seeds each user with a near-identical coffee-preference memory
under the same logical key, then proves both boundaries: each tenant's
`list()` sees only its own key, and each tenant's semantic search returns only
its own memory, with `top_k=5` leaving the other tenant's vector unreachable
rather than merely outranked.

## Run it

Provision the vector-indexed table from the
[semantic-memory example](../semantic-memory/README.md#run-it), then:

```bash
pip install strands-dynamodb-storage boto3
python multi_tenant.py --table agent-storage
```

Or the TypeScript equivalent:

```bash
npm install strands-dynamodb-storage @aws-sdk/client-bedrock-runtime tsx
npx tsx multi-tenant.ts --table agent-storage
```

```text
seeded one coffee memory per user (same logical key memories/m1)

alice.list('memories/') -> ['memories/m1']
bob.list('memories/')   -> ['memories/m1']

search as alice: ['Alice drinks oat-milk lattes, decaf after noon']
  Bob's memory in alice's results: False
search as bob: ['Bob drinks double espressos, no milk ever']
  Alice's memory in bob's results: False
```

The vector index is eventually consistent, so a memory written moments before
the search may take a short time to become searchable; the script writes and
searches back to back, and a rerun covers any lag.

## Scoping is not authorization

The partition value is supplied by the caller: a principal holding
`dynamodb:SearchVectors` on the table can search any partition. This
mechanism guarantees that well-behaved application code cannot accidentally
cross tenants; keeping a hostile caller out belongs to AWS IAM and your
application layer. The work each search performs also tracks the size of one
tenant's partition rather than the whole table, so a busy tenant does not
make every other tenant's recall slower or more expensive.

## Clean up

The script deletes its own demo items. Drop the table when finished:

```bash
aws dynamodb delete-table --table-name agent-storage
```
