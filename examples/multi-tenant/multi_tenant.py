"""Serve many tenants from one DynamoDB table, with per-tenant isolation.

A memory store serving many users has to keep one user's memories out of
another user's results. With this package that is a schema decision made at
construction time, not a filter appended to every query:

- The constructor ``prefix`` pins every write, read, and listing inside the
  tenant's own key space, so two tenants resolve the same logical key to
  physically distinct partitions.
- The vector index is partitioned by ``pk`` (its search schema HASH element),
  so a search scoped to one tenant's partition never ranges over another
  tenant's vectors, no matter how similar they are.

This script seeds two users with near-identical memories, then shows that
listings and semantic searches for one user never surface the other's data,
even when the other user's memory is the closest match in the whole table.

Prerequisites: vector-indexed table (see README.md), Bedrock access for
Titan embeddings.

Usage:
  python multi_tenant.py --table agent-storage
"""

import argparse
import asyncio
import json
from typing import Any

import boto3
from strands_dynamodb_storage import DynamoDBStorage, SearchQuery

ALICE = "user/alice"
BOB = "user/bob"


def make_embedder(region: str) -> Any:
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    def embed(text: str) -> list[float]:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text, "dimensions": 1024}),
        )
        return json.loads(response["body"].read())["embedding"]  # type: ignore[no-any-return]

    return embed


async def main_async(table: str, region: str) -> None:
    embed = make_embedder(region)

    # One store per tenant. Same table, same code path; only the prefix differs.
    alice = DynamoDBStorage(table, region_name=region, prefix=ALICE)
    bob = DynamoDBStorage(table, region_name=region, prefix=BOB)

    # Near-identical content on both sides. If scoping leaked, Bob's espresso
    # memory would be the nearest neighbor for Alice's coffee question.
    await alice.write("memories/m1", b"Alice drinks oat-milk lattes, decaf after noon",
                      vector=embed("Alice drinks oat-milk lattes, decaf after noon"))
    await bob.write("memories/m1", b"Bob drinks double espressos, no milk ever",
                    vector=embed("Bob drinks double espressos, no milk ever"))
    print("seeded one coffee memory per user (same logical key memories/m1)\n")

    # 1. The byte contract is scoped: each tenant lists only its own keys.
    print(f"alice.list('memories/') -> {await alice.list('memories/')}")
    print(f"bob.list('memories/')   -> {await bob.list('memories/')}\n")

    # 2. Search is scoped by partition: each tenant's query is answered only
    #    from that tenant's vectors.
    question = "how does this user take their coffee?"
    for name, store, pk in (("alice", alice, ALICE), ("bob", bob, BOB)):
        results = await store.search(SearchQuery(vector=embed(question), top_k=5, pk=pk, include_values=True))
        answers = [r.data.decode() for r in results if r.data is not None]
        print(f"search as {name}: {answers}")
        other = "Bob" if name == "alice" else "Alice"
        leaked = any(other in a for a in answers)
        print(f"  {other}'s memory in {name}'s results: {leaked}")
        assert not leaked, "cross-tenant leak"

    # Clean up the demo items.
    await alice.delete("memories/m1")
    await bob.delete("memories/m1")
    print("\ndemo items deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    asyncio.run(main_async(args.table, args.region))


if __name__ == "__main__":
    main()
