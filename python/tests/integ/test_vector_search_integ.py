# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for native vector search — real Amazon DynamoDB.

Gated by RUN_INTEG=1 and AWS credentials. Provisions its own table with a
vector index (SearchSchema HASH on pk so searches can be partition-pinned),
waits for the index to finish backfilling, then exercises the native
``search()`` path end to end. Tears the table down afterwards.

Run:
    RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import boto3
import pytest
from strands.vended_memory_stores.file_memory_store import FileMemoryStore

from strands_dynamodb_storage import DynamoDBStorage, DynamoDBVectorSearchStrategy, SearchQuery

RUN = os.environ.get("RUN_INTEG") == "1"
REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = f"strands-integ-vector-{int(time.time())}-{uuid.uuid4().hex[:6]}"
INDEX = "vector_index"

pytestmark = [
    pytest.mark.skipif(not RUN, reason="integ tests run only with RUN_INTEG=1"),
    pytest.mark.asyncio,
]


@pytest.fixture(scope="module")
def vector_table():
    client = boto3.client("dynamodb", region_name=REGION)
    client.create_table(
        TableName=TABLE,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        VectorIndexes=[
            {
                "IndexName": INDEX,
                "VectorAttribute": {"AttributeName": "vector"},
                "SearchSchema": [{"AttributeName": "pk", "SearchSchemaElementType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "Dimensions": 4,
                "DistanceFunction": "COSINE",
            }
        ],
    )
    client.get_waiter("table_exists").wait(TableName=TABLE)
    # ACTIVE alone is not ready: wait until Backfilling is false/absent.
    deadline = time.time() + 600
    while time.time() < deadline:
        indexes = client.describe_table(TableName=TABLE)["Table"].get("VectorIndexes", [])
        idx = next((i for i in indexes if i.get("IndexName") == INDEX), None)
        if idx and idx.get("IndexStatus") == "ACTIVE" and not idx.get("Backfilling", False):
            break
        time.sleep(5)
    else:
        pytest.fail("vector index did not become ready within 600s")
    yield client
    client.delete_table(TableName=TABLE)


async def test_native_search_end_to_end(vector_table):
    storage = DynamoDBStorage(TABLE, region_name=REGION, prefix="tenant/a/", index_name=INDEX)
    await storage.write("memories/m1", b"likes window seats", vector=[1.0, 0.0, 0.0, 0.0], metadata={"kind": "pref"})
    await storage.write("memories/m2", b"allergic to peanuts", vector=[0.0, 1.0, 0.0, 0.0])

    # The index is eventually consistent: poll until both items are visible.
    deadline = time.time() + 300
    results = []
    while time.time() < deadline:
        results = await storage.search(SearchQuery(vector=[1.0, 0.0, 0.0, 0.0], top_k=2, pk="tenant/a"))
        if len(results) == 2:
            break
        time.sleep(5)
    assert len(results) == 2, "expected both items in search results within 300s"
    assert results[0].key == "memories/m1"  # nearest first
    assert results[0].metadata == {"kind": "pref"}
    assert results[0].score <= results[1].score  # COSINE distance: lower is closer

    # include_values round-trips the stored bytes
    [top] = await storage.search(SearchQuery(vector=[1.0, 0.0, 0.0, 0.0], top_k=1, pk="tenant/a", include_values=True))
    assert top.data == b"likes window seats"


async def test_index_backfills_a_value_in_place(vector_table):
    """index() adds the vector to a value written before the strategy, leaving the item's own data alone."""
    vocabulary = ("dark", "mode", "coffee", "tea")

    def embed(text: str) -> list[float]:
        tokens = text.lower().split()
        return [0.01 + tokens.count(word) for word in vocabulary]

    plain = DynamoDBStorage(TABLE, region_name=REGION, prefix="tenant/c/", index_name=INDEX)
    await plain.write("memory/facts/a", b"dark mode", metadata={"kind": "pref"})
    strategy = DynamoDBVectorSearchStrategy(embedder=embed)
    view = DynamoDBStorage(
        TABLE, region_name=REGION, prefix="tenant/c/", index_name=INDEX, search_strategy=strategy
    ).namespace("memory/facts")

    await strategy.index(view, "a", await view.read("a"))

    item = vector_table.get_item(TableName=TABLE, Key={"pk": {"S": "tenant/c"}, "sk": {"S": "memory/facts/a"}})["Item"]
    assert [float(n["N"]) for n in item["vector"]["L"]] == embed("dark mode")
    assert item["data"]["B"] == b"dark mode"
    assert item["meta"]["M"] == {"kind": {"S": "pref"}}, "only the vector is written, so metadata survives"
    # The condition rejects a second backfill: the item already carries a vector.
    assert await view._set_vector("a", embed("coffee tea")) is False

    # The index is eventually consistent: poll until the backfilled item is visible.
    found = []
    deadline = time.time() + 300
    while time.time() < deadline:
        found = await view.search("dark")
        if found:
            break
        await asyncio.sleep(5)
    assert [result.key for result in found] == ["a"]


async def test_vector_search_strategy_end_to_end(vector_table):
    """DynamoDBVectorSearchStrategy against the real index: FileMemoryStore recalls by meaning, best first."""
    vocabulary = ("dark", "mode", "coffee", "tea")

    def embed(text: str) -> list[float]:
        tokens = text.lower().replace("#", " ").split()
        return [0.01 + tokens.count(word) for word in vocabulary]

    storage = DynamoDBStorage(
        TABLE,
        region_name=REGION,
        prefix="tenant/b/",
        index_name=INDEX,
        search_strategy=DynamoDBVectorSearchStrategy(embedder=embed),
    )
    knowledge = FileMemoryStore(name="facts", storage=storage, writable=True)  # type: ignore[abstract]
    await knowledge.add("# Display\nprefers dark mode")
    await knowledge.add("# Drinks\nlikes coffee and tea")

    # The index is eventually consistent: poll until both items are visible.
    found = []
    deadline = time.time() + 300
    while time.time() < deadline:
        found = await knowledge.search("dark")
        if len(found) == 2:
            break
        await asyncio.sleep(5)
    assert [entry.content.splitlines()[0] for entry in found] == ["# Display", "# Drinks"]

    raw = await storage.namespace("memory/facts").search("dark")
    assert raw[0].score > raw[1].score, "converted COSINE scores rank the closest value first"
    assert all(0 < result.score <= 1 for result in raw)

    await storage.namespace("memory/facts").delete(raw[0].key)
    deadline = time.time() + 300
    while time.time() < deadline:
        remaining = [result.key for result in await storage.namespace("memory/facts").search("dark")]
        if raw[0].key not in remaining:
            break
        await asyncio.sleep(5)
    assert raw[0].key not in remaining
