# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for DynamoDBMemoryStore: real Amazon DynamoDB and Amazon Bedrock.

Gated by RUN_INTEG=1 and AWS credentials with ``bedrock:InvokeModel`` on Amazon Titan
Text Embeddings V2 and Cohere Embed v3 and v4 (cross-Region inference profile
``us.cohere.embed-v4:0``). Provisions its own table with a 1024-dimension vector index,
waits for the index to finish backfilling, exercises the memory store through the
Strands Memory Manager, and tears the table down afterwards.

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
from strands.memory import MemoryManager
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import BedrockEmbedder, DynamoDBMemoryStore, DynamoDBStorage

RUN = os.environ.get("RUN_INTEG") == "1"
REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = f"strands-integ-memory-{int(time.time())}-{uuid.uuid4().hex[:6]}"

pytestmark = [
    pytest.mark.skipif(not RUN, reason="integ tests run only with RUN_INTEG=1"),
    pytest.mark.asyncio,
]

MEMORIES = ["Prefers a window seat on flights", "Has a severe peanut allergy", "Drinks black coffee, no sugar"]


@pytest.fixture(scope="module")
def memory_table():
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
                "IndexName": "vector_index",
                "VectorAttribute": {"AttributeName": "vector"},
                "SearchSchema": [{"AttributeName": "pk", "SearchSchemaElementType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "Dimensions": 1024,
                "DistanceFunction": "COSINE",
            }
        ],
    )
    try:
        client.get_waiter("table_exists").wait(TableName=TABLE)
        deadline = time.time() + 600
        while time.time() < deadline:
            indexes = client.describe_table(TableName=TABLE)["Table"].get("VectorIndexes", [])
            if indexes and indexes[0].get("IndexStatus") == "ACTIVE" and not indexes[0].get("Backfilling", False):
                break
            time.sleep(5)
        else:
            pytest.fail("vector index did not become ready within 600s")
        yield client
    finally:
        client.delete_table(TableName=TABLE)


async def search_until(store: DynamoDBMemoryStore, query: str, count: int):
    """The vector index is eventually consistent: poll until ``count`` memories are visible."""
    deadline = time.time() + 300
    entries = []
    while time.time() < deadline:
        entries = await store.search(query, {"max_search_results": 10})
        if len(entries) >= count:
            return entries
        await asyncio.sleep(5)
    pytest.fail(f"expected {count} memories in search results within 300s; got {len(entries)}")


async def test_memory_manager_end_to_end_with_the_default_embedder(memory_table):
    storage = DynamoDBStorage(TABLE, region_name=REGION).namespace(f"user/default-{uuid.uuid4().hex[:6]}")
    store = DynamoDBMemoryStore(storage, name="prefs")
    manager = MemoryManager(stores=[store], injection=False)
    for memory in MEMORIES:
        await manager.add(memory)

    entries = await search_until(store, "Where does the user like to sit on a plane?", 3)
    assert entries[0].content == "Prefers a window seat on flights"
    assert entries[0].metadata["_score"] <= entries[1].metadata["_score"]  # COSINE distance: lower is closer

    [top] = await manager.search("What can't the user eat?", {"max_search_results": 1})
    assert top.content == "Has a severe peanut allergy"
    assert top.store_name == "prefs"

    # Adding the same text again overwrites the item instead of duplicating it.
    key = await store.add(MEMORIES[0], {"source": "profile"})
    partition = storage.partition_key(key)
    items = memory_table.query(
        TableName=TABLE,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": partition}},
        ConsistentRead=True,
    )["Items"]
    assert len(items) == 3

    await storage.delete(key)
    assert await storage.read(key) is None


@pytest.mark.parametrize("model_id", ["us.cohere.embed-v4:0", "cohere.embed-english-v3"])
async def test_cohere_embedders(memory_table, model_id):
    storage = DynamoDBStorage(TABLE, region_name=REGION).namespace(f"user/cohere-{uuid.uuid4().hex[:6]}")
    store = DynamoDBMemoryStore(storage, embedder=BedrockEmbedder(model_id, region_name=REGION))
    for memory in MEMORIES:
        await store.add(memory)
    entries = await search_until(store, "Where does the user like to sit on a plane?", 3)
    assert entries[0].content == "Prefers a window seat on flights"


async def test_mismatched_dimensions_fail_without_writing(memory_table):
    storage = DynamoDBStorage(TABLE, region_name=REGION).namespace(f"user/small-{uuid.uuid4().hex[:6]}")
    store = DynamoDBMemoryStore(storage, embedder=BedrockEmbedder(dimensions=512, region_name=REGION))
    with pytest.raises(StorageError) as write_error:
        await store.add("Prefers a window seat")
    assert "Expected: 1024, Actual: 512" in str(write_error.value.__cause__)
    with pytest.raises(StorageError) as search_error:
        await store.search("window seat")
    assert "does not match vector index dimension 1024" in str(search_error.value.__cause__)
    items = memory_table.query(
        TableName=TABLE,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": storage.partition_key("memory/dynamodb/x")}},
        ConsistentRead=True,
    )["Items"]
    assert items == []
