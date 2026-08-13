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

import os
import time
import uuid

import boto3
import pytest

from strands_dynamodb_storage import DynamoDBStorage, SearchQuery

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
