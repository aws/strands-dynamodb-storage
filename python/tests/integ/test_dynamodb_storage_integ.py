# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests — real Amazon DynamoDB + S3 (never DynamoDB Local).

Gated by RUN_INTEG=1 and AWS credentials. Provisions its own table (pk/sk) and — unless
INTEG_S3_BUCKET is supplied — its own S3 bucket, then tears everything down.

Run:
    RUN_INTEG=1 AWS_REGION=us-east-1 .venv/bin/python -m pytest tests/integ -q

Native vector search() is covered separately in test_vector_search_integ.py (it
provisions its own vector-indexed table).
"""

from __future__ import annotations

import os
import time
import uuid

import boto3
import pytest
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import DynamoDBListQuery, DynamoDBStorage

RUN = os.environ.get("RUN_INTEG") == "1"
REGION = os.environ.get("AWS_REGION", "us-east-1")
STAMP = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
TABLE = os.environ.get("INTEG_TABLE", f"strands-ddb-storage-py-integ-{STAMP}")
PROVIDED_BUCKET = os.environ.get("INTEG_S3_BUCKET")
BUCKET = (PROVIDED_BUCKET or f"strands-ddb-storage-py-integ-{STAMP}").lower()

pytestmark = pytest.mark.skipif(not RUN, reason="set RUN_INTEG=1 (needs AWS credentials) to run")

_created_bucket = False


@pytest.fixture(scope="module", autouse=True)
def provision():
    ddb = boto3.client("dynamodb", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    global _created_bucket
    ddb.create_table(
        TableName=TABLE,
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.get_waiter("table_exists").wait(TableName=TABLE)
    ddb.update_time_to_live(TableName=TABLE, TimeToLiveSpecification={"AttributeName": "expireAt", "Enabled": True})
    if not PROVIDED_BUCKET:
        kwargs = {"Bucket": BUCKET}
        if REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
        s3.create_bucket(**kwargs)
        _created_bucket = True
    yield
    try:
        ddb.delete_table(TableName=TABLE)
    except Exception:
        pass
    if _created_bucket:
        try:
            token = None
            while True:
                resp = s3.list_objects_v2(Bucket=BUCKET, **({"ContinuationToken": token} if token else {}))
                for obj in resp.get("Contents", []):
                    s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
                if not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")
            s3.delete_bucket(Bucket=BUCKET)
        except Exception:
            pass


def store(**kw) -> DynamoDBStorage:
    return DynamoDBStorage(TABLE, region_name=REGION, **kw)


async def test_point_ops_and_list_and_namespace():
    s = store()
    base = f"sessions/pt-{STAMP}"
    await s.write(f"{base}/a", b"hello")
    assert await s.read(f"{base}/a") == b"hello"
    await s.write(f"{base}/b", b"two")
    assert await s.list(DynamoDBListQuery(pk=base)) == [f"{base}/a", f"{base}/b"]
    view = s.namespace(base)
    await view.write("scopes/agent/x", b"v")
    assert await view.read("scopes/agent/x") == b"v"
    assert await s.read(f"{base}/scopes/agent/x") == b"v"
    await s.delete(f"{base}/a")
    assert await s.read(f"{base}/a") is None


async def test_s3_offload_and_compression():
    s = store(s3_bucket=BUCKET, compression="gzip")
    big = b"A" * 800_000  # compressible -> should stay inline
    await s.write(f"sessions/z-{STAMP}/inline", big)
    assert await s.read(f"sessions/z-{STAMP}/inline") == big
    payload = os.urandom(400_001)  # incompressible -> offloads to S3
    await s.write(f"sessions/z-{STAMP}/big", payload)
    assert await s.read(f"sessions/z-{STAMP}/big") == payload
    await s.delete(f"sessions/z-{STAMP}/big")


async def test_ttl_stamp_and_filter():
    s = store(ttl_seconds=3600)
    pk = f"sessions/ttl-{STAMP}"
    await s.write(f"{pk}/live", b"L")
    await s.write(f"{pk}/dead", b"D", ttl_seconds=-1)
    assert await s.read(f"{pk}/dead") is None
    assert await s.list(DynamoDBListQuery(pk=pk)) == [f"{pk}/live"]


async def test_oversized_without_bucket_raises():
    with pytest.raises(StorageError, match="s3_bucket"):
        await store().write(f"sessions/nb-{STAMP}/big", b"Z" * 400_001)


async def test_cross_tenant_isolation():
    a = store(prefix=f"tenant-a-{STAMP}")
    b = store(prefix=f"tenant-b-{STAMP}")
    await a.write("sessions/s1/secret", b"a-data")
    await b.write("sessions/s1/secret", b"b-data")
    assert await a.read("sessions/s1/secret") == b"a-data"
    assert await b.read("sessions/s1/secret") == b"b-data"
    assert await b.list("sessions/s1/") == ["sessions/s1/secret"]
