# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoDBStorage, using moto for offline DynamoDB + S3."""

from __future__ import annotations

import gzip
import math
import time
from unittest import mock

import boto3
import pytest
from moto import mock_aws
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import DynamoDBListQuery, DynamoDBStorage, SearchQuery

REGION = "us-east-1"
TABLE = "test-table"
BUCKET = "test-bucket"
SENTINEL = "\u0000"


@pytest.fixture
def aws():
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName=TABLE,
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            BillingMode="PAY_PER_REQUEST",
        )
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        yield ddb, s3


def make(aws, **kw) -> DynamoDBStorage:
    ddb, s3 = aws
    kw.setdefault("client", ddb)
    if kw.get("s3_bucket"):
        kw.setdefault("s3_client", s3)
    return DynamoDBStorage(TABLE, **kw)


def _split(full: str) -> tuple[str, str]:
    segs = [s for s in full.split("/") if s]
    if len(segs) <= 1:
        return (segs[0] if segs else full), SENTINEL
    return "/".join(segs[:2]), ("/".join(segs[2:]) or SENTINEL)


def raw_item(ddb, full_key: str):
    pk, sk = _split(full_key)
    return ddb.get_item(TableName=TABLE, Key={"pk": {"S": pk}, "sk": {"S": sk}}).get("Item")


def s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / ((na * nb) or 1)


def cosine_adapter(ddb):
    """Brute-force cosine over stored vectors — stands in for native SearchVectors."""

    async def adapter(params):
        resp = ddb.scan(TableName=TABLE)
        out = []
        for it in resp.get("Items", []):
            if "vector" not in it:
                continue
            if params.get("pk") and it["pk"]["S"] != params["pk"]:
                continue
            vec = [float(n["N"]) for n in it["vector"]["L"]]
            meta = None
            if "meta" in it:
                meta = {
                    k: (v["S"] if "S" in v else (float(v["N"]) if "N" in v else v.get("BOOL")))
                    for k, v in it["meta"]["M"].items()
                }
            out.append({"key": it["k"]["S"], "score": _cosine(params["vector"], vec), "metadata": meta})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[: params["top_k"]]

    return adapter


# --------------------------------------------------------------------- point ops


async def test_round_trip_and_overwrite(aws):
    s = make(aws)
    await s.write("sessions/s1/a", b"hello")
    assert await s.read("sessions/s1/a") == b"hello"
    await s.write("sessions/s1/a", b"world")
    assert await s.read("sessions/s1/a") == b"world"


async def test_missing_key_returns_none(aws):
    assert await make(aws).read("sessions/s1/missing") is None


async def test_delete_and_noop(aws):
    s = make(aws)
    await s.write("sessions/s1/a", b"x")
    await s.delete("sessions/s1/a")
    assert await s.read("sessions/s1/a") is None
    await s.delete("sessions/s1/a")  # no-op


async def test_single_segment_key(aws):
    s = make(aws)
    await s.write("singleton", b"one")
    assert await s.read("singleton") == b"one"
    await s.delete("singleton")
    assert await s.read("singleton") is None


async def test_rejects_empty_and_traversal_keys(aws):
    s = make(aws)
    with pytest.raises(StorageError, match="must not be empty"):
        await s.write("", b"x")
    with pytest.raises(StorageError, match=r"'\.\.'"):
        await s.read("sessions/../etc")


# ----------------------------------------------------------------------- listing


async def test_list_prefix_sorted(aws):
    s = make(aws)
    base = "sessions/s1/scopes/agent/a1/immutable_history"
    await s.write(f"{base}/snapshot_2.json", b"2")
    await s.write(f"{base}/snapshot_1.json", b"1")
    await s.write(f"{base}/snapshot_3.json", b"3")
    assert await s.list(f"{base}/") == [f"{base}/snapshot_1.json", f"{base}/snapshot_2.json", f"{base}/snapshot_3.json"]


async def test_list_structured_query_and_between_and_start_after(aws):
    s = make(aws)
    pk = "sessions/s1"
    for n in ["1", "2", "3", "4"]:
        await s.write(f"{pk}/k/{n}", n.encode())
    assert sorted(await s.list(DynamoDBListQuery(pk=pk))) == [f"{pk}/k/{n}" for n in ["1", "2", "3", "4"]]
    assert await s.list(DynamoDBListQuery(pk=pk, sk_between=("k/2", "k/3"))) == [f"{pk}/k/2", f"{pk}/k/3"]
    assert await s.list(DynamoDBListQuery(pk=pk, start_after=f"{pk}/k/2")) == [f"{pk}/k/3", f"{pk}/k/4"]


async def test_list_rejects_broad_prefix_and_dual_sk(aws):
    s = make(aws)
    with pytest.raises(StorageError, match="too broad"):
        await s.list("sessions/")
    with pytest.raises(StorageError, match="not both"):
        await s.list(DynamoDBListQuery(pk="x/y", sk_prefix="a", sk_between=("a", "b")))


# -------------------------------------------------------------------- S3 offload


async def test_s3_offload_round_trip_and_cleanup(aws):
    _, s3 = aws
    s = make(aws, s3_bucket=BUCKET)
    payload = b"Z" * 400_001
    await s.write("sessions/s1/big", payload)
    item = raw_item(aws[0], "sessions/s1/big")
    assert item["s3"]["BOOL"] is True and "data" not in item
    assert s3_exists(s3, "sessions/s1/big")
    assert await s.read("sessions/s1/big") == payload
    await s.delete("sessions/s1/big")
    assert not s3_exists(s3, "sessions/s1/big")
    assert await s.read("sessions/s1/big") is None


async def test_oversized_without_bucket_raises(aws):
    with pytest.raises(StorageError, match="s3_bucket"):
        await make(aws).write("sessions/s1/big", b"Z" * 400_001)


async def test_small_value_inline(aws):
    _, s3 = aws
    s = make(aws, s3_bucket=BUCKET)
    await s.write("sessions/s1/small", b"tiny")
    assert not s3_exists(s3, "sessions/s1/small")
    assert await s.read("sessions/s1/small") == b"tiny"


async def test_shrink_overwrite_reclaims_offloaded_s3_object(aws):
    """Offloaded -> inline overwrite must delete the now-unreferenced S3 object."""
    ddb, s3 = aws
    s = make(aws, s3_bucket=BUCKET)
    await s.write("sessions/s1/doc", b"Z" * 400_001)
    assert s3_exists(s3, "sessions/s1/doc")
    await s.write("sessions/s1/doc", b"small now")
    item = raw_item(ddb, "sessions/s1/doc")
    assert "s3" not in item and item["data"]["B"] == b"small now"
    assert not s3_exists(s3, "sessions/s1/doc")  # reclaimed, not orphaned
    assert await s.read("sessions/s1/doc") == b"small now"
    await s.delete("sessions/s1/doc")  # still clean end-to-end


async def test_inline_overwrite_never_touches_s3(aws):
    """Negative control: inline -> inline overwrite must not attempt S3 cleanup."""
    s = make(aws, s3_bucket=BUCKET)
    await s.write("sessions/s1/doc", b"one")
    with mock.patch.object(s, "_s3_delete", wraps=s._s3_delete) as spy:
        await s.write("sessions/s1/doc", b"two")
        spy.assert_not_called()
    assert await s.read("sessions/s1/doc") == b"two"


async def test_offloaded_overwrite_keeps_object_readable(aws):
    """Offloaded -> offloaded overwrite reuses the deterministic object key."""
    _, s3 = aws
    s = make(aws, s3_bucket=BUCKET)
    await s.write("sessions/s1/doc", b"A" * 400_001)
    await s.write("sessions/s1/doc", b"B" * 400_002)
    assert s3_exists(s3, "sessions/s1/doc")
    assert await s.read("sessions/s1/doc") == b"B" * 400_002


async def test_s3_cleanup_failure_does_not_fail_the_write(aws):
    """The shrink-overwrite reclamation is best-effort: the write must survive it."""
    _, s3 = aws
    s = make(aws, s3_bucket=BUCKET)
    await s.write("sessions/s1/doc", b"Z" * 400_001)
    with mock.patch.object(s, "_s3_delete", side_effect=RuntimeError("s3 down")):
        await s.write("sessions/s1/doc", b"small now")  # must not raise
    assert await s.read("sessions/s1/doc") == b"small now"


# ------------------------------------------------------------------- compression


async def test_gzip_inline(aws):
    s = make(aws, compression="gzip")
    original = b"A" * 5000
    await s.write("sessions/s1/big", original)
    item = raw_item(aws[0], "sessions/s1/big")
    assert item["z"]["BOOL"] is True
    assert len(item["data"]["B"]) < 5000
    assert await s.read("sessions/s1/big") == original


async def test_large_compressible_stays_inline(aws):
    _, s3 = aws
    s = make(aws, compression="gzip", s3_bucket=BUCKET)
    original = b"A" * 800_000
    await s.write("sessions/s1/big", original)
    assert not s3_exists(s3, "sessions/s1/big")  # cost win
    assert raw_item(aws[0], "sessions/s1/big")["z"]["BOOL"] is True
    assert await s.read("sessions/s1/big") == original


async def test_no_shrink_stored_uncompressed(aws):
    s = make(aws, compression="gzip")
    await s.write("sessions/s1/tiny", b"hi")
    assert "z" not in raw_item(aws[0], "sessions/s1/tiny")
    assert await s.read("sessions/s1/tiny") == b"hi"


async def test_reader_without_compression_reads_compressed(aws):
    writer = make(aws, compression="gzip")
    await writer.write("sessions/s1/a", b"A" * 2000)
    reader = make(aws)  # compression off
    assert await reader.read("sessions/s1/a") == b"A" * 2000


async def test_incompressible_offloads(aws):
    import os

    _, s3 = aws
    s = make(aws, compression="gzip", s3_bucket=BUCKET)
    payload = os.urandom(400_001)
    await s.write("sessions/s1/big", payload)
    assert s3_exists(s3, "sessions/s1/big")
    assert await s.read("sessions/s1/big") == payload


# --------------------------------------------------------------------------- TTL


async def test_ttl_stamps_future_expiry(aws):
    s = make(aws, ttl_seconds=3600)
    before = int(time.time())
    await s.write("sessions/s1/a", b"v")
    exp = int(raw_item(aws[0], "sessions/s1/a")["expireAt"]["N"])
    assert exp >= before + 3600


async def test_no_ttl_when_disabled(aws):
    s = make(aws)
    await s.write("sessions/s1/a", b"v")
    assert "expireAt" not in raw_item(aws[0], "sessions/s1/a")


async def test_per_write_ttl_override(aws):
    s = make(aws, ttl_seconds=60)
    before = int(time.time())
    await s.write("sessions/s1/a", b"v", ttl_seconds=7200)
    assert int(raw_item(aws[0], "sessions/s1/a")["expireAt"]["N"]) >= before + 7200


async def test_custom_ttl_attribute(aws):
    s = make(aws, ttl_seconds=60, ttl_attribute="ttl")
    await s.write("sessions/s1/a", b"v")
    item = raw_item(aws[0], "sessions/s1/a")
    assert "ttl" in item and "expireAt" not in item


async def test_ttl_read_and_list_filter(aws):
    s = make(aws, ttl_seconds=3600)
    await s.write("sessions/s1/live", b"L")
    await s.write("sessions/s1/dead", b"D", ttl_seconds=-1)
    assert await s.read("sessions/s1/dead") is None
    assert await s.list(DynamoDBListQuery(pk="sessions/s1")) == ["sessions/s1/live"]


async def test_ttl_opt_in_reader_without_ttl_returns_item(aws):
    writer = make(aws, ttl_seconds=3600)
    await writer.write("sessions/s1/a", b"v", ttl_seconds=-1)
    reader = make(aws)  # no ttl -> no filter
    assert await reader.read("sessions/s1/a") == b"v"


async def test_no_stamp_without_ttl_even_with_override(aws):
    s = make(aws)
    await s.write("sessions/s1/a", b"v", ttl_seconds=60)
    assert "expireAt" not in raw_item(aws[0], "sessions/s1/a")


async def test_float_ttl_duration_stamps_integer(aws):
    """A fractional duration must floor to an integer stamp (type hints aren't enforced)."""
    s = make(aws, ttl_seconds=3600)
    await s.write("sessions/s1/a", b"v", ttl_seconds=90.5)  # type: ignore[arg-type]
    raw = raw_item(aws[0], "sessions/s1/a")["expireAt"]["N"]
    assert "." not in raw
    assert int(raw) >= int(time.time()) + 90


async def test_read_tolerates_foreign_float_ttl(aws):
    """Another producer on the shared table may stamp a fractional epoch value.

    An expired float filters like an expired int; a future float passes through;
    neither crashes read().
    """
    ddb, _ = aws
    s = make(aws, ttl_seconds=3600)
    await s.write("sessions/s1/dead", b"D")
    await s.write("sessions/s1/live", b"L")
    for key, stamp in (("sessions/s1/dead", time.time() - 60.5), ("sessions/s1/live", time.time() + 3600.5)):
        pk, sk = _split(key)
        ddb.update_item(
            TableName=TABLE,
            Key={"pk": {"S": pk}, "sk": {"S": sk}},
            UpdateExpression="SET expireAt = :v",
            ExpressionAttributeValues={":v": {"N": str(stamp)}},
        )
    assert await s.read("sessions/s1/dead") is None
    assert await s.read("sessions/s1/live") == b"L"


async def test_read_ignores_wrong_type_ttl_attribute(aws):
    """A TTL attribute of the wrong type is another writer's bug, not a read() crash.

    DynamoDB's N type won't accept a non-numeric string, so the worst a foreign
    writer can do beyond a float is stamp the attribute as a different type (S
    here): the "N" lookup misses and the item is treated as not expired.
    """
    ddb, _ = aws
    s = make(aws, ttl_seconds=3600)
    await s.write("sessions/s1/a", b"v")
    pk, sk = _split("sessions/s1/a")
    ddb.update_item(
        TableName=TABLE,
        Key={"pk": {"S": pk}, "sk": {"S": sk}},
        UpdateExpression="SET expireAt = :v",
        ExpressionAttributeValues={":v": {"S": "not-a-number"}},
    )
    assert await s.read("sessions/s1/a") == b"v"


# --------------------------------------------------------------------- namespace


async def test_constructor_prefix_transparent(aws):
    s = make(aws, prefix="tenant-x")
    await s.write("sessions/s1/a", b"v")
    assert await s.read("sessions/s1/a") == b"v"
    assert await s.list("sessions/s1/") == ["sessions/s1/a"]
    assert raw_item(aws[0], "tenant-x/sessions/s1/a") is not None


async def test_namespace_view(aws):
    s = make(aws)
    view = s.namespace("sessions/s1")
    await view.write("scopes/agent/a1/x", b"v")
    assert await view.read("scopes/agent/a1/x") == b"v"
    assert await s.read("sessions/s1/scopes/agent/a1/x") == b"v"
    assert await view.list("scopes/agent/a1/") == ["scopes/agent/a1/x"]


def test_namespace_region_config_composes():
    s = DynamoDBStorage(TABLE, region_name=REGION)
    view = s.namespace("sessions/s1")
    assert isinstance(view, DynamoDBStorage)
    assert isinstance(view.namespace("scopes/a"), DynamoDBStorage)


# ------------------------------------------------------------------ vector search


async def test_vector_write_inline(aws):
    s = make(aws, vector_search=cosine_adapter(aws[0]))
    await s.write("memory/u1/m1", b"likes window seats", vector=[1, 0, 0], metadata={"kind": "pref"})
    item = raw_item(aws[0], "memory/u1/m1")
    assert [float(n["N"]) for n in item["vector"]["L"]] == [1.0, 0.0, 0.0]
    assert item["meta"]["M"]["kind"]["S"] == "pref"
    assert await s.read("memory/u1/m1") == b"likes window seats"


async def test_search_ranking_values_metadata(aws):
    s = make(aws, vector_search=cosine_adapter(aws[0]))
    await s.write("memory/u1/a", b"a", vector=[1, 0, 0], metadata={"source": "profile"})
    await s.write("memory/u1/b", b"b", vector=[0.9, 0.1, 0])
    await s.write("memory/u1/c", b"c", vector=[0, 1, 0])
    results = await s.search(SearchQuery(vector=[1, 0, 0], top_k=2, pk="memory/u1", include_values=True))
    assert [r.key for r in results] == ["memory/u1/a", "memory/u1/b"]
    assert results[0].score >= results[1].score
    assert results[0].data == b"a"
    assert results[0].metadata == {"source": "profile"}


async def test_search_native_calls_search_vectors(aws):
    """Without an adapter, search() issues DynamoDB SearchVectors natively."""
    ddb, _ = aws
    storage = make(aws, prefix="tenant/a/")
    with mock.patch.object(
        ddb,
        "search_vectors",
        create=True,
        return_value={
            "SearchResults": [
                {
                    "Item": {
                        "pk": {"S": "tenant/a"},
                        "sk": {"S": "memories/m1"},
                        "meta": {"M": {"source": {"S": "profile"}, "rank": {"N": "2"}}},
                    },
                    "Score": 0.125,
                }
            ]
        },
    ) as sv:
        results = await storage.search(SearchQuery(vector=[1.0, 0.0, 0.0], top_k=3, pk="tenant/a"))
    request = sv.call_args.kwargs
    assert request["TopK"] == 3
    assert request["SearchVector"] == [{"N": "1.0"}, {"N": "0.0"}, {"N": "0.0"}]
    assert request["SearchConditionExpression"] == "#pk = :pk"
    assert request["ExpressionAttributeValues"] == {":pk": {"S": "tenant/a"}}
    assert len(results) == 1
    assert results[0].key == "memories/m1"  # namespace prefix stripped
    assert results[0].score == 0.125
    assert results[0].metadata == {"source": "profile", "rank": 2}


async def test_search_native_topk_bounds(aws):
    storage = make(aws)
    with pytest.raises(StorageError, match="top_k"):
        await storage.search(SearchQuery(vector=[1.0], top_k=101))
    with pytest.raises(StorageError, match="top_k"):
        await storage.search(SearchQuery(vector=[1.0], top_k=0))


async def test_search_native_rejects_nonfinite_vector(aws):
    storage = make(aws)
    with pytest.raises(StorageError, match="non-finite"):
        await storage.search(SearchQuery(vector=[float("nan"), 0.0], top_k=1))


async def test_write_rejects_nonfinite_vector(aws):
    """Mirror of the query-side check: same input, same clear client-side message."""
    storage = make(aws)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(StorageError, match="non-finite"):
            await storage.write("sessions/s1/m", b"v", vector=[bad, 0.0])
    # and the item was never written
    assert raw_item(aws[0], "sessions/s1/m") is None


async def test_search_native_filter_overfetches_and_postfilters(aws):
    ddb, _ = aws
    storage = make(aws, prefix="tenant/a/")

    def item(sk, source, score):
        return {
            "Item": {
                "pk": {"S": "tenant/a"},
                "sk": {"S": sk},
                "meta": {"M": {"source": {"S": source}}},
            },
            "Score": score,
        }

    with mock.patch.object(
        ddb,
        "search_vectors",
        create=True,
        return_value={
            "SearchResults": [
                item("m1", "web", 0.1),
                item("m2", "profile", 0.2),
                item("m3", "profile", 0.3),
            ]
        },
    ) as sv:
        results = await storage.search(SearchQuery(vector=[1.0], top_k=2, pk="tenant/a", filter={"source": "profile"}))
    # Over-fetched (top_k * factor, capped at 100), then post-filtered to top_k.
    assert sv.call_args.kwargs["TopK"] == 20
    assert [r.key for r in results] == ["m2", "m3"]


async def test_search_native_projects_response_floor(aws):
    """The request asks only for keys + metadata; payload attrs only with include_values."""
    ddb, _ = aws
    storage = make(aws)
    resp = {"SearchResults": []}
    with mock.patch.object(ddb, "search_vectors", create=True, return_value=resp) as sv:
        await storage.search(SearchQuery(vector=[1.0], top_k=1, pk="tenant/a"))
    req = sv.call_args.kwargs
    assert req["ProjectionExpression"] == "#pk, #sk, #m"
    assert req["ExpressionAttributeNames"] == {"#pk": "pk", "#sk": "sk", "#m": "meta"}
    with mock.patch.object(ddb, "search_vectors", create=True, return_value=resp) as sv:
        await storage.search(SearchQuery(vector=[1.0], top_k=1, pk="tenant/a", include_values=True))
    req = sv.call_args.kwargs
    assert req["ProjectionExpression"] == "#pk, #sk, #m, #d, #s3, #z"
    assert req["ExpressionAttributeNames"]["#d"] == "data"


async def test_search_include_values_reuses_projected_payload(aws):
    """A projected inline payload is decoded from the hit; no GetItem is issued."""
    ddb, _ = aws
    storage = make(aws)
    resp = {
        "SearchResults": [
            {
                "Item": {
                    "pk": {"S": "tenant/a"},
                    "sk": {"S": "m1"},
                    "data": {"B": b"inline payload"},
                },
                "Score": 0.1,
            },
            {
                "Item": {
                    "pk": {"S": "tenant/a"},
                    "sk": {"S": "m2"},
                    "data": {"B": gzip.compress(b"compressed payload")},
                    "z": {"BOOL": True},
                },
                "Score": 0.2,
            },
        ]
    }
    with (
        mock.patch.object(ddb, "search_vectors", create=True, return_value=resp),
        mock.patch.object(ddb, "get_item", wraps=ddb.get_item) as gi,
    ):
        results = await storage.search(SearchQuery(vector=[1.0], top_k=2, pk="tenant/a", include_values=True))
    assert [r.data for r in results] == [b"inline payload", b"compressed payload"]
    assert gi.call_count == 0


async def test_search_include_values_falls_back_for_offloaded(aws):
    """An s3-flagged hit carries no inline payload; the point read fetches from S3."""
    ddb, _ = aws
    storage = make(aws, s3_bucket=BUCKET)
    big = b"A" * 380_001
    await storage.write("tenant/a/big", big, vector=[1.0])
    resp = {
        "SearchResults": [
            {"Item": {"pk": {"S": "tenant/a"}, "sk": {"S": "big"}, "s3": {"BOOL": True}}, "Score": 0.1},
        ]
    }
    with (
        mock.patch.object(ddb, "search_vectors", create=True, return_value=resp),
        mock.patch.object(ddb, "get_item", wraps=ddb.get_item) as gi,
    ):
        results = await storage.search(SearchQuery(vector=[1.0], top_k=1, pk="tenant/a", include_values=True))
    assert results[0].data == big
    assert gi.call_count == 1


async def test_search_native_drops_foreign_namespace_matches(aws):
    ddb, _ = aws
    storage = make(aws, prefix="tenant/a/")
    with mock.patch.object(
        ddb,
        "search_vectors",
        create=True,
        return_value={
            "SearchResults": [
                {"Item": {"pk": {"S": "tenant/b"}, "sk": {"S": "m1"}}, "Score": 0.1},
            ]
        },
    ):
        results = await storage.search(SearchQuery(vector=[1.0], top_k=1))
    assert results == []


# ------------------------------------------------------------ construction/errors


def test_rejects_both_client_and_region(aws):
    with pytest.raises(StorageError, match="both client and region"):
        DynamoDBStorage(TABLE, client=aws[0], region_name=REGION)


async def test_wraps_backend_failures(aws):
    ddb, _ = aws
    bad = DynamoDBStorage("nonexistent-table", client=ddb)
    with pytest.raises(StorageError):
        await bad.write("sessions/s1/a", b"x")
    with pytest.raises(StorageError):
        await bad.read("sessions/s1/a")
    with pytest.raises(StorageError):
        await bad.delete("sessions/s1/a")
    with pytest.raises(StorageError):
        await bad.list(DynamoDBListQuery(pk="sessions/s1"))


async def test_wraps_search_adapter_failure(aws):
    async def boom(_params):
        raise RuntimeError("adapter boom")

    s = make(aws, vector_search=boom)
    with pytest.raises(StorageError):
        await s.search(SearchQuery(vector=[1, 0, 0], top_k=1))


async def test_list_rejects_traversal_prefix(aws):
    with pytest.raises(StorageError, match=r"'\.\.'"):
        await make(aws).list("sessions/../etc")


def test_public_exports():
    import strands_dynamodb_storage as pkg

    assert pkg.DynamoDBStorage is DynamoDBStorage
    for name in ("DynamoDBListQuery", "SearchQuery", "SearchResult", "VectorSearchAdapter"):
        assert hasattr(pkg, name)


# --------------------------------------------------- AutoSDE regression coverage


async def test_limit_with_start_after_returns_full_limit(aws):
    # start_after must be applied during collection so a limit is filled with
    # post-cursor keys, not truncated afterwards (would otherwise return < limit).
    s = make(aws)
    pk = "sessions/s1"
    for n in ["1", "2", "3", "4", "5"]:
        await s.write(f"{pk}/k/{n}", n.encode())
    result = await s.list(DynamoDBListQuery(pk=pk, start_after=f"{pk}/k/1", limit=2))
    assert result == [f"{pk}/k/2", f"{pk}/k/3"]


async def test_limit_with_ttl_filter_returns_live_items(aws):
    # With a TTL filter active, Limit must not be pushed to DynamoDB (it caps pre-filter
    # evaluation), so a limit returns that many *live* items rather than under-returning.
    s = make(aws, ttl_seconds=3600)
    pk = "sessions/s1"
    await s.write(f"{pk}/a_dead", b"x", ttl_seconds=-1)
    await s.write(f"{pk}/b_live", b"x")
    await s.write(f"{pk}/c_dead", b"x", ttl_seconds=-1)
    await s.write(f"{pk}/d_live", b"x")
    await s.write(f"{pk}/e_live", b"x")
    result = await s.list(DynamoDBListQuery(pk=pk, limit=2))
    assert result == [f"{pk}/b_live", f"{pk}/d_live"]


def test_namespace_forwards_session_and_config():
    session = object()
    config = object()
    s = DynamoDBStorage(TABLE, boto_session=session, boto_client_config=config)
    view = s.namespace("sessions/s1")
    assert view._boto_session is session
    assert view._boto_client_config is config


async def test_cross_tenant_isolation(aws):
    # Two tenants scoped by a constructor-bound prefix. The same logical key must
    # resolve to physically distinct partitions, and neither tenant can read or list
    # into the other's data, which is the isolation guarantee this backend must hold.
    a = make(aws, prefix="tenant-a")
    b = make(aws, prefix="tenant-b")
    await a.write("sessions/s1/secret", b"a-data")
    await a.write("sessions/s1/other", b"a-other")
    await b.write("sessions/s1/secret", b"b-data")
    assert await a.read("sessions/s1/secret") == b"a-data"
    assert await b.read("sessions/s1/secret") == b"b-data"  # same key, isolated value
    assert await a.list("sessions/s1/") == ["sessions/s1/other", "sessions/s1/secret"]
    assert await b.list("sessions/s1/") == ["sessions/s1/secret"]  # cannot see tenant-a keys
    # underlying partitions are physically distinct
    assert raw_item(aws[0], "tenant-a/sessions/s1/secret")["data"]["B"] != b"b-data"


async def test_structured_query_cannot_cross_namespace(aws):
    # A DynamoDBListQuery names the partition directly, so the namespace boundary is
    # enforced explicitly rather than by construction. Naming another tenant's
    # partition is rejected, not silently answered with different data.
    a = make(aws, prefix="tenant-a")
    b = make(aws, prefix="tenant-b")
    await a.write("sessions/s1/secret", b"a-data")
    with pytest.raises(StorageError, match="outside this storage namespace"):
        await b.list(DynamoDBListQuery(pk="tenant-a/sessions"))
    # the owning instance can still query its own partition
    assert await a.list(DynamoDBListQuery(pk="tenant-a/sessions")) == ["sessions/s1/secret"]


async def test_structured_query_unprefixed_storage_unrestricted(aws):
    # Without a namespace there is no boundary to enforce; naming any partition is fine.
    s = make(aws)
    await s.write("sessions/s1/a", b"1")
    assert await s.list(DynamoDBListQuery(pk="sessions/s1")) == ["sessions/s1/a"]


async def test_search_pk_cannot_cross_namespace(aws):
    async def never_called(params):  # pragma: no cover - must not run
        raise AssertionError("adapter must not be reached for an out-of-scope pk")

    b = make(aws, prefix="tenant-b", vector_search=never_called)
    with pytest.raises(StorageError, match="outside this storage namespace"):
        await b.search(SearchQuery(vector=[1.0, 0.0], top_k=1, pk="tenant-a/sessions"))


async def test_search_drops_matches_outside_namespace(aws):
    # pk is optional, so the index can return matches from other namespaces. Those are
    # dropped rather than de-prefixed blindly into something resembling our own key.
    async def leaky(params):
        return [
            {"key": "tenant-a/sessions/s1/secret", "score": 0.99},
            {"key": "tenant-b/sessions/s1/mine", "score": 0.98},
        ]

    b = make(aws, prefix="tenant-b", vector_search=leaky)
    results = await b.search(SearchQuery(vector=[1.0, 0.0], top_k=2))
    assert [r.key for r in results] == ["sessions/s1/mine"]


async def test_list_drops_stored_keys_outside_namespace(aws):
    # Defence in depth on the read path: an item physically inside our partition whose
    # stored key attribute belongs elsewhere must not be de-prefixed into our namespace.
    ddb, _ = aws
    b = make(aws, prefix="tenant-b")
    await b.write("sessions/s1/mine", b"ok")
    pk, sk = _split("tenant-b/sessions/s1/planted")
    ddb.put_item(
        TableName=TABLE,
        Item={"pk": {"S": pk}, "sk": {"S": sk}, "k": {"S": "tenant-a/sessions/s1/secret"}, "data": {"B": b"x"}},
    )
    assert await b.list(DynamoDBListQuery(pk="tenant-b/sessions")) == ["sessions/s1/mine"]


async def test_structured_query_rejects_prefix_confusion(aws):
    # 'tenant' must not be treated as an ancestor of 'tenant-b/' by string prefixing.
    b = make(aws, prefix="tenant-b")
    with pytest.raises(StorageError, match="outside this storage namespace"):
        await b.list(DynamoDBListQuery(pk="tenant"))


async def test_namespaced_view_can_query_its_own_partition(aws):
    # A deep namespace lives inside a shallower physical partition; querying it is
    # allowed, and rows outside the namespace are filtered out.
    s = make(aws)
    view = s.namespace("sessions/s1")
    await view.write("scopes/agent/a1/x", b"v")
    await s.write("sessions/s2/other", b"sibling")
    assert await view.list(DynamoDBListQuery(pk="sessions/s1")) == ["scopes/agent/a1/x"]
