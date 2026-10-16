# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LexicalIndex tests against moto: a base table plus a separate pk/sk index table."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import re
from contextlib import contextmanager
from typing import Any
from unittest import mock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import (
    DynamoDBStorage,
    IdentifierQuery,
    LexicalIndex,
    LexicalIndexLimits,
    LexicalQuery,
    LexicalSearchResponse,
    RepairReport,
    RevisionConflictError,
    SearchableDocument,
)
from strands_dynamodb_storage.lexical_terms import (
    identifier_posting_pk,
    manifest_pk,
    posting_partition_keys,
    term_posting_pk,
)

REGION = "us-east-1"
BASE_TABLE = "base-table"
INDEX_TABLE = "index-table"
BUCKET = "test-bucket"
SENTINEL = "\u0000"
NOW = 1_800_000_000.0
IO_METHODS = ("get_item", "put_item", "query", "batch_get_item", "transact_write_items")
LONE_SURROGATE = chr(0xD800)
INDEX_ATTRIBUTE_NAMES = ("pk", "sk", "k", "data", "s3", "meta", "z", "rev", "tv", "terms", "ids", "lxrev", "lxscope")


@pytest.fixture
def aws():
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name=REGION)
        for table in (BASE_TABLE, INDEX_TABLE):
            ddb.create_table(
                TableName=table,
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


def make_storage(aws, **kw) -> DynamoDBStorage:
    ddb, s3 = aws
    kw.setdefault("client", ddb)
    if kw.get("s3_bucket"):
        kw.setdefault("s3_client", s3)
    return DynamoDBStorage(BASE_TABLE, **kw)


def make_index(storage: DynamoDBStorage, **limits: Any) -> LexicalIndex:
    return LexicalIndex(storage, index_table_name=INDEX_TABLE, limits=LexicalIndexLimits(**limits) if limits else None)


def doc(key: str, text: str = "", **fields: Any) -> SearchableDocument:
    data = fields.pop("data", b"value")
    return SearchableDocument(key=key, data=data, text=text, **fields)


def query(text: str, top_k: int = 10, **fields: Any) -> LexicalQuery:
    return LexicalQuery(text=text, top_k=top_k, **fields)


def keys(response: LexicalSearchResponse) -> list[str]:
    return [result.key for result in response.results]


@contextmanager
def frozen_clock(at: float):
    with mock.patch("strands_dynamodb_storage.dynamodb_storage.time") as clock:
        clock.time.return_value = at
        yield clock


def cancellation(*codes: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "TransactionCanceledException", "Message": "Transaction cancelled"},
            "CancellationReasons": [{"Code": code} for code in codes],
        },
        "TransactWriteItems",
    )


def _split(full: str) -> tuple[str, str]:
    segs = [s for s in full.split("/") if s]
    if len(segs) <= 1:
        return (segs[0] if segs else full), SENTINEL
    return "/".join(segs[:2]), ("/".join(segs[2:]) or SENTINEL)


def base_item(ddb, doc_id: str):
    pk, sk = _split(doc_id)
    return ddb.get_item(TableName=BASE_TABLE, Key={"pk": {"S": pk}, "sk": {"S": sk}}).get("Item")


def index_item(ddb, pk: str, sk: str):
    return ddb.get_item(TableName=INDEX_TABLE, Key={"pk": {"S": pk}, "sk": {"S": sk}}).get("Item")


def all_items(ddb, table: str) -> list[dict[str, Any]]:
    return ddb.scan(TableName=table)["Items"]


def postings_of(ddb, doc_id: str) -> dict[str, dict[str, Any]]:
    return {
        item["pk"]["S"]: item
        for item in all_items(ddb, INDEX_TABLE)
        if item["sk"]["S"] == doc_id and not item["pk"]["S"].startswith("m|")
    }


def assert_index_consistent(ddb, scope: str, doc_id: str) -> dict[str, Any]:
    """Manifest terms/ids equal the stored posting set, every posting and the base carry the manifest rev."""
    manifest = index_item(ddb, manifest_pk(scope), doc_id)
    terms = [element["S"] for element in manifest["terms"]["L"]]
    identifiers = [element["S"] for element in manifest["ids"]["L"]]
    postings = postings_of(ddb, doc_id)
    assert set(postings) == set(posting_partition_keys(scope, terms, identifiers))
    assert all(posting["rev"]["S"] == manifest["rev"]["S"] for posting in postings.values())
    assert base_item(ddb, doc_id)["lxrev"]["S"] == manifest["rev"]["S"]
    return manifest


def s3_object_count(s3) -> int:
    return s3.list_objects_v2(Bucket=BUCKET).get("KeyCount", 0)


@contextmanager
def assert_no_io(ddb):
    with contextlib.ExitStack() as stack:
        calls = [stack.enter_context(mock.patch.object(ddb, name, wraps=getattr(ddb, name))) for name in IO_METHODS]
        yield
    assert not any(call.called for call in calls)


def rival_index() -> LexicalIndex:
    """An index on its own client, so patched methods of the fixture client never intercept its calls."""
    return LexicalIndex(
        DynamoDBStorage(BASE_TABLE, client=boto3.client("dynamodb", region_name=REGION)),
        index_table_name=INDEX_TABLE,
    )


def sent(mocked: mock.MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs for call in mocked.call_args_list]


def action_target(action: dict[str, Any]) -> tuple[str, str, str, str]:
    """``(kind, table, pk, sk)`` of one TransactWriteItems action."""
    ((kind, body),) = action.items()
    key = body["Item"] if kind == "Put" else body["Key"]
    return kind, body["TableName"], key["pk"]["S"], key["sk"]["S"]


def describe(action: dict[str, Any]) -> str:
    kind, table, pk, _ = action_target(action)
    return f"{kind} {table} {pk}"


def projected(request: dict[str, Any]) -> set[str]:
    return set(request["RequestItems"][BASE_TABLE]["ExpressionAttributeNames"].values())


def fail_first(real, error: Exception):
    calls: list[dict[str, Any]] = []

    def side_effect(**request):
        calls.append(request)
        if len(calls) == 1:
            raise error
        return real(**request)

    return side_effect


def serve_one_key_per_round(read, *, stall_after_round: int | None = None):
    """A BatchGetItem that serves one key per round, then (optionally) stops making progress."""
    rounds: list[dict[str, Any]] = []

    def side_effect(**request):
        rounds.append(request)
        table_request = request["RequestItems"][BASE_TABLE]
        if stall_after_round is not None and len(rounds) > stall_after_round:
            return {"Responses": {BASE_TABLE: []}, "UnprocessedKeys": request["RequestItems"]}
        response = read(RequestItems={BASE_TABLE: {**table_request, "Keys": table_request["Keys"][:1]}})
        remaining = table_request["Keys"][1:]
        response["UnprocessedKeys"] = {BASE_TABLE: {**table_request, "Keys": remaining}} if remaining else {}
        return response

    return side_effect, rounds


def cause_chain(error: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    while error is not None:
        chain.append(error)
        error = error.__cause__
    return chain


async def remove_after_bypass(index: LexicalIndex, storage: DynamoDBStorage) -> RepairReport:
    await storage.write("docs/a", b"bypass")
    return await index.repair()


# ---------------------------------------------------------------- upsert / retrieval


async def test_upsert_search_and_lookup_round_trip(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    revision = await index.upsert(
        doc("docs/inv-1", "Invoice FC-00123 overdue", data=b"one", identifiers=["FC-00123"], metadata={"kind": "inv"})
    )
    await index.upsert(doc("docs/inv-2", "Invoice paid", data=b"two"))

    response = await index.search(query("overdue invoice", include_values=True))
    assert keys(response) == ["docs/inv-1", "docs/inv-2"]
    first, second = response.results
    assert (first.score, first.matched_terms, first.data, first.metadata) == (1.0, 2, b"one", {"kind": "inv"})
    assert (second.score, second.matched_terms, second.data) == (0.5, 1, b"two")
    assert response.truncated is False and response.truncation_reasons == [] and response.candidates_examined == 2

    lookup = await index.lookup(IdentifierQuery(identifier="FC-00123", top_k=5))
    assert keys(lookup) == ["docs/inv-1"]
    assert (lookup.results[0].score, lookup.results[0].matched_terms, lookup.results[0].data) == (1.0, 1, None)

    assert await index.revision("docs/inv-1") == revision and len(revision) == 32
    assert await index.revision("docs/missing") is None
    assert await storage.read("docs/inv-1") == b"one"
    manifest = assert_index_consistent(ddb, "", "docs/inv-1")
    assert manifest["tv"]["S"] == "lexical-v1"
    assert base_item(ddb, "docs/inv-1")["lxscope"]["S"] == "0:"


async def test_leading_zeroes_keep_identifiers_and_terms_distinct(aws):
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "Invoice FC-00123", identifiers=["FC-00123"]))
    await index.upsert(doc("docs/b", "Invoice FC-123", identifiers=["FC-123"]))
    assert keys(await index.lookup(IdentifierQuery("FC-00123", top_k=5))) == ["docs/a"]
    assert keys(await index.lookup(IdentifierQuery("FC-123", top_k=5))) == ["docs/b"]
    response = await index.search(query("FC-123"))
    assert keys(response) == ["docs/b", "docs/a"]
    assert [result.matched_terms for result in response.results] == [3, 1]


async def test_identifier_lookup_is_case_sensitive(aws):
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/upper", identifiers=["ERR_AUTH_403"]))
    await index.upsert(doc("docs/lower", identifiers=["err_auth_403"]))
    assert keys(await index.lookup(IdentifierQuery("ERR_AUTH_403", top_k=5))) == ["docs/upper"]
    assert keys(await index.lookup(IdentifierQuery("err_auth_403", top_k=5))) == ["docs/lower"]


async def test_lookup_orders_by_key_utf8_bytes(aws):
    index = make_index(make_storage(aws))
    for key in ("docs/\u00e9", "docs/z", "docs/b", "docs/a"):
        await index.upsert(doc(key, identifiers=["ID-1"]))
    assert keys(await index.lookup(IdentifierQuery("ID-1", top_k=10))) == ["docs/a", "docs/b", "docs/z", "docs/\u00e9"]
    assert keys(await index.lookup(IdentifierQuery("ID-1", top_k=2))) == ["docs/a", "docs/b"]


async def test_require_all_terms_drops_partial_matches(aws):
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/both", "alpha beta"))
    await index.upsert(doc("docs/one", "alpha"))
    assert keys(await index.search(query("alpha beta"))) == ["docs/both", "docs/one"]
    assert keys(await index.search(query("alpha beta", require_all_terms=True))) == ["docs/both"]


async def test_document_without_text_is_stored_but_not_searchable(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    await index.upsert(doc("docs/blob", data=b"payload"))
    assert await storage.read("docs/blob") == b"payload"
    assert postings_of(ddb, "docs/blob") == {}
    assert index_item(ddb, manifest_pk(""), "docs/blob")["terms"]["L"] == []


async def test_overwrite_with_new_terms_removes_old_postings(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha beta", identifiers=["OLD-1"]))
    await index.upsert(doc("docs/a", "gamma"))
    assert set(postings_of(ddb, "docs/a")) == {term_posting_pk("", "gamma")}
    assert_index_consistent(ddb, "", "docs/a")
    stale = await index.search(query("alpha"))
    assert keys(stale) == [] and stale.candidates_examined == 0
    assert keys(await index.lookup(IdentifierQuery("OLD-1", top_k=5))) == []
    assert keys(await index.search(query("gamma"))) == ["docs/a"]


async def test_overwrite_with_same_terms_advances_shared_postings(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    first = await index.upsert(doc("docs/a", "alpha beta", data=b"one"))
    second = await index.upsert(doc("docs/a", "alpha beta", data=b"two"))
    assert first != second
    postings = postings_of(ddb, "docs/a")
    assert len(postings) == 2 and {posting["rev"]["S"] for posting in postings.values()} == {second}
    assert_index_consistent(ddb, "", "docs/a")
    response = await index.search(query("alpha", include_values=True))
    assert [result.data for result in response.results] == [b"two"]


async def test_delete_then_recreate(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    first = await index.upsert(doc("docs/a", "alpha", identifiers=["ID-1"]))
    assert await index.delete("docs/a") is True
    assert base_item(ddb, "docs/a") is None
    assert all_items(ddb, INDEX_TABLE) == []
    assert await index.revision("docs/a") is None
    assert keys(await index.search(query("alpha"))) == []
    assert await index.delete("docs/a") is False

    second = await index.upsert(doc("docs/a", "alpha", data=b"again"))
    assert second != first
    response = await index.search(query("alpha", include_values=True))
    assert [(result.key, result.data) for result in response.results] == [("docs/a", b"again")]


async def test_delete_removes_a_never_indexed_base_item(aws):
    storage = make_storage(aws)
    index = make_index(storage)
    await storage.write("docs/plain", b"plain")
    assert await index.delete("docs/plain") is True
    assert await storage.read("docs/plain") is None


# ------------------------------------------------------------------- concurrency


async def test_concurrent_writer_forces_one_retry_and_converges(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    rival = rival_index()
    await index.upsert(doc("docs/a", "alpha shared"))
    commit = ddb.transact_write_items
    requests: list[dict[str, Any]] = []

    def rival_commits_first(**request):
        requests.append(request)
        if len(requests) == 1:
            asyncio.run(rival.upsert(doc("docs/a", "beta shared", data=b"rival")))
        return commit(**request)

    with mock.patch.object(ddb, "transact_write_items", side_effect=rival_commits_first):
        revision = await index.upsert(doc("docs/a", "gamma shared", data=b"mine"))

    assert len(requests) == 2
    assert requests[0]["ClientRequestToken"] != requests[1]["ClientRequestToken"] == revision
    manifest = assert_index_consistent(ddb, "", "docs/a")
    assert [element["S"] for element in manifest["terms"]["L"]] == ["gamma", "shared"]
    assert set(postings_of(ddb, "docs/a")) == {term_posting_pk("", "gamma"), term_posting_pk("", "shared")}
    assert keys(await index.search(query("beta"))) == []
    assert keys(await index.search(query("alpha"))) == []
    response = await index.search(query("gamma", include_values=True))
    assert [result.data for result in response.results] == [b"mine"]


async def test_expected_revision_conflicts(aws):
    index = make_index(make_storage(aws))
    first = await index.upsert(doc("docs/a", "alpha"))
    with pytest.raises(RevisionConflictError):
        await index.upsert(doc("docs/a", "beta"), expected_revision="0" * 32)
    assert await index.revision("docs/a") == first
    second = await index.upsert(doc("docs/a", "beta"), expected_revision=first)
    with pytest.raises(RevisionConflictError):
        await index.delete("docs/a", expected_revision=first)
    with pytest.raises(RevisionConflictError):
        await index.upsert(doc("docs/new", "alpha"), expected_revision=first)
    assert await index.delete("docs/a", expected_revision=second) is True
    assert issubclass(RevisionConflictError, StorageError)


async def test_conflict_retries_exhausted(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_conflict_retries=2)
    with mock.patch.object(
        ddb, "transact_write_items", side_effect=cancellation("None", "TransactionConflict")
    ) as transact:
        with pytest.raises(StorageError, match="concurrent modification; retries exhausted"):
            await index.upsert(doc("docs/a", "alpha"))
    assert transact.call_count == 3
    assert all_items(ddb, BASE_TABLE) == [] and all_items(ddb, INDEX_TABLE) == []


@pytest.mark.parametrize(
    "failure",
    [cancellation("None", "ValidationError"), ClientError({"Error": {"Code": "ValidationException"}}, "Transact")],
    ids=["validation-cancellation", "validation-exception"],
)
async def test_non_conflict_transaction_failure_is_storage_error(aws, failure):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with mock.patch.object(ddb, "transact_write_items", side_effect=failure) as transact:
        with pytest.raises(StorageError) as error:
            await index.upsert(doc("docs/a", "supersecret words", identifiers=["SECRET-ID-9"]))
    assert transact.call_count == 1
    message = str(error.value)
    assert "upsert" in message and INDEX_TABLE in message and BASE_TABLE in message
    assert "supersecret" not in message and "SECRET-ID-9" not in message
    assert not isinstance(error.value, RevisionConflictError)


# --------------------------------------------------------------------------- TTL


async def test_expired_documents_are_excluded_including_expiry_equal_to_now(aws):
    index = make_index(make_storage(aws, ttl_seconds=3600))
    with frozen_clock(NOW) as clock:
        await index.upsert(doc("docs/edge", "alpha", ttl_seconds=0))
        await index.upsert(doc("docs/live", "alpha", ttl_seconds=10))
        assert keys(await index.search(query("alpha"))) == ["docs/live"]
        clock.time.return_value = NOW + 10
        assert keys(await index.search(query("alpha"))) == []
        report = await index.repair()
    assert (report.documents_checked, report.documents_removed) == (2, 2)


async def test_postings_carry_the_document_ttl_and_manifest_does_not(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws, ttl_seconds=3600))
    with frozen_clock(NOW):
        await index.upsert(doc("docs/a", "alpha", identifiers=["ID-1"]))
    expected = {"N": str(int(NOW + 3600))}
    assert base_item(ddb, "docs/a")["expireAt"] == expected
    postings = postings_of(ddb, "docs/a")
    assert len(postings) == 2 and all(posting["expireAt"] == expected for posting in postings.values())
    assert "expireAt" not in index_item(ddb, manifest_pk(""), "docs/a")


async def test_ttl_disabled_storage_writes_no_ttl_anywhere(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha", ttl_seconds=60))
    assert "expireAt" not in base_item(ddb, "docs/a")
    assert all("expireAt" not in item for item in all_items(ddb, INDEX_TABLE))


# ------------------------------------------------------------ bypass and validity


async def test_direct_storage_write_hides_document_and_repair_removes_entries(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    await index.upsert(doc("docs/a", "alpha", identifiers=["ID-1"]))
    await storage.write("docs/a", b"bypass")

    response = await index.search(query("alpha"))
    assert keys(response) == [] and response.candidates_examined == 1
    assert keys(await index.lookup(IdentifierQuery("ID-1", top_k=5))) == []

    report = await index.repair()
    assert report == RepairReport(documents_checked=1, documents_removed=1, postings_rebuilt=0, cursor=None)
    assert all_items(ddb, INDEX_TABLE) == []
    assert await storage.read("docs/a") == b"bypass"


async def test_stale_posting_revision_is_excluded(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha beta"))
    ddb.put_item(
        TableName=INDEX_TABLE,
        Item={"pk": {"S": term_posting_pk("", "alpha")}, "sk": {"S": "docs/a"}, "rev": {"S": "stale"}},
    )
    assert keys(await index.search(query("alpha"))) == []
    response = await index.search(query("alpha beta"))
    assert [(result.key, result.matched_terms, result.score) for result in response.results] == [("docs/a", 1, 0.5)]


async def test_filter_uses_current_metadata(aws):
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha", metadata={"lang": "en"}))
    await index.upsert(doc("docs/a", "alpha", metadata={"lang": "es"}))
    assert keys(await index.search(query("alpha", filter={"lang": "en"}))) == []
    response = await index.search(query("alpha", filter={"lang": "es"}))
    assert [(result.key, result.metadata) for result in response.results] == [("docs/a", {"lang": "es"})]


@pytest.mark.parametrize(
    "filter_, matches",
    [
        ({"flag": True}, True),
        ({"flag": 1}, False),
        ({"count": True}, False),
        ({"count": 1}, True),
        ({"count": 1.0}, True),
        ({"count": "1"}, False),
        ({"ratio": 1.5}, True),
        ({"name": ""}, True),
        ({"name": "x"}, False),
        ({"missing": 1}, False),
        ({}, True),
    ],
)
async def test_filter_is_strict_about_value_kinds(aws, filter_, matches):
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha", metadata={"flag": True, "count": 1, "ratio": 1.5, "name": ""}))
    await index.upsert(doc("docs/bare", "alpha"))
    expected = ["docs/a"] if matches else []
    if filter_ == {}:
        expected.append("docs/bare")
    assert keys(await index.search(query("alpha", filter=filter_))) == expected
    assert keys(await index.search(query("alpha", filter=None))) == ["docs/a", "docs/bare"]


# ------------------------------------------------------------------ scope isolation


async def test_namespaced_scopes_are_isolated(aws):
    root = make_storage(aws)
    tenant_a = make_index(root.namespace("tenant-a"))
    tenant_b = make_index(root.namespace("tenant-b"))
    await tenant_a.upsert(doc("docs/x", "shared secret", data=b"A", identifiers=["ID-1"]))
    await tenant_b.upsert(doc("docs/x", "shared secret", data=b"B", identifiers=["ID-1"]))

    for index, expected in ((tenant_a, b"A"), (tenant_b, b"B")):
        response = await index.search(query("shared", include_values=True))
        assert [(result.key, result.data) for result in response.results] == [("docs/x", expected)]
        lookup = await index.lookup(IdentifierQuery("ID-1", top_k=5, include_values=True))
        assert [result.data for result in lookup.results] == [expected]

    unscoped = await make_index(root).search(query("shared"))
    assert keys(unscoped) == [] and unscoped.candidates_examined == 0


async def test_cross_scope_overwrite_and_delete_are_rejected(aws):
    root = make_storage(aws)
    tenant_a = make_index(root.namespace("tenant-a"))
    root_index = make_index(root)
    await tenant_a.upsert(doc("docs/x", "alpha", data=b"A"))
    with pytest.raises(StorageError, match="owned by another lexical scope"):
        await root_index.upsert(doc("tenant-a/docs/x", "hijack", data=b"root"))
    with pytest.raises(StorageError, match="owned by another lexical scope"):
        await root_index.delete("tenant-a/docs/x")
    response = await tenant_a.search(query("alpha", include_values=True))
    assert [result.data for result in response.results] == [b"A"]


async def test_out_of_scope_postings_are_dropped_before_base_reads(aws):
    ddb, _ = aws
    root = make_storage(aws)
    tenant_a = make_index(root.namespace("tenant-a"))
    tenant_b = make_index(root.namespace("tenant-b"))
    await tenant_a.upsert(doc("docs/own", "alpha"))
    await tenant_b.upsert(doc("docs/x", "alpha"))
    ddb.put_item(
        TableName=INDEX_TABLE,
        Item={"pk": {"S": term_posting_pk("tenant-a/", "alpha")}, "sk": {"S": "tenant-b/docs/x"}, "rev": {"S": "r"}},
    )
    with mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get:
        response = await tenant_a.search(query("alpha"))
    assert keys(response) == ["docs/own"] and response.candidates_examined == 1
    requested = [key["pk"]["S"] for key in batch_get.call_args.kwargs["RequestItems"][BASE_TABLE]["Keys"]]
    assert requested == ["tenant-a/docs"]


# ------------------------------------------------------------------------- limits


@pytest.mark.parametrize(
    "document, message",
    [
        (doc("docs/a", " ".join(f"w{n}" for n in range(48)), identifiers=["I-1", "I-2"]), "50 postings"),
        (doc("docs/a", "x" * 65), "max_term_bytes"),
        (doc("docs/a", identifiers=["I" * 129]), "max_identifier_bytes"),
        (doc("docs/a", "ab " * 22_000), "max_text_bytes"),
        (doc("docs/a", data=os.urandom(400_001)), "S3 offload is not supported"),
        (doc("docs/" + "k" * 1100), "sort key limit"),
        (doc("docs/a", identifiers=[" padded"]), "whitespace"),
        (doc("docs/a", identifiers="FC-1"), "not a string"),
        (doc("docs/a", vector=[float("nan")]), "non-finite"),
        (doc("docs/../a"), "'..'"),
    ],
    ids=[
        "postings",
        "term-bytes",
        "identifier-bytes",
        "text-bytes",
        "s3-size",
        "key-bytes",
        "identifier-whitespace",
        "identifiers-as-string",
        "vector",
        "key-traversal",
    ],
)
async def test_upsert_validation_fails_before_any_write(aws, document, message):
    ddb, s3 = aws
    index = make_index(make_storage(aws, s3_bucket=BUCKET))
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        with pytest.raises(StorageError, match=message) as error:
            await index.upsert(document)
    transact.assert_not_called()
    assert all_items(ddb, BASE_TABLE) == [] and all_items(ddb, INDEX_TABLE) == []
    assert s3_object_count(s3) == 0
    assert "w47" not in str(error.value) and "padded" not in str(error.value)


@pytest.mark.parametrize(
    "lexical_query, message",
    [
        (query("alpha", top_k=0), "top_k"),
        (query("alpha", top_k=101), "top_k"),
        (query("alpha", top_k=1.5), "top_k"),
        (query("-- !! __"), "no searchable terms"),
        (query("x" * 65), "max_term_bytes"),
        (query(" ".join(f"t{n}" for n in range(17))), "17 distinct terms"),
    ],
    ids=["top-k-zero", "top-k-over", "top-k-fractional", "no-terms", "term-bytes", "query-terms"],
)
async def test_search_validation_fails_before_any_read(aws, lexical_query, message):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with mock.patch.object(ddb, "query", wraps=ddb.query) as query_call:
        with pytest.raises(StorageError, match=message):
            await index.search(lexical_query)
    query_call.assert_not_called()


@pytest.mark.parametrize(
    "identifier_query, message",
    [
        (IdentifierQuery("ID-1", top_k=0), "top_k"),
        (IdentifierQuery("ID-1", top_k=101), "top_k"),
        (IdentifierQuery("ID-1", top_k=1.5), "top_k"),
        (IdentifierQuery("ID\t1", top_k=1), "control characters"),
        (IdentifierQuery("A-1 ", top_k=1), "whitespace"),
        (IdentifierQuery("I" * 129, top_k=1), "max_identifier_bytes"),
    ],
    ids=["top-k", "top-k-over", "top-k-fractional", "control-char", "trailing-whitespace", "identifier-bytes"],
)
async def test_lookup_validation_fails_before_any_read(aws, identifier_query, message):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with mock.patch.object(ddb, "query", wraps=ddb.query) as query_call:
        with pytest.raises(StorageError, match=message):
            await index.lookup(identifier_query)
    query_call.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_postings_per_document": 50},
        {"page_size": 1001},
        {"max_candidates": 1001},
        {"max_concurrency": 17},
        {"max_postings_per_document": 10, "max_query_terms": 11},
        {"max_term_bytes": 0},
        {"max_conflict_retries": -1},
        {"page_size": True},
        {"max_text_bytes": 1.5},
    ],
)
def test_invalid_limits_are_rejected(overrides):
    with pytest.raises(StorageError, match="LexicalIndexLimits"):
        LexicalIndexLimits(**overrides)


def test_retry_limits_may_be_zero():
    limits = LexicalIndexLimits(max_conflict_retries=0, max_unprocessed_retries=0)
    assert limits.max_conflict_retries == 0 and limits.max_unprocessed_retries == 0


def test_scope_longer_than_sort_key_limit_is_rejected(aws):
    with pytest.raises(StorageError, match="scope"):
        make_index(make_storage(aws, prefix="t" * 1100))


def test_scope_property_exposes_storage_prefix(aws):
    assert make_index(make_storage(aws)).scope == ""
    assert make_index(make_storage(aws).namespace("tenant/a")).scope == "tenant/a/"


async def test_repair_rejects_non_positive_max_documents(aws):
    with pytest.raises(StorageError, match="max_documents"):
        await make_index(make_storage(aws)).repair(max_documents=0)


# --------------------------------------------------------------------- truncation


async def test_max_candidates_truncates_and_admitted_candidates_keep_matching(aws):
    index = make_index(make_storage(aws), max_candidates=2)
    await index.upsert(doc("docs/a", "alpha"))
    await index.upsert(doc("docs/b", "alpha beta"))
    await index.upsert(doc("docs/c", "alpha beta"))
    response = await index.search(query("alpha beta"))
    assert [(result.key, result.matched_terms) for result in response.results] == [("docs/b", 2), ("docs/a", 1)]
    assert response.truncated is True
    assert response.truncation_reasons == ["max_candidates"]
    assert response.candidates_examined == 2


async def test_max_pages_per_term_truncates(aws):
    index = make_index(make_storage(aws), page_size=1, max_pages_per_term=2)
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    response = await index.search(query("alpha"))
    assert keys(response) == ["docs/a", "docs/b"]
    assert response.truncation_reasons == ["max_pages_per_term"] and response.truncated is True


async def test_truncation_reasons_are_sorted_and_unique(aws):
    index = make_index(make_storage(aws), page_size=1, max_pages_per_term=2, max_candidates=1)
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha beta"))
    response = await index.search(query("alpha beta"))
    assert response.truncation_reasons == ["max_candidates", "max_pages_per_term"]
    assert keys(response) == ["docs/a"]


# ---------------------------------------------------------------- unprocessed keys


async def test_unprocessed_keys_are_retried(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha"))
    read = ddb.batch_get_item
    calls: list[dict[str, Any]] = []

    def unprocessed_once(**request):
        calls.append(request)
        if len(calls) == 1:
            return {"Responses": {}, "UnprocessedKeys": request["RequestItems"]}
        return read(**request)

    with mock.patch.object(ddb, "batch_get_item", side_effect=unprocessed_once):
        response = await index.search(query("alpha"))
    assert keys(response) == ["docs/a"] and len(calls) == 2
    assert calls[0]["RequestItems"][BASE_TABLE]["ConsistentRead"] is True


async def test_unprocessed_keys_exhausted_raises(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_unprocessed_retries=1)
    await index.upsert(doc("docs/a", "alpha"))

    def always_unprocessed(**request):
        return {"Responses": {}, "UnprocessedKeys": request["RequestItems"]}

    with mock.patch.object(ddb, "batch_get_item", side_effect=always_unprocessed) as batch_get:
        with pytest.raises(StorageError, match="unprocessed"):
            await index.search(query("alpha"))
    assert batch_get.call_count == 2


# ------------------------------------------------------------------------- repair


async def test_repair_resumes_from_cursor(aws):
    storage = make_storage(aws)
    index = make_index(storage)
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    await storage.write("docs/a", b"bypass")
    await storage.write("docs/c", b"bypass")

    first = await index.repair(max_documents=2)
    assert first == RepairReport(documents_checked=2, documents_removed=1, postings_rebuilt=0, cursor="docs/b")
    second = await index.repair(max_documents=2, cursor=first.cursor)
    assert second == RepairReport(documents_checked=1, documents_removed=1, postings_rebuilt=0, cursor=None)
    assert keys(await index.search(query("alpha"))) == ["docs/b"]


async def test_repair_rebuilds_missing_postings(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws, ttl_seconds=3600))
    with frozen_clock(NOW):
        revision = await index.upsert(doc("docs/a", "alpha beta", identifiers=["ID-1"]))
        ddb.delete_item(TableName=INDEX_TABLE, Key={"pk": {"S": term_posting_pk("", "beta")}, "sk": {"S": "docs/a"}})
        assert keys(await index.search(query("beta"))) == []

        report = await index.repair(rebuild_postings=True)
        assert report == RepairReport(documents_checked=1, documents_removed=0, postings_rebuilt=1, cursor=None)
        assert keys(await index.search(query("beta"))) == ["docs/a"]
    rebuilt = index_item(ddb, term_posting_pk("", "beta"), "docs/a")
    assert rebuilt["rev"]["S"] == revision and rebuilt["expireAt"] == {"N": str(int(NOW + 3600))}
    assert_index_consistent(ddb, "", "docs/a")


async def test_repair_skips_documents_changed_concurrently(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    await index.upsert(doc("docs/a", "alpha"))
    await storage.write("docs/a", b"bypass")
    with mock.patch.object(ddb, "transact_write_items", side_effect=cancellation("ConditionalCheckFailed")):
        report = await index.repair()
    assert report == RepairReport(documents_checked=1, documents_removed=0, postings_rebuilt=0, cursor=None)
    assert index_item(ddb, manifest_pk(""), "docs/a") is not None


async def test_repair_keeps_valid_documents_untouched(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws).namespace("tenant-a"))
    await index.upsert(doc("docs/a", "alpha"))
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        report = await index.repair()
    assert report == RepairReport(documents_checked=1, documents_removed=0, postings_rebuilt=0, cursor=None)
    transact.assert_not_called()


# ---------------------------------------------------------------- stored values


async def test_empty_value_is_preserved_with_include_values(aws):
    storage = make_storage(aws)
    index = make_index(storage)
    await index.upsert(doc("docs/empty", "alpha", data=b""))
    response = await index.search(query("alpha", include_values=True))
    assert [result.data for result in response.results] == [b""]
    assert await storage.read("docs/empty") == b""


async def test_gzip_compressed_value_round_trips(aws):
    ddb, _ = aws
    storage = make_storage(aws, compression="gzip")
    index = make_index(storage)
    original = b"A" * 5000
    await index.upsert(doc("docs/big", "alpha", data=original))
    item = base_item(ddb, "docs/big")
    assert item["z"]["BOOL"] is True and gzip.decompress(item["data"]["B"]) == original
    response = await index.search(query("alpha", include_values=True))
    assert [result.data for result in response.results] == [original]
    assert await storage.read("docs/big") == original


async def test_upsert_and_delete_reclaim_previously_offloaded_objects(aws):
    _, s3 = aws
    storage = make_storage(aws, s3_bucket=BUCKET)
    index = make_index(storage)
    await storage.write("docs/a", b"Z" * 400_001)
    await storage.write("docs/b", b"Z" * 400_001)
    assert s3_object_count(s3) == 2

    await index.upsert(doc("docs/a", "alpha", data=b"small"))
    assert await index.delete("docs/b") is True
    assert s3_object_count(s3) == 0
    assert await storage.read("docs/a") == b"small"


def test_public_exports():
    import strands_dynamodb_storage as package

    for name in (
        "LexicalIndex",
        "LexicalIndexLimits",
        "SearchableDocument",
        "LexicalQuery",
        "IdentifierQuery",
        "LexicalSearchResult",
        "LexicalSearchResponse",
        "RepairReport",
        "RevisionConflictError",
        "LEXICAL_TOKENIZER_VERSION",
    ):
        assert name in package.__all__ and hasattr(package, name)
    assert not hasattr(package, "_DocumentItemPort")


# -------------------------------------------------------------- transaction shapes


async def test_upsert_writes_one_ordered_transaction_keyed_by_the_revision(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws).namespace("tenant/a"))
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        revision = await index.upsert(doc("tickets/t1", "Alpha beta", identifiers=["T-1"]))
    (request,) = sent(transact)
    doc_id = "tenant/a/tickets/t1"
    assert re.fullmatch("[0-9a-f]{32}", revision) and request["ClientRequestToken"] == revision
    assert [describe(action) for action in request["TransactItems"]] == [
        f"Put {BASE_TABLE} tenant/a",
        f"Put {INDEX_TABLE} m|9:tenant/a/",
        f"Put {INDEX_TABLE} t|9:tenant/a/5:alpha",
        f"Put {INDEX_TABLE} t|9:tenant/a/4:beta",
        f"Put {INDEX_TABLE} i|9:tenant/a/3:T-1",
    ]
    assert index_item(ddb, "m|9:tenant/a/", doc_id) == {
        "pk": {"S": "m|9:tenant/a/"},
        "sk": {"S": doc_id},
        "rev": {"S": revision},
        "tv": {"S": "lexical-v1"},
        "terms": {"L": [{"S": "alpha"}, {"S": "beta"}]},
        "ids": {"L": [{"S": "T-1"}]},
    }
    assert index_item(ddb, "i|9:tenant/a/3:T-1", doc_id) == {
        "pk": {"S": "i|9:tenant/a/3:T-1"},
        "sk": {"S": doc_id},
        "rev": {"S": revision},
    }
    base = base_item(ddb, doc_id)
    assert (base["k"], base["lxrev"], base["lxscope"]) == ({"S": doc_id}, {"S": revision}, {"S": "9:tenant/a/"})
    assert_index_consistent(ddb, "tenant/a/", doc_id)


async def test_overwrite_deletes_exactly_the_stale_postings(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha beta", identifiers=["ID-1"]))
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        await index.upsert(doc("docs/a", "gamma beta", identifiers=["ID-2"]))
    (request,) = sent(transact)
    assert [describe(action) for action in request["TransactItems"] if "Delete" in action] == [
        f"Delete {INDEX_TABLE} {term_posting_pk('', 'alpha')}",
        f"Delete {INDEX_TABLE} {identifier_posting_pk('', 'ID-1')}",
    ]
    assert_index_consistent(ddb, "", "docs/a")


async def test_delete_and_repair_send_no_client_request_token(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    await storage.write("docs/b", b"bypass")
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        assert await index.delete("docs/a") is True
        report = await index.repair(rebuild_postings=True)
    assert (report.documents_removed, report.postings_rebuilt) == (1, 1)
    requests = sent(transact)
    assert len(requests) == 3 and all("ClientRequestToken" not in request for request in requests)


async def test_reads_are_consistent_where_required_and_postings_page_by_page_size(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with mock.patch.object(ddb, "get_item", wraps=ddb.get_item) as get_item:
        await index.upsert(doc("docs/a", "alpha"))
        await index.revision("docs/a")
    assert [request["ConsistentRead"] for request in sent(get_item)] == [True, True, True]
    with (
        mock.patch.object(ddb, "query", wraps=ddb.query) as query_call,
        mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get,
    ):
        await index.search(query("alpha"))
        await index.repair()
    posting_query, manifest_query = sent(query_call)
    assert (posting_query["ConsistentRead"], posting_query["Limit"]) == (False, LexicalIndexLimits().page_size)
    assert manifest_query["ConsistentRead"] is True
    assert batch_get.call_args.kwargs["RequestItems"][BASE_TABLE]["ConsistentRead"] is True


# ------------------------------------------------------------------ delete races


async def test_delete_is_replanned_when_a_first_upsert_races_it(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    await storage.write("docs/a", b"plain")
    commit = ddb.transact_write_items
    requests: list[dict[str, Any]] = []

    def first_upsert_commits_first(**request):
        requests.append(request)
        if len(requests) == 1:
            asyncio.run(rival_index().upsert(doc("docs/a", "alpha", identifiers=["ID-1"])))
        return commit(**request)

    with mock.patch.object(ddb, "transact_write_items", side_effect=first_upsert_commits_first):
        assert await index.delete("docs/a") is True

    assert [describe(action) for action in requests[0]["TransactItems"]] == [
        f"Delete {BASE_TABLE} docs/a",
        f"ConditionCheck {INDEX_TABLE} {manifest_pk('')}",
    ]
    assert [describe(action) for action in requests[1]["TransactItems"]] == [
        f"Delete {BASE_TABLE} docs/a",
        f"Delete {INDEX_TABLE} {manifest_pk('')}",
        f"Delete {INDEX_TABLE} {term_posting_pk('', 'alpha')}",
        f"Delete {INDEX_TABLE} {identifier_posting_pk('', 'ID-1')}",
    ]
    assert all_items(ddb, BASE_TABLE) == [] and all_items(ddb, INDEX_TABLE) == []


# --------------------------------------------------------- retries and cancellation


@pytest.mark.parametrize("code", ["ThrottlingError", "ProvisionedThroughputExceeded", "RequestLimitExceeded"])
async def test_throttled_transactions_are_retried_by_upsert_and_delete(aws, code):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    commit = ddb.transact_write_items
    with mock.patch.object(ddb, "transact_write_items", side_effect=fail_first(commit, cancellation("None", code))):
        await index.upsert(doc("docs/a", "alpha"))
    assert keys(await index.search(query("alpha"))) == ["docs/a"]
    with mock.patch.object(ddb, "transact_write_items", side_effect=fail_first(commit, cancellation(code))) as transact:
        assert await index.delete("docs/a") is True
    assert transact.call_count == 2
    assert all_items(ddb, BASE_TABLE) == [] and all_items(ddb, INDEX_TABLE) == []


@pytest.mark.parametrize(
    "last_code, reason",
    [("TransactionConflict", "concurrent modification"), ("ThrottlingError", "transaction throttled")],
)
async def test_exhausted_retries_chain_the_last_cancellation(aws, last_code, reason):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_conflict_retries=1)
    first, last = cancellation("ConditionalCheckFailed"), cancellation(last_code)
    with mock.patch.object(ddb, "transact_write_items", side_effect=[first, last]):
        with pytest.raises(StorageError, match=f"{reason}; retries exhausted") as error:
            await index.upsert(doc("docs/a", "alpha"))
    assert error.value.__cause__ is last


@pytest.mark.parametrize(
    "failure",
    [
        cancellation("ThrottlingError"),
        cancellation("None", "ProvisionedThroughputExceeded"),
        cancellation("ValidationError"),
        ClientError({"Error": {"Code": "ValidationException"}}, "TransactWriteItems"),
    ],
    ids=["throttling", "provisioned-throughput", "validation-cancellation", "validation-exception"],
)
async def test_repair_raises_on_the_first_non_conflict_failure(aws, failure):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    for name in ("a", "b"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
        await storage.write(f"docs/{name}", b"bypass")
    expected = f"Lexical index repair failed (index table '{INDEX_TABLE}', base table '{BASE_TABLE}')"
    with mock.patch.object(ddb, "transact_write_items", side_effect=failure) as transact:
        with pytest.raises(StorageError, match=re.escape(expected)) as error:
            await index.repair()
    assert transact.call_count == 1
    assert failure in cause_chain(error.value)
    assert [item["sk"]["S"] for item in all_items(ddb, INDEX_TABLE) if item["pk"]["S"] == manifest_pk("")] == [
        "docs/a",
        "docs/b",
    ]


async def test_repair_skips_a_document_changed_during_rebuild(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha"))
    commit = ddb.transact_write_items

    def rival_upserts_first(**request):
        asyncio.run(rival_index().upsert(doc("docs/a", "beta")))
        return commit(**request)

    with mock.patch.object(ddb, "transact_write_items", side_effect=rival_upserts_first) as transact:
        report = await index.repair(rebuild_postings=True)
    assert report == RepairReport(documents_checked=1, documents_removed=0, postings_rebuilt=0, cursor=None)
    assert transact.call_count == 1
    assert keys(await index.search(query("beta"))) == ["docs/a"]
    assert_index_consistent(ddb, "", "docs/a")


async def test_repair_treats_an_empty_cursor_as_the_beginning(aws):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    for name in ("a", "b"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    await storage.write("docs/a", b"bypass")
    with mock.patch.object(ddb, "query", wraps=ddb.query) as query_call:
        report = await index.repair(cursor="")
    assert report == RepairReport(documents_checked=2, documents_removed=1, postings_rebuilt=0, cursor=None)
    assert all("ExclusiveStartKey" not in request for request in sent(query_call))


# ------------------------------------------------------------ unprocessed keys


async def test_unprocessed_rounds_with_progress_never_exhaust_retries(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_unprocessed_retries=0)
    for name in ("a", "b", "c", "d"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    side_effect, rounds = serve_one_key_per_round(ddb.batch_get_item)
    with mock.patch.object(ddb, "batch_get_item", side_effect=side_effect):
        response = await index.search(query("alpha"))
    assert keys(response) == ["docs/a", "docs/b", "docs/c", "docs/d"] and len(rounds) == 4


async def test_stalled_unprocessed_rounds_exhaust_retries_after_progress(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_unprocessed_retries=1)
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha"))
    side_effect, rounds = serve_one_key_per_round(ddb.batch_get_item, stall_after_round=1)
    with mock.patch.object(ddb, "batch_get_item", side_effect=side_effect):
        with pytest.raises(
            StorageError,
            match=re.escape(f"left 2 keys unprocessed in base table '{BASE_TABLE}' after 1 retries without progress"),
        ):
            await index.search(query("alpha"))
    assert len(rounds) == 3


# ----------------------------------------------------------------- stored values


async def test_include_values_reads_values_for_the_kept_results_only(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws, compression="gzip"))
    for name in ("a", "b", "c"):
        await index.upsert(doc(f"docs/{name}", "alpha", data=name.encode() * 100))
    with mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get:
        response = await index.search(query("alpha", top_k=1, include_values=True))
    assert [(result.key, result.data) for result in response.results] == [("docs/a", b"a" * 100)]
    validation, values = sent(batch_get)
    assert projected(validation) == {"pk", "sk", "k", "lxrev", "lxscope", "meta", "s3"}
    assert len(validation["RequestItems"][BASE_TABLE]["Keys"]) == 3
    assert projected(values) == {"pk", "sk", "lxrev", "data", "z", "s3"}
    assert values["RequestItems"][BASE_TABLE]["Keys"] == [{"pk": {"S": "docs/a"}, "sk": {"S": SENTINEL}}]
    assert values["RequestItems"][BASE_TABLE]["ConsistentRead"] is True


async def test_validation_projects_the_ttl_attribute_but_never_data(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws, ttl_seconds=3600))
    await index.upsert(doc("docs/a", "alpha"))
    with mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get:
        await index.search(query("alpha"))
    (validation,) = sent(batch_get)
    assert projected(validation) == {"pk", "sk", "k", "lxrev", "lxscope", "meta", "s3", "expireAt"}


async def test_values_pass_drops_a_result_whose_revision_changed(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha", data=b"one"))
    await index.upsert(doc("docs/b", "alpha", data=b"two"))
    read = ddb.batch_get_item
    calls: list[dict[str, Any]] = []

    def rewrite_between_reads(**request):
        calls.append(request)
        if len(calls) == 2:
            asyncio.run(rival_index().upsert(doc("docs/a", "alpha", data=b"changed")))
        return read(**request)

    with mock.patch.object(ddb, "batch_get_item", side_effect=rewrite_between_reads):
        response = await index.search(query("alpha", include_values=True))
    assert [(result.key, result.data) for result in response.results] == [("docs/b", b"two")]
    assert len(calls) == 2 and response.candidates_examined == 2


async def test_values_pass_drops_a_result_without_data(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    await index.upsert(doc("docs/a", "alpha"))
    await index.upsert(doc("docs/b", "alpha", data=b"two"))
    ddb.update_item(
        TableName=BASE_TABLE,
        Key={"pk": {"S": "docs/a"}, "sk": {"S": SENTINEL}},
        UpdateExpression="REMOVE #data",
        ExpressionAttributeNames={"#data": "data"},
    )
    assert keys(await index.search(query("alpha"))) == ["docs/a", "docs/b"]
    response = await index.search(query("alpha", include_values=True))
    assert [(result.key, result.data) for result in response.results] == [("docs/b", b"two")]


async def test_lookup_returns_a_gzip_compressed_value(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws, compression="gzip"))
    original = b"B" * 5000
    await index.upsert(doc("docs/big", identifiers=["ID-9"], data=original))
    assert base_item(ddb, "docs/big")["z"]["BOOL"] is True
    response = await index.lookup(IdentifierQuery("ID-9", top_k=1, include_values=True))
    assert [(result.key, result.data) for result in response.results] == [("docs/big", original)]


async def test_failed_s3_cleanup_does_not_fail_the_upsert(aws):
    _, s3 = aws
    storage = make_storage(aws, s3_bucket=BUCKET)
    index = make_index(storage)
    await storage.write("docs/a", b"Z" * 400_001)
    failure = ClientError({"Error": {"Code": "InternalError"}}, "DeleteObject")
    with mock.patch.object(s3, "delete_object", side_effect=failure) as delete_object:
        revision = await index.upsert(doc("docs/a", "alpha", data=b"small"))
    delete_object.assert_called_once()
    assert await index.revision("docs/a") == revision
    response = await index.search(query("alpha", include_values=True))
    assert [result.data for result in response.results] == [b"small"]
    assert s3_object_count(s3) == 1


# ------------------------------------------------------------ batch validation reads


async def test_batch_get_reads_a_shared_base_key_once(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    revision = await index.upsert(doc("x/y", "alpha"))
    ddb.put_item(
        TableName=INDEX_TABLE,
        Item={"pk": {"S": term_posting_pk("", "alpha")}, "sk": {"S": f"x/y/{SENTINEL}"}, "rev": {"S": revision}},
    )
    with mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get:
        response = await index.search(query("alpha"))
    assert keys(response) == ["x/y"] and response.candidates_examined == 2
    (request,) = sent(batch_get)
    assert request["RequestItems"][BASE_TABLE]["Keys"] == [{"pk": {"S": "x/y"}, "sk": {"S": SENTINEL}}]


async def test_validation_reads_in_chunks_of_at_most_100_keys(aws):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    for number in range(101):
        await index.upsert(doc(f"docs/{number}", "alpha"))
    with mock.patch.object(ddb, "batch_get_item", wraps=ddb.batch_get_item) as batch_get:
        response = await index.search(query("alpha", top_k=100))
    assert len(response.results) == 100 and response.candidates_examined == 101
    assert sorted(len(request["RequestItems"][BASE_TABLE]["Keys"]) for request in sent(batch_get)) == [1, 100]


# ------------------------------------------------------- key limits and validation


@pytest.mark.parametrize(
    "operation",
    [
        lambda index: index.upsert(doc("docs/a", "q" * 2100)),
        lambda index: index.upsert(doc("docs/a", identifiers=["Q" * 2100])),
        lambda index: index.search(query("q" * 2100)),
        lambda index: index.lookup(IdentifierQuery("Q" * 2100, top_k=1)),
    ],
    ids=["upsert-term", "upsert-identifier", "search", "lookup"],
)
async def test_partition_keys_above_2048_bytes_fail_before_any_io(aws, operation):
    ddb, _ = aws
    index = make_index(make_storage(aws), max_term_bytes=4096, max_identifier_bytes=4096)
    with assert_no_io(ddb):
        with pytest.raises(
            StorageError, match=re.escape("above the DynamoDB partition-key limit of 2048 bytes")
        ) as error:
            await operation(index)
    assert "qqqq" not in str(error.value).lower()


@pytest.mark.parametrize(
    "operation",
    [lambda index, key: index.delete(key), lambda index, key: index.revision(key)],
    ids=["delete", "revision"],
)
async def test_oversized_key_is_rejected_before_any_io(aws, operation):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with assert_no_io(ddb):
        with pytest.raises(StorageError, match="sort key limit"):
            await operation(index, "docs/" + "k" * 1100)


@pytest.mark.parametrize(
    "operation",
    [
        lambda index: index.upsert(doc("docs/a", f"secret {LONE_SURROGATE} words")),
        lambda index: index.upsert(doc("docs/a", "ok", identifiers=[f"SECRET-{LONE_SURROGATE}"])),
        lambda index: index.search(query(f"secret {LONE_SURROGATE}")),
        lambda index: index.lookup(IdentifierQuery(f"SECRET-{LONE_SURROGATE}", top_k=1)),
    ],
    ids=["upsert-text", "upsert-identifier", "search", "lookup"],
)
async def test_lone_surrogates_are_rejected_before_any_io(aws, operation):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    with assert_no_io(ddb):
        with pytest.raises(StorageError, match="must be well-formed Unicode; it contains a lone surrogate") as error:
            await operation(index)
    assert "secret" not in str(error.value).lower()


@pytest.mark.parametrize("name", INDEX_ATTRIBUTE_NAMES)
def test_ttl_attribute_colliding_with_index_attributes_is_rejected(aws, name):
    with pytest.raises(StorageError, match="ttl_attribute"):
        make_index(make_storage(aws, ttl_seconds=60, ttl_attribute=name))
    assert make_index(make_storage(aws, ttl_attribute=name)).scope == ""


def test_empty_index_table_name_is_rejected(aws):
    with pytest.raises(StorageError, match="index_table_name"):
        LexicalIndex(make_storage(aws), index_table_name="")


# --------------------------------------------------------------- corrupt manifests


@pytest.mark.parametrize("stored_revision", [None, {"N": "1"}], ids=["missing", "number"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda index: index.revision("docs/a"),
        lambda index: index.upsert(doc("docs/a", "alpha")),
        lambda index: index.delete("docs/a"),
        lambda index: index.repair(),
    ],
    ids=["revision", "upsert", "delete", "repair"],
)
async def test_manifest_without_a_string_revision_is_an_error(aws, operation, stored_revision):
    ddb, _ = aws
    index = make_index(make_storage(aws))
    manifest = {"pk": {"S": manifest_pk("")}, "sk": {"S": "docs/a"}, "terms": {"L": []}, "ids": {"L": []}}
    if stored_revision is not None:
        manifest["rev"] = stored_revision
    ddb.put_item(TableName=INDEX_TABLE, Item=manifest)
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        with pytest.raises(StorageError, match="has no string revision"):
            await operation(index)
    transact.assert_not_called()
    assert index_item(ddb, manifest_pk(""), "docs/a") == manifest


@pytest.mark.parametrize(
    "operation",
    [
        lambda index, storage: index.upsert(doc("docs/a", "gamma")),
        lambda index, storage: index.delete("docs/a"),
        lambda index, storage: index.repair(rebuild_postings=True),
        remove_after_bypass,
    ],
    ids=["upsert", "delete", "repair-rebuild", "repair-remove"],
)
async def test_duplicate_manifest_terms_touch_each_posting_once(aws, operation):
    ddb, _ = aws
    storage = make_storage(aws)
    index = make_index(storage)
    revision = await index.upsert(doc("docs/a", "alpha", identifiers=["ID-1"]))
    ddb.put_item(
        TableName=INDEX_TABLE,
        Item={
            "pk": {"S": manifest_pk("")},
            "sk": {"S": "docs/a"},
            "rev": {"S": revision},
            "tv": {"S": "lexical-v1"},
            "terms": {"L": [{"S": "alpha"}, {"S": "alpha"}]},
            "ids": {"L": [{"S": "ID-1"}, {"S": "ID-1"}]},
        },
    )
    with mock.patch.object(ddb, "transact_write_items", wraps=ddb.transact_write_items) as transact:
        await operation(index, storage)
    (request,) = sent(transact)
    touched = [action_target(action)[1:] for action in request["TransactItems"]]
    assert len(touched) == len(set(touched))
    assert (INDEX_TABLE, term_posting_pk("", "alpha"), "docs/a") in touched
    assert (INDEX_TABLE, identifier_posting_pk("", "ID-1"), "docs/a") in touched


# ------------------------------------------------------------- document item port


async def test_port_inline_item_equals_what_write_stores(aws):
    ddb, _ = aws
    storage = make_storage(aws, compression="gzip", ttl_seconds=60)
    data = b"x" * 2048
    options: dict[str, Any] = {
        "vector": [0.5, 1.0],
        "metadata": {"kind": "note", "n": 2, "ok": True},
        "ttl_seconds": 30,
    }
    with frozen_clock(NOW):
        await storage.write("docs/a", data, **options)
        port = storage._document_item_port()
        item = port.inline_item(port.locate("docs/a"), data, **options)
    assert item == base_item(ddb, "docs/a")
    assert item["z"] == {"BOOL": True}


def test_port_exposes_table_scope_ttl_and_key_mapping(aws):
    port = make_storage(aws, prefix="tenant/a", ttl_seconds=60)._document_item_port()
    assert (port.table_name, port.scope, port.ttl_attribute) == (BASE_TABLE, "tenant/a/", "expireAt")
    location = port.locate("//docs//x/")
    assert (location.key, location.doc_id, location.pk, location.sk) == (
        "docs/x",
        "tenant/a/docs/x",
        "tenant/a",
        "docs/x",
    )
    assert port.relative_key("tenant/a/docs/x") == "docs/x"
    assert port.relative_key("tenant/b/docs/x") is None
    assert make_storage(aws)._document_item_port().ttl_attribute is None


def test_port_expiry_applies_only_when_ttl_is_enabled(aws):
    expired = {"expireAt": {"N": "1"}}
    assert make_storage(aws)._document_item_port().is_expired(expired) is False
    assert make_storage(aws, ttl_seconds=60)._document_item_port().is_expired(expired) is True
