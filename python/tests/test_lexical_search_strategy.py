# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LexicalSearchStrategy tests against moto: DynamoDBStorage(search_strategy=...) and SDK consumers."""

from __future__ import annotations

from typing import Any, Optional
from unittest import mock

import boto3
import pytest
from moto import mock_aws
from strands.storage import InMemoryStorage
from strands.storage.storage import StorageSearchResult
from strands.types.exceptions import StorageError
from strands.vended_memory_stores import FileMemoryStore

from strands_dynamodb_storage import (
    DynamoDBStorage,
    IdentifierQuery,
    LexicalIndex,
    LexicalIndexLimits,
    LexicalSearchStrategy,
    SearchableText,
    SearchQuery,
    SearchResult,
)
from strands_dynamodb_storage.lexical_terms import manifest_pk

REGION = "us-east-1"
BASE_TABLE = "base-table"
INDEX_TABLE = "index-table"
BUCKET = "test-bucket"
SENTINEL = "\u0000"


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
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
        yield ddb


def decode_text(key: str, data: bytes) -> Optional[SearchableText]:
    return SearchableText(text=data.decode("utf-8"))


def make_strategy(extract=decode_text, **options: Any) -> LexicalSearchStrategy:
    return LexicalSearchStrategy(index_table_name=INDEX_TABLE, extract=extract, **options)


def make_storage(ddb, strategy: Optional[LexicalSearchStrategy] = None, **kw: Any) -> DynamoDBStorage:
    if strategy is not None:
        kw["search_strategy"] = strategy
    return DynamoDBStorage(BASE_TABLE, client=ddb, **kw)


def base_item(ddb, doc_id: str) -> Optional[dict[str, Any]]:
    segments = [s for s in doc_id.split("/") if s]
    pk = "/".join(segments[:2]) if len(segments) > 1 else segments[0]
    sk = "/".join(segments[2:]) or SENTINEL
    return ddb.get_item(TableName=BASE_TABLE, Key={"pk": {"S": pk}, "sk": {"S": sk}}).get("Item")


def index_items(ddb) -> list[dict[str, Any]]:
    return ddb.scan(TableName=INDEX_TABLE)["Items"]


def manifests(ddb, scope: str = "") -> list[str]:
    return [item["sk"]["S"] for item in index_items(ddb) if item["pk"]["S"] == manifest_pk(scope)]


def hits(results) -> list[tuple[str, float]]:
    return [(result.key, result.score) for result in results]


async def test_write_indexes_and_string_search_finds_it(aws):
    storage = make_storage(aws, make_strategy())
    await storage.write("docs/a", b"login fails on api-prod-7")
    await storage.write("docs/b", b"billing is overdue")

    results = await storage.search("why does login fail?")

    assert hits(results) == [("docs/a", 0.25)]
    assert all(isinstance(result, SearchResult) for result in results)
    assert results[0].data == b"login fails on api-prod-7"
    assert results[0].metadata is None
    assert "lxrev" in base_item(aws, "docs/a")
    assert sorted(manifests(aws)) == ["docs/a", "docs/b"]


async def test_strategy_search_returns_sdk_results(aws):
    strategy = make_strategy()
    storage = make_storage(aws, strategy)
    await storage.write("docs/a", b"alpha beta")

    results = await strategy.search(storage, "alpha")

    assert all(isinstance(result, StorageSearchResult) for result in results)
    assert [(result.key, result.score, result.data) for result in results] == [("docs/a", 1.0, b"alpha beta")]


async def test_include_values_false_returns_no_data(aws):
    storage = make_storage(aws, make_strategy(include_values=False))
    await storage.write("docs/a", b"alpha")

    results = await storage.search("alpha")

    assert hits(results) == [("docs/a", 1.0)]
    assert results[0].data is None


async def test_top_k_defaults_to_the_constructor_and_is_overridable(aws):
    strategy = make_strategy(top_k=1)
    storage = make_storage(aws, strategy)
    for key in ("docs/a", "docs/b", "docs/c"):
        await storage.write(key, b"alpha")

    assert [result.key for result in await storage.search("alpha")] == ["docs/a"]
    assert [result.key for result in await strategy.search(storage, "alpha", top_k=2)] == ["docs/a", "docs/b"]


async def test_namespaced_views_share_the_strategy_but_not_the_scope(aws):
    storage = make_storage(aws, make_strategy())
    tenant_a = storage.namespace("tenant-a")
    tenant_b = storage.namespace("tenant-b")
    await tenant_a.write("notes/n1", b"alpha from a")
    await tenant_b.write("notes/n1", b"alpha from b")

    assert tenant_a._search_strategy is storage._search_strategy
    assert [(r.key, r.data) for r in await tenant_a.search("alpha")] == [("notes/n1", b"alpha from a")]
    assert [(r.key, r.data) for r in await tenant_b.search("alpha")] == [("notes/n1", b"alpha from b")]
    assert sorted(manifests(aws, "tenant-a/") + manifests(aws, "tenant-b/")) == [
        "tenant-a/notes/n1",
        "tenant-b/notes/n1",
    ]


async def test_extract_none_leaves_the_value_unindexed(aws):
    storage = make_storage(aws, make_strategy(extract=lambda key, data: None))
    await storage.write("docs/a", b"alpha")

    assert await storage.read("docs/a") == b"alpha"
    assert await storage.search("alpha") == []
    assert index_items(aws) == []


async def test_extract_none_hides_a_previously_indexed_version_until_repair(aws):
    indexing = make_storage(aws, make_strategy())
    skipping = make_storage(aws, make_strategy(extract=lambda key, data: None))
    await indexing.write("docs/a", b"alpha")
    await skipping.write("docs/a", b"alpha again")

    assert await indexing.search("alpha") == []
    assert manifests(aws) == ["docs/a"]
    report = await LexicalIndex(indexing, index_table_name=INDEX_TABLE).repair()
    assert report.documents_removed == 1
    assert index_items(aws) == []


async def test_identifiers_from_the_extractor_are_indexed_for_lookup(aws):
    def extract(key: str, data: bytes) -> Optional[SearchableText]:
        return SearchableText(text="invoice", identifiers=["FC-00123"])

    storage = make_storage(aws, make_strategy(extract=extract))
    await storage.write("docs/a", b"{}")

    response = await LexicalIndex(storage, index_table_name=INDEX_TABLE).lookup(IdentifierQuery("FC-00123", top_k=1))

    assert [result.key for result in response.results] == ["docs/a"]


async def test_indexing_failure_keeps_the_value_but_not_searchable(aws):
    storage = make_storage(aws, make_strategy())
    await storage.write("docs/a", b"alpha")
    too_many_terms = " ".join(f"term{n}" for n in range(50)).encode()

    with pytest.raises(StorageError, match=r"^Wrote 'docs/a' but indexing failed$") as raised:
        await storage.write("docs/a", too_many_terms)

    assert isinstance(raised.value.__cause__, StorageError)
    assert "max_postings_per_document" in str(raised.value.__cause__)
    assert await storage.read("docs/a") == too_many_terms
    assert await storage.search("alpha") == []
    assert await storage.search("term1") == []


async def test_offloaded_value_is_stored_but_not_indexed(aws):
    def extract(key: str, data: bytes) -> Optional[SearchableText]:
        return SearchableText(text="alpha")

    s3 = boto3.client("s3", region_name=REGION)
    storage = make_storage(aws, make_strategy(extract=extract), s3_bucket=BUCKET, s3_client=s3)
    large = b"x" * 400_000

    with pytest.raises(StorageError, match="Wrote 'docs/a' but indexing failed") as raised:
        await storage.write("docs/a", large)

    assert "S3 offload is not supported" in str(raised.value.__cause__)
    assert await storage.read("docs/a") == large
    assert await storage.search("alpha") == []


async def test_extractor_errors_are_reported_as_indexing_failures(aws):
    def broken(key: str, data: bytes) -> Optional[SearchableText]:
        raise ValueError("cannot parse")

    storage = make_storage(aws, make_strategy(extract=broken))

    with pytest.raises(StorageError, match="Wrote 'docs/a' but indexing failed") as raised:
        await storage.write("docs/a", b"alpha")

    assert isinstance(raised.value.__cause__, ValueError)
    assert await storage.read("docs/a") == b"alpha"


async def test_long_query_uses_the_first_max_query_terms(aws):
    storage = make_storage(aws, make_strategy(limits=LexicalIndexLimits(max_query_terms=3)))
    await storage.write("docs/a", b"alpha")
    await storage.write("docs/d", b"delta")

    assert hits(await storage.search("alpha beta gamma delta")) == [("docs/a", pytest.approx(1 / 3))]


async def test_default_query_term_limit_never_raises(aws):
    storage = make_storage(aws, make_strategy())
    await storage.write("docs/a", b"alpha")
    long_query = "alpha " + " ".join(f"word{n}" for n in range(40))

    assert hits(await storage.search(long_query)) == [("docs/a", pytest.approx(1 / 16))]


async def test_oversized_query_terms_are_skipped(aws):
    storage = make_storage(aws, make_strategy())
    await storage.write("docs/a", b"alpha")

    assert hits(await storage.search("alpha " + "x" * 65)) == [("docs/a", 1.0)]
    assert await storage.search("x" * 65) == []


@pytest.mark.parametrize("text", ["", "   ", "?! ... //"])
async def test_query_without_terms_returns_nothing(aws, text):
    strategy = make_strategy()
    storage = make_storage(aws, strategy)
    await storage.write("docs/a", b"alpha")

    assert await storage.search(text) == []
    assert await strategy.search(storage, text) == []


async def test_string_search_without_strategy_raises(aws):
    storage = make_storage(aws)

    with pytest.raises(StorageError) as raised:
        await storage.search("what does the user prefer?")

    assert str(raised.value) == (
        "DynamoDBStorage.search requires a SearchQuery with a pre-computed embedding vector. "
        "Plain-string queries are not supported without a search_strategy: embed the text first and pass "
        "SearchQuery(vector=..., top_k=...), or configure a search_strategy (for example LexicalSearchStrategy)."
    )


async def test_search_query_keeps_the_vector_path_with_a_strategy(aws):
    calls: list[dict[str, Any]] = []

    async def vector_search(params: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append(params)
        return [{"key": "memory/u1/m1", "score": 0.25}]

    strategy = make_strategy()
    storage = make_storage(aws, strategy, vector_search=vector_search)

    with mock.patch.object(strategy, "search", new=mock.AsyncMock()) as strategy_search:
        results = await storage.search(SearchQuery(vector=[1.0, 0.0], top_k=1))

    strategy_search.assert_not_awaited()
    assert hits(results) == [("memory/u1/m1", 0.25)]
    assert calls[0]["vector"] == [1.0, 0.0]


async def test_hook_upsert_preserves_vector_metadata_and_compression(aws):
    storage = make_storage(aws, make_strategy(), compression="gzip")
    body = b"alpha " * 100
    await storage.write("memory/u1/m1", body, vector=[0.5, -1.0], metadata={"kind": "preference", "rank": 2})

    item = base_item(aws, "memory/u1/m1")
    assert item["vector"] == {"L": [{"N": "0.5"}, {"N": "-1.0"}]}
    assert item["meta"] == {"M": {"kind": {"S": "preference"}, "rank": {"N": "2"}}}
    assert item["z"] == {"BOOL": True}
    assert "lxrev" in item
    assert await storage.read("memory/u1/m1") == body
    assert [(r.key, r.data) for r in await storage.search("alpha")] == [("memory/u1/m1", body)]


async def test_hook_upsert_copies_the_ttl_stamp_to_postings(aws):
    storage = make_storage(aws, make_strategy(), ttl_seconds=3600)
    await storage.write("docs/a", b"alpha", ttl_seconds=60)

    stamp = base_item(aws, "docs/a")["expireAt"]
    postings = [item for item in index_items(aws) if not item["pk"]["S"].startswith("m|")]
    assert postings and all(item["expireAt"] == stamp for item in postings)


async def test_non_dynamodb_storage_is_rejected(aws):
    strategy = make_strategy()

    with pytest.raises(StorageError, match="requires a DynamoDBStorage; got InMemoryStorage"):
        await strategy.search(InMemoryStorage(), "alpha")
    with pytest.raises(StorageError, match="Wrote 'docs/a' but indexing failed") as raised:
        await InMemoryStorage(search_strategy=strategy).write("docs/a", b"alpha")
    assert "requires a DynamoDBStorage" in str(raised.value.__cause__)


async def test_foreign_strategy_errors_are_wrapped(aws):
    class FailingStrategy:
        async def index(self, storage: DynamoDBStorage, key: str, data: bytes, **kwargs: Any) -> None:
            return None

        async def search(self, storage: DynamoDBStorage, query: str, **kwargs: Any) -> list[StorageSearchResult]:
            raise RuntimeError("backend down")

    storage = DynamoDBStorage(BASE_TABLE, client=aws, search_strategy=FailingStrategy())

    with pytest.raises(StorageError, match=f"Failed to search DynamoDB table '{BASE_TABLE}'") as raised:
        await storage.search("alpha")
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"index_table_name": ""}, "non-empty index_table_name"),
        ({"extract": None}, "extract callable"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 101}, "top_k"),
    ],
)
def test_constructor_validates_its_options(options, message):
    config: dict[str, Any] = {"index_table_name": INDEX_TABLE, "extract": decode_text, **options}

    with pytest.raises(StorageError, match=message):
        LexicalSearchStrategy(**config)


def test_strategy_is_marked_sandbox_safe():
    assert make_strategy().requires_host_fs is False


async def test_sdk_file_memory_store_adds_and_searches_entries(aws):
    storage = make_storage(aws, make_strategy())
    store = FileMemoryStore(storage=storage, name="agent-memory")

    key = await store.add("# User preferences\nPrefers dark mode in every editor")
    await store.add("# Project setup\nUses Python 3.12")
    entries = await store.search("which mode does the user prefer?")

    assert key == "user-preferences.md"
    assert [(entry.content, entry.metadata) for entry in entries] == [
        (
            "# User preferences\nPrefers dark mode in every editor",
            {"path": "user-preferences.md", "score": pytest.approx(2 / 6)},
        )
    ]
    assert sorted(manifests(aws, "memory/agent-memory/")) == [
        "memory/agent-memory/project-setup.md",
        "memory/agent-memory/user-preferences.md",
    ]


async def test_sdk_file_memory_store_reindexes_merged_entries(aws):
    store = FileMemoryStore(storage=make_storage(aws, make_strategy()), name="agent-memory")
    await store.add("# User preferences\nPrefers dark mode")
    await store.add("# User preferences\nLikes window seats")

    entries = await store.search("window seats")

    assert [entry.content for entry in entries] == ["# User preferences\nPrefers dark mode\nLikes window seats"]
