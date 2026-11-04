# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoDBVectorSearchStrategy (moto, with the native search stood in by an adapter)."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import boto3
import pytest
from moto import mock_aws
from strands.types.exceptions import StorageError
from strands.vended_memory_stores.file_memory_store import FileMemoryStore
from test_dynamodb_storage import REGION, TABLE, make, raw_item

from strands_dynamodb_storage import BedrockEmbedder, DynamoDBVectorSearchStrategy
from strands_dynamodb_storage import vector_search as vector_search_module

needs_storage_search = pytest.mark.skipif(
    importlib.util.find_spec("strands.storage.search") is None, reason="needs strands-agents>=1.54.0"
)

VOCABULARY = ("dark", "mode", "coffee", "tea")


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
        yield ddb, boto3.client("s3", region_name=REGION)


def words(text: str) -> list[float]:
    """One dimension per vocabulary word, plus a floor so no vector is zero."""
    tokens = text.lower().split()
    return [0.01 + tokens.count(word) for word in VOCABULARY]


class PurposeEmbedder:
    """Records ``(text, purpose)`` for each call, the package's own embedder shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def embed(self, text: str, purpose: str) -> list[float]:
        self.calls.append((text, purpose))
        return words(text)


def distance_adapter(ddb, requests: list[dict[str, Any]]):
    """Brute-force cosine *distance* over stored vectors, lower closer, as the service scores COSINE."""

    async def adapter(params):
        requests.append(params)
        out = []
        for item in ddb.scan(TableName=TABLE).get("Items", []):
            if "vector" not in item or (params.get("pk") and item["pk"]["S"] != params["pk"]):
                continue
            stored = [float(n["N"]) for n in item["vector"]["L"]]
            dot = sum(a * b for a, b in zip(params["vector"], stored, strict=True))
            norms = math.sqrt(sum(a * a for a in params["vector"])) * math.sqrt(sum(b * b for b in stored))
            out.append({"key": item["k"]["S"], "score": 1 - dot / norms})
        out.sort(key=lambda match: match["score"])
        return out[: params["top_k"]]

    return adapter


def vector_storage(aws, strategy: DynamoDBVectorSearchStrategy, requests: list[dict[str, Any]] | None = None):
    adapter = distance_adapter(aws[0], [] if requests is None else requests)
    return make(aws, search_strategy=strategy, vector_search=adapter)


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize("max_results", [0, 101, True, 1.5])
def test_max_results_must_be_an_integer_from_1_to_100(max_results):
    with pytest.raises(ValueError, match="max_results"):
        DynamoDBVectorSearchStrategy(max_results=max_results)


def test_an_unknown_distance_function_is_rejected():
    with pytest.raises(ValueError, match="distance_function"):
        DynamoDBVectorSearchStrategy(distance_function="MANHATTAN")  # type: ignore[arg-type]


# ---------------------------------------------------------------- write


async def test_write_stores_the_vector_on_the_same_item(aws):
    embedder = PurposeEmbedder()
    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=embedder))

    await storage.namespace("memory/facts").write("prefs.md", b"dark mode")

    item = raw_item(aws[0], "memory/facts/prefs.md")
    assert [float(n["N"]) for n in item["vector"]["L"]] == words("dark mode")
    assert item["data"]["B"] == b"dark mode"
    assert embedder.calls == [("dark mode", "document")]


async def test_a_caller_vector_is_stored_as_given(aws):
    embedder = PurposeEmbedder()
    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=embedder))

    await storage.write("memory/facts/a", b"dark mode", vector=[1.0, 0.0, 0.0, 0.0])

    assert [float(n["N"]) for n in raw_item(aws[0], "memory/facts/a")["vector"]["L"]] == [1.0, 0.0, 0.0, 0.0]
    assert embedder.calls == []


@pytest.mark.parametrize("data", [b"", b"   ", b"\xff\xfe"])
async def test_an_empty_or_binary_value_is_stored_without_a_vector_and_clears_the_old_one(aws, data):
    embedder = PurposeEmbedder()
    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=embedder))
    await storage.write("memory/facts/a", b"dark mode")

    await storage.write("memory/facts/a", data)

    assert "vector" not in raw_item(aws[0], "memory/facts/a")
    assert await storage.read("memory/facts/a") == data
    assert embedder.calls == [("dark mode", "document")]


async def test_a_failed_embedding_writes_nothing(aws):
    class Failing:
        async def embed(self, text, purpose):
            raise RuntimeError("model down")

    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=Failing()))

    with pytest.raises(StorageError, match="Failed to embed 'memory/facts/a' for search, so nothing was written") as e:
        await storage.write("memory/facts/a", b"dark mode")

    assert "model down" in str(e.value.__cause__)
    assert raw_item(aws[0], "memory/facts/a") is None


async def test_an_embedder_returning_no_vector_writes_nothing(aws):
    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=lambda text: []))

    with pytest.raises(StorageError, match="nothing was written"):
        await storage.write("memory/facts/a", b"dark mode")

    assert raw_item(aws[0], "memory/facts/a") is None


async def test_write_embeds_once_and_does_not_call_index(aws, monkeypatch):
    embedder = PurposeEmbedder()
    strategy = DynamoDBVectorSearchStrategy(embedder=embedder)

    async def unexpected(*args, **kwargs):
        raise AssertionError("write must not call index for this strategy")

    monkeypatch.setattr(strategy, "index", unexpected)

    await vector_storage(aws, strategy).write("memory/facts/a", b"dark mode")

    assert embedder.calls == [("dark mode", "document")]


async def test_index_backfills_a_value_written_before_the_strategy(aws):
    await make(aws).write("memory/facts/a", b"dark mode")
    assert "vector" not in raw_item(aws[0], "memory/facts/a")
    strategy = DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder())
    view = vector_storage(aws, strategy).namespace("memory/facts")

    await strategy.index(view, "a", await view.read("a"))

    assert [float(n["N"]) for n in raw_item(aws[0], "memory/facts/a")["vector"]["L"]] == words("dark mode")
    assert await view.read("a") == b"dark mode"


# ---------------------------------------------------------------- search


@needs_storage_search
async def test_search_ranks_by_meaning_with_higher_scores_closer_and_returns_the_bytes(aws):
    embedder = PurposeEmbedder()
    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=embedder)).namespace("memory/facts")
    await view.write("prefs.md", b"dark mode")
    await view.write("drinks.md", b"coffee tea")

    found = await view.search("dark")

    assert [(result.key, result.data) for result in found] == [("prefs.md", b"dark mode"), ("drinks.md", b"coffee tea")]
    assert found[0].score > found[1].score
    assert all(0 < result.score <= 1 for result in found)
    assert embedder.calls[-1] == ("dark", "query")


@needs_storage_search
async def test_search_is_pinned_to_the_storage_partition(aws):
    requests: list[dict[str, Any]] = []
    root = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder()), requests)
    await root.write("memory/facts/a", b"dark mode")
    await root.write("memory/other/a", b"dark mode")

    found = await root.namespace("memory/facts").search("dark")

    assert requests[-1]["pk"] == "memory/facts"
    assert [result.key for result in found] == ["a"]


@needs_storage_search
async def test_a_storage_shallower_than_a_partition_searches_unpinned(aws):
    requests: list[dict[str, Any]] = []
    root = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder()), requests)
    await root.write("memory/facts/a", b"dark mode")

    await root.namespace("memory").search("dark")

    assert "pk" not in requests[-1]


@needs_storage_search
async def test_max_results_bounds_the_search(aws):
    requests: list[dict[str, Any]] = []
    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder(), max_results=1), requests)
    view = view.namespace("memory/facts")
    await view.write("a", b"dark mode")
    await view.write("b", b"dark")

    assert len(await view.search("dark")) == 1
    assert requests[-1]["top_k"] == 1


async def test_a_blank_query_returns_nothing_without_embedding(aws):
    embedder = PurposeEmbedder()
    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=embedder)).namespace("memory/facts")

    assert await view.search("   ") == []
    assert embedder.calls == []


@needs_storage_search
async def test_a_deleted_value_stops_matching(aws):
    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder())).namespace("memory/facts")
    await view.write("a", b"dark mode")

    await view.delete("a")

    assert await view.search("dark") == []


@needs_storage_search
async def test_dot_product_scores_are_returned_as_the_index_gives_them(aws):
    async def adapter(params):
        return [{"key": "memory/facts/a", "score": 7.5}]

    view = make(
        aws,
        search_strategy=DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder(), distance_function="DOT_PRODUCT"),
        vector_search=adapter,
    ).namespace("memory/facts")

    assert [result.score for result in await view.search("dark")] == [7.5]


# ---------------------------------------------------------------- embedder shapes


@needs_storage_search
async def test_a_text_only_embedder_is_called_with_the_text_alone(aws):
    calls: list[str] = []

    class TextOnly:
        async def embed(self, text: str) -> list[float]:
            calls.append(text)
            return words(text)

    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=TextOnly())).namespace("memory/facts")
    await view.write("a", b"dark mode")
    await view.search("dark")

    assert calls == ["dark mode", "dark"]


@needs_storage_search
async def test_a_plain_function_embeds_values_and_queries(aws):
    view = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=words)).namespace("memory/facts")
    await view.write("a", b"dark mode")

    assert [result.key for result in await view.search("dark")] == ["a"]


async def test_the_default_embedder_is_bedrock_on_the_storage_defaults(aws, monkeypatch):
    built: list[dict[str, Any]] = []

    class FakeBedrock:
        def __init__(self, **kwargs):
            built.append(kwargs)

        async def embed(self, text, purpose="document"):
            return words(text)

    monkeypatch.setattr(vector_search_module, "BedrockEmbedder", FakeBedrock)
    strategy = DynamoDBVectorSearchStrategy()
    storage = vector_storage(aws, strategy)

    await storage.write("memory/facts/a", b"dark mode")
    await storage.write("memory/facts/b", b"tea")

    assert built == [{"region_name": REGION, "boto_session": None, "boto_client_config": None}]


def test_bedrock_embedder_is_an_sdk_embedder_where_the_sdk_has_one():
    search = pytest.importorskip("strands.storage.search")
    if not hasattr(search, "Embedder"):
        pytest.skip("this strands-agents release has no Embedder")

    assert isinstance(BedrockEmbedder(client=object()), search.Embedder)


@needs_storage_search
async def test_file_memory_store_recalls_by_meaning(aws):
    storage = vector_storage(aws, DynamoDBVectorSearchStrategy(embedder=PurposeEmbedder()))
    knowledge = FileMemoryStore(name="facts", storage=storage, writable=True)  # type: ignore[abstract]
    await knowledge.add("# Display\nprefers dark mode")
    await knowledge.add("# Drinks\nlikes coffee and tea")

    found = await knowledge.search("dark")

    assert found[0].content.startswith("# Display")
