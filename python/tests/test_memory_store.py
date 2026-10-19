# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoDBMemoryStore, using moto for offline DynamoDB and fake embedders."""

from __future__ import annotations

import hashlib
import logging
import math
from unittest import mock

import boto3
import pytest
from moto import mock_aws
from strands.memory import MemoryManager
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import BedrockEmbedder, DynamoDBMemoryStore, DynamoDBStorage, Embedder

REGION = "us-east-1"
TABLE = "memory-table"


@pytest.fixture
def ddb():
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        client.create_table(
            TableName=TABLE,
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield client


def _cosine_distance(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (norm or 1.0)


def scan_adapter(client):
    """Brute-force COSINE distance over stored vectors, standing in for SearchVectors (lower is closer)."""

    async def adapter(params):
        out = []
        for item in client.scan(TableName=TABLE).get("Items", []):
            if "vector" not in item or item["pk"]["S"] != params.get("pk"):
                continue
            vector = [float(n["N"]) for n in item["vector"]["L"]]
            meta = None
            if "meta" in item:
                meta = {
                    k: (v["S"] if "S" in v else v.get("BOOL") if "BOOL" in v else float(v["N"]))
                    for k, v in item["meta"]["M"].items()
                }
            out.append({"key": item["k"]["S"], "score": _cosine_distance(params["vector"], vector), "metadata": meta})
        out.sort(key=lambda match: match["score"])
        return out[: params["top_k"]]

    return adapter


class KeywordEmbedder:
    """Deterministic embedder: one dimension per keyword, recording every call."""

    WORDS = ("window", "seat", "aisle", "peanut", "allergy", "coffee")

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def embed(self, text, purpose):
        self.calls.append((text, purpose))
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in self.WORDS] + [0.01]


def make_storage(ddb, **kw) -> DynamoDBStorage:
    kw.setdefault("client", ddb)
    return DynamoDBStorage(TABLE, **kw)


def make_store(ddb, *, prefix="user/u1", embedder=None, **kw) -> DynamoDBMemoryStore:
    storage = (
        make_storage(ddb, vector_search=scan_adapter(ddb)).namespace(prefix)
        if prefix
        else make_storage(ddb, vector_search=scan_adapter(ddb))
    )
    return DynamoDBMemoryStore(storage, embedder=embedder or KeywordEmbedder(), **kw)


def item_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


# ------------------------------------------------------------------- protocol


def test_exposes_memory_store_protocol_fields(ddb):
    store = make_store(ddb, name="prefs", description="User preferences", max_search_results=4)
    assert store.name == "prefs"
    assert store.description == "User preferences"
    assert store.max_search_results == 4
    assert store.writable is True
    assert store.extraction is None


async def test_writable_false_keeps_the_memory_manager_from_writing(ddb):
    read_only = make_store(ddb, name="read-only", writable=False, embedder=KeywordEmbedder())
    writable = make_store(ddb, name="notes", embedder=KeywordEmbedder())
    assert read_only.writable is False
    await MemoryManager(stores=[read_only, writable], injection=False).add("Prefers a window seat")
    assert read_only._embedder.calls == []
    assert [call[1] for call in writable._embedder.calls] == ["document"]


def test_description_none_means_no_description(ddb):
    assert make_store(ddb, description=None).description is None
    assert make_store(ddb).description == "Long-term memories stored in Amazon DynamoDB, searched by meaning"


@pytest.mark.parametrize(("option", "value"), [("description", 5), ("writable", None), ("writable", "yes")])
def test_rejects_a_description_or_writable_of_the_wrong_type(ddb, option, value):
    with pytest.raises(TypeError, match=option):
        make_store(ddb, **{option: value})


async def test_works_inside_memory_manager(ddb):
    store = make_store(ddb, name="prefs")
    manager = MemoryManager(stores=[store], injection=False)
    await manager.add("Prefers a window seat")
    await manager.add("Has a peanut allergy")
    entries = await manager.search("which seat does the user like?")
    assert entries[0].content == "Prefers a window seat"
    assert entries[0].store_name == "prefs"


def test_memory_manager_accepts_extraction(ddb):
    store = make_store(ddb, extraction=True)
    assert store.extraction is True  # the MemoryManager reads it to decide whether to extract
    MemoryManager(stores=[store], injection=False)  # add() satisfies the extractor's write sink


# ------------------------------------------------------------------- add


async def test_add_writes_one_item_with_text_vector_and_metadata(ddb):
    embedder = KeywordEmbedder()
    store = make_store(ddb, name="prefs", embedder=embedder)
    key = await store.add("Prefers a window seat", {"source": "chat", "turn": 3, "confirmed": True})
    assert key == f"memory/prefs/{item_id('Prefers a window seat')}"
    raw = ddb.get_item(TableName=TABLE, Key={"pk": {"S": "user/u1"}, "sk": {"S": key}})["Item"]
    assert raw["k"]["S"] == f"user/u1/{key}"
    assert raw["data"]["B"] == b"Prefers a window seat"
    assert [float(n["N"]) for n in raw["vector"]["L"]] == await KeywordEmbedder().embed("Prefers a window seat", "x")
    assert raw["meta"]["M"] == {"source": {"S": "chat"}, "turn": {"N": "3"}, "confirmed": {"BOOL": True}}
    assert embedder.calls == [("Prefers a window seat", "document")]


async def test_add_same_text_overwrites_instead_of_duplicating(ddb):
    store = make_store(ddb)
    first = await store.add("Prefers a window seat", {"source": "chat"})
    second = await store.add("Prefers a window seat", {"source": "profile"})
    assert first == second
    items = ddb.scan(TableName=TABLE)["Items"]
    assert len(items) == 1
    assert items[0]["meta"]["M"] == {"source": {"S": "profile"}}


async def test_add_without_metadata_writes_no_metadata_attribute_and_logs_nothing(ddb, caplog):
    store = make_store(ddb)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat")
        await store.add("Likes coffee", {})
        await store.add("Has a peanut allergy", None)
    assert all("meta" not in item for item in ddb.scan(TableName=TABLE)["Items"])
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"_key": "x"}, "is reserved for search results"),
        ({1: "x"}, "is not a string"),
        ({"": "x"}, "is empty, which DynamoDB rejects"),
        ({"__proto__": "x"}, "cannot be written by the AWS SDK for JavaScript"),
        ({"s": "a\udcffb"}, "contains a lone surrogate"),
        ({"k": "\udcffy"}, "contains a lone surrogate"),
        ({"a\ud800": "v"}, "contains a lone surrogate"),
        ({"k\udfff": "v"}, "contains a lone surrogate"),
        ({"v": "x\udfff"}, "contains a lone surrogate"),
        ({"constructor": "x"}, "cannot be written by the AWS SDK for JavaScript"),
        ({"tags": ["a"]}, "has type list"),
        ({"n": 2**53}, "is out of range"),
    ],
)
async def test_add_logs_why_each_metadata_entry_was_dropped(ddb, caplog, metadata, reason):
    store = make_store(ddb)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", metadata)
    assert reason in caplog.text


async def test_drop_warning_for_a_lone_surrogate_key_can_be_written_as_utf8(ddb, caplog):
    store = make_store(ddb)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {"a\udcff": "v"})
    assert "key=<a\\udcff>" in caplog.text
    caplog.text.encode("utf-8")  # strict, as a FileHandler(encoding="utf-8") writes it


async def test_drop_warning_names_the_store_and_the_key_as_typescript_does(ddb, caplog):
    store = make_store(ddb, name="prefs")
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {"_key": "x"})
    assert "store=<prefs>, key=<_key> | dropping metadata entry: is reserved for search results" in caplog.text


async def test_add_round_trips_non_ascii_text(ddb):
    store = make_store(ddb)
    await store.add("Préfère la fenêtre: window seat 窓側")
    [entry] = await store.search("window")
    assert entry.content == "Préfère la fenêtre: window seat 窓側"


async def test_add_stamps_ttl_when_storage_has_ttl(ddb):
    storage = make_storage(ddb, ttl_seconds=3600, vector_search=scan_adapter(ddb)).namespace("user/u1")
    store = DynamoDBMemoryStore(storage, embedder=KeywordEmbedder())
    await store.add("Prefers a window seat")
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert int(item["expireAt"]["N"]) > 0


@pytest.mark.parametrize(
    ("content", "error"),
    [("", ValueError), ("   \n", ValueError), ("\ufeff", ValueError), ("\x85", ValueError), ("\x1c\u3000", ValueError)]
    + [("\ufeff ", ValueError), (" \ufeff", ValueError)]
    + [("a\ud800", ValueError), ("a\udfffb", ValueError), (None, TypeError), (b"bytes", TypeError), (42, TypeError)],
)
async def test_add_rejects_invalid_content(ddb, content, error):
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    message = {
        TypeError: "content must be a string",
        ValueError: "content must not be empty|content must be valid Unicode text; it contains a lone surrogate",
    }[error]
    with pytest.raises(error, match=message):
        await store.add(content)
    assert embedder.calls == []


@pytest.mark.parametrize(
    "dropped",
    [
        {"_key": "x"},
        {"_score": 1},
        {"tags": ["a", "b"]},
        {"nested": {"a": 1}},
        {"missing": None},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"huge": 1e200},
        {"tiny": 1e-140},
        {"digits": 10**40 + 1},
        {"unsafe": 2**53},
        {"neg_unsafe": -(2**53)},
        {"below_min": 1e-131},
        {"large_float": 1e20},
        {1: "non-string key"},
        {"": "empty key"},
    ],
)
async def test_add_stores_the_memory_and_drops_unsupported_metadata_with_a_warning(ddb, caplog, dropped):
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {"source": "chat", **dropped})
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert item["meta"]["M"] == {"source": {"S": "chat"}}
    assert "dropping metadata entry" in caplog.text
    assert len(embedder.calls) == 1


async def test_add_stores_str_int_and_float_subclasses_as_plain_values(ddb):
    import enum

    class Color(str, enum.Enum):
        RED = "red"

    class Priority(enum.IntEnum):
        HIGH = 3

    class Weight(float):
        def __str__(self):
            return "W"

    store = make_store(ddb)
    await store.add("Prefers a window seat", {"color": Color.RED, "priority": Priority.HIGH, "weight": Weight(0.5)})
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert item["meta"]["M"] == {"color": {"S": "red"}, "priority": {"N": "3"}, "weight": {"N": "0.5"}}


async def test_str_subclass_metadata_keys_are_written_and_logged_by_their_value(ddb, caplog):
    import enum

    class Color(str, enum.Enum):
        RED = "red"
        BLUE = "blue"

    store = make_store(ddb)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {Color.RED: "v", Color.BLUE: ["x"]})
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert item["meta"]["M"] == {"red": {"S": "v"}}
    assert "key=<blue> | dropping metadata entry: has type list" in caplog.text


async def test_add_keeps_numbers_both_packages_round_trip(ddb):
    store = make_store(ddb)
    kept = {
        "zero": 0,
        "neg_zero": -0.0,
        "neg": -1.5,
        "max": 2**53 - 1,
        "min_neg": -(2**53 - 1),
        "tiny": 1e-130,
        "fraction": 0.25,
    }
    await store.add("Prefers a window seat", kept)
    [item] = ddb.scan(TableName=TABLE)["Items"]
    from strands_dynamodb_storage.dynamodb_storage import _unmarshal_meta

    # The decoder search results go through returns every kept value exactly.
    assert _unmarshal_meta(item["meta"]) == kept


@pytest.mark.parametrize("metadata", [[("a", 1)], "source=chat", 42])
async def test_add_drops_metadata_that_is_not_a_dict(ddb, caplog, metadata):
    store = make_store(ddb)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", metadata)
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert "meta" not in item
    assert "must be a dict" in caplog.text


async def test_extraction_with_structured_model_metadata_still_stores_the_memory(ddb):
    """The SDK forwards whatever metadata a model emits; a list must not fail the batch."""
    store = make_store(ddb, name="prefs")
    manager = MemoryManager(stores=[store], injection=False)
    await manager.add("Prefers a window seat", {"metadata": {"tags": ["travel"], "confidence": None, "source": "x"}})
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert item["meta"]["M"] == {"source": {"S": "x"}}


# ------------------------------------------------------------------- search


async def test_search_returns_closest_first_with_key_and_score(ddb):
    embedder = KeywordEmbedder()
    store = make_store(ddb, name="prefs", embedder=embedder)
    await store.add("Prefers a window seat", {"source": "chat"})
    await store.add("Has a peanut allergy")
    await store.add("Drinks coffee black")
    entries = await store.search("window seat please")
    assert [e.content for e in entries][0] == "Prefers a window seat"
    top = entries[0]
    assert top.metadata["source"] == "chat"
    assert top.metadata["_key"] == f"memory/prefs/{item_id('Prefers a window seat')}"
    scores = [e.metadata["_score"] for e in entries]
    assert scores == sorted(scores)  # COSINE distance: lower is closer
    assert embedder.calls[-1] == ("window seat please", "query")


async def test_search_key_is_accepted_by_storage_delete(ddb):
    store = make_store(ddb)
    await store.add("Prefers a window seat")
    [entry] = await store.search("window")
    await store._storage.delete(entry.metadata["_key"])
    assert await store.search("window") == []


async def test_search_pins_the_partition_and_projects_values_on_the_native_path(ddb):
    storage = make_storage(ddb).namespace("user/u1")
    store = DynamoDBMemoryStore(storage, name="prefs", embedder=KeywordEmbedder())
    response = {
        "SearchResults": [
            {
                "Item": {
                    "pk": {"S": "user/u1"},
                    "sk": {"S": "memory/prefs/abc"},
                    "data": {"B": b"Prefers a window seat"},
                    "meta": {"M": {"source": {"S": "chat"}}},
                },
                "Score": 0.05,
            },
            {
                # Another store's memory and an application item in the same partition.
                "Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/facts/def"}, "data": {"B": b"other store"}},
                "Score": 0.06,
            },
            {
                "Item": {"pk": {"S": "user/u1"}, "sk": {"S": "notes/n1"}, "data": {"B": b"app item"}},
                "Score": 0.07,
            },
        ]
    }
    with (
        mock.patch.object(ddb, "search_vectors", create=True, return_value=response) as sv,
        mock.patch.object(ddb, "get_item", wraps=ddb.get_item) as get_item,
    ):
        entries = await store.search("window", {"max_search_results": 7})
    get_item.assert_not_called()
    request = sv.call_args.kwargs
    assert request["TopK"] == 7
    assert request["ExpressionAttributeValues"] == {":pk": {"S": "user/u1"}}
    assert "#d" in request["ProjectionExpression"]
    assert sv.call_count == 1
    assert [(e.content, e.metadata) for e in entries] == [
        ("Prefers a window seat", {"source": "chat", "_key": "memory/prefs/abc", "_score": 0.05})
    ]


async def test_search_skips_an_offloaded_match_whose_item_is_gone_and_decodes_invalid_utf8(ddb):
    store = DynamoDBMemoryStore(make_storage(ddb).namespace("user/u1"), name="prefs", embedder=KeywordEmbedder())
    response = {
        "SearchResults": [
            # Offloaded to S3 (no inline payload) and deleted between SearchVectors and the point read.
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/prefs/gone"}, "s3": {"BOOL": True}}, "Score": 0.01},
            {
                "Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/prefs/bin"}, "data": {"B": b"\xff\xfe"}},
                "Score": 0.02,
            },
        ]
    }
    with mock.patch.object(ddb, "search_vectors", create=True, return_value=response):
        entries = await store.search("window")
    assert [(e.content, e.metadata["_key"]) for e in entries] == [("\ufffd\ufffd", "memory/prefs/bin")]


async def test_search_reads_no_value_for_a_match_outside_the_store(ddb):
    """Matches the store discards are dropped before the point read; its own still get one."""
    store = DynamoDBMemoryStore(make_storage(ddb).namespace("user/u1"), name="prefs", embedder=KeywordEmbedder())
    response = {
        "SearchResults": [
            # Another store's offloaded memory, an application item whose payload the index
            # does not project, and a key that contains this store's prefix later on.
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/facts/big"}, "s3": {"BOOL": True}}, "Score": 0.01},
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "notes/n1"}}, "Score": 0.02},
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "x/memory/prefs/y"}}, "Score": 0.03},
            # A store whose name starts with this one's: "prefs2" is not under "memory/prefs/".
            {
                "Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/prefs2/sib"}, "data": {"B": b"sibling"}},
                "Score": 0.031,
            },
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/prefs2/off"}, "s3": {"BOOL": True}}, "Score": 0.032},
            # This store's own offloaded memory, deleted before the read.
            {"Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/prefs/gone"}, "s3": {"BOOL": True}}, "Score": 0.04},
            {
                "Item": {
                    "pk": {"S": "user/u1"},
                    "sk": {"S": "memory/prefs/abc"},
                    "data": {"B": b"Prefers a window seat"},
                },
                "Score": 0.05,
            },
        ]
    }
    with (
        mock.patch.object(ddb, "search_vectors", create=True, return_value=response),
        mock.patch.object(ddb, "get_item", wraps=ddb.get_item) as get_item,
    ):
        entries = await store.search("window")
    assert [call.kwargs["Key"]["sk"]["S"] for call in get_item.call_args_list] == ["memory/prefs/gone"]
    assert [(e.content, e.metadata["_key"]) for e in entries] == [("Prefers a window seat", "memory/prefs/abc")]


async def test_content_with_a_byte_order_mark_reads_back_unchanged(ddb):
    store = make_store(ddb)
    content = "\ufeffPrefers a window seat"
    await store.add(content)
    [entry] = await store.search("window")
    assert entry.content == content
    response = {
        "SearchResults": [
            {
                "Item": {"pk": {"S": "user/u1"}, "sk": {"S": "memory/dynamodb/x"}, "data": {"B": content.encode()}},
                "Score": 0.1,
            }
        ]
    }
    with mock.patch.object(ddb, "search_vectors", create=True, return_value=response):
        [native] = await DynamoDBMemoryStore(make_storage(ddb).namespace("user/u1"), embedder=KeywordEmbedder()).search(
            "window"
        )
    assert native.content == content


async def test_rejects_a_lone_surrogate_before_looking_at_metadata(ddb, caplog):
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        with pytest.raises(ValueError, match="lone surrogate"):
            await store.add("a\ud800b", {"tags": ["dropped"]})
    assert caplog.text == ""
    assert embedder.calls == []


@pytest.mark.parametrize("content", ["  Prefers a window seat\n", "Cafe\u0301 by the window", "\U0001f600 window seat"])
async def test_stores_and_embeds_content_exactly_as_given(ddb, content):
    """No trimming and no Unicode normalization: the decomposed e + U+0301 stays as given."""
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    await store.add(content)
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert embedder.calls == [(content, "document")]
    assert item["data"]["B"] == content.encode()
    assert item["sk"]["S"].endswith(item_id(content))


async def test_stores_in_deep_sibling_namespaces_compete_for_one_partition(ddb):
    """tenant/a/user/u1 and tenant/a/user/u2 share partition tenant/a, so u2's nearer memories fill TopK."""
    root = make_storage(ddb, vector_search=scan_adapter(ddb))
    u1 = DynamoDBMemoryStore(root.namespace("tenant/a/user/u1"), embedder=KeywordEmbedder())
    u2 = DynamoDBMemoryStore(root.namespace("tenant/a/user/u2"), embedder=KeywordEmbedder())
    await u1.add("Likes coffee by the window")
    for memory in ("Window seat", "Window seat please", "Always a window seat"):
        await u2.add(memory)
    assert u1._partition == u2._partition == "tenant/a"
    assert await u1.search("window seat", {"max_search_results": 3}) == []
    assert [e.content for e in await u1.search("window seat", {"max_search_results": 4})] == [
        "Likes coffee by the window"
    ]
    # Two-segment namespaces get partitions of their own.
    v1 = DynamoDBMemoryStore(root.namespace("user/v1"), embedder=KeywordEmbedder())
    v2 = DynamoDBMemoryStore(root.namespace("user/v2"), embedder=KeywordEmbedder())
    await v1.add("Likes coffee by the window")
    for memory in ("Window seat", "Window seat please", "Always a window seat"):
        await v2.add(memory)
    assert [e.content for e in await v1.search("window seat", {"max_search_results": 1})] == [
        "Likes coffee by the window"
    ]


async def test_search_keeps_only_this_stores_memories_in_a_shared_partition(ddb):
    prefs = make_store(ddb, name="prefs")
    facts = make_store(ddb, name="facts")
    await prefs.add("Prefers a window seat")
    await facts.add("Window seats are on the left")
    # An application item whose key contains memory/prefs/ other than at the start.
    vector = await KeywordEmbedder().embed("window seat", "document")
    await prefs._storage.write("notes/memory/prefs/x", b"app item", vector=vector)
    assert [e.content for e in await prefs.search("window")] == ["Prefers a window seat"]
    assert [e.content for e in await facts.search("window")] == ["Window seats are on the left"]


async def test_search_is_scoped_to_the_storage_namespace(ddb):
    alice = make_store(ddb, prefix="user/alice")
    bob = make_store(ddb, prefix="user/bob")
    await alice.add("Prefers a window seat")
    assert await bob.search("window") == []


@pytest.mark.parametrize(
    ("options", "store_default", "expected"),
    [
        (None, None, 10),
        ({}, None, 10),
        (None, 4, 4),
        ({"max_search_results": None}, 4, 4),
        ({"max_search_results": 2}, 4, 2),
        ({"max_search_results": 100}, None, 100),
        ({"max_search_results": 500}, None, 100),
    ],
)
async def test_search_result_count(ddb, options, store_default, expected):
    storage = make_storage(ddb).namespace("user/u1")
    store = DynamoDBMemoryStore(storage, embedder=KeywordEmbedder(), max_search_results=store_default)
    with mock.patch.object(ddb, "search_vectors", create=True, return_value={"SearchResults": []}) as sv:
        await store.search("window", options)
    assert sv.call_args.kwargs["TopK"] == expected


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
async def test_search_rejects_invalid_result_count(ddb, bad):
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    with pytest.raises(ValueError):
        await store.search("window", {"max_search_results": bad})
    assert embedder.calls == []


async def test_search_checks_the_limit_before_returning_nothing_for_a_blank_query(ddb):
    embedder = KeywordEmbedder()
    with pytest.raises(ValueError):
        await make_store(ddb, embedder=embedder).search("   ", {"max_search_results": 0})
    assert embedder.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "\ufeff", "\x85", "\x1f"])
async def test_search_with_empty_query_returns_nothing_without_calls(ddb, query):
    embedder = KeywordEmbedder()
    store = make_store(ddb, embedder=embedder)
    with mock.patch.object(ddb, "search_vectors", create=True) as sv:
        assert await store.search(query) == []
    assert embedder.calls == []
    sv.assert_not_called()


@pytest.mark.parametrize("query", [None, 3, b"window"])
async def test_search_rejects_non_string_query(ddb, query):
    embedder = KeywordEmbedder()
    with pytest.raises(TypeError, match="query must be a string"):
        await make_store(ddb, embedder=embedder).search(query)
    assert embedder.calls == []


async def test_search_failure_raises_storage_error(ddb):
    storage = make_storage(ddb).namespace("user/u1")
    store = DynamoDBMemoryStore(storage, embedder=KeywordEmbedder())
    with mock.patch.object(ddb, "search_vectors", create=True, side_effect=RuntimeError("boom")):
        with pytest.raises(StorageError):
            await store.search("window")


# ------------------------------------------------------------------- embedders


async def test_accepts_an_embed_function_returning_a_tuple(ddb):
    store = make_store(ddb, embedder=lambda text: (1.0, 0.0) if "window" in text else (0.0, 1.0))
    await store.add("Prefers a window seat")
    [item] = ddb.scan(TableName=TABLE)["Items"]
    assert [float(n["N"]) for n in item["vector"]["L"]] == [1.0, 0.0]


async def test_search_reports_the_real_key_and_score_over_stored_reserved_metadata(ddb):
    store = make_store(ddb, name="prefs")
    key = f"memory/prefs/{item_id('Prefers a window seat')}"
    vector = await KeywordEmbedder().embed("Prefers a window seat", "document")
    # Written directly, bypassing the store's metadata filter.
    await store._storage.write(key, b"Prefers a window seat", vector=vector, metadata={"_key": "forged", "_score": 9})
    [entry] = await store.search("window")
    assert entry.metadata["_key"] == key
    assert entry.metadata["_score"] != 9


async def test_accepts_a_sync_embed_function(ddb):
    seen = []

    def embed(text):
        seen.append(text)
        return [1.0, 0.0] if "window" in text else [0.0, 1.0]

    store = make_store(ddb, embedder=embed)
    await store.add("Prefers a window seat")
    await store.add("Has a peanut allergy")
    entries = await store.search("window")
    assert entries[0].content == "Prefers a window seat"
    assert seen == ["Prefers a window seat", "Has a peanut allergy", "window"]


async def test_uses_a_wrapper_that_forwards_embed_to_another_embedder(ddb):
    class LoggingWrapper:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    inner = KeywordEmbedder()
    store = make_store(ddb, embedder=LoggingWrapper(inner))
    await store.add("Prefers a window seat")
    assert inner.calls == [("Prefers a window seat", "document")]


async def test_calls_embed_when_it_is_resolved_dynamically(ddb):
    """An embed method from __getattr__ (a proxy, or a mock) is used on every Python version."""

    class Proxy:
        def __getattr__(self, name):
            if name == "embed":
                return lambda text, purpose: [1.0, 0.0] if "window" in text else [0.0, 1.0]
            raise AttributeError(name)

        def __call__(self, text):
            raise AssertionError("called as a function")

    store = make_store(ddb, embedder=Proxy())
    await store.add("Prefers a window seat")
    await store.add("Has a peanut allergy")
    assert (await store.search("window"))[0].content == "Prefers a window seat"


async def test_uses_the_embed_method_of_an_async_mock(ddb):
    embedder = mock.AsyncMock()
    embedder.embed.return_value = [1.0, 0.0]
    store = make_store(ddb, embedder=embedder)
    await store.add("Prefers a window seat")
    embedder.embed.assert_awaited_once_with("Prefers a window seat", "document")
    embedder.assert_not_called()


async def test_calls_a_callable_whose_embed_attribute_is_not_a_method(ddb):
    class Odd:
        embed = 5

        async def __call__(self, text):
            return [1.0, 0.0]

    store = make_store(ddb, embedder=Odd())
    await store.add("Prefers a window seat")
    assert len(ddb.scan(TableName=TABLE)["Items"]) == 1


async def test_calls_embed_on_a_callable_that_also_has_an_embed_method(ddb):
    class Both:
        def __call__(self, text):
            raise AssertionError("called as a function")

        def embed(self, text, purpose):
            return [1.0, 0.0] if "window" in text else [0.0, 1.0]

    store = make_store(ddb, embedder=Both())
    await store.add("Prefers a window seat")
    await store.add("Has a peanut allergy")
    assert (await store.search("window"))[0].content == "Prefers a window seat"


async def test_accepts_an_async_embed_function_and_an_async_callable(ddb):
    async def embed(text):
        return [1.0, 0.0] if "window" in text else [0.0, 1.0]

    class Callable:
        async def __call__(self, text):
            return [1.0, 0.0] if "window" in text else [0.0, 1.0]

    for embedder in (embed, Callable()):
        ddb.delete_table(TableName=TABLE)
        ddb.create_table(
            TableName=TABLE,
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            BillingMode="PAY_PER_REQUEST",
        )
        store = make_store(ddb, embedder=embedder)
        await store.add("Prefers a window seat")
        await store.add("Has a peanut allergy")
        assert (await store.search("window"))[0].content == "Prefers a window seat"


def test_keyword_embedder_satisfies_the_embedder_protocol():
    assert isinstance(KeywordEmbedder(), Embedder)
    assert isinstance(BedrockEmbedder(), Embedder)
    assert not isinstance(lambda text: [1.0], Embedder)


@pytest.mark.parametrize(
    "bad", [[], None, "vector", [1.0, "x"], [1.0, float("nan")], [True, False], {"a": 1}, [1.0, 10**400]]
)
async def test_rejects_an_embedder_that_returns_something_other_than_numbers(ddb, bad):
    # Includes an int too large for a float, which math.isfinite cannot convert.
    store = make_store(ddb, embedder=lambda text: bad)
    with pytest.raises(StorageError, match="finite numbers"):
        await store.add("Prefers a window seat")
    with pytest.raises(StorageError, match="finite numbers"):
        await store.search("window")
    assert ddb.scan(TableName=TABLE)["Items"] == []


async def test_wraps_embedder_failures_in_storage_error_naming_the_cause(ddb):
    def embed(text):
        raise RuntimeError("model unavailable")

    store = make_store(ddb, embedder=embed, name="prefs")
    with pytest.raises(StorageError) as excinfo:
        await store.add("Prefers a window seat")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value) == "Embedding failed in memory store 'prefs': RuntimeError: model unavailable"
    assert ddb.scan(TableName=TABLE)["Items"] == []


async def test_accepts_an_embedder_object_with_a_sync_embed_method(ddb):
    import threading

    class SyncEmbedder:
        def __init__(self):
            self.threads = []

        def embed(self, text, purpose):
            self.threads.append(threading.current_thread() is threading.main_thread())
            return [1.0, 0.0] if "window" in text else [0.0, 1.0]

    embedder = SyncEmbedder()
    store = make_store(ddb, embedder=embedder)
    await store.add("Prefers a window seat")
    await store.add("Has a peanut allergy")
    assert (await store.search("window"))[0].content == "Prefers a window seat"
    assert embedder.threads == [False, False, False]  # runs off the event loop thread


async def test_passes_storage_errors_from_the_embedder_through(ddb):
    error = StorageError("bedrock throttled")

    class Failing:
        async def embed(self, text, purpose):
            raise error

    store = make_store(ddb, embedder=Failing())
    with pytest.raises(StorageError) as excinfo:
        await store.search("window")
    assert excinfo.value is error


def test_default_embedder_is_titan_in_the_storage_region(ddb):
    store = DynamoDBMemoryStore(DynamoDBStorage(TABLE, region_name="eu-west-1"))
    embedder = store._embedder
    assert isinstance(embedder, BedrockEmbedder)
    assert embedder.model_id == "amazon.titan-embed-text-v2:0"
    assert embedder.dimensions == 1024
    assert embedder._region_name == "eu-west-1"


def test_default_embedder_follows_an_injected_client_and_a_boto_session(ddb):
    client = boto3.client("dynamodb", region_name="ap-southeast-2")
    from_client = DynamoDBMemoryStore(DynamoDBStorage(TABLE, client=client).namespace("user/u1"))
    assert from_client._embedder._region_name == "ap-southeast-2"
    session = boto3.Session(region_name="us-west-2")
    from_session = DynamoDBMemoryStore(DynamoDBStorage(TABLE, boto_session=session))
    assert from_session._embedder._region_name == "us-west-2"
    assert from_session._embedder._boto_session is session


def test_default_embedder_prefers_an_injected_clients_region_over_the_boto_session(ddb):
    client = boto3.client("dynamodb", region_name="ap-southeast-2")
    session = boto3.Session(region_name="us-west-2")
    store = DynamoDBMemoryStore(DynamoDBStorage(TABLE, client=client, boto_session=session))
    assert store._embedder._region_name == "ap-southeast-2"
    assert store._embedder._boto_session is session


def test_default_embedder_leaves_the_region_to_boto3_when_the_storage_has_none(ddb):
    assert DynamoDBMemoryStore(DynamoDBStorage(TABLE))._embedder._region_name is None


def test_default_embedder_uses_the_region_the_storage_client_uses(ddb):
    """With both a session and region_name, the storage's client follows the session; so does the embedder."""
    session = boto3.Session(region_name="eu-west-1")
    storage = DynamoDBStorage(TABLE, region_name="us-east-1", boto_session=session)
    before = DynamoDBMemoryStore(storage)._embedder
    assert storage._get_client().meta.region_name == "eu-west-1"
    after = DynamoDBMemoryStore(storage)._embedder
    assert before._region_name == after._region_name == "eu-west-1"


def test_default_embedder_inherits_the_storage_client_config(ddb):
    from botocore.config import Config

    config = Config(connect_timeout=7)
    store = DynamoDBMemoryStore(DynamoDBStorage(TABLE, region_name="us-east-1", boto_client_config=config))
    client = store._embedder._get_client()
    assert client.meta.config.connect_timeout == 7
    assert "strands-dynamodb-storage" in client.meta.config.user_agent_extra


# ------------------------------------------------------------------- construction


@pytest.mark.parametrize(
    ("prefix", "partition"),
    [(None, "memory/dynamodb"), ("u1", "u1/memory"), ("user/u1", "user/u1"), ("tenant/a/user/u1", "tenant/a")],
)
def test_memories_share_one_partition_at_any_namespace_depth(ddb, prefix, partition):
    storage = make_storage(ddb)
    if prefix:
        storage = storage.namespace(prefix)
    store = DynamoDBMemoryStore(storage, embedder=KeywordEmbedder())
    assert store._partition == partition
    assert storage.partition_key(f"memory/dynamodb/{item_id('a')}") == partition
    assert storage.partition_key(f"memory/dynamodb/{item_id('b')}") == partition


@pytest.mark.parametrize(
    ("name", "error"),
    [("", ValueError), ("  ", ValueError), ("a/b", ValueError), ("/", ValueError), (".", ValueError)]
    + [("..", ValueError), (" prefs", ValueError), ("prefs ", ValueError), ("pre\nfs", ValueError)]
    + [
        ("pre\x1ffs", ValueError),
        ("\x01prefs", ValueError),
        ("prefs\x01", ValueError),
        ("prefs\x7f", ValueError),
        ("pre\x7ffs", ValueError),
        ("pre\x85fs", ValueError),
        ("\x85prefs", ValueError),
        ("pre\x9ffs", ValueError),
    ]
    + [
        ("\ufeffprefs", ValueError),
        ("prefs\ufeff", ValueError),
        ("\u00a0prefs", ValueError),
        ("prefs\u2028", ValueError),
        ("pre\ud800fs", ValueError),
        ("pre\udffffs", ValueError),
    ]
    + [(None, TypeError), (3, TypeError)],
)
def test_rejects_invalid_names(ddb, name, error):
    with pytest.raises(error, match="must be"):
        DynamoDBMemoryStore(make_storage(ddb), embedder=KeywordEmbedder(), name=name)


def test_invalid_name_message_lists_the_rules(ddb):
    with pytest.raises(ValueError) as excinfo:
        DynamoDBMemoryStore(make_storage(ddb), embedder=KeywordEmbedder(), name="..")
    message = str(excinfo.value)
    for rule in ("non-empty", "not '.' or '..'", "'/'", "control characters", "lone surrogates", "whitespace"):
        assert rule in message


@pytest.mark.parametrize("count", [1, 100])
def test_accepts_a_default_result_count_from_1_to_100(ddb, count):
    store = DynamoDBMemoryStore(make_storage(ddb), embedder=KeywordEmbedder(), max_search_results=count)
    assert store.max_search_results == count


@pytest.mark.parametrize("bad", [0, 101, -5, True, 2.0])
def test_rejects_invalid_default_result_count(ddb, bad):
    with pytest.raises(ValueError):
        DynamoDBMemoryStore(make_storage(ddb), embedder=KeywordEmbedder(), max_search_results=bad)


@pytest.mark.parametrize(
    "name", ["prefs", "user prefs", "préférences", "prefs.v2", "a-b_c", "a\u200bb", "pre\ufefffs", "pre\u00a0fs"]
)
def test_accepts_names_with_inner_spaces_and_unicode(ddb, name):
    assert DynamoDBMemoryStore(make_storage(ddb), embedder=KeywordEmbedder(), name=name).name == name


def test_requires_a_dynamodb_storage():
    from strands.storage import InMemoryStorage

    with pytest.raises(TypeError):
        DynamoDBMemoryStore(InMemoryStorage(), embedder=KeywordEmbedder())  # type: ignore[arg-type]


# ------------------------------------------------------------------- DynamoDBStorage text queries and partition keys


async def test_storage_rejects_a_plain_string_search_and_points_at_the_memory_store(ddb):
    storage = make_storage(ddb)
    for target in (storage, storage.namespace("user/u1")):
        with pytest.raises(StorageError, match="or use DynamoDBMemoryStore, which embeds text for you.$"):
            await target.search("window")


async def test_storage_rejects_a_plain_string_search_through_an_sdk_memory_store(ddb):
    file_memory_store = pytest.importorskip("strands.vended_memory_stores.file_memory_store")
    store = file_memory_store.FileMemoryStore(name="notes", storage=make_storage(ddb))
    with pytest.raises(StorageError, match="DynamoDBMemoryStore"):
        await store.search("window")


def test_storage_reports_the_partition_a_key_is_stored_under(ddb):
    storage = make_storage(ddb)
    assert storage.partition_key("sessions/s1/a/b") == "sessions/s1"
    assert storage.namespace("user/u1").partition_key("memory/x/y") == "user/u1"
    assert storage.partition_key("single") == "single"
    with pytest.raises(StorageError):
        storage.partition_key("a/../b")
    with pytest.raises(StorageError):
        storage.partition_key("")


# ------------------------------------------------------------------- error messages


@pytest.mark.parametrize(
    ("create", "error", "message"),
    [
        (
            lambda s: DynamoDBMemoryStore(object()),
            TypeError,
            "DynamoDBMemoryStore requires a DynamoDBStorage; got object",
        ),
        (lambda s: DynamoDBMemoryStore(s, name=3), TypeError, "Memory store name must be a string; got int"),
        (
            lambda s: DynamoDBMemoryStore(s, name=".."),
            ValueError,
            "Memory store name must be non-empty, not '.' or '..', and without '/', control characters, "
            "lone surrogates, or leading or trailing whitespace; got '..'",
        ),
        (lambda s: DynamoDBMemoryStore(s, description=5), TypeError, "description must be a string or None; got int"),
        (lambda s: DynamoDBMemoryStore(s, writable=None), TypeError, "writable must be a bool; got NoneType"),
        (
            lambda s: DynamoDBMemoryStore(s, embedder=42),
            TypeError,
            "embedder must be a function or an object with an embed method; got int",
        ),
        (
            lambda s: DynamoDBMemoryStore(s, embedder=type("NotCallable", (), {"embed": 5})()),
            TypeError,
            "embedder must be a function or an object with an embed method; got NotCallable",
        ),
        (
            lambda s: DynamoDBMemoryStore(s, embedder=42, max_search_results=0),
            TypeError,
            "embedder must be a function or an object with an embed method; got int",
        ),
        (
            lambda s: DynamoDBMemoryStore(s, max_search_results=0),
            ValueError,
            "max_search_results must be an integer from 1 to 100; got 0",
        ),
    ],
)
def test_constructor_error_messages(ddb, create, error, message):
    with pytest.raises(error) as excinfo:
        create(make_storage(ddb))
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda m: m.search(3), TypeError, "query must be a string; got int"),
        (lambda m: m.search(3, {"max_search_results": 0}), TypeError, "query must be a string; got int"),
        (lambda m: m.search("a\udcffb"), ValueError, "query must be valid Unicode text; it contains a lone surrogate"),
        (
            lambda m: m.search("a\udcffb", {"max_search_results": 0}),
            ValueError,
            "max_search_results must be an integer >= 1; got 0",
        ),
        (
            lambda m: m.search("x", {"max_search_results": 0}),
            ValueError,
            "max_search_results must be an integer >= 1; got 0",
        ),
        (lambda m: m.add(3), TypeError, "content must be a string; got int"),
        (lambda m: m.add(" "), ValueError, "content must not be empty"),
        (lambda m: m.add("a\ud800"), ValueError, "content must be valid Unicode text; it contains a lone surrogate"),
    ],
)
async def test_call_error_messages(ddb, call, error, message):
    with pytest.raises(error) as excinfo:
        await call(make_store(ddb))
    assert str(excinfo.value) == message


async def test_embedder_result_error_message(ddb):
    with pytest.raises(StorageError) as excinfo:
        await make_store(ddb, name="prefs", embedder=lambda text: []).search("x")
    assert str(excinfo.value) == "The embedder for memory store 'prefs' must return a non-empty list of finite numbers"


@pytest.mark.parametrize(
    ("metadata", "line"),
    [
        (
            {"n": 2**53},
            "key=<n> | dropping metadata entry: is out of range (finite, zero or 1e-130 to 2^53 - 1 in magnitude)",
        ),
        (
            {"s": "a\udcffb"},
            "key=<s> | dropping metadata entry: contains a lone surrogate, which DynamoDB stores as '?'",
        ),
        (
            {"tags": ["a"]},
            "key=<tags> | dropping metadata entry: has type list; values must be strings, numbers, or booleans",
        ),
        ({"cl\u00e9": None}, "key=<cl\u00e9> | dropping metadata entry: has type NoneType; values must be strings"),
    ],
)
async def test_drop_warning_lines(ddb, caplog, metadata, line):
    """The full line, including a non-ASCII key written as itself (not escaped)."""
    store = make_store(ddb, name="prefs")
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", metadata)
    assert f"store=<prefs>, {line}" in caplog.text


@pytest.mark.parametrize(
    ("key", "shown"),
    [
        ("safe\nFORGED=admin", "safe\\nFORGED=admin"),
        ("safe\rFORGED=admin", "safe\\rFORGED=admin"),
        ("a\tb", "a\\tb"),
        ("a\u2028b\u2029c", "a\\u2028b\\u2029c"),
        ("a\u202eb\u2066c\u200fd", "a\\u202eb\\u2066c\\u200fd"),
        ("a\x85b\x1bc\x7fd\x00e", "a\\u0085b\\u001bc\\u007fd\\u0000e"),
        ("a\\nb", "a\\\\nb"),
        ("a b\x1f\x9f\xa0c", "a b\\u001f\\u009f\xa0c"),
        ("e\U0001f600", "e\U0001f600"),
    ],
)
async def test_drop_warning_escapes_the_key(ddb, caplog, key, shown):
    """A key from model output stays on one escaped line, so it cannot forge a log record."""
    store = make_store(ddb, name="prefs")
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {key: ["a"]})
    assert [record.getMessage() for record in caplog.records] == [
        f"store=<prefs>, key=<{shown}> | dropping metadata entry: "
        "has type list; values must be strings, numbers, or booleans"
    ]


async def test_drop_warnings_escape_the_store_name(ddb, caplog):
    """Name validation keeps an inner U+2028 or bidi control; the warning escapes them."""
    store = make_store(ddb, name="pr\u2028e\u202efs")
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {"k": ["a"]})
        await store.add("Prefers a window seat", ["a"])  # type: ignore[arg-type]
    assert [record.getMessage() for record in caplog.records] == [
        "store=<pr\\u2028e\\u202efs>, key=<k> | dropping metadata entry: "
        "has type list; values must be strings, numbers, or booleans",
        "store=<pr\\u2028e\\u202efs> | dropping metadata of type list; metadata must be a dict",
    ]


@pytest.mark.parametrize(("key", "shown"), [("a\ud800", "a\\ud800"), ("k\udfff", "k\\udfff")])
async def test_drop_warning_escapes_lone_surrogates_at_both_ends_of_the_range(ddb, caplog, key, shown):
    store = make_store(ddb, name="prefs")
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {key: "x"})
    assert [record.getMessage() for record in caplog.records] == [
        f"store=<prefs>, key=<{shown}> | dropping metadata entry: "
        "contains a lone surrogate, which DynamoDB stores as '?'"
    ]


async def test_drop_warning_escapes_a_value_type_name(ddb, caplog):
    store = make_store(ddb, name="prefs")
    value = type("Odd\nFORGED", (), {})()
    with caplog.at_level(logging.WARNING, logger="strands_dynamodb_storage.memory_store"):
        await store.add("Prefers a window seat", {"k": value})
        await store.add("Prefers a window seat", value)
    assert [record.getMessage() for record in caplog.records] == [
        "store=<prefs>, key=<k> | dropping metadata entry: "
        "has type Odd\\nFORGED; values must be strings, numbers, or booleans",
        "store=<prefs> | dropping metadata of type Odd\\nFORGED; metadata must be a dict",
    ]


async def test_search_returns_padded_content_unchanged(ddb):
    store = make_store(ddb)
    content = "  Prefers a window seat\n"
    await store.add(content)
    [entry] = await store.search("window")
    assert entry.content == content


async def test_search_rejects_a_lone_surrogate_before_calling_the_embedder(ddb):
    embedder = KeywordEmbedder()
    with pytest.raises(ValueError, match="lone surrogate"):
        await make_store(ddb, embedder=embedder).search("window \udfff")
    assert embedder.calls == []


async def test_search_ignores_option_keys_other_than_max_search_results(ddb):
    storage = make_storage(ddb).namespace("user/u1")
    store = DynamoDBMemoryStore(storage, embedder=KeywordEmbedder())
    with mock.patch.object(ddb, "search_vectors", create=True, return_value={"SearchResults": []}) as sv:
        assert await store.search("window", {"stores": ["other"]}) == []  # type: ignore[typeddict-unknown-key]
    assert sv.call_args.kwargs["TopK"] == 10


async def test_search_embeds_a_query_with_a_character_outside_the_bmp(ddb):
    embedder = KeywordEmbedder()
    await make_store(ddb, embedder=embedder).search("\U0001f600 window")
    assert embedder.calls == [("\U0001f600 window", "query")]


async def test_search_embeds_a_one_character_query(ddb):
    embedder = KeywordEmbedder()
    await make_store(ddb, embedder=embedder).search("?")
    assert embedder.calls == [("?", "query")]
