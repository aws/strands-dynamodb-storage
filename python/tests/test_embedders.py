# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BedrockEmbedder request formats, response parsing, and configuration."""

from __future__ import annotations

import io
import json

import boto3
import pytest
from strands.types.exceptions import StorageError

from strands_dynamodb_storage import DEFAULT_EMBEDDING_MODEL_ID, BedrockEmbedder


class FakeBedrock:
    """Records invoke_model calls and replies with a canned JSON body."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"body": io.BytesIO(json.dumps(self.reply).encode())}

    def body(self, index=-1):
        return json.loads(self.calls[index]["body"])


def titan_reply(dims=1024):
    return {"embedding": [0.5] * dims, "embeddingsByType": {"float": [0.5] * dims}, "inputTextTokenCount": 3}


def cohere_reply(dims=1024):
    return {"id": "x", "texts": ["t"], "embeddings": {"float": [[0.25] * dims]}, "response_type": "embeddings_by_type"}


# ------------------------------------------------------------------- request formats


async def test_titan_v2_request_and_response():
    fake = FakeBedrock(titan_reply())
    embedder = BedrockEmbedder(client=fake)
    vector = await embedder.embed("Prefers a window seat", "query")
    assert vector == [0.5] * 1024
    [call] = fake.calls
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    assert call["contentType"] == "application/json"
    assert call["accept"] == "application/json"
    assert fake.body() == {"inputText": "Prefers a window seat", "dimensions": 1024, "normalize": True}


@pytest.mark.parametrize("dims", [256, 512, 1024])
async def test_titan_v2_dimensions(dims):
    fake = FakeBedrock(titan_reply(dims))
    assert len(await BedrockEmbedder(dimensions=dims, client=fake).embed("text")) == dims
    assert fake.body()["dimensions"] == dims


@pytest.mark.parametrize("model_id", ["cohere.embed-english-v3", "cohere.embed-multilingual-v3"])
async def test_cohere_v3_uses_input_types_and_no_output_dimension(model_id):
    fake = FakeBedrock(cohere_reply())
    embedder = BedrockEmbedder(model_id, client=fake)
    assert await embedder.embed("Prefers a window seat", "document") == [0.25] * 1024
    assert fake.body() == {
        "texts": ["Prefers a window seat"],
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    await embedder.embed("which seat?", "query")
    assert fake.body()["input_type"] == "search_query"


@pytest.mark.parametrize("model_id", ["cohere.embed-v4:0", "us.cohere.embed-v4:0", "global.cohere.embed-v4:0"])
async def test_cohere_v4_sends_output_dimension(model_id):
    fake = FakeBedrock(cohere_reply(1536))
    embedder = BedrockEmbedder(model_id, dimensions=1536, client=fake)
    assert len(await embedder.embed("text", "query")) == 1536
    assert fake.calls[-1]["modelId"] == model_id
    assert fake.body() == {
        "texts": ["text"],
        "input_type": "search_query",
        "embedding_types": ["float"],
        "output_dimension": 1536,
    }


async def test_embed_defaults_to_document_purpose():
    fake = FakeBedrock(cohere_reply())
    await BedrockEmbedder("cohere.embed-english-v3", client=fake).embed("text")
    assert fake.body()["input_type"] == "search_document"


# ------------------------------------------------------------------- model IDs and dimensions


@pytest.mark.parametrize(
    "model_id",
    [
        "amazon.titan-embed-text-v2:0",
        "amazon.titan-embed-text-v2:0:8k",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
        "cohere.embed-v4:0",
        "us.cohere.embed-v4:0",
        "eu.cohere.embed-v4:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.cohere.embed-v4:0",
    ],
)
def test_accepts_supported_model_ids(model_id):
    BedrockEmbedder(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-image-v1",
        "amazon.nova-2-multimodal-embeddings-v1:0",
        "cohere.command-r-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123",
        "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123",
        "",
        ".cohere.embed-v4:0",
        "cohere.embed-v5:0",
        "amazon.titan-embed-text-v20",
        "cohere.embed-english-v30",
        "amazon.titan-embed-text-v2-foo",
        "Amazon.Titan-Embed-Text-V2:0",
        "x/amazon.titan-embed-text-v2:0",
    ],
)
def test_rejects_unsupported_model_ids(model_id):
    with pytest.raises(ValueError, match="Unsupported Bedrock embedding model"):
        BedrockEmbedder(model_id)


@pytest.mark.parametrize(
    ("model_id", "dims"),
    [
        ("amazon.titan-embed-text-v2:0", 1536),
        ("amazon.titan-embed-text-v2:0", 300),
        ("cohere.embed-english-v3", 512),
        ("cohere.embed-v4:0", 2048),
        ("amazon.titan-embed-text-v2:0", True),
        ("amazon.titan-embed-text-v2:0", 1024.0),
    ],
)
def test_rejects_dimensions_the_model_does_not_support(model_id, dims):
    with pytest.raises(ValueError, match="supports dimensions"):
        BedrockEmbedder(model_id, dimensions=dims)


def test_rejects_client_with_region():
    with pytest.raises(StorageError, match="both client and region_name"):
        BedrockEmbedder(client=FakeBedrock(titan_reply()), region_name="us-east-1")
    with pytest.raises(StorageError, match="both client and region_name"):
        BedrockEmbedder(client=FakeBedrock(titan_reply()), region_name="")


@pytest.mark.parametrize("model_id", [None, 3, b"amazon.titan-embed-text-v2:0"])
def test_rejects_a_model_id_that_is_not_a_string(model_id):
    with pytest.raises(TypeError, match="model_id must be a string"):
        BedrockEmbedder(model_id)  # type: ignore[arg-type]


def test_default_model_id_is_exported():
    assert DEFAULT_EMBEDDING_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert BedrockEmbedder().model_id == DEFAULT_EMBEDDING_MODEL_ID


APP_PROFILE = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"


@pytest.mark.parametrize(
    ("family", "reply", "body"),
    [
        (
            "amazon.titan-embed-text-v2",
            titan_reply(),
            {"inputText": "text", "dimensions": 1024, "normalize": True},
        ),
        (
            "cohere.embed-v3",
            cohere_reply(),
            {"texts": ["text"], "input_type": "search_query", "embedding_types": ["float"]},
        ),
        (
            "cohere.embed-v4",
            cohere_reply(),
            {"texts": ["text"], "input_type": "search_query", "embedding_types": ["float"], "output_dimension": 1024},
        ),
    ],
)
async def test_model_family_makes_an_application_inference_profile_usable(family, reply, body):
    fake = FakeBedrock(reply)
    embedder = BedrockEmbedder(APP_PROFILE, model_family=family, client=fake)
    assert len(await embedder.embed("text", "query")) == 1024
    assert fake.calls[-1]["modelId"] == APP_PROFILE
    assert fake.body() == body


CUSTOM_MODEL = "arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.titan-embed-text-v2:0/abc123"


def test_custom_model_arn_needs_model_family():
    with pytest.raises(ValueError, match="Unsupported Bedrock embedding model"):
        BedrockEmbedder(CUSTOM_MODEL)
    assert BedrockEmbedder(CUSTOM_MODEL, model_family="amazon.titan-embed-text-v2").model_id == CUSTOM_MODEL


def test_model_family_must_be_known_and_agree_with_the_model_id():
    with pytest.raises(ValueError, match="model_family must be one of"):
        BedrockEmbedder(APP_PROFILE, model_family="titan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_family must be one of"):
        BedrockEmbedder(APP_PROFILE, model_family=["cohere.embed-v4"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        BedrockEmbedder("amazon.titan-embed-text-v2:0", model_family="cohere.embed-v4")
    with pytest.raises(ValueError, match="supports dimensions"):
        BedrockEmbedder(APP_PROFILE, model_family="cohere.embed-v3", dimensions=512)
    assert BedrockEmbedder("cohere.embed-v4:0", model_family="cohere.embed-v4").model_id == "cohere.embed-v4:0"


def test_unsupported_model_error_mentions_model_family():
    with pytest.raises(ValueError, match="pass model_family"):
        BedrockEmbedder(APP_PROFILE)


# ------------------------------------------------------------------- input validation


@pytest.mark.parametrize(("text", "error"), [("", ValueError), (None, TypeError), (b"bytes", TypeError)])
async def test_rejects_empty_or_non_string_text(text, error):
    fake = FakeBedrock(titan_reply())
    with pytest.raises(error):
        await BedrockEmbedder(client=fake).embed(text)
    assert fake.calls == []


async def test_sends_whitespace_text_to_the_model():
    # Titan V2 embeds whitespace-only input; the memory store rejects it before calling.
    fake = FakeBedrock(titan_reply())
    assert len(await BedrockEmbedder(client=fake).embed(" ")) == 1024
    assert fake.body()["inputText"] == " "


@pytest.mark.parametrize("model_id", ["amazon.titan-embed-text-v2:0", "cohere.embed-v4:0"])
@pytest.mark.parametrize("purpose", ["search_query", "DOCUMENT", None, ["document"], {"a": 1}])
async def test_rejects_unknown_purpose(model_id, purpose):
    fake = FakeBedrock(titan_reply())
    with pytest.raises(ValueError, match="purpose must be 'document' or 'query'"):
        await BedrockEmbedder(model_id, client=fake).embed("text", purpose)
    assert fake.calls == []


# ------------------------------------------------------------------- response handling


async def test_wraps_bedrock_errors_in_storage_error_naming_the_cause():
    failure = RuntimeError("ThrottlingException")
    with pytest.raises(StorageError) as excinfo:
        await BedrockEmbedder(client=FakeBedrock(failure)).embed("text")
    assert excinfo.value.__cause__ is failure
    assert str(excinfo.value) == (
        "Bedrock embedding request to model 'amazon.titan-embed-text-v2:0' failed: RuntimeError: ThrottlingException"
    )


async def test_wraps_client_creation_errors_in_storage_error(monkeypatch):
    from botocore.exceptions import NoRegionError

    for variable in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_CONFIG_FILE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
    with pytest.raises(StorageError, match="You must specify a region") as excinfo:
        await BedrockEmbedder(boto_session=boto3.Session()).embed("text")
    assert isinstance(excinfo.value.__cause__, NoRegionError)


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({}, "without an embedding"),
        (None, "without an embedding"),
        ({"embeddingsByType": {"float": [0.5] * 1024}}, "without an embedding"),
        ({"embedding": None}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 512}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 1025}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 1023 + [None]}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 1023 + ["x"]}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 1023 + [float("nan")]}, "not 1024 finite numbers"),
        ({"embedding": [True] * 1024}, "not 1024 finite numbers"),
        ({"embedding": [0.5] * 1023 + [10**400]}, "not 1024 finite numbers"),
    ],
)
async def test_rejects_malformed_titan_responses(reply, message):
    with pytest.raises(StorageError, match=f"model 'amazon.titan-embed-text-v2:0' returned .*{message}"):
        await BedrockEmbedder(client=FakeBedrock(reply)).embed("text")


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"embeddings": [[0.25] * 1024]}, "without an embedding"),
        ({"embeddings": {"float": []}}, "without an embedding"),
        ({"embeddings": {"float": "x"}}, "without an embedding"),
        ({"embeddings": {"float": {0: [0.25] * 1024}}}, "without an embedding"),
        ({"embeddings": None}, "without an embedding"),
        ({"embeddings": {"int8": [[1] * 1024]}}, "without an embedding"),
        ({"embeddings": {"float": [None]}}, "not 1024 finite numbers"),
        ({"embeddings": {"float": [[0.25] * 256]}}, "not 1024 finite numbers"),
        ({"embeddings": {"float": [[0.25] * 1025]}}, "not 1024 finite numbers"),
    ],
)
async def test_rejects_malformed_cohere_responses(reply, message):
    with pytest.raises(StorageError, match=f"model 'cohere.embed-english-v3' returned .*{message}"):
        await BedrockEmbedder("cohere.embed-english-v3", client=FakeBedrock(reply)).embed("text")


# ------------------------------------------------------------------- client creation


def test_creates_a_bedrock_runtime_client_in_the_given_region_with_the_user_agent_marker():
    embedder = BedrockEmbedder(region_name="eu-west-1")
    client = embedder._get_client()
    assert client.meta.service_model.service_name == "bedrock-runtime"
    assert client.meta.region_name == "eu-west-1"
    assert "strands-dynamodb-storage" in client.meta.config.user_agent_extra
    assert embedder._get_client() is client


def test_creates_the_client_from_a_boto_session():
    session = boto3.Session(region_name="ap-southeast-2")
    client = BedrockEmbedder(boto_session=session)._get_client()
    assert client.meta.region_name == "ap-southeast-2"


def test_keeps_a_caller_supplied_user_agent():
    from botocore.config import Config

    client = BedrockEmbedder(region_name="us-east-1", boto_client_config=Config(user_agent_extra="mine"))._get_client()
    assert client.meta.config.user_agent_extra == "mine"


# ------------------------------------------------------------------- error messages

UNSUPPORTED = (
    "Unsupported Bedrock embedding model 'amazon.titan-embed-text-v1': BedrockEmbedder supports "
    "amazon.titan-embed-text-v2, cohere.embed-english-v3, cohere.embed-multilingual-v3, and cohere.embed-v4. "
    "For an application inference profile or provisioned model ARN, pass model_family; for other models, "
    "pass an embed function to DynamoDBMemoryStore."
)


@pytest.mark.parametrize(
    ("create", "error", "message"),
    [
        (lambda: BedrockEmbedder(3), TypeError, "model_id must be a string; got int"),
        (
            lambda: BedrockEmbedder(APP_PROFILE, model_family="titan"),
            ValueError,
            "model_family must be one of ['amazon.titan-embed-text-v2', 'cohere.embed-v3', 'cohere.embed-v4']; "
            "got 'titan'",
        ),
        (
            lambda: BedrockEmbedder("amazon.titan-embed-text-v2:0", model_family="cohere.embed-v4"),
            ValueError,
            "model_family 'cohere.embed-v4' does not match model 'amazon.titan-embed-text-v2:0' "
            "(amazon.titan-embed-text-v2)",
        ),
        (lambda: BedrockEmbedder("amazon.titan-embed-text-v1"), ValueError, UNSUPPORTED),
        (
            lambda: BedrockEmbedder(dimensions=300),
            ValueError,
            "Model 'amazon.titan-embed-text-v2:0' supports dimensions [256, 512, 1024]; got 300",
        ),
        (
            lambda: BedrockEmbedder(client=FakeBedrock(titan_reply()), region_name="us-east-1"),
            StorageError,
            "Cannot specify both client and region_name; configure the region on the client instead",
        ),
    ],
)
def test_constructor_error_messages(create, error, message):
    with pytest.raises(error) as excinfo:
        create()
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((3,), TypeError, "BedrockEmbedder.embed requires a string; got int"),
        (("",), ValueError, "BedrockEmbedder.embed requires a non-empty string"),
        (("x", "search_query"), ValueError, "purpose must be 'document' or 'query'; got 'search_query'"),
    ],
)
async def test_embed_error_messages(args, error, message):
    with pytest.raises(error) as excinfo:
        await BedrockEmbedder(client=FakeBedrock(titan_reply())).embed(*args)
    assert str(excinfo.value) == message
