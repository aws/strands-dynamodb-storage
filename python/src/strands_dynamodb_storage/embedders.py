# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text embedders for :class:`~strands_dynamodb_storage.DynamoDBMemoryStore`."""

from __future__ import annotations

import asyncio
import builtins
import json
import math
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Optional, Protocol, Union, runtime_checkable

from strands.types.exceptions import StorageError

from .dynamodb_storage import _client_config

EmbeddingPurpose = Literal["document", "query"]
"""What a text is embedded for: ``"document"`` for a memory being stored, ``"query"`` for a search."""


_Vector = Union[builtins.list[float], tuple[float, ...]]
"""An embedding vector as an embedder returns it: a list or tuple of floats."""


@runtime_checkable
class Embedder(Protocol):
    """Turns text into an embedding vector.

    ``purpose`` lets a model that embeds stored text and search queries differently
    (Cohere Embed) use the matching input type. A model that does not distinguish
    the two ignores it. ``embed`` may be a plain method or a coroutine method; a
    plain method runs in a worker thread.
    """

    def embed(self, text: str, purpose: EmbeddingPurpose) -> Union[_Vector, Awaitable[_Vector]]:
        """Return the embedding vector of ``text``."""
        ...


EmbedFunction = Callable[[str], Union[_Vector, Awaitable[_Vector]]]
"""A plain function from text to vector, sync or async, used for both memories and queries.

It returns a list or tuple of floats; convert a NumPy array with ``.tolist()``.
"""

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
"""Default :class:`BedrockEmbedder` model: Amazon Titan Text Embeddings V2."""

BedrockModelFamily = Literal["amazon.titan-embed-text-v2", "cohere.embed-v3", "cohere.embed-v4"]
"""Request format of a Bedrock embedding model, for identifiers that do not name the model."""

_TITAN_V2: BedrockModelFamily = "amazon.titan-embed-text-v2"
_COHERE_V3: BedrockModelFamily = "cohere.embed-v3"
_COHERE_V4: BedrockModelFamily = "cohere.embed-v4"
_SUPPORTED_DIMENSIONS: dict[BedrockModelFamily, tuple[int, ...]] = {
    _TITAN_V2: (256, 512, 1024),
    _COHERE_V3: (1024,),
    _COHERE_V4: (256, 512, 1024, 1536),
}
_COHERE_INPUT_TYPE = {"document": "search_document", "query": "search_query"}


_MODEL_PREFIXES: tuple[tuple[str, BedrockModelFamily], ...] = (
    ("amazon.titan-embed-text-v2", _TITAN_V2),
    ("cohere.embed-english-v3", _COHERE_V3),
    ("cohere.embed-multilingual-v3", _COHERE_V3),
    ("cohere.embed-v4", _COHERE_V4),
)

_MISSING = object()


def _is_finite_number(value: Any) -> bool:
    """Whether ``value`` is an int or float (not a bool) that converts to a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _model_family(model_id: str) -> Optional[BedrockModelFamily]:
    """Map a Bedrock model ID, inference profile ID, or ARN to its request format.

    Returns ``None`` for a model this embedder does not recognize, including
    application inference profiles and provisioned models, whose ARNs do not name
    the underlying model.
    """
    base = model_id.rsplit("/", 1)[-1] if model_id.startswith("arn:") else model_id
    # A cross-Region inference profile ID prefixes the model ID with a geography,
    # for example "us.cohere.embed-v4:0".
    geography, separator, rest = base.partition(".")
    if separator and geography and geography not in ("amazon", "cohere"):
        base = rest
    for prefix, family in _MODEL_PREFIXES:
        # The model name ends at the end of the ID or at a ":" before the version.
        if base == prefix or base.startswith(prefix + ":"):
            return family
    return None


class BedrockEmbedder:
    """Embeds text with an Amazon Bedrock embedding model.

    Supports Amazon Titan Text Embeddings V2 (the default) and Cohere Embed v3
    (English and multilingual) and v4, by model ID, cross-Region inference profile
    ID, or ARN. For an application inference profile or a provisioned model, whose
    ARN does not name the model, pass ``model_family``. Cohere models embed stored
    memories and search queries with separate input types; Titan does not
    distinguish them. For any other model, pass an embed function to
    :class:`DynamoDBMemoryStore` instead.

    ``dimensions`` must equal the ``Dimensions`` of the table's vector index.

    Requires ``bedrock:InvokeModel`` on the model (and, for an inference profile,
    on the profile).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        *,
        dimensions: int = 1024,
        model_family: Optional[BedrockModelFamily] = None,
        region_name: Optional[str] = None,
        client: Any = None,
        boto_session: Any = None,
        boto_client_config: Any = None,
    ) -> None:
        """Create an embedder. No AWS call is made until the first :meth:`embed`.

        Args:
            model_id: Bedrock model ID, cross-Region inference profile ID, or ARN.
            dimensions: Length of the returned vectors. Titan Text Embeddings V2
                supports 256, 512, and 1024; Cohere Embed v3 supports 1024; Cohere
                Embed v4 supports 256, 512, 1024, and 1536.
            model_family: The model's request format, required when ``model_id`` is
                an application inference profile or provisioned model ARN, which
                does not name the model.
            region_name: AWS Region of the Bedrock runtime endpoint. Cannot be
                combined with ``client``.
            client: Pre-configured ``bedrock-runtime`` client.
            boto_session: boto3 session to create the client from.
            boto_client_config: botocore ``Config`` for the created client.

        Raises:
            TypeError: If ``model_id`` is not a string.
            ValueError: If the model is not supported, ``model_family`` is unknown or
                contradicts ``model_id``, or ``dimensions`` is not one the model supports.
            StorageError: If both ``client`` and ``region_name`` are set.
        """
        if not isinstance(model_id, str):
            raise TypeError(f"model_id must be a string; got {type(model_id).__name__}")
        detected = _model_family(model_id)
        if model_family is not None and (
            not isinstance(model_family, str) or model_family not in _SUPPORTED_DIMENSIONS
        ):
            raise ValueError(f"model_family must be one of {list(_SUPPORTED_DIMENSIONS)}; got {model_family!r}")
        if model_family is not None and detected is not None and model_family != detected:
            raise ValueError(f"model_family '{model_family}' does not match model '{model_id}' ({detected})")
        family = model_family or detected
        if family is None:
            raise ValueError(
                f"Unsupported Bedrock embedding model '{model_id}': BedrockEmbedder supports "
                "amazon.titan-embed-text-v2, cohere.embed-english-v3, cohere.embed-multilingual-v3, "
                "and cohere.embed-v4. For an application inference profile or provisioned model ARN, "
                "pass model_family; for other models, pass an embed function to DynamoDBMemoryStore."
            )
        supported = _SUPPORTED_DIMENSIONS[family]
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions not in supported:
            raise ValueError(f"Model '{model_id}' supports dimensions {list(supported)}; got {dimensions!r}")
        if client is not None and region_name is not None:
            raise StorageError("Cannot specify both client and region_name; configure the region on the client instead")
        self.model_id = model_id
        self.dimensions = dimensions
        self._family = family
        self._region_name = region_name
        self._client = client
        self._boto_session = boto_session
        self._boto_client_config = boto_client_config

    async def embed(self, text: str, purpose: EmbeddingPurpose = "document") -> builtins.list[float]:
        """Return the embedding vector of ``text``.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If ``text`` is empty or ``purpose`` is not ``"document"`` or ``"query"``.
            StorageError: If the Bedrock call fails or returns an unexpected response.
        """
        if not isinstance(text, str):
            raise TypeError(f"BedrockEmbedder.embed requires a string; got {type(text).__name__}")
        if text == "":
            raise ValueError("BedrockEmbedder.embed requires a non-empty string")
        if not isinstance(purpose, str) or purpose not in _COHERE_INPUT_TYPE:
            raise ValueError(f"purpose must be 'document' or 'query'; got {purpose!r}")
        body = json.dumps(self._request_body(text, purpose))

        def invoke(client: Any) -> Any:
            response = client.invoke_model(
                modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
            )
            return json.loads(response["body"].read())

        try:
            payload = await asyncio.to_thread(invoke, self._get_client())
        except Exception as error:
            # The service message (input too long, access denied, throttled) is the actionable
            # part, and the Memory Manager logs only the error's text, so it goes in the message.
            raise StorageError(
                f"Bedrock embedding request to model '{self.model_id}' failed: {type(error).__name__}: {error}"
            ) from error
        return self._parse(payload)

    def _request_body(self, text: str, purpose: EmbeddingPurpose) -> dict[str, Any]:
        if self._family == _TITAN_V2:
            return {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        body: dict[str, Any] = {
            "texts": [text],
            "input_type": _COHERE_INPUT_TYPE[purpose],
            "embedding_types": ["float"],
        }
        if self._family == _COHERE_V4:
            body["output_dimension"] = self.dimensions
        return body

    def _parse(self, payload: Any) -> builtins.list[float]:
        vector: Any = _MISSING
        if self._family == _TITAN_V2:
            if isinstance(payload, dict):
                vector = payload.get("embedding", _MISSING)
        else:
            embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
            floats = embeddings.get("float") if isinstance(embeddings, dict) else None
            if isinstance(floats, list) and floats:
                vector = floats[0]
        if vector is _MISSING:
            raise StorageError(f"Bedrock model '{self.model_id}' returned a response without an embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != self.dimensions
            or not all(_is_finite_number(v) for v in vector)
        ):
            raise StorageError(
                f"Bedrock model '{self.model_id}' returned an embedding that is not {self.dimensions} finite numbers"
            )
        return [float(v) for v in vector]

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            session = self._boto_session if self._boto_session is not None else boto3.Session()
            self._client = session.client(
                "bedrock-runtime",
                region_name=self._region_name,
                config=_client_config(self._boto_client_config),
            )
        return self._client
