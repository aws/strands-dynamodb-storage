# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DynamoDBVectorSearchStrategy``: the native vector index behind ``DynamoDBStorage``'s text search."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Literal, Optional

from strands.types.exceptions import StorageError

from .dynamodb_storage import _MAX_TOP_K, DynamoDBStorage, SearchQuery
from .embedders import BedrockEmbedder, EmbeddingPurpose, _as_vector, _call_embedder, _takes_purpose

if TYPE_CHECKING:
    from strands.storage.search import StorageSearchResult

    from .embedders import _AnyEmbedder

DistanceFunction = Literal["COSINE", "EUCLIDEAN", "DOT_PRODUCT"]
"""The ``DistanceFunction`` of the table's vector index."""


class DynamoDBVectorSearchStrategy:
    """Vector search for :class:`DynamoDBStorage`, backed by the table's native vector index.

    The DynamoDB counterpart of the SDK's ``S3VectorSearchStrategy``. Pass it as
    ``search_strategy``: each ``write`` embeds the value and stores its vector on the same item,
    in the same request, and a plain-string ``search`` embeds the query and runs ``SearchVectors``.
    ``delete`` removes the vector with the item. Empty and non-UTF-8 values are stored without a
    vector, which clears any earlier one.

    The table needs the vector index from the README, with ``Dimensions`` matching the embedder
    (1024 for the default, Titan Text Embeddings V2). Searches are pinned to the storage's
    partition, so use a storage namespaced at least two segments deep, as ``FileMemoryStore``'s
    ``memory/<name>`` is. Every write is embedded, so give the strategy a storage of its own rather
    than the agent's. Each value is embedded whole and must fit the embedding model's input. Values
    written before the strategy was configured have no vector until rewritten, or passed to :meth:`index`.

    Example:
        ```python
        from strands.vended_memory_stores.file_memory_store import FileMemoryStore

        storage = DynamoDBStorage("agent-storage", search_strategy=DynamoDBVectorSearchStrategy())
        knowledge = FileMemoryStore(name="facts", storage=storage)
        ```
    """

    def __init__(
        self,
        *,
        embedder: Optional[_AnyEmbedder] = None,
        max_results: int = 10,
        distance_function: DistanceFunction = "COSINE",
    ) -> None:
        """Configure the strategy.

        Args:
            embedder: Any SDK ``Embedder`` (strands-agents 1.64.0 or later), a :class:`BedrockEmbedder`
                or other object with ``embed(text, purpose)``, or a function from text to vector.
                Defaults to ``BedrockEmbedder()`` with the storage's Region, ``boto_session``, and
                ``boto_client_config``, created on first use.
            max_results: Most results one search returns, from 1 to 100.
            distance_function: The vector index's ``DistanceFunction``, so scores can be returned
                with higher meaning closer.

        Raises:
            ValueError: If ``max_results`` is out of range or ``distance_function`` is unknown.
        """
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= _MAX_TOP_K:
            raise ValueError(f"max_results must be an integer from 1 to {_MAX_TOP_K}; got {max_results!r}")
        if distance_function not in ("COSINE", "EUCLIDEAN", "DOT_PRODUCT"):
            raise ValueError(f"distance_function must be COSINE, EUCLIDEAN or DOT_PRODUCT; got {distance_function!r}")
        self._embedder = embedder
        self._takes_purpose = _takes_purpose(getattr(embedder, "embed", None))
        self._max_results = max_results
        self._distance_function = distance_function

    async def index(self, storage: DynamoDBStorage, key: str, data: bytes, **kwargs: Any) -> None:
        """Add the vector for ``data`` to the item stored under ``key``.

        ``write`` already stores the vector in the same request and does not call this; use it to
        backfill a value written before the strategy was configured. Only the vector is written,
        so the stored value, its metadata and its TTL are left as they are. A no-op when the item
        already has a vector, when it is gone, and when ``data`` is empty or not UTF-8.

        Args:
            storage: The storage the value is in.
            key: The key, relative to ``storage``.
            data: The value's bytes, as :meth:`DynamoDBStorage.read` returns them.
            **kwargs: Unused; accepted for forward compatibility.

        Raises:
            StorageError: If embedding or the write fails.
        """
        vector = await self._document_vector(storage, data)
        if vector is None:
            return
        await storage._set_vector(key, vector)

    async def search(self, storage: DynamoDBStorage, query: str, **kwargs: Any) -> builtins.list[StorageSearchResult]:
        """Return the stored values nearest ``query``, best first, with their bytes as ``data``.

        Args:
            storage: The storage to search.
            query: A natural-language string. A blank one returns ``[]`` without embedding.
            **kwargs: Unused; accepted for forward compatibility.

        Raises:
            StorageError: If embedding or the search fails.
        """
        from strands.storage.search import StorageSearchResult

        if not query.strip():
            return []
        vector = await self._embed(storage, query, "query")
        if vector is None:
            raise StorageError("The search strategy's embedder must return a non-empty list of finite numbers")
        matches = await storage.search(
            SearchQuery(vector=vector, top_k=self._max_results, pk=_partition(storage), include_values=True)
        )
        return [StorageSearchResult(key=m.key, score=self._similarity(m.score), data=m.data) for m in matches]

    async def _document_vector(self, storage: DynamoDBStorage, data: bytes) -> Optional[builtins.list[float]]:
        """The vector :meth:`DynamoDBStorage.write` stores with ``data``, or ``None`` for empty or non-UTF-8 data.

        Raises:
            StorageError: If the embedder fails or returns something other than a vector.
        """
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if not text.strip():
            return None
        vector = await self._embed(storage, text, "document")
        if vector is None:
            raise StorageError("The search strategy's embedder must return a non-empty list of finite numbers")
        return vector

    async def _embed(
        self, storage: DynamoDBStorage, text: str, purpose: EmbeddingPurpose
    ) -> Optional[builtins.list[float]]:
        if self._embedder is None:
            region, session, client_config = storage._aws_defaults()
            self._embedder = BedrockEmbedder(region_name=region, boto_session=session, boto_client_config=client_config)
            self._takes_purpose = True
        try:
            result = await _call_embedder(self._embedder, text, purpose, takes_purpose=self._takes_purpose)
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Embedding failed: {type(error).__name__}: {error}") from error
        return _as_vector(result)

    def _similarity(self, score: float) -> float:
        """The index score with higher meaning closer, as the SDK's search results rank."""
        if self._distance_function == "DOT_PRODUCT":
            return score
        return 1 / (1 + score)


def _partition(storage: DynamoDBStorage) -> Optional[str]:
    """The partition every key under ``storage`` shares, or ``None`` when its prefix spans several."""
    if storage.prefix.count("/") < 2:
        return None
    return storage.partition_key("_")
