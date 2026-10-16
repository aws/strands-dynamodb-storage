# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-style search strategy over the lexical document index (preview).

Lets consumers that only know the byte ``Storage`` API (``write`` then ``search("text")``, like the
SDK ``FileMemoryStore``) use :class:`LexicalIndex` through ``DynamoDBStorage(search_strategy=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from strands.types.exceptions import StorageError

from .dynamodb_storage import DynamoDBStorage
from .lexical_index import (
    LexicalIndex,
    LexicalIndexLimits,
    LexicalSearchResult,
    SearchableDocument,
    _require_table_name,
    _require_top_k,
)
from .lexical_terms import text_terms, utf8_byte_length

if TYPE_CHECKING:
    from strands.storage.storage import StorageSearchResult


@dataclass(frozen=True)
class SearchableText:
    """What an extractor found searchable in a stored value: free text and exact identifiers.

    ``text`` is tokenized like :attr:`SearchableDocument.text` and ``identifiers`` are indexed like
    :attr:`SearchableDocument.identifiers`; together they must fit ``max_postings_per_document``.
    """

    text: str = ""
    identifiers: Sequence[str] = ()


SearchableTextExtractor = Callable[[str, bytes], Optional[SearchableText]]
"""Decides what is searchable in a written value: ``(key, data) -> SearchableText | None``.

``key`` is the normalized key relative to the storage scope. Return ``None`` to leave the value
unindexed. Bytes are never decoded implicitly: the extractor chooses the encoding and the fields.
"""


class LexicalSearchStrategy:
    """Search strategy that indexes written values in a :class:`LexicalIndex` and answers string queries.

    Follows the SDK storage-backend pattern: ``DynamoDBStorage(search_strategy=...)`` calls
    :meth:`index` after every successful ``write()`` and delegates plain-string ``search()`` to
    :meth:`search`. A :class:`LexicalIndex` is built per call from the storage it receives, so each
    namespaced view searches only its own scope. It works only through DynamoDB, never the host
    filesystem, so it is marked sandbox-safe (``requires_host_fs = False``).

    Example:
        ```python
        strategy = LexicalSearchStrategy(
            index_table_name="agent-lexical-index",
            extract=lambda key, data: SearchableText(text=data.decode("utf-8")),
        )
        storage = DynamoDBStorage("agent-data", region_name="us-east-1", search_strategy=strategy)
        ```
    """

    requires_host_fs: Literal[False] = False

    def __init__(
        self,
        *,
        index_table_name: str,
        extract: SearchableTextExtractor,
        top_k: int = 10,
        include_values: bool = True,
        limits: Optional[LexicalIndexLimits] = None,
    ) -> None:
        """Configure the strategy.

        Args:
            index_table_name: Index table with key schema ``pk`` (S, HASH) and ``sk`` (S, RANGE).
            extract: Required; see :data:`SearchableTextExtractor`.
            top_k: Default maximum number of results per search (1..100); a ``top_k`` keyword
                argument to :meth:`search` overrides it.
            include_values: Whether results carry the stored ``data``.
            limits: Document, query and retry bounds; defaults to :class:`LexicalIndexLimits`.

        Raises:
            StorageError: If ``index_table_name`` is empty, ``extract`` is not callable, or
                ``top_k`` is outside 1..100.
        """
        self._index_table_name = _require_table_name(index_table_name)
        self._extract = _require_extractor(extract)
        _require_top_k(top_k)
        self._top_k = top_k
        self._include_values = include_values
        self._limits = limits if limits is not None else LexicalIndexLimits()

    async def index(self, storage: DynamoDBStorage, key: str, data: bytes, **kwargs: Any) -> None:
        """Index a value ``storage`` just wrote by re-storing it with its index entries.

        ``extract(key, data)`` returning ``None`` skips indexing; a previously indexed version of
        the key stays hidden from search (the write dropped its index markers) and
        :meth:`LexicalIndex.repair` removes its entries. Otherwise the value is upserted with the
        ``vector``, ``metadata`` and ``ttl_seconds`` keyword arguments ``write()`` forwards, so they
        are preserved. This costs one more write than ``write()`` alone and, since a failed hook
        leaves the written value in place, is at-least-once; :meth:`LexicalIndex.upsert` remains
        the single-write atomic path.

        Raises:
            StorageError: If ``storage`` is not a :class:`DynamoDBStorage`, the extracted document
                breaks a :class:`LexicalIndexLimits` bound, or DynamoDB fails. Errors raised by
                ``extract`` propagate unchanged.
        """
        lexical_index = self._index_for(storage)
        searchable = self._extract(key, data)
        if searchable is None:
            return
        await lexical_index.upsert(_searchable_document(key, data, searchable, kwargs))

    async def search(self, storage: DynamoDBStorage, query: str, **kwargs: Any) -> list[StorageSearchResult]:
        """Rank the values in ``storage``'s scope by the fraction of query terms they contain.

        Only the first ``max_query_terms`` distinct query terms are used, after skipping terms
        above ``max_term_bytes`` (no indexed document can contain them); indexed text is never
        truncated. A query without usable terms returns ``[]``. ``top_k`` defaults to the
        constructor's. Truncation reasons of the lexical response have no place in the SDK result
        and are dropped.

        Returns:
            SDK ``StorageSearchResult`` objects, best first, carrying ``data`` only with
            ``include_values``. The type is imported on use because the package's SDK floor
            predates it: this method needs strands-agents 1.54 or later.

        Raises:
            StorageError: If ``storage`` is not a :class:`DynamoDBStorage`, ``top_k`` is outside
                1..100, the query is not well-formed Unicode, keys stay unprocessed, or DynamoDB fails.
        """
        lexical_index = self._index_for(storage)
        top_k = kwargs.get("top_k", self._top_k)
        _require_top_k(top_k)
        terms = self._query_terms(query)
        if not terms:
            return []
        response = await lexical_index._search_terms(terms, top_k=top_k, include_values=self._include_values)
        return _storage_search_results(response.results)

    def _index_for(self, storage: DynamoDBStorage) -> LexicalIndex:
        if not isinstance(storage, DynamoDBStorage):
            raise StorageError(f"LexicalSearchStrategy requires a DynamoDBStorage; got {type(storage).__name__}")
        return LexicalIndex(storage, index_table_name=self._index_table_name, limits=self._limits)

    def _query_terms(self, query: str) -> list[str]:
        """The first ``max_query_terms`` distinct terms of ``query`` that fit ``max_term_bytes``."""
        indexable = [term for term in text_terms(query) if utf8_byte_length(term) <= self._limits.max_term_bytes]
        return indexable[: self._limits.max_query_terms]


def _searchable_document(
    key: str, data: bytes, searchable: SearchableText, write_options: dict[str, Any]
) -> SearchableDocument:
    return SearchableDocument(
        key=key,
        data=data,
        text=searchable.text,
        identifiers=searchable.identifiers,
        metadata=write_options.get("metadata"),
        vector=write_options.get("vector"),
        ttl_seconds=write_options.get("ttl_seconds"),
    )


def _storage_search_results(results: Sequence[LexicalSearchResult]) -> list[StorageSearchResult]:
    """Map lexical results to the SDK result type, imported here so the package still imports on older SDKs."""
    from strands.storage.storage import StorageSearchResult

    return [StorageSearchResult(key=result.key, score=result.score, data=result.data) for result in results]


def _require_extractor(extract: SearchableTextExtractor) -> SearchableTextExtractor:
    if not callable(extract):
        raise StorageError("LexicalSearchStrategy requires an extract callable: (key, data) -> SearchableText | None")
    return extract
