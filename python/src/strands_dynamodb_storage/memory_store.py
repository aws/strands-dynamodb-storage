# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-term memory for the Strands Memory Manager, backed by DynamoDB vector search."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import inspect
import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

from strands.memory import MemoryEntry, MemoryStore, SearchOptions
from strands.types.exceptions import StorageError

from .dynamodb_storage import _MAX_TOP_K, DynamoDBStorage, SearchQuery
from .embedders import BedrockEmbedder, Embedder, EmbeddingPurpose, EmbedFunction, _is_finite_number

if TYPE_CHECKING:
    from strands.memory import ExtractionConfig

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SEARCH_RESULTS = 10
_RESERVED_METADATA_KEYS = frozenset({"_key", "_score"})
# Dropped in both packages so the TypeScript package can write every memory this one writes.
_UNWRITABLE_KEYS = frozenset({"__proto__", "constructor"})


class DynamoDBMemoryStore(MemoryStore):
    """A Strands ``MemoryStore`` that keeps memories in DynamoDB and finds them by meaning.

    Each memory is one item in the storage's table, written with its text, its
    embedding, and optional metadata, under the key ``memory/<name>/<id>`` inside
    the storage namespace. ``<id>`` is derived from the text, so adding the same
    text again overwrites the item instead of creating a duplicate. A search embeds
    the query and runs one ``SearchVectors`` call, pinned to the partition that
    holds this store's memories. Memories stored inline come back from the index
    projection with no base-table read; a memory offloaded to S3, or a match from an
    index that does not project the data attribute, costs one point read (GetItem,
    plus the S3 read for an offloaded memory). Matches that are not this store's
    memories are dropped before any read.

    Embedding happens here, not in the storage: session snapshots and other values
    written through the same ``DynamoDBStorage`` are never embedded.

    The partition is the first two segments of the full key, so every namespace
    that shares its first two segments with this one (``user/u1/agent-a`` and
    ``user/u1/agent-b`` both use ``user/u1``) shares the partition. Every item there
    that carries a vector competes for the same ``SearchVectors`` results, and this
    store keeps only its own, so a search can return fewer than requested. Give each
    memory store a partition of its own, for example a two-segment namespace such
    as ``storage.namespace("user/u1")``.

    Example::

        storage = DynamoDBStorage("agent-data", region_name="us-east-1").namespace("user/u1")
        memory = DynamoDBMemoryStore(storage)
        agent = Agent(memory_manager=MemoryManager(stores=[memory]))
    """

    def __init__(
        self,
        storage: DynamoDBStorage,
        *,
        embedder: Union[Embedder, EmbedFunction, None] = None,
        name: str = "dynamodb",
        description: Optional[str] = "Long-term memories stored in Amazon DynamoDB, searched by meaning",
        max_search_results: Optional[int] = None,
        writable: bool = True,
        extraction: Union[ExtractionConfig, bool, None] = None,
    ) -> None:
        """Create a memory store over ``storage``. No AWS call is made until the first add or search.

        Args:
            storage: The ``DynamoDBStorage`` (usually a namespaced view) to keep memories in.
                Its table needs the vector index described in the package README.
            embedder: An :class:`Embedder` such as :class:`BedrockEmbedder`, or a plain
                function from text to vector. Defaults to ``BedrockEmbedder()`` (Amazon
                Titan Text Embeddings V2, 1024 dimensions) in the storage's Region, with
                its ``boto_session`` and ``boto_client_config``.
            name: Store name, unique within a ``MemoryManager``; also the key segment
                ``memory/<name>/``. Must not be ``.`` or ``..`` or contain ``/``, control
                characters, lone surrogates, or leading or trailing whitespace.
            description: Description shown to the model in the memory tools; ``None``
                for none.
            max_search_results: Default number of results per search, 1 to 100. When it
                is unset, a ``MemoryManager`` search asks for its own default (3), and a
                direct :meth:`search` call without options returns up to 10.
            writable: Whether the ``MemoryManager`` may write to this store.
            extraction: Automatic extraction configuration, passed through to the
                ``MemoryManager``.

        Raises:
            TypeError: If ``storage`` is not a ``DynamoDBStorage``, ``name`` is not a string,
                ``description`` is not a string or ``None``, ``writable`` is not a bool, or
                ``embedder`` is neither callable nor has an ``embed`` method.
            ValueError: If ``name`` or ``max_search_results`` is invalid.
        """
        if not isinstance(storage, DynamoDBStorage):
            raise TypeError(f"DynamoDBMemoryStore requires a DynamoDBStorage; got {type(storage).__name__}")
        if not isinstance(name, str):
            raise TypeError(f"Memory store name must be a string; got {type(name).__name__}")
        if (
            name.strip() == ""
            or name != name.strip()
            or name[0] == "\ufeff"
            or name[-1] == "\ufeff"
            or "/" in name
            or name in (".", "..")
            or any(_is_control(c) for c in name)
            or _has_lone_surrogate(name)
        ):
            raise ValueError(
                "Memory store name must be non-empty, not '.' or '..', and without '/', control characters, "
                f"lone surrogates, or leading or trailing whitespace; got {name!r}"
            )
        if description is not None and not isinstance(description, str):
            raise TypeError(f"description must be a string or None; got {type(description).__name__}")
        if not isinstance(writable, bool):
            raise TypeError(f"writable must be a bool; got {type(writable).__name__}")
        if embedder is not None and not callable(embedder) and not callable(getattr(embedder, "embed", None)):
            raise TypeError(
                f"embedder must be a function or an object with an embed method; got {type(embedder).__name__}"
            )
        if max_search_results is not None:
            _check_limit(max_search_results, "max_search_results")
        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.writable = writable
        self.extraction = extraction
        self._storage = storage
        self._key_prefix = f"memory/{name}/"
        # Every memory key shares this partition: the id is at least the third key segment.
        self._partition = storage.partition_key(f"{self._key_prefix}0")
        if embedder is None:
            region, session, client_config = storage._aws_defaults()
            embedder = BedrockEmbedder(region_name=region, boto_session=session, boto_client_config=client_config)
        self._embedder = embedder

    async def search(self, query: str, options: Optional[SearchOptions] = None) -> builtins.list[MemoryEntry]:
        """Return the stored memories closest in meaning to ``query``, closest first.

        Each entry's ``metadata`` holds the metadata stored with the memory, plus
        ``_key`` (the storage key, which :meth:`DynamoDBStorage.delete` accepts) and
        ``_score`` (the raw vector index score: for COSINE and EUCLIDEAN, lower is
        closer). A per-call ``max_search_results`` above 100 is capped at 100, the
        ``SearchVectors`` limit. Memories stored inline come back from the index
        projection; a memory large enough to be offloaded to S3 is read back from S3.
        A query that is empty or whitespace returns an empty list without calling
        the embedder or DynamoDB.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If ``max_search_results`` is less than 1 or not an integer, or
                ``query`` contains a lone surrogate.
            StorageError: If embedding or the search fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string; got {type(query).__name__}")
        limit = self._resolve_limit(options)
        if _is_blank(query):
            return []
        if _has_lone_surrogate(query):
            # Bedrock would embed it as "?", a different query from the one asked.
            raise ValueError("query must be valid Unicode text; it contains a lone surrogate")
        vector = await self._embed(query, "query")
        # The partition can hold other vector items (another store's memories, or the
        # application's own); keep only this store's, dropped before any value is read.
        results = await self._storage._search(
            SearchQuery(vector=vector, top_k=limit, pk=self._partition, include_values=True),
            key_prefix=self._key_prefix,
        )
        entries: builtins.list[MemoryEntry] = []
        for result in results:
            if result.data is None:
                continue
            metadata: dict[str, Any] = dict(result.metadata) if result.metadata else {}
            metadata["_key"] = result.key
            metadata["_score"] = result.score
            entries.append(MemoryEntry(content=result.data.decode("utf-8", errors="replace"), metadata=metadata))
        return entries

    async def add(self, content: str, metadata: Optional[dict[str, Any]] = None) -> str:
        """Embed ``content`` and store it as a memory; return its storage key.

        Adding text that is already stored overwrites that item, replacing its
        metadata and restarting its TTL, so retried writes do not create duplicates.

        ``metadata`` values that are strings, booleans, or numbers from 1e-130 to
        2**53 - 1 in magnitude (or zero) are stored. Anything else is dropped with a
        warning, so a memory whose metadata came from a model is still stored: lists,
        nested objects, ``None``, other numbers, keys or strings with a lone surrogate,
        non-string or empty keys, the keys ``__proto__`` and ``constructor``, and the
        keys ``_key`` and ``_score``, which are reserved for search results.

        Raises:
            TypeError: If ``content`` is not a string.
            ValueError: If ``content`` is empty or whitespace, or contains a lone surrogate.
            StorageError: If embedding or the write fails.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string; got {type(content).__name__}")
        if _is_blank(content):
            raise ValueError("content must not be empty")
        try:
            payload = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("content must be valid Unicode text; it contains a lone surrogate") from error
        checked = self._storable_metadata(metadata)
        key = f"{self._key_prefix}{hashlib.sha256(payload).hexdigest()[:32]}"
        vector = await self._embed(content, "document")
        await self._storage.write(key, payload, vector=vector, metadata=checked)
        return key

    def _resolve_limit(self, options: Optional[SearchOptions]) -> int:
        requested = options.get("max_search_results") if options else None
        if requested is None:
            requested = self.max_search_results
        if requested is None:
            return _DEFAULT_MAX_SEARCH_RESULTS
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError(f"max_search_results must be an integer >= 1; got {requested!r}")
        return min(requested, _MAX_TOP_K)

    async def _embed(self, text: str, purpose: EmbeddingPurpose) -> builtins.list[float]:
        embedder = self._embedder
        try:
            # getattr, not isinstance(embedder, Embedder): from Python 3.12 the Protocol check
            # skips attributes resolved by __getattr__ (as on unittest.mock objects).
            embed = getattr(embedder, "embed", None)
            if callable(embed):
                result: Any = await _call(embed, text, purpose)
            else:
                result = await _call(embedder, text)
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(
                f"Embedding failed in memory store '{self.name}': {type(error).__name__}: {error}"
            ) from error
        if not isinstance(result, (list, tuple)) or len(result) == 0 or not all(_is_finite_number(v) for v in result):
            raise StorageError(
                f"The embedder for memory store '{self.name}' must return a non-empty list of finite numbers"
            )
        return [float(v) for v in result]

    def _storable_metadata(self, metadata: Any) -> Optional[dict[str, Union[str, int, float, bool]]]:
        """Keep the metadata entries the storage can store; drop the rest with a warning."""
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            logger.warning(
                "store=<%s> | dropping metadata of type %s; metadata must be a dict",
                _log_safe(self.name),
                _log_safe(type(metadata).__name__),
            )
            return None
        kept: dict[str, Union[str, int, float, bool]] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or key in _RESERVED_METADATA_KEYS or key == "" or key in _UNWRITABLE_KEYS:
                if key in _RESERVED_METADATA_KEYS:
                    reason = "is reserved for search results"
                elif key in _UNWRITABLE_KEYS:
                    reason = "cannot be written by the AWS SDK for JavaScript"
                else:
                    reason = "is empty, which DynamoDB rejects" if key == "" else "is not a string"
            elif _has_lone_surrogate(key) or (isinstance(value, str) and _has_lone_surrogate(value)):
                reason = "contains a lone surrogate, which DynamoDB stores as '?'"
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_dynamodb_number(value):
                reason = "is out of range (finite, zero or 1e-130 to 2^53 - 1 in magnitude)"
            elif not isinstance(value, (str, int, float, bool)):
                reason = f"has type {_log_safe(type(value).__name__)}; values must be strings, numbers, or booleans"
            else:
                # Plain values: str() of an enum member that subclasses str or int gives
                # "Color.RED", not the value (for IntEnum on Python 3.10, which then fails the write).
                kept[key] = (
                    value
                    if isinstance(value, bool)
                    else str.__str__(value)
                    if isinstance(value, str)
                    else int(value)
                    if isinstance(value, int)
                    else float(value)
                )
                continue
            # A key comes from model output: escape it so it cannot break or forge a log line.
            shown = _log_safe(str.__str__(key) if isinstance(key, str) else repr(key))
            logger.warning("store=<%s>, key=<%s> | dropping metadata entry: %s", _log_safe(self.name), shown, reason)
        return kept or None


_MAX_SAFE_INTEGER = 2**53 - 1


def _has_lone_surrogate(text: str) -> bool:
    """Whether ``text`` holds a code point from U+D800 to U+DFFF, which UTF-8 cannot encode."""
    return any(0xD800 <= ord(c) <= 0xDFFF for c in text)


def _is_blank(text: str) -> bool:
    """Whether ``text`` is only whitespace, by the union of Python's and JavaScript's definitions."""
    return all(c.isspace() or c == "\ufeff" for c in text)


def _is_control(character: str) -> bool:
    """C0 and C1 control characters and DEL, the set both packages reject in store names."""
    return ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F


_LOG_ESCAPES = {"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
# Line and paragraph separators, and the Bidi_Control characters, which can reorder the
# rest of a line in a viewer that applies the Unicode bidirectional algorithm.
_LOG_UNICODE_ESCAPES = frozenset("\u2028\u2029\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _log_safe(text: str) -> str:
    """``text`` for a log line: backslash, control characters, line separators,
    bidirectional controls and lone surrogates escaped, so a value from model output
    cannot split the line or reorder how it displays."""
    return "".join(
        _LOG_ESCAPES[c]
        if c in _LOG_ESCAPES
        else f"\\u{ord(c):04x}"
        if _is_control(c) or c in _LOG_UNICODE_ESCAPES or 0xD800 <= ord(c) <= 0xDFFF
        else c
        for c in text
    )


def _is_dynamodb_number(value: Union[int, float]) -> bool:
    """Whether ``value`` round-trips as a DynamoDB number in both the Python and TypeScript packages.

    Finite; zero or at least 1e-130 in magnitude (DynamoDB's lower bound); at most 2**53 - 1, the
    largest integer metadata reads return exactly and the bound the TypeScript package applies,
    because the JavaScript SDK refuses larger plain numbers.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value == 0 or 1e-130 <= abs(value) <= _MAX_SAFE_INTEGER


async def _call(function: Any, *args: Any) -> Any:
    """Await ``function(*args)``; a sync callable (for example a boto3 call) runs in a worker thread."""
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    result = await asyncio.to_thread(function, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _check_limit(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_TOP_K:
        raise ValueError(f"{label} must be an integer from 1 to {_MAX_TOP_K}; got {value!r}")
