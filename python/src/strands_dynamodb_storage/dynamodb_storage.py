# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon DynamoDB ``Storage`` backend for the Strands Agents SDK."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import gzip
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

from strands.types.exceptions import StorageError

from .keys import normalize_key, normalize_prefix

# Single-table attribute names.
_PK = "pk"
_SK = "sk"
_KEY_ATTR = "k"
_DATA_ATTR = "data"
_S3_ATTR = "s3"
_META_ATTR = "meta"
_Z_ATTR = "z"

# Leave margin below DynamoDB's 400 KB item limit for keys/metadata.
_OFFLOAD_THRESHOLD_BYTES = 380_000
# Sentinel sort key for single-segment keys, so point ops round-trip.
_SENTINEL_SK = "\u0000"
# Service maximum for SearchVectors TopK ("must be between 1 and 100 inclusive").
_MAX_TOP_K = 100
# Over-fetch factor when a client-side metadata filter is applied, so filtering
# doesn't silently drop results below the requested top_k.
_FILTER_OVERFETCH = 10

_MetaValue = Union[str, int, float, bool]

# Optional override for the native vector search call (DynamoDB SearchVectors).
# Receives a params dict and returns match dicts: {"key", "score", "metadata"?,
# "expire_at"?}. An adapter opts in to TTL filtering by returning the item's
# epoch-seconds expiry as "expire_at".
VectorSearchAdapter = Callable[[dict[str, Any]], Awaitable[builtins.list[dict[str, Any]]]]


@dataclass
class DynamoDBListQuery:
    """Structured list query naming the partition directly.

    The ``Storage`` ``list`` argument is generic so backends like DynamoDB can accept
    a richer query than a string prefix. A ``DynamoDBListQuery`` names the partition
    (``pk``), so listing is a native ``Query`` against one partition. SDK-internal
    callers still pass a plain string prefix.
    """

    pk: str
    sk_prefix: Optional[str] = None
    sk_between: Optional[tuple[str, str]] = None
    limit: Optional[int] = None
    start_after: Optional[str] = None


@dataclass
class SearchQuery:
    """Vector similarity query for the optional native vector index.

    ``pk`` is required when the index's ``SearchSchema`` declares a HASH element
    (the documented tenant-scoping setup does; the service rejects an unpinned
    search against such an index) and must be omitted when it doesn't.
    """

    vector: builtins.list[float]
    top_k: int
    pk: Optional[str] = None
    filter: Optional[dict[str, _MetaValue]] = None
    include_values: bool = False


@dataclass
class SearchResult:
    """A single nearest-neighbour match."""

    key: str
    score: float
    data: Optional[bytes] = None
    metadata: Optional[dict[str, Any]] = None


class _StrategyMatch(Protocol):
    """Read-only shape of one search-strategy result; the SDK's ``StorageSearchResult`` satisfies it."""

    @property
    def key(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def data(self) -> Optional[bytes]: ...


class StringSearchStrategy(Protocol):
    """Structural shape of the SDK ``SearchStrategy`` as :class:`DynamoDBStorage` uses it.

    Declared here instead of importing the SDK type because the supported SDK floor predates
    it. :class:`LexicalSearchStrategy` implements it.
    """

    async def index(self, storage: DynamoDBStorage, key: str, data: bytes, **kwargs: Any) -> None:
        """Index a value that ``storage`` just wrote; ``kwargs`` carry ``vector``, ``metadata`` and ``ttl_seconds``."""
        ...

    async def search(self, storage: DynamoDBStorage, query: str, **kwargs: Any) -> Sequence[_StrategyMatch]:
        """Rank the values stored in ``storage`` for a natural-language ``query``, best first."""
        ...


def _clean_prefix(prefix: str) -> str:
    """Collapse/strip a prefix and add a single trailing slash (empty stays empty)."""
    joined = "/".join(seg for seg in prefix.split("/") if seg)
    return f"{joined}/" if joined else ""


def _unmarshal_meta(attr: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Inverse of :func:`_marshal_meta` for the flat metadata map."""
    if attr is None:
        return None
    out: dict[str, Any] = {}
    for key, value in attr.get("M", {}).items():
        if "BOOL" in value:
            out[key] = value["BOOL"]
        elif "N" in value:
            raw = value["N"]
            number = float(raw)
            out[key] = int(number) if number.is_integer() and "." not in raw and "e" not in raw.lower() else number
        else:
            out[key] = value.get("S")
    return out


def _marshal_meta(meta: dict[str, _MetaValue]) -> dict[str, Any]:
    """Marshal a flat metadata map to a DynamoDB ``M`` AttributeValue."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        # bool is a subclass of int — check it first.
        if isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
        else:
            out[key] = {"S": str(value)}
    return {"M": out}


@dataclass(frozen=True)
class _DocumentLocation:
    """Internal: where a document lives. ``key`` is scope-relative; ``doc_id`` is the full stored key."""

    key: str
    doc_id: str
    pk: str
    sk: str


@dataclass(frozen=True)
class _DocumentItemPort:
    """Internal port through which the lexical index reads and writes base-table items.

    Not part of the public API; obtained from :meth:`DynamoDBStorage._document_item_port`. It only
    delegates to private storage methods, so indexed items are built by the same code as
    :meth:`DynamoDBStorage.write`. ``ttl_attribute`` is set only when the storage opted in to TTL,
    and ``client()`` returns the low-level boto3 client (synchronous; call it through
    ``asyncio.to_thread``).
    """

    _storage: DynamoDBStorage

    @property
    def table_name(self) -> str:
        """Base table name."""
        return self._storage._table_name

    @property
    def scope(self) -> str:
        """Storage prefix exactly as held (``"tenant/a/"`` or ``""``)."""
        return self._storage._prefix

    @property
    def ttl_attribute(self) -> Optional[str]:
        """TTL attribute name; ``None`` unless the storage opted in to TTL."""
        return self._storage._ttl_attribute if self._storage._ttl_enabled else None

    def client(self) -> Any:
        """The storage's (lazily built) low-level DynamoDB client."""
        return self._storage._get_client()

    def locate(self, key: str) -> _DocumentLocation:
        """See :meth:`DynamoDBStorage._locate`."""
        return self._storage._locate(key)

    def relative_key(self, doc_id: str) -> Optional[str]:
        """The key relative to the storage prefix, or ``None`` when ``doc_id`` is outside it."""
        return self._storage._strip_prefix(doc_id)

    def is_expired(self, item: dict[str, Any]) -> bool:
        """See :meth:`DynamoDBStorage._is_ttl_expired`."""
        return self._storage._is_ttl_expired(item)

    def inline_item(
        self,
        location: _DocumentLocation,
        data: bytes,
        *,
        vector: Optional[builtins.list[float]] = None,
        metadata: Optional[dict[str, _MetaValue]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        """See :meth:`DynamoDBStorage._inline_item`."""
        return self._storage._inline_item(location, data, vector=vector, metadata=metadata, ttl_seconds=ttl_seconds)

    async def delete_offloaded(self, doc_id: str) -> None:
        """See :meth:`DynamoDBStorage._reclaim_offloaded`."""
        await self._storage._reclaim_offloaded(doc_id)


class DynamoDBStorage:
    """DynamoDB ``Storage`` backend (single-table design).

    Each key is one item: the ``/``-separated key is split into a partition key
    (leading scope) and a sort key (remainder), with the raw bytes in a binary
    ``data`` attribute. Point operations are single-item Put/Get/Delete; listing is a
    native ``Query`` — a string prefix resolves to a partition plus ``begins_with``,
    while a :class:`DynamoDBListQuery` names the partition directly.

    Values above the 400 KB item limit are offloaded to S3 when ``s3_bucket`` is set;
    a small pointer item stays in DynamoDB. Optional transparent gzip compression,
    optional DynamoDB-native TTL (with read/list filtering of expired items), and an
    optional native vector ``search`` complete the feature set. boto3 is imported
    lazily, so applications that never construct this class don't pay the import cost.

    Example:
        ```python
        from strands import Agent
        from strands.session import SessionManager
        from strands_dynamodb_storage import DynamoDBStorage

        storage = DynamoDBStorage("agent-data", region_name="us-east-1")
        agent = Agent(session_manager=SessionManager(storage=storage))
        ```
    """

    def __init__(
        self,
        table_name: str,
        *,
        region_name: Optional[str] = None,
        client: Any = None,
        prefix: str = "",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "",
        s3_client: Any = None,
        compression: str = "none",
        ttl_seconds: Optional[int] = None,
        ttl_attribute: str = "expireAt",
        index_name: str = "vector_index",
        vector_attribute: str = "vector",
        vector_search: Optional[VectorSearchAdapter] = None,
        search_strategy: Optional[StringSearchStrategy] = None,
        boto_session: Any = None,
        boto_client_config: Any = None,
    ) -> None:
        """Initialize the backend.

        Args:
            table_name: Target table (partition key ``pk`` string, sort key ``sk`` string).
            region_name: AWS region override. Cannot be combined with ``client``.
            client: Pre-configured low-level DynamoDB client.
            prefix: Key prefix prepended to every key (a namespace within the table).
            s3_bucket: Bucket for offloading values above the item-size limit.
                If DynamoDB TTL reaps an offloaded item, only the pointer item is
                removed; configure an S3 lifecycle rule to reap orphaned objects.
            s3_prefix: Key prefix for offloaded S3 objects.
            s3_client: Pre-configured S3 client. Ignored unless ``s3_bucket`` is set.
            compression: ``"gzip"`` for transparent compression, or ``"none"``.
            ttl_seconds: Default TTL. Setting it opts in to TTL (stamp + read/list/search
                filter). A per-write ``ttl_seconds`` override applies only when the
                instance opted in here.
            ttl_attribute: Item attribute holding the epoch-seconds TTL.
            index_name: Vector index name (for ``search``).
            vector_attribute: Item attribute holding the embedding vector.
            vector_search: Adapter performing the native vector search.
            search_strategy: Strategy answering plain-string :meth:`search` queries, such as
                :class:`LexicalSearchStrategy`. As in the SDK's storage backends, every successful
                :meth:`write` is then passed to its ``index`` hook. Namespaced views share it.
            boto_session: Pre-configured boto3 session. Cannot be combined with region_name.
            boto_client_config: Botocore Config for the created clients.

        Raises:
            StorageError: If both ``client`` and ``region_name`` are provided.
        """
        if client is not None and region_name is not None:
            raise StorageError("Cannot specify both client and region_name; configure the region on the client instead")

        self._table_name = table_name
        self._region_name = region_name
        self._client = client
        self._prefix = _clean_prefix(prefix)
        self._s3_bucket = s3_bucket
        self._s3_prefix = _clean_prefix(s3_prefix)
        self._s3_client = s3_client
        self._compress = compression == "gzip"
        self._ttl_seconds = ttl_seconds
        self._ttl_enabled = ttl_seconds is not None
        self._ttl_attribute = ttl_attribute
        self._index_name = index_name
        self._vector_attribute = vector_attribute
        self._vector_search = vector_search
        self._search_strategy = search_strategy
        self._boto_session = boto_session
        self._boto_client_config = boto_client_config

    async def write(
        self,
        key: str,
        data: bytes,
        *,
        vector: Optional[builtins.list[float]] = None,
        metadata: Optional[dict[str, _MetaValue]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.

        Values above the item-size limit offload to S3 when ``s3_bucket`` is set.
        An optional ``vector`` (kept inline for the native index) and ``metadata``
        enable :meth:`search`. With a ``search_strategy``, the stored value and these
        options are then passed to the strategy's ``index`` hook.

        Raises:
            StorageError: If the key is invalid, the value is oversized with no S3
                bucket configured, or the write fails; or, with the value already
                stored, ``"Wrote '<key>' but indexing failed"`` if the hook fails.
        """
        normalized = normalize_key(key)
        full = f"{self._prefix}{normalized}"
        pk, sk = self._split(full)

        extra = self._item_extras(vector, metadata, ttl_seconds)
        payload, compressed = self._encode_value(data)
        if compressed:
            extra[_Z_ATTR] = {"BOOL": True}

        try:
            if len(payload) > _OFFLOAD_THRESHOLD_BYTES:
                if not self._s3_bucket:
                    raise StorageError(
                        f"Value for '{normalized}' is {len(payload)} bytes, above the "
                        f"{_OFFLOAD_THRESHOLD_BYTES}-byte limit; configure s3_bucket to offload large values"
                    )
                await self._s3_put(full, payload)
                item = {_PK: {"S": pk}, _SK: {"S": sk}, _KEY_ATTR: {"S": full}, _S3_ATTR: {"BOOL": True}, **extra}
                await self._put(item)
            else:
                item = {_PK: {"S": pk}, _SK: {"S": sk}, _KEY_ATTR: {"S": full}, _DATA_ATTR: {"B": payload}, **extra}
                # An overwrite can shrink a previously offloaded value back inline.
                # Ask for the replaced item so the now-unreferenced S3 object can be
                # reclaimed: the new item carries no s3 flag, so without this the
                # object is orphaned forever (a later delete() never reaches it).
                old = await self._put(item, return_old=bool(self._s3_bucket))
                if old.get(_S3_ATTR, {}).get("BOOL"):
                    await self._reclaim_offloaded(full)
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to write '{normalized}' to DynamoDB table '{self._table_name}'") from error
        await self._index_written(normalized, data, vector=vector, metadata=metadata, ttl_seconds=ttl_seconds)

    async def read(self, key: str) -> Optional[bytes]:
        """Retrieve the bytes stored under ``key``, or ``None`` if absent.

        Raises:
            StorageError: If the key is invalid or the read fails.
        """
        normalized = normalize_key(key)
        full = f"{self._prefix}{normalized}"
        pk, sk = self._split(full)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.get_item, TableName=self._table_name, Key={_PK: {"S": pk}, _SK: {"S": sk}}
            )
            item = response.get("Item")
            if not item:
                return None
            # Hide items whose TTL has passed but DynamoDB has not yet reaped (only
            # when TTL is opted in). Runs before any S3 fetch.
            if self._is_ttl_expired(item):
                return None
            compressed = item.get(_Z_ATTR, {}).get("BOOL") is True
            if item.get(_S3_ATTR, {}).get("BOOL"):
                raw = await self._s3_get(full)
            else:
                raw = item.get(_DATA_ATTR, {}).get("B")
                raw = bytes(raw) if raw is not None else None
            if raw is None:
                return None
            return gzip.decompress(raw) if compressed else raw
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to read '{normalized}' from DynamoDB table '{self._table_name}'") from error

    async def delete(self, key: str) -> None:
        """Delete the value under ``key`` (and any offloaded S3 object). No-op if absent.

        Raises:
            StorageError: If the key is invalid or the delete fails.
        """
        normalized = normalize_key(key)
        full = f"{self._prefix}{normalized}"
        pk, sk = self._split(full)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.delete_item,
                TableName=self._table_name,
                Key={_PK: {"S": pk}, _SK: {"S": sk}},
                ReturnValues="ALL_OLD",
            )
            if response.get("Attributes", {}).get(_S3_ATTR, {}).get("BOOL"):
                await self._s3_delete(full)
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to delete '{normalized}' from DynamoDB table '{self._table_name}'") from error

    async def list(self, query: Union[str, DynamoDBListQuery]) -> builtins.list[str]:
        """List keys, sorted ascending.

        Accepts a **string prefix** (resolved to a partition plus ``begins_with``; must
        include at least the scope and identifier) or a :class:`DynamoDBListQuery`
        naming the partition directly.

        Raises:
            StorageError: If the query is invalid or the list fails.
        """
        try:
            if isinstance(query, str):
                return await self._list_by_prefix(query)
            return await self._list_by_query(query)
        except StorageError:
            raise
        except Exception as error:
            label = f"'{query}'" if isinstance(query, str) else f"pk='{query.pk}'"
            raise StorageError(f"Failed to list DynamoDB table '{self._table_name}' under {label}") from error

    def namespace(self, prefix: str) -> DynamoDBStorage:
        """Return a prefixed view sharing this client/config.

        Returns a real :class:`DynamoDBStorage` (not the SDK's generic string-only
        namespace view), so structured :class:`DynamoDBListQuery` listing and
        :meth:`search` survive namespacing. Nested calls compose.
        """
        sub = normalize_prefix(prefix)
        kwargs: dict[str, Any] = {
            "prefix": f"{self._prefix}{sub}",
            "s3_prefix": self._s3_prefix,
            "index_name": self._index_name,
            "vector_attribute": self._vector_attribute,
        }
        if self._client is not None:
            kwargs["client"] = self._client
        elif self._region_name is not None:
            kwargs["region_name"] = self._region_name
        if self._s3_bucket is not None:
            kwargs["s3_bucket"] = self._s3_bucket
        if self._s3_client is not None:
            kwargs["s3_client"] = self._s3_client
        if self._compress:
            kwargs["compression"] = "gzip"
        if self._ttl_seconds is not None:
            kwargs["ttl_seconds"] = self._ttl_seconds
        kwargs["ttl_attribute"] = self._ttl_attribute
        if self._vector_search is not None:
            kwargs["vector_search"] = self._vector_search
        if self._search_strategy is not None:
            kwargs["search_strategy"] = self._search_strategy
        if self._boto_session is not None:
            kwargs["boto_session"] = self._boto_session
        if self._boto_client_config is not None:
            kwargs["boto_client_config"] = self._boto_client_config
        return DynamoDBStorage(self._table_name, **kwargs)

    async def search(self, query: Union[SearchQuery, str]) -> builtins.list[SearchResult]:
        """Nearest-neighbour vector search over items written with a ``vector``.

        Optional part of the ``Storage`` contract — consumers feature-detect
        (``hasattr(storage, "search")``) and fall back to client-side KNN when absent.
        Calls DynamoDB ``SearchVectors`` natively (requires boto3 >= 1.43.64); a
        ``vector_search`` adapter, when configured, overrides the native call.

        A plain-string ``query`` is delegated to the configured ``search_strategy`` and its
        matches are returned as :class:`SearchResult` with ``key``, ``score`` and ``data``
        (``metadata`` is ``None``); a :class:`SearchQuery` always takes the vector path.

        The score is the raw ``Score`` from the vector index and its direction
        depends on the index's distance function: for COSINE and EUCLIDEAN lower
        is closer; for DOT_PRODUCT higher is more similar. Results are returned
        in the service's most-similar-first order. Like a global secondary
        index, the vector index is eventually consistent.

        With TTL enabled, matches whose TTL attribute has passed are dropped, the
        same check ``read`` applies, without any extra reads: the native path
        projects the TTL attribute from the vector index, and an adapter opts in
        by returning ``expire_at`` on each match. A match that carries no expiry
        (an index that does not project the TTL attribute, or an adapter that
        does not return it) is kept. Expired matches are not back-filled, so a
        search can return fewer than ``top_k`` results.

        Raises:
            StorageError: If the search fails, ``top_k`` is out of range, or ``query`` is a
                plain string and no ``search_strategy`` is configured.
        """
        if isinstance(query, str):
            return await self._search_text(query)
        if query.pk is not None:
            self._assert_pk_in_scope(query.pk)
        try:
            if self._vector_search is not None:
                params: dict[str, Any] = {
                    "table_name": self._table_name,
                    "index_name": self._index_name,
                    "vector_attribute": self._vector_attribute,
                    "vector": query.vector,
                    "top_k": query.top_k,
                }
                if query.pk is not None:
                    params["pk"] = query.pk
                if query.filter is not None:
                    params["filter"] = query.filter
                matches = await self._vector_search(params)
            else:
                matches = await self._native_vector_search(query)
            results: builtins.list[SearchResult] = []
            for match in matches:
                # Defence in depth: pk is optional, so the index may return matches from
                # outside this namespace. Drop them rather than de-prefixing blindly —
                # a foreign key would otherwise be handed back looking like one of ours.
                key = self._strip_prefix(match["key"])
                if key is None:
                    continue
                if self._ttl_enabled and self._has_passed(match.get("expire_at")):
                    continue
                data = None
                if query.include_values:
                    # The native path decodes the projected payload into the match;
                    # offloaded values, narrow projections, and adapter matches fall
                    # back to a point read (which also fetches from S3 and decompresses).
                    data = match["data"] if "data" in match else await self.read(key)
                results.append(SearchResult(key=key, score=match["score"], data=data, metadata=match.get("metadata")))
            return results
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to search DynamoDB table '{self._table_name}'") from error

    async def _search_text(self, query: str) -> builtins.list[SearchResult]:
        """Delegate a plain-string query to the search strategy, as the SDK's storage backends do.

        Matches keep the strategy's ``key``, ``score`` and ``data``; ``metadata`` is always ``None``.
        Errors other than ``StorageError`` are wrapped, like every other failure of this backend.

        Raises:
            StorageError: If no search strategy is configured, or the strategy fails.
        """
        if self._search_strategy is None:
            raise StorageError(
                "DynamoDBStorage.search requires a SearchQuery with a pre-computed embedding vector. "
                "Plain-string queries are not supported without a search_strategy: embed the text first "
                "and pass SearchQuery(vector=..., top_k=...), or configure a search_strategy "
                "(for example LexicalSearchStrategy)."
            )
        try:
            matches = await self._search_strategy.search(self, query)
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to search DynamoDB table '{self._table_name}'") from error
        return [SearchResult(key=match.key, score=match.score, data=match.data) for match in matches]

    async def _index_written(
        self,
        key: str,
        data: bytes,
        *,
        vector: Optional[builtins.list[float]],
        metadata: Optional[dict[str, _MetaValue]],
        ttl_seconds: Optional[int],
    ) -> None:
        """Pass a completed write to the search strategy's ``index`` hook (a no-op without a strategy).

        ``key`` is the normalized key and the write's ``vector``, ``metadata`` and ``ttl_seconds``
        are forwarded, so an indexing strategy that re-stores the value keeps them. Any failure,
        ``StorageError`` included, becomes ``StorageError("Wrote '<key>' but indexing failed")``
        chaining the cause (the SDK's wording): the value is already stored, and the caller must
        be able to tell that apart from a failed write.
        """
        if self._search_strategy is None:
            return
        try:
            await self._search_strategy.index(
                self, key, data, vector=vector, metadata=metadata, ttl_seconds=ttl_seconds
            )
        except Exception as error:
            raise StorageError(f"Wrote '{key}' but indexing failed") from error

    async def _native_vector_search(self, query: SearchQuery) -> builtins.list[dict[str, Any]]:
        """Issue DynamoDB ``SearchVectors`` directly (GA path, no adapter).

        Scoping and filtering:

        * ``query.pk`` pins ``SearchConditionExpression`` to one partition
          (already namespace-validated by the caller).
        * ``query.filter`` is applied client-side after the search, with the
          request over-fetching (capped at the service TopK limit of 100) so
          filtering doesn't silently drop results below ``top_k``. A filtered
          search can still return fewer than ``top_k`` matches once the cap
          truncates the over-fetch.
        * With TTL enabled, the TTL attribute is projected and expired matches
          are dropped before counting towards ``top_k``.
        """
        if query.top_k < 1 or query.top_k > _MAX_TOP_K:
            raise StorageError(
                f"top_k must be between 1 and {_MAX_TOP_K} (SearchVectors TopK limit); got {query.top_k}"
            )
        if not all(math.isfinite(v) for v in query.vector):
            raise StorageError("Query vector contains non-finite values (nan/inf); the DynamoDB N type rejects them.")
        top_k = query.top_k
        if query.filter is not None:
            top_k = min(query.top_k * _FILTER_OVERFETCH, _MAX_TOP_K)

        # Project only what the response is for: keys (to rebuild the storage key)
        # and metadata (results + client-side filter), adding the payload and its
        # s3/gzip flags only when the caller asked for values. Everything else the
        # index projects (ProjectionType ALL projects the whole item) is billed
        # response bytes for data nobody reads. Every name is aliased: `data` is a
        # DynamoDB reserved word.
        names: dict[str, str] = {"#pk": _PK, "#sk": _SK, "#m": _META_ATTR}
        projection = "#pk, #sk, #m"
        if query.include_values:
            names.update({"#d": _DATA_ATTR, "#s3": _S3_ATTR, "#z": _Z_ATTR})
            projection += ", #d, #s3, #z"
        if self._ttl_enabled:
            names["#ttl"] = self._ttl_attribute
            projection += ", #ttl"
        request: dict[str, Any] = {
            "TableName": self._table_name,
            "IndexName": self._index_name,
            "SearchVector": [{"N": str(v)} for v in query.vector],
            "TopK": top_k,
            "ProjectionExpression": projection,
            "ExpressionAttributeNames": names,
        }
        if query.pk is not None:
            request["SearchConditionExpression"] = "#pk = :pk"
            request["ExpressionAttributeValues"] = {":pk": {"S": query.pk}}

        client = self._get_client()
        response = await asyncio.to_thread(client.search_vectors, **request)

        matches: builtins.list[dict[str, Any]] = []
        for result in response.get("SearchResults", []):
            item = result["Item"]
            pk = item[_PK]["S"]
            sk = item.get(_SK, {}).get("S", "")
            full_key = pk if sk in ("", _SENTINEL_SK) else f"{pk}/{sk}"
            if self._ttl_enabled and self._is_expired(item):
                continue
            metadata = _unmarshal_meta(item.get(_META_ATTR))
            if query.filter is not None and not self._matches_search_filter(metadata, query.filter):
                continue
            match: dict[str, Any] = {"key": full_key, "score": float(result["Score"])}
            if metadata is not None:
                match["metadata"] = metadata
            if query.include_values and not item.get(_S3_ATTR, {}).get("BOOL"):
                blob = item.get(_DATA_ATTR, {}).get("B")
                if blob is not None:
                    raw = bytes(blob)
                    match["data"] = gzip.decompress(raw) if item.get(_Z_ATTR, {}).get("BOOL") is True else raw
            matches.append(match)
            if len(matches) >= query.top_k:
                break
        return matches

    @staticmethod
    def _matches_search_filter(metadata: Optional[dict[str, Any]], filter_dict: dict[str, _MetaValue]) -> bool:
        """Exact-equality match of every filter entry against item metadata."""
        if metadata is None:
            return False
        return all(metadata.get(k) == v for k, v in filter_dict.items())

    # ------------------------------------------------------------------ internals

    def _item_extras(
        self,
        vector: Optional[builtins.list[float]],
        metadata: Optional[dict[str, _MetaValue]],
        ttl_seconds: Optional[int],
    ) -> dict[str, Any]:
        """Optional item attributes shared by every write path: vector, metadata and TTL stamp.

        The embedding stays inline even when the payload offloads, because the native vector
        index can only index an on-item attribute. Non-finite components are rejected here,
        mirroring the query-side check in :meth:`_native_vector_search`: the service rejects
        them anyway, but as an opaque write failure. The TTL stamp is floored as a whole, so
        a float duration (e.g. 90.5) never emits a fractional value that other readers of the
        shared table may not parse; it is written only when this instance opted in to TTL.

        Raises:
            StorageError: If the vector contains non-finite values.
        """
        extras: dict[str, Any] = {}
        if vector is not None:
            if not all(math.isfinite(component) for component in vector):
                raise StorageError("Vector contains non-finite values (nan/inf); the DynamoDB N type rejects them.")
            extras[self._vector_attribute] = {"L": [{"N": str(component)} for component in vector]}
        if metadata is not None:
            extras[_META_ATTR] = _marshal_meta(metadata)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        if self._ttl_enabled and effective_ttl is not None:
            extras[self._ttl_attribute] = {"N": str(int(time.time() + effective_ttl))}
        return extras

    def _encode_value(self, data: bytes) -> tuple[bytes, bool]:
        """Return ``(payload, compressed)``: gzip is kept only when compression is on and it shrinks."""
        if not self._compress:
            return data, False
        compressed = gzip.compress(data)
        if len(compressed) < len(data):
            return compressed, True
        return data, False

    def _document_item_port(self) -> _DocumentItemPort:
        """Internal: the narrow view of this storage that :class:`LexicalIndex` writes through.

        Not part of the public API. The index reaches base-table behaviour only through this
        port, so it builds items with the same helpers as :meth:`write` and never touches other
        private members of the storage.
        """
        return _DocumentItemPort(self)

    def _locate(self, key: str) -> _DocumentLocation:
        """Normalize ``key`` and resolve its full stored key and base-table primary key.

        Raises:
            StorageError: If the key is invalid.
        """
        normalized = normalize_key(key)
        doc_id = f"{self._prefix}{normalized}"
        pk, sk = self._split(doc_id)
        return _DocumentLocation(key=normalized, doc_id=doc_id, pk=pk, sk=sk)

    def _inline_item(
        self,
        location: _DocumentLocation,
        data: bytes,
        *,
        vector: Optional[builtins.list[float]] = None,
        metadata: Optional[dict[str, _MetaValue]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build the item :meth:`write` would store for an inline value, with the same extras and encoding.

        Indexed documents must stay inline so the lexical transaction writes them atomically, so a
        value that would need S3 offload is rejected even when a bucket is configured.

        Raises:
            StorageError: If the vector is non-finite or the encoded payload is above the inline threshold.
        """
        extras = self._item_extras(vector, metadata, ttl_seconds)
        payload, compressed = self._encode_value(data)
        if compressed:
            extras[_Z_ATTR] = {"BOOL": True}
        if len(payload) > _OFFLOAD_THRESHOLD_BYTES:
            raise StorageError(
                f"Value for '{location.key}' is {len(payload)} bytes, above the {_OFFLOAD_THRESHOLD_BYTES}-byte "
                "inline limit; S3 offload is not supported for indexed documents"
            )
        return {
            _PK: {"S": location.pk},
            _SK: {"S": location.sk},
            _KEY_ATTR: {"S": location.doc_id},
            _DATA_ATTR: {"B": payload},
            **extras,
        }

    async def _reclaim_offloaded(self, full_key: str) -> None:
        """Best-effort delete of the offloaded S3 object of ``full_key``; a no-op without a bucket.

        Never raises: the DynamoDB write it follows is already durable, so a failed cleanup must
        not fail it. An S3 lifecycle rule is the backstop for missed reclamations.
        """
        if not self._s3_bucket:
            return
        with contextlib.suppress(Exception):
            await self._s3_delete(full_key)

    def _is_ttl_expired(self, item: dict[str, Any]) -> bool:
        """True when this storage opted in to TTL and the item's stamp is at or before now."""
        return self._ttl_enabled and self._is_expired(item)

    def _split(self, full_key: str) -> tuple[str, str]:
        """Split a full key into (partition key, sort key)."""
        segments = [seg for seg in full_key.split("/") if seg]
        if len(segments) <= 1:
            return (segments[0] if segments else full_key), _SENTINEL_SK
        return "/".join(segments[:2]), ("/".join(segments[2:]) or _SENTINEL_SK)

    def _is_expired(self, item: dict[str, Any]) -> bool:
        """True when the item's TTL attribute holds a past epoch-seconds value.

        Parses leniently: the table is shared infrastructure, and another producer
        may stamp a fractional epoch value (``time.time()`` untruncated). DynamoDB's
        reaper compares numerically, so a float is a valid TTL; a value this code
        cannot parse is treated as not expired -- filtering is this method's job,
        not validating other writers' data.
        """
        return self._has_passed(item.get(self._ttl_attribute, {}).get("N"))

    @staticmethod
    def _has_passed(epoch_seconds: Any) -> bool:
        """True when ``epoch_seconds`` parses as a number at or before now; False otherwise."""
        if epoch_seconds is None or isinstance(epoch_seconds, bool):
            return False
        try:
            return float(epoch_seconds) <= time.time()
        except (TypeError, ValueError):
            return False

    def _assert_pk_in_scope(self, pk: str) -> None:
        """Reject a caller-supplied partition key that falls outside this namespace.

        ``write``/``read``/``delete`` and the string-prefix ``list`` all derive the
        physical partition from ``prefix + key``, so the namespace is enforced by
        construction. A :class:`DynamoDBListQuery` and :class:`SearchQuery` name the
        partition directly, so the same boundary is enforced here instead. The
        partition is rejected rather than silently rewritten: a caller asking for
        another namespace's partition has a bug or is probing, and neither should be
        answered with quietly different data.

        A partition at or below the namespace is in scope. So is an ancestor partition:
        because the physical partition is only the first two key segments, a deeply
        namespaced view legitimately lives inside a shallower partition, and any rows
        outside the namespace are dropped by :meth:`_strip_prefix`. A partition on a
        different branch (another tenant) is rejected.
        """
        if not self._prefix:
            return
        candidate = pk if pk.endswith("/") else f"{pk}/"
        if pk.startswith(self._prefix) or self._prefix.startswith(candidate):
            return
        raise StorageError(
            f"Partition key '{pk}' is outside this storage namespace '{self._prefix}'. "
            f"Prefix the partition key with '{self._prefix}' (for example "
            f"'{self._prefix}sessions') or use the storage instance that owns it."
        )

    def _strip_prefix(self, stored: str) -> Optional[str]:
        """Strip the namespace prefix from a stored key, or ``None`` if out of scope.

        Guards the de-prefix: slicing blindly would turn another namespace's key into
        something that looks like one of ours.
        """
        if not self._prefix:
            return stored
        if not stored.startswith(self._prefix):
            return None
        return stored[len(self._prefix) :]

    async def _list_by_prefix(self, prefix: str) -> builtins.list[str]:
        normalized = normalize_prefix(prefix)
        full = f"{self._prefix}{normalized}"
        segments = [seg for seg in full.split("/") if seg]
        if len(segments) < 2:
            raise StorageError(
                f"DynamoDB list prefix '{prefix}' is too broad; include at least the scope and identifier "
                f"(e.g. 'sessions/<id>/'). Cross-partition listing requires a DynamoDBListQuery."
            )
        pk = "/".join(segments[:2])
        sk_prefix = "/".join(segments[2:])
        # Restore the trailing slash the segment split drops:
        # 'a/b/turns/' must not match 'a/b/turnstile'.
        if sk_prefix and full.endswith("/"):
            sk_prefix += "/"
        keys = await self._list_by_query(DynamoDBListQuery(pk=pk, sk_prefix=sk_prefix or None))
        # begins_with on the sort key can't exclude the partition's sentinel row;
        # enforce true string-prefix semantics on the full key.
        return [key for key in keys if key.startswith(normalized)]

    async def _list_by_query(self, query: DynamoDBListQuery) -> builtins.list[str]:
        if query.sk_prefix is not None and query.sk_between is not None:
            raise StorageError("DynamoDBListQuery accepts sk_prefix or sk_between, not both")
        self._assert_pk_in_scope(query.pk)

        names: dict[str, str] = {"#pk": _PK, "#k": _KEY_ATTR}
        values: dict[str, Any] = {":pk": {"S": query.pk}}
        condition = "#pk = :pk"
        if query.sk_prefix is not None:
            names["#sk"] = _SK
            values[":sk"] = {"S": query.sk_prefix}
            condition += " AND begins_with(#sk, :sk)"
        elif query.sk_between is not None:
            names["#sk"] = _SK
            values[":from"] = {"S": query.sk_between[0]}
            values[":to"] = {"S": query.sk_between[1]}
            condition += " AND #sk BETWEEN :from AND :to"

        # Hide expired items — only when TTL is opted in. attribute_not_exists keeps
        # rows written before TTL was enabled from being dropped.
        filter_expression: Optional[str] = None
        if self._ttl_enabled:
            names["#ttl"] = self._ttl_attribute
            values[":now"] = {"N": str(int(time.time()))}
            filter_expression = "(attribute_not_exists(#ttl) OR #ttl > :now)"

        client = self._get_client()
        keys: builtins.list[str] = []
        exclusive_start_key: Optional[dict[str, Any]] = None
        while True:
            kwargs: dict[str, Any] = {
                "TableName": self._table_name,
                "KeyConditionExpression": condition,
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
                "ProjectionExpression": "#k",
            }
            if filter_expression:
                kwargs["FilterExpression"] = filter_expression
            # DynamoDB's Limit caps items *evaluated* before the FilterExpression, and our
            # start_after skip is applied client-side — so passing Limit while either is
            # active can under-return. Only push Limit down when neither is in play;
            # otherwise the client-side count below is authoritative.
            if query.limit and not filter_expression and query.start_after is None:
                kwargs["Limit"] = query.limit
            if exclusive_start_key:
                kwargs["ExclusiveStartKey"] = exclusive_start_key

            response = await asyncio.to_thread(client.query, **kwargs)
            for item in response.get("Items", []):
                stored = item.get(_KEY_ATTR, {}).get("S")
                if stored is None:
                    continue
                key = self._strip_prefix(stored)
                if key is None:
                    continue
                # Apply start_after during collection (before the limit check) so a limit
                # is filled with post-cursor keys rather than truncated after the fact.
                if query.start_after is not None and key <= query.start_after:
                    continue
                keys.append(key)
                if query.limit and len(keys) >= query.limit:
                    return sorted(keys)
            exclusive_start_key = response.get("LastEvaluatedKey")
            if not exclusive_start_key:
                break
        return sorted(keys)

    async def _put(self, item: dict[str, Any], *, return_old: bool = False) -> dict[str, Any]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"TableName": self._table_name, "Item": item}
        if return_old:
            kwargs["ReturnValues"] = "ALL_OLD"
        response = await asyncio.to_thread(client.put_item, **kwargs)
        attributes: dict[str, Any] = response.get("Attributes", {})
        return attributes

    def _s3_key(self, full_key: str) -> str:
        return f"{self._s3_prefix}{full_key}"

    async def _s3_put(self, full_key: str, data: bytes) -> None:
        client = self._get_s3_client()
        await asyncio.to_thread(client.put_object, Bucket=self._s3_bucket, Key=self._s3_key(full_key), Body=data)

    async def _s3_get(self, full_key: str) -> Optional[bytes]:
        client = self._get_s3_client()
        response = await asyncio.to_thread(client.get_object, Bucket=self._s3_bucket, Key=self._s3_key(full_key))
        return await asyncio.to_thread(response["Body"].read)

    async def _s3_delete(self, full_key: str) -> None:
        client = self._get_s3_client()
        await asyncio.to_thread(client.delete_object, Bucket=self._s3_bucket, Key=self._s3_key(full_key))

    def _make_config(self) -> Any:
        from botocore.config import Config

        config = self._boto_client_config
        if config is None:
            return Config(user_agent_extra="strands-dynamodb-storage")
        if not getattr(config, "user_agent_extra", None):
            return config.merge(Config(user_agent_extra="strands-dynamodb-storage"))
        return config

    def _session(self) -> Any:
        import boto3

        return self._boto_session if self._boto_session is not None else boto3.Session(region_name=self._region_name)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = self._session().client("dynamodb", config=self._make_config())
        return self._client

    def _get_s3_client(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client
        self._s3_client = self._session().client("s3", config=self._make_config())
        return self._s3_client
