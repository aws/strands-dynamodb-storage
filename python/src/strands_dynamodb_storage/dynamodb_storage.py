# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon DynamoDB ``Storage`` backend for the Strands Agents SDK."""

from __future__ import annotations

import asyncio
import builtins
import gzip
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

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
# Receives a params dict and returns match dicts: {"key", "score", "metadata"?}.
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
            ttl_seconds: Default TTL. Setting it opts in to TTL (stamp + read/list
                filter). A per-write ``ttl_seconds`` override applies only when the
                instance opted in here.
            ttl_attribute: Item attribute holding the epoch-seconds TTL.
            index_name: Vector index name (for ``search``).
            vector_attribute: Item attribute holding the embedding vector.
            vector_search: Adapter performing the native vector search.
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
        enable :meth:`search`.

        Raises:
            StorageError: If the key is invalid, the value is oversized with no S3
                bucket configured, or the write fails.
        """
        normalized = normalize_key(key)
        full = f"{self._prefix}{normalized}"
        pk, sk = self._split(full)

        extra: dict[str, Any] = {}
        # The embedding stays inline even when the payload offloads, because the
        # native vector index can only index an on-item attribute.
        if vector is not None:
            extra[self._vector_attribute] = {"L": [{"N": str(component)} for component in vector]}
        if metadata is not None:
            extra[_META_ATTR] = _marshal_meta(metadata)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        if self._ttl_enabled and effective_ttl is not None:
            # Floor the whole stamp: a float duration (e.g. 90.5) must not emit a
            # fractional value that other readers of the shared table may not parse.
            extra[self._ttl_attribute] = {"N": str(int(time.time() + effective_ttl))}

        payload = data
        compressed = False
        if self._compress:
            gz = gzip.compress(data)
            if len(gz) < len(data):
                payload = gz
                compressed = True
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
                    try:
                        await self._s3_delete(full)
                    except Exception:  # noqa: BLE001
                        # Best-effort: the write itself is durable, so a failed
                        # cleanup must not fail it. An S3 lifecycle rule is the
                        # backstop for missed reclamations.
                        pass
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to write '{normalized}' to DynamoDB table '{self._table_name}'") from error

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
            if self._ttl_enabled and self._is_expired(item):
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
        if self._boto_session is not None:
            kwargs["boto_session"] = self._boto_session
        if self._boto_client_config is not None:
            kwargs["boto_client_config"] = self._boto_client_config
        return DynamoDBStorage(self._table_name, **kwargs)

    async def search(self, query: SearchQuery) -> builtins.list[SearchResult]:
        """Nearest-neighbour vector search over items written with a ``vector``.

        Optional part of the ``Storage`` contract — consumers feature-detect
        (``hasattr(storage, "search")``) and fall back to client-side KNN when absent.
        Calls DynamoDB ``SearchVectors`` natively (requires boto3 >= 1.43.64); a
        ``vector_search`` adapter, when configured, overrides the native call.

        The score is the raw ``Score`` from the vector index and its direction
        depends on the index's distance function: for COSINE and EUCLIDEAN lower
        is closer; for DOT_PRODUCT higher is more similar. Results are returned
        in the service's most-similar-first order. Like a global secondary
        index, the vector index is eventually consistent. Unlike ``read``/``list``,
        results are not filtered for TTL expiry: because TTL deletion is
        asynchronous, a search can briefly return items whose expiry has passed
        but which DynamoDB has not yet physically deleted.

        Raises:
            StorageError: If the search fails or ``top_k`` is out of range.
        """
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
                data = None
                if query.include_values:
                    data = await self.read(key)
                results.append(SearchResult(key=key, score=match["score"], data=data, metadata=match.get("metadata")))
            return results
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(f"Failed to search DynamoDB table '{self._table_name}'") from error

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

        request: dict[str, Any] = {
            "TableName": self._table_name,
            "IndexName": self._index_name,
            "SearchVector": [{"N": str(v)} for v in query.vector],
            "TopK": top_k,
        }
        if query.pk is not None:
            request["SearchConditionExpression"] = "#pk = :pk"
            request["ExpressionAttributeNames"] = {"#pk": _PK}
            request["ExpressionAttributeValues"] = {":pk": {"S": query.pk}}

        client = self._get_client()
        response = await asyncio.to_thread(client.search_vectors, **request)

        matches: builtins.list[dict[str, Any]] = []
        for result in response.get("SearchResults", []):
            item = result["Item"]
            pk = item[_PK]["S"]
            sk = item.get(_SK, {}).get("S", "")
            full_key = pk if sk in ("", _SENTINEL_SK) else f"{pk}/{sk}"
            metadata = _unmarshal_meta(item.get(_META_ATTR))
            if query.filter is not None and not self._matches_search_filter(metadata, query.filter):
                continue
            match: dict[str, Any] = {"key": full_key, "score": float(result["Score"])}
            if metadata is not None:
                match["metadata"] = metadata
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
        raw = item.get(self._ttl_attribute, {}).get("N")
        if raw is None:
            return False
        try:
            return float(raw) <= time.time()
        except ValueError:
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
        return await self._list_by_query(DynamoDBListQuery(pk=pk, sk_prefix=sk_prefix or None))

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
