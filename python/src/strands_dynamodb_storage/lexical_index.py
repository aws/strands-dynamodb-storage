# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in lexical document index over :class:`DynamoDBStorage` (preview).

The index owns the write path of indexed documents: the base item, its manifest and its
postings change in one ``TransactWriteItems`` call. Retrieval treats postings as hints and
validates every candidate against a strongly consistent read of the base item.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import functools
import gzip
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Optional, TypeVar

from strands.types.exceptions import StorageError

from .dynamodb_storage import (
    _DATA_ATTR,
    _KEY_ATTR,
    _META_ATTR,
    _PK,
    _S3_ATTR,
    _SK,
    _Z_ATTR,
    DynamoDBStorage,
    _DocumentItemPort,
    _DocumentLocation,
    _MetaValue,
    _unmarshal_meta,
)
from .lexical_terms import (
    LEXICAL_TOKENIZER_VERSION,
    identifier_posting_pk,
    manifest_pk,
    normalize_identifier,
    normalize_identifiers,
    posting_partition_keys,
    scope_segment,
    term_posting_pk,
    text_terms,
    utf8_byte_length,
)

_REVISION_ATTR = "rev"
_TOKENIZER_ATTR = "tv"
_TERMS_ATTR = "terms"
_IDENTIFIERS_ATTR = "ids"
_INDEXED_REVISION_ATTR = "lxrev"
_INDEXED_SCOPE_ATTR = "lxscope"
_RESERVED_ATTRIBUTE_NAMES = frozenset(
    {
        _PK,
        _SK,
        _KEY_ATTR,
        _DATA_ATTR,
        _S3_ATTR,
        _META_ATTR,
        _Z_ATTR,
        _REVISION_ATTR,
        _TOKENIZER_ATTR,
        _TERMS_ATTR,
        _IDENTIFIERS_ATTR,
        _INDEXED_REVISION_ATTR,
        _INDEXED_SCOPE_ATTR,
    }
)

_POSTING_ATTRIBUTES = (_SK, _REVISION_ATTR)
_MANIFEST_ATTRIBUTES = (_PK, _SK, _REVISION_ATTR, _TERMS_ATTR, _IDENTIFIERS_ATTR)
_OWNERSHIP_ATTRIBUTES = (_PK, _INDEXED_SCOPE_ATTR, _S3_ATTR)
_VALIDATION_ATTRIBUTES = (_PK, _SK, _KEY_ATTR, _INDEXED_REVISION_ATTR, _INDEXED_SCOPE_ATTR, _META_ATTR, _S3_ATTR)
_VALUE_ATTRIBUTES = (_PK, _SK, _INDEXED_REVISION_ATTR, _DATA_ATTR, _Z_ATTR, _S3_ATTR)
_REPAIR_ATTRIBUTES = (_PK, _INDEXED_REVISION_ATTR, _INDEXED_SCOPE_ATTR)

_MAX_TRANSACTION_ACTIONS = 100
_MAX_POSTINGS_PER_DOCUMENT = (_MAX_TRANSACTION_ACTIONS - 2) // 2
_MAX_TRANSACTION_BYTES = 4_000_000
_MAX_PAGE_SIZE = 1000
_MAX_CANDIDATES = 1000
_MAX_CONCURRENCY = 16
_MAX_TOP_K = 100
_MAX_SORT_KEY_BYTES = 1024
_MAX_PARTITION_KEY_BYTES = 2048
_MAX_BATCH_GET_KEYS = 100
_BACKOFF_BASE_SECONDS = 0.05
_CONFLICT_BACKOFF_CAP_SECONDS = 0.5
_UNPROCESSED_BACKOFF_CAP_SECONDS = 1.0
_RETRY_LIMIT_NAMES = frozenset({"max_conflict_retries", "max_unprocessed_retries"})
_CONFLICT_CANCELLATION_CODES = frozenset({"ConditionalCheckFailed", "TransactionConflict"})
_THROTTLING_CANCELLATION_CODES = frozenset({"ThrottlingError", "ProvisionedThroughputExceeded", "RequestLimitExceeded"})

_PAGES_TRUNCATION = "max_pages_per_term"
_CANDIDATES_TRUNCATION = "max_candidates"

_T = TypeVar("_T")


@dataclass(frozen=True)
class SearchableDocument:
    """A document written through :meth:`LexicalIndex.upsert`.

    ``data`` is stored as :meth:`DynamoDBStorage.write` stores an inline value. ``text`` is
    tokenized into terms and ``identifiers`` are indexed for exact, case-sensitive lookup;
    neither is stored, so a tokenizer change requires re-upserting from your source of truth.
    """

    key: str
    data: bytes
    text: str = ""
    identifiers: Sequence[str] = ()
    metadata: Optional[dict[str, _MetaValue]] = None
    vector: Optional[list[float]] = None
    ttl_seconds: Optional[int] = None


@dataclass(frozen=True)
class LexicalIndexLimits:
    """Bounds on documents, queries and retries, validated on construction.

    Every limit is a positive integer except the two retry counts, which may be 0.
    ``max_postings_per_document`` is capped so that an overwrite with disjoint postings
    (``2 + old + new`` actions) fits in one TransactWriteItems call. ``max_conflict_retries``
    bounds the retries of a transaction cancelled by a concurrent writer or by throttling;
    ``max_unprocessed_retries`` bounds consecutive ``BatchGetItem`` rounds without progress.

    Raises:
        StorageError: If a limit is not an integer, is out of range, or ``max_query_terms``
            exceeds ``max_postings_per_document``.
    """

    max_postings_per_document: int = _MAX_POSTINGS_PER_DOCUMENT
    max_term_bytes: int = 64
    max_identifier_bytes: int = 128
    max_text_bytes: int = 65_536
    max_query_terms: int = 16
    page_size: int = 100
    max_pages_per_term: int = 5
    max_candidates: int = 300
    max_concurrency: int = 4
    max_conflict_retries: int = 3
    max_unprocessed_retries: int = 5

    def __post_init__(self) -> None:
        for limit in fields(self):
            minimum = 0 if limit.name in _RETRY_LIMIT_NAMES else 1
            _require_integer(f"LexicalIndexLimits.{limit.name}", getattr(self, limit.name), minimum)
        _require_at_most(
            "max_postings_per_document",
            self.max_postings_per_document,
            _MAX_POSTINGS_PER_DOCUMENT,
            f" (an overwrite with disjoint postings needs 2 + old + new <= {_MAX_TRANSACTION_ACTIONS} "
            "TransactWriteItems actions)",
        )
        _require_at_most("page_size", self.page_size, _MAX_PAGE_SIZE)
        _require_at_most("max_candidates", self.max_candidates, _MAX_CANDIDATES)
        _require_at_most("max_concurrency", self.max_concurrency, _MAX_CONCURRENCY)
        _require_at_most(
            "max_query_terms", self.max_query_terms, self.max_postings_per_document, " (max_postings_per_document)"
        )


@dataclass
class LexicalQuery:
    """Term query: documents matching any (or, with ``require_all_terms``, every) query term.

    With ``include_values``, values are read by a second consistent read of the ranked
    ``top_k`` results only; a result whose document changed in between is dropped, so fewer
    than ``top_k`` results may be returned.
    """

    text: str
    top_k: int
    require_all_terms: bool = False
    filter: Optional[dict[str, _MetaValue]] = None
    include_values: bool = False


@dataclass
class IdentifierQuery:
    """Exact, case-sensitive identifier lookup; ``include_values`` behaves as in :class:`LexicalQuery`."""

    identifier: str
    top_k: int
    filter: Optional[dict[str, _MetaValue]] = None
    include_values: bool = False


@dataclass
class LexicalSearchResult:
    """A validated match. ``score`` is ``matched_terms`` divided by the number of query terms."""

    key: str
    score: float
    matched_terms: int
    data: Optional[bytes] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class LexicalSearchResponse:
    """Ranked results plus why the candidate set may be incomplete.

    ``truncation_reasons`` holds ``"max_pages_per_term"`` and/or ``"max_candidates"``, sorted.
    """

    results: list[LexicalSearchResult]
    truncated: bool
    truncation_reasons: list[str]
    candidates_examined: int


@dataclass
class RepairReport:
    """Outcome of one :meth:`LexicalIndex.repair` pass; pass ``cursor`` back to resume."""

    documents_checked: int
    documents_removed: int
    postings_rebuilt: int
    cursor: Optional[str]


class RevisionConflictError(StorageError):
    """``expected_revision`` did not match the document's current revision."""


class _TransactionCancelled(Exception):
    """A cancelled ``TransactWriteItems`` that upsert and delete retry after re-reading state.

    ``cancellation`` is the service error; it becomes the cause of the error raised once
    retries are exhausted.
    """

    reason = "transaction cancelled"

    def __init__(self, cancellation: Exception) -> None:
        super().__init__(self.reason)
        self.cancellation = cancellation


class _WriteConflict(_TransactionCancelled):
    """Cancelled by a failed condition or a concurrent transaction; repair skips the document."""

    reason = "concurrent modification"


class _WriteThrottled(_TransactionCancelled):
    """Cancelled because an item was throttled; repair treats it as a failure."""

    reason = "transaction throttled"


class _RepairOutcome(enum.Enum):
    KEPT = enum.auto()
    REMOVED = enum.auto()
    REBUILT = enum.auto()
    SKIPPED = enum.auto()


@dataclass(frozen=True)
class _Manifest:
    doc_id: str
    revision: str
    terms: tuple[str, ...]
    identifiers: tuple[str, ...]


@dataclass(frozen=True)
class _UpsertPlan:
    location: _DocumentLocation
    terms: list[str]
    identifiers: list[str]
    postings: list[str]
    base_item: dict[str, Any]


@dataclass(frozen=True)
class _PostingList:
    postings: list[tuple[str, str]]
    truncated: bool


@dataclass(frozen=True)
class _Retrieval:
    partitions: list[str]
    top_k: int
    filter: Optional[dict[str, _MetaValue]]
    include_values: bool
    required_matches: int


@dataclass
class _Candidate:
    location: _DocumentLocation
    partitions_by_revision: dict[str, set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Match:
    """A validated result and the base-item revision it was validated at."""

    location: _DocumentLocation
    revision: str
    result: LexicalSearchResult


@dataclass
class _CandidatePool:
    """Deterministic merge of posting lists, independent of fetch timing.

    Postings are admitted in query-term order, then sort-key order. Doc ids that are outside
    the scope (or not canonical) are dropped here, before any base-table read. Once the pool
    is full, new doc ids are skipped while admitted candidates keep accumulating matches.
    """

    max_candidates: int
    locate: Callable[[str], Optional[_DocumentLocation]]
    candidates: dict[str, _Candidate] = field(default_factory=dict)
    truncation_reasons: set[str] = field(default_factory=set)

    def add(self, partition_index: int, doc_id: str, revision: str) -> None:
        candidate = self.candidates.get(doc_id) or self._admit(doc_id)
        if candidate is not None:
            candidate.partitions_by_revision.setdefault(revision, set()).add(partition_index)

    def _admit(self, doc_id: str) -> Optional[_Candidate]:
        location = self.locate(doc_id)
        if location is None:
            return None
        if len(self.candidates) >= self.max_candidates:
            self.truncation_reasons.add(_CANDIDATES_TRUNCATION)
            return None
        candidate = self.candidates[doc_id] = _Candidate(location)
        return candidate


class LexicalIndex:
    """Opt-in lexical index (terms and exact identifiers) over a :class:`DynamoDBStorage` (preview).

    Documents are written through :meth:`upsert`, which stores the base item, a manifest and one
    posting per term/identifier atomically in a separate, user-provisioned index table (``pk``/``sk``
    strings). Not BM25, not hybrid, not phrase or substring search; ``DynamoDBStorage.search()`` is
    unchanged. A direct ``storage.write()`` over an indexed document drops its index markers, which
    hides it from retrieval until it is re-upserted; :meth:`repair` removes the leftover entries.

    Example:
        ```python
        index = LexicalIndex(storage.namespace("tenant-a"), index_table_name="agent-data-lexical")
        await index.upsert(SearchableDocument(key="docs/inv-1", data=body, text=text, identifiers=["FC-00123"]))
        response = await index.search(LexicalQuery(text="overdue invoice", top_k=5))
        ```
    """

    def __init__(
        self,
        storage: DynamoDBStorage,
        *,
        index_table_name: str,
        limits: Optional[LexicalIndexLimits] = None,
    ) -> None:
        """Bind the index to a storage namespace.

        Args:
            storage: Storage whose prefix is the index scope; bind the authenticated tenant with
                ``storage.namespace(...)``.
            index_table_name: Index table with key schema ``pk`` (S, HASH) and ``sk`` (S, RANGE).
            limits: Document, query and retry bounds; defaults to :class:`LexicalIndexLimits`.

        Raises:
            StorageError: If ``index_table_name`` is empty, the scope is longer than 1024 UTF-8
                bytes, or the storage's TTL attribute (when TTL is enabled) collides with an
                attribute the index writes.
        """
        self._port: _DocumentItemPort = storage._document_item_port()
        self._index_table_name = _require_table_name(index_table_name)
        self._limits = limits if limits is not None else LexicalIndexLimits()
        _require_size(
            "The storage scope", utf8_byte_length(self._port.scope), "the sort key limit", _MAX_SORT_KEY_BYTES
        )
        self._manifest_partition = manifest_pk(self._port.scope)
        _require_partition_keys([self._manifest_partition])
        _require_unreserved_ttl_attribute(self._port.ttl_attribute)
        self._scope_marker = scope_segment(self._port.scope)

    @property
    def scope(self) -> str:
        """The storage prefix this index is confined to (``""`` for an unprefixed storage)."""
        return self._port.scope

    async def upsert(self, document: SearchableDocument, *, expected_revision: Optional[str] = None) -> str:
        """Store ``document`` and replace its postings atomically.

        All validation runs before any I/O and nothing is truncated.

        Args:
            document: The document, its searchable text and identifiers.
            expected_revision: When set, the write succeeds only if the document's current
                revision equals it (optimistic concurrency).

        Returns:
            The new revision.

        Raises:
            RevisionConflictError: If ``expected_revision`` does not match.
            StorageError: If validation fails (including text or identifiers that are not
                well-formed Unicode and index keys above the DynamoDB key limits), the key is
                owned by another lexical scope, the manifest is corrupt, retries after
                concurrent modification or throttling are exhausted (the last cancellation is
                the cause), or DynamoDB fails.
        """
        return await self._guard("upsert", document.key, lambda: self._upsert(document, expected_revision))

    async def delete(self, key: str, *, expected_revision: Optional[str] = None) -> bool:
        """Delete the document, its manifest and postings atomically.

        Returns:
            ``True`` if anything was deleted.

        Raises:
            RevisionConflictError: If ``expected_revision`` does not match.
            StorageError: If the key is invalid, above the sort-key limit or owned by another
                lexical scope, the manifest is corrupt, retries are exhausted, or DynamoDB fails.
        """
        return await self._guard("delete", key, lambda: self._delete(key, expected_revision))

    async def revision(self, key: str) -> Optional[str]:
        """Current revision of an indexed document (strongly consistent), or ``None``.

        Raises:
            StorageError: If the key is invalid or above the sort-key limit, the manifest is
                corrupt, or the read fails.
        """
        return await self._guard("revision", key, lambda: self._revision(key))

    async def search(self, query: LexicalQuery) -> LexicalSearchResponse:
        """Rank documents by the fraction of query terms they contain.

        Ties break by key (UTF-8 byte order). Candidates come from eventually consistent posting
        queries and are validated with a strongly consistent read of each base item; values are
        read afterwards for the kept results only (see :class:`LexicalQuery`).

        Raises:
            StorageError: If ``top_k`` is outside 1..100, the query is not well-formed Unicode,
                has no terms, too many terms, an oversized term or partition key, keys stay
                unprocessed, or DynamoDB fails.
        """
        return await self._guard("search", None, lambda: self._search(query))

    async def lookup(self, query: IdentifierQuery) -> LexicalSearchResponse:
        """Find documents indexed with exactly ``query.identifier``, ordered by key.

        Raises:
            StorageError: If ``top_k`` is outside 1..100, the identifier is invalid or oversized,
                keys stay unprocessed, or DynamoDB fails.
        """
        return await self._guard("lookup", None, lambda: self._lookup(query))

    async def repair(
        self, *, max_documents: int = 100, cursor: Optional[str] = None, rebuild_postings: bool = False
    ) -> RepairReport:
        """Remove index entries whose base item is gone, stale, foreign or expired.

        Walks this scope's manifests sequentially (never a Scan), resuming after ``cursor``; an
        empty cursor starts from the beginning. With ``rebuild_postings`` it also re-puts the
        postings of valid documents. It cannot re-tokenize, because source text is not stored.
        A document changed concurrently is skipped silently and still counted as checked.

        Raises:
            StorageError: If ``max_documents`` is not a positive integer, a manifest is corrupt,
                or a transaction fails for any reason other than a concurrent writer (throttling
                included); the first such failure stops the pass.
        """
        return await self._guard("repair", None, lambda: self._repair(max_documents, cursor, rebuild_postings))

    async def _guard(self, operation: str, key: Optional[str], action: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``action``, wrapping unexpected errors in a :class:`StorageError` naming the operation and tables.

        The message may name the key but never document text, terms or identifiers.
        """
        try:
            return await action()
        except StorageError:
            raise
        except Exception as error:
            raise StorageError(self._failure_message(operation, key)) from error

    def _failure_message(self, operation: str, key: Optional[str]) -> str:
        subject = f" for '{key}'" if key is not None else ""
        return (
            f"Lexical index {operation} failed{subject} "
            f"(index table '{self._index_table_name}', base table '{self._port.table_name}')"
        )

    async def _upsert(self, document: SearchableDocument, expected_revision: Optional[str]) -> str:
        plan = self._plan_upsert(document)
        return await self._retry_cancelled(
            "upsert", plan.location.key, lambda: self._attempt_upsert(plan, expected_revision)
        )

    def _plan_upsert(self, document: SearchableDocument) -> _UpsertPlan:
        location = self._locate(document.key)
        terms = self._document_terms(document.text)
        identifiers = self._document_identifiers(document.identifiers)
        self._require_posting_budget(len(terms), len(identifiers))
        postings = posting_partition_keys(self.scope, terms, identifiers)
        _require_partition_keys(postings)
        base_item = self._port.inline_item(
            location,
            document.data,
            vector=document.vector,
            metadata=document.metadata,
            ttl_seconds=document.ttl_seconds,
        )
        return _UpsertPlan(
            location=location, terms=terms, identifiers=identifiers, postings=postings, base_item=base_item
        )

    def _locate(self, key: str) -> _DocumentLocation:
        """Location of ``key``; its stored key must fit the DynamoDB sort-key limit (checked before any I/O)."""
        location = self._port.locate(key)
        _require_size(
            "The stored document key", utf8_byte_length(location.doc_id), "the sort key limit", _MAX_SORT_KEY_BYTES
        )
        return location

    def _document_terms(self, text: str) -> list[str]:
        _require_size("Document text", utf8_byte_length(text), "max_text_bytes", self._limits.max_text_bytes)
        terms = text_terms(text)
        for term in terms:
            _require_size("A term", utf8_byte_length(term), "max_term_bytes", self._limits.max_term_bytes)
        return terms

    def _document_identifiers(self, identifiers: Sequence[str]) -> list[str]:
        """Normalize identifiers; a bare string is rejected because it would index each character."""
        if isinstance(identifiers, str):
            raise StorageError("SearchableDocument.identifiers must be a sequence of identifiers, not a string")
        normalized = normalize_identifiers(identifiers)
        for identifier in normalized:
            _require_size(
                "An identifier", utf8_byte_length(identifier), "max_identifier_bytes", self._limits.max_identifier_bytes
            )
        return normalized

    def _require_posting_budget(self, term_count: int, identifier_count: int) -> None:
        postings = term_count + identifier_count
        limit = self._limits.max_postings_per_document
        if postings > limit:
            raise StorageError(
                f"Document needs {postings} postings ({term_count} terms + {identifier_count} identifiers), "
                f"above max_postings_per_document ({limit}); split the document or reduce distinct terms"
            )

    async def _attempt_upsert(self, plan: _UpsertPlan, expected_revision: Optional[str]) -> str:
        manifest, base = await self._read_current(plan.location)
        self._check_preconditions(plan.location, manifest, base, expected_revision)
        revision = uuid.uuid4().hex
        await self._transact(self._upsert_actions(plan, manifest, revision), token=revision)
        await self._reclaim_offloaded(plan.location, base)
        return revision

    def _upsert_actions(self, plan: _UpsertPlan, manifest: Optional[_Manifest], revision: str) -> list[dict[str, Any]]:
        """Base put, manifest put, every new posting (shared ones get the new revision), then stale deletes."""
        doc_id = plan.location.doc_id
        retained = set(plan.postings)
        stale_postings = [pk for pk in self._manifest_postings(manifest) if pk not in retained]
        ttl_attributes = self._ttl_attributes(plan.base_item)
        return [
            self._put_base_action(plan.base_item, revision),
            self._put_manifest_action(plan, manifest, revision),
            *(self._put_posting_action(pk, doc_id, revision, ttl_attributes) for pk in plan.postings),
            *(self._delete_posting_action(pk, doc_id) for pk in stale_postings),
        ]

    async def _delete(self, key: str, expected_revision: Optional[str]) -> bool:
        location = self._locate(key)
        return await self._retry_cancelled(
            "delete", location.key, lambda: self._attempt_delete(location, expected_revision)
        )

    async def _attempt_delete(self, location: _DocumentLocation, expected_revision: Optional[str]) -> bool:
        manifest, base = await self._read_current(location)
        self._check_preconditions(location, manifest, base, expected_revision)
        if manifest is None and base is None:
            return False
        await self._transact(self._delete_actions(location, manifest, base))
        await self._reclaim_offloaded(location, base)
        return True

    def _delete_actions(
        self, location: _DocumentLocation, manifest: Optional[_Manifest], base: Optional[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Delete the base item (when present), then the manifest and its postings.

        Without a manifest, a ``ConditionCheck`` that none exists takes their place: a first
        upsert committed between the reads and this transaction would otherwise keep its
        manifest and postings while its base item is deleted.
        """
        base_actions = [self._delete_base_action(location)] if base is not None else []
        if manifest is None:
            return [*base_actions, self._manifest_absent_check(location.doc_id)]
        return [*base_actions, *self._manifest_removal_actions(manifest)]

    async def _revision(self, key: str) -> Optional[str]:
        manifest = await self._read_manifest(self._locate(key).doc_id)
        return manifest.revision if manifest is not None else None

    async def _read_current(self, location: _DocumentLocation) -> tuple[Optional[_Manifest], Optional[dict[str, Any]]]:
        """Consistent reads of the manifest and of the base item's ownership markers, in parallel."""
        manifest: Optional[_Manifest]
        base: Optional[dict[str, Any]]
        manifest, base = await asyncio.gather(
            self._read_manifest(location.doc_id),
            self._read_base_item(location, _OWNERSHIP_ATTRIBUTES),
        )
        return manifest, base

    def _check_preconditions(
        self,
        location: _DocumentLocation,
        manifest: Optional[_Manifest],
        base: Optional[dict[str, Any]],
        expected_revision: Optional[str],
    ) -> None:
        current_revision = manifest.revision if manifest is not None else None
        if expected_revision is not None and current_revision != expected_revision:
            raise RevisionConflictError(
                f"Lexical index revision conflict for '{location.key}': it changed since the expected revision"
            )
        if base is not None and _string_attr(base, _INDEXED_SCOPE_ATTR) not in (None, self._scope_marker):
            raise StorageError(
                f"'{location.key}' is owned by another lexical scope; write it through that scope's index"
            )

    async def _reclaim_offloaded(self, location: _DocumentLocation, base: Optional[dict[str, Any]]) -> None:
        """Delete the S3 object a replaced offloaded value left behind (best-effort, via the port)."""
        if base is not None and _is_offloaded(base):
            await self._port.delete_offloaded(location.doc_id)

    async def _retry_cancelled(self, operation: str, key: str, attempt: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``attempt`` until it commits, re-reading state after each cancelled transaction.

        Conflicts and throttling are both retried, up to ``max_conflict_retries`` times, sleeping
        ``min(0.05 * 2**n, 0.5)`` seconds after failed attempt ``n`` only when another attempt
        follows. The error raised once retries are exhausted chains the last cancellation.
        """
        for attempt_number in range(self._limits.max_conflict_retries):
            try:
                return await attempt()
            except _TransactionCancelled:
                await asyncio.sleep(_backoff(attempt_number, _CONFLICT_BACKOFF_CAP_SECONDS))
        try:
            return await attempt()
        except _TransactionCancelled as cancelled:
            raise StorageError(
                f"{self._failure_message(operation, key)}: {cancelled.reason}; retries exhausted"
            ) from cancelled.cancellation

    async def _transact(self, actions: list[dict[str, Any]], *, token: Optional[str] = None) -> None:
        """Run one ``TransactWriteItems``; conflict and throttling cancellations raise ``_TransactionCancelled``."""
        _require_transaction_bounds(actions)
        request: dict[str, Any] = {"TransactItems": actions}
        if token is not None:
            request["ClientRequestToken"] = token
        try:
            await asyncio.to_thread(self._client().transact_write_items, **request)
        except Exception as error:
            cancelled = _classify_cancellation(error)
            if cancelled is None:
                raise
            raise cancelled(error) from error

    def _put_base_action(self, base_item: dict[str, Any], revision: str) -> dict[str, Any]:
        item = {
            **base_item,
            _INDEXED_REVISION_ATTR: {"S": revision},
            _INDEXED_SCOPE_ATTR: {"S": self._scope_marker},
        }
        return {"Put": {"TableName": self._port.table_name, "Item": item, **self._owned_by_scope_condition()}}

    def _delete_base_action(self, location: _DocumentLocation) -> dict[str, Any]:
        return {
            "Delete": {
                "TableName": self._port.table_name,
                "Key": _primary_key(location.pk, location.sk),
                **self._owned_by_scope_condition(),
            }
        }

    def _owned_by_scope_condition(self) -> dict[str, Any]:
        return {
            "ConditionExpression": "attribute_not_exists(#lxscope) OR #lxscope = :scope",
            "ExpressionAttributeNames": {"#lxscope": _INDEXED_SCOPE_ATTR},
            "ExpressionAttributeValues": {":scope": {"S": self._scope_marker}},
        }

    def _put_manifest_action(self, plan: _UpsertPlan, manifest: Optional[_Manifest], revision: str) -> dict[str, Any]:
        item = {
            **self._manifest_key(plan.location.doc_id),
            _REVISION_ATTR: {"S": revision},
            _TOKENIZER_ATTR: {"S": LEXICAL_TOKENIZER_VERSION},
            _TERMS_ATTR: _string_list_value(plan.terms),
            _IDENTIFIERS_ATTR: _string_list_value(plan.identifiers),
        }
        condition = _revision_condition(manifest.revision) if manifest is not None else _absent_condition()
        return {"Put": {"TableName": self._index_table_name, "Item": item, **condition}}

    def _manifest_absent_check(self, doc_id: str) -> dict[str, Any]:
        return {
            "ConditionCheck": {
                "TableName": self._index_table_name,
                "Key": self._manifest_key(doc_id),
                **_absent_condition(),
            }
        }

    def _manifest_removal_actions(self, manifest: _Manifest) -> list[dict[str, Any]]:
        return [
            {
                "Delete": {
                    "TableName": self._index_table_name,
                    "Key": self._manifest_key(manifest.doc_id),
                    **_revision_condition(manifest.revision),
                }
            },
            *(self._delete_posting_action(pk, manifest.doc_id) for pk in self._manifest_postings(manifest)),
        ]

    def _rebuild_actions(self, manifest: _Manifest, base: dict[str, Any]) -> list[dict[str, Any]]:
        ttl_attributes = self._ttl_attributes(base)
        return [
            {
                "ConditionCheck": {
                    "TableName": self._index_table_name,
                    "Key": self._manifest_key(manifest.doc_id),
                    **_revision_condition(manifest.revision),
                }
            },
            *(
                self._put_posting_action(pk, manifest.doc_id, manifest.revision, ttl_attributes)
                for pk in self._manifest_postings(manifest)
            ),
        ]

    def _put_posting_action(
        self, partition: str, doc_id: str, revision: str, ttl_attributes: dict[str, Any]
    ) -> dict[str, Any]:
        item = {**_primary_key(partition, doc_id), _REVISION_ATTR: {"S": revision}, **ttl_attributes}
        return {"Put": {"TableName": self._index_table_name, "Item": item}}

    def _delete_posting_action(self, partition: str, doc_id: str) -> dict[str, Any]:
        return {"Delete": {"TableName": self._index_table_name, "Key": _primary_key(partition, doc_id)}}

    def _manifest_key(self, doc_id: str) -> dict[str, Any]:
        return _primary_key(self._manifest_partition, doc_id)

    def _manifest_postings(self, manifest: Optional[_Manifest]) -> list[str]:
        if manifest is None:
            return []
        return posting_partition_keys(self.scope, manifest.terms, manifest.identifiers)

    def _ttl_attributes(self, item: dict[str, Any]) -> dict[str, Any]:
        """The base item's TTL stamp, copied onto postings so they expire with the document."""
        name = self._port.ttl_attribute
        if name is None or name not in item:
            return {}
        return {name: item[name]}

    def _ttl_attribute_names(self) -> list[str]:
        return [self._port.ttl_attribute] if self._port.ttl_attribute is not None else []

    async def _read_manifest(self, doc_id: str) -> Optional[_Manifest]:
        response = await asyncio.to_thread(
            self._client().get_item,
            TableName=self._index_table_name,
            Key=self._manifest_key(doc_id),
            ConsistentRead=True,
        )
        item = response.get("Item")
        return self._to_manifest(item) if item else None

    def _to_manifest(self, item: dict[str, Any]) -> _Manifest:
        """Parse a manifest item; one without a string ``rev`` is corrupt, never treated as absent."""
        doc_id = item[_SK]["S"]
        revision = _string_attr(item, _REVISION_ATTR)
        if revision is None:
            raise StorageError(
                f"Lexical index manifest for '{doc_id}' in index table '{self._index_table_name}' "
                "has no string revision"
            )
        return _Manifest(
            doc_id=doc_id,
            revision=revision,
            terms=_string_list(item.get(_TERMS_ATTR)),
            identifiers=_string_list(item.get(_IDENTIFIERS_ATTR)),
        )

    async def _read_base_item(self, location: _DocumentLocation, attributes: Sequence[str]) -> Optional[dict[str, Any]]:
        """Consistent projected read of the base item; ``attributes`` must include ``pk`` to detect existence."""
        names = _attribute_names(attributes)
        response = await asyncio.to_thread(
            self._client().get_item,
            TableName=self._port.table_name,
            Key=_primary_key(location.pk, location.sk),
            ConsistentRead=True,
            ProjectionExpression=", ".join(names),
            ExpressionAttributeNames=names,
        )
        item: Optional[dict[str, Any]] = response.get("Item") or None
        return item

    async def _search(self, query: LexicalQuery) -> LexicalSearchResponse:
        _require_top_k(query.top_k)
        return await self._search_terms(
            self._query_terms(query.text),
            top_k=query.top_k,
            include_values=query.include_values,
            filter=query.filter,
            require_all_terms=query.require_all_terms,
        )

    async def _search_terms(
        self,
        terms: Sequence[str],
        *,
        top_k: int,
        include_values: bool,
        filter: Optional[dict[str, _MetaValue]] = None,
        require_all_terms: bool = False,
    ) -> LexicalSearchResponse:
        """Internal: rank documents by already tokenized query ``terms``; shared by :meth:`search` and the SDK strategy.

        Callers validate first: ``top_k`` within 1..100, and ``terms`` distinct, non-empty and within
        ``max_term_bytes`` and ``max_query_terms`` (:meth:`search` rejects a query that is not, the
        strategy trims it). Unexpected errors are wrapped as in :meth:`search`.

        Raises:
            StorageError: If a partition key is oversized, keys stay unprocessed, or DynamoDB fails.
        """
        retrieval = _Retrieval(
            partitions=[term_posting_pk(self.scope, term) for term in terms],
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            required_matches=len(terms) if require_all_terms else 1,
        )
        return await self._guard("search", None, lambda: self._retrieve(retrieval))

    def _query_terms(self, text: str) -> list[str]:
        terms = text_terms(text)
        for term in terms:
            _require_size("A query term", utf8_byte_length(term), "max_term_bytes", self._limits.max_term_bytes)
        if not terms:
            raise StorageError("Lexical query has no searchable terms")
        if len(terms) > self._limits.max_query_terms:
            raise StorageError(
                f"Lexical query has {len(terms)} distinct terms, above max_query_terms "
                f"({self._limits.max_query_terms}); shorten the query"
            )
        return terms

    async def _lookup(self, query: IdentifierQuery) -> LexicalSearchResponse:
        _require_top_k(query.top_k)
        identifier = normalize_identifier(query.identifier)
        _require_size(
            "The identifier", utf8_byte_length(identifier), "max_identifier_bytes", self._limits.max_identifier_bytes
        )
        return await self._retrieve(
            _Retrieval(
                partitions=[identifier_posting_pk(self.scope, identifier)],
                top_k=query.top_k,
                filter=query.filter,
                include_values=query.include_values,
                required_matches=1,
            )
        )

    async def _retrieve(self, retrieval: _Retrieval) -> LexicalSearchResponse:
        """Merge the posting lists, validate every candidate, rank, and read values for the kept results only."""
        _require_partition_keys(retrieval.partitions)
        posting_lists = await self._gather_bounded(
            [functools.partial(self._read_postings, partition) for partition in retrieval.partitions]
        )
        pool = _merge_postings(posting_lists, self._limits.max_candidates, self._candidate_location)
        candidates = list(pool.candidates.values())
        items = await self._read_items([candidate.location for candidate in candidates], self._validation_attributes())
        matches = sorted(
            (
                match
                for candidate in candidates
                if (match := self._to_match(candidate, items.get(_location_id(candidate.location)), retrieval))
                is not None
            ),
            key=_ranking,
        )[: retrieval.top_k]
        results = await self._with_values(matches) if retrieval.include_values else [match.result for match in matches]
        return LexicalSearchResponse(
            results=results,
            truncated=bool(pool.truncation_reasons),
            truncation_reasons=sorted(pool.truncation_reasons),
            candidates_examined=len(candidates),
        )

    async def _gather_bounded(self, tasks: Sequence[Callable[[], Awaitable[_T]]]) -> list[_T]:
        """Run ``tasks`` with at most ``max_concurrency`` in flight, returning results in input order."""
        semaphore = asyncio.Semaphore(self._limits.max_concurrency)

        async def run(task: Callable[[], Awaitable[_T]]) -> _T:
            async with semaphore:
                return await task()

        return list(await asyncio.gather(*(run(task) for task in tasks)))

    async def _read_postings(self, partition: str) -> _PostingList:
        """Page through one posting partition (eventually consistent), up to ``max_pages_per_term`` pages."""
        postings: list[tuple[str, str]] = []
        start_key: Optional[dict[str, Any]] = None
        for _ in range(self._limits.max_pages_per_term):
            request = self._partition_query(
                partition, _POSTING_ATTRIBUTES, limit=self._limits.page_size, start_key=start_key, consistent=False
            )
            response = await asyncio.to_thread(self._client().query, **request)
            postings.extend(_posting_entries(response.get("Items", [])))
            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                return _PostingList(postings=postings, truncated=False)
        return _PostingList(postings=postings, truncated=True)

    def _partition_query(
        self,
        partition: str,
        attributes: Sequence[str],
        *,
        limit: int,
        start_key: Optional[dict[str, Any]],
        consistent: bool,
    ) -> dict[str, Any]:
        projected = _attribute_names(attributes)
        request: dict[str, Any] = {
            "TableName": self._index_table_name,
            "KeyConditionExpression": "#pk = :pk",
            "ExpressionAttributeNames": {"#pk": _PK, **projected},
            "ExpressionAttributeValues": {":pk": {"S": partition}},
            "ProjectionExpression": ", ".join(projected),
            "ConsistentRead": consistent,
            "Limit": limit,
        }
        if start_key:
            request["ExclusiveStartKey"] = start_key
        return request

    def _candidate_location(self, doc_id: str) -> Optional[_DocumentLocation]:
        """Base-table location of a posting's doc id, or ``None`` if it is outside this scope or not canonical."""
        key = self._port.relative_key(doc_id)
        if key is None:
            return None
        try:
            location = self._port.locate(key)
        except StorageError:
            return None
        return location if location.doc_id == doc_id else None

    async def _read_items(
        self, locations: Sequence[_DocumentLocation], attributes: Sequence[str]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Strongly consistent ``BatchGetItem`` of ``locations`` (unique keys, chunks of at most 100) by primary key."""
        keys = list({_location_id(location): _primary_key(*_location_id(location)) for location in locations}.values())
        chunks = [keys[start : start + _MAX_BATCH_GET_KEYS] for start in range(0, len(keys), _MAX_BATCH_GET_KEYS)]
        batches = await self._gather_bounded(
            [functools.partial(self._batch_get, chunk, attributes) for chunk in chunks]
        )
        return {(item[_PK]["S"], item[_SK]["S"]): item for batch in batches for item in batch}

    def _validation_attributes(self) -> list[str]:
        return [*_VALIDATION_ATTRIBUTES, *self._ttl_attribute_names()]

    async def _batch_get(self, keys: list[dict[str, Any]], attributes: Sequence[str]) -> list[dict[str, Any]]:
        """Read ``keys``, retrying ``UnprocessedKeys`` until every key is processed.

        ``max_unprocessed_retries`` bounds consecutive rounds that return no item: a round with
        progress resets the count, so a read DynamoDB serves piecemeal never exhausts it. Each
        retry first sleeps ``min(0.05 * 2**n, 1.0)`` s, ``n`` being the stalled rounds before the
        current one (0 after a round with progress). Keys still unprocessed once the budget is
        spent raise: the lexical answer is never silently partial.
        """
        table = self._port.table_name
        names = _attribute_names(attributes)
        request_items: dict[str, Any] = {
            table: {
                "Keys": keys,
                "ConsistentRead": True,
                "ProjectionExpression": ", ".join(names),
                "ExpressionAttributeNames": names,
            }
        }
        items: list[dict[str, Any]] = []
        stalled_rounds = 0
        while True:
            response = await asyncio.to_thread(self._client().batch_get_item, RequestItems=request_items)
            returned = response.get("Responses", {}).get(table, [])
            items.extend(returned)
            request_items = response.get("UnprocessedKeys") or {}
            pending = request_items.get(table, {}).get("Keys") or []
            if not pending:
                return items
            stalled_rounds = 0 if returned else stalled_rounds + 1
            self._require_unprocessed_budget(stalled_rounds, len(pending))
            await asyncio.sleep(_backoff(max(stalled_rounds - 1, 0), _UNPROCESSED_BACKOFF_CAP_SECONDS))

    def _require_unprocessed_budget(self, stalled_rounds: int, pending_keys: int) -> None:
        retries = self._limits.max_unprocessed_retries
        if stalled_rounds > retries:
            raise StorageError(
                f"Lexical index read left {pending_keys} keys unprocessed in base table "
                f"'{self._port.table_name}' after {retries} retries without progress"
            )

    def _to_match(
        self, candidate: _Candidate, item: Optional[dict[str, Any]], retrieval: _Retrieval
    ) -> Optional[_Match]:
        if item is None or not self._is_valid_candidate(candidate, item):
            return None
        metadata = _unmarshal_meta(item.get(_META_ATTR))
        if not _matches_filter(metadata, retrieval.filter):
            return None
        revision = item[_INDEXED_REVISION_ATTR]["S"]
        matched = len(candidate.partitions_by_revision[revision])
        if matched < retrieval.required_matches:
            return None
        result = LexicalSearchResult(
            key=candidate.location.key,
            score=matched / len(retrieval.partitions),
            matched_terms=matched,
            metadata=metadata,
        )
        return _Match(location=candidate.location, revision=revision, result=result)

    def _is_valid_candidate(self, candidate: _Candidate, item: dict[str, Any]) -> bool:
        """The base item is this candidate's document, owned by this scope, at a posted revision, inline and live."""
        return (
            _string_attr(item, _KEY_ATTR) == candidate.location.doc_id
            and _string_attr(item, _INDEXED_SCOPE_ATTR) == self._scope_marker
            and _string_attr(item, _INDEXED_REVISION_ATTR) in candidate.partitions_by_revision
            and not _is_offloaded(item)
            and not self._port.is_expired(item)
        )

    async def _with_values(self, matches: Sequence[_Match]) -> list[LexicalSearchResult]:
        """Read and decode the values of the kept results with a second strongly consistent read.

        A result whose base item vanished, moved to another revision, was offloaded or holds no
        ``data`` since validation is dropped, so fewer than ``top_k`` results may be returned.
        """
        items = await self._read_items([match.location for match in matches], _VALUE_ATTRIBUTES)
        return [
            result
            for match in matches
            if (result := _with_value(match, items.get(_location_id(match.location)))) is not None
        ]

    async def _repair(self, max_documents: int, cursor: Optional[str], rebuild_postings: bool) -> RepairReport:
        _require_integer("max_documents", max_documents, 1)
        manifests, more_may_remain = await self._read_manifest_page(max_documents, cursor)
        outcomes: Counter[_RepairOutcome] = Counter()
        for manifest in manifests:
            outcomes[await self._repair_document(manifest, rebuild_postings)] += 1
        return RepairReport(
            documents_checked=len(manifests),
            documents_removed=outcomes[_RepairOutcome.REMOVED],
            postings_rebuilt=outcomes[_RepairOutcome.REBUILT],
            cursor=manifests[-1].doc_id if more_may_remain else None,
        )

    async def _read_manifest_page(self, max_documents: int, cursor: Optional[str]) -> tuple[list[_Manifest], bool]:
        """Up to ``max_documents`` manifests after ``cursor`` (strongly consistent), and whether more may remain.

        A ``None`` or empty cursor starts from the beginning of the manifest partition.
        """
        manifests: list[_Manifest] = []
        start_key = self._manifest_key(cursor) if cursor else None
        while len(manifests) < max_documents:
            request = self._partition_query(
                self._manifest_partition,
                _MANIFEST_ATTRIBUTES,
                limit=max_documents - len(manifests),
                start_key=start_key,
                consistent=True,
            )
            response = await asyncio.to_thread(self._client().query, **request)
            manifests.extend(self._to_manifest(item) for item in response.get("Items", []))
            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                return manifests, False
        return manifests, True

    async def _repair_document(self, manifest: _Manifest, rebuild_postings: bool) -> _RepairOutcome:
        base = await self._read_indexed_base(manifest.doc_id)
        try:
            if base is None or not self._is_manifest_current(manifest, base):
                await self._transact(self._manifest_removal_actions(manifest))
                return _RepairOutcome.REMOVED
            if rebuild_postings:
                await self._transact(self._rebuild_actions(manifest, base))
                return _RepairOutcome.REBUILT
        except _WriteConflict:
            return _RepairOutcome.SKIPPED
        return _RepairOutcome.KEPT

    async def _read_indexed_base(self, doc_id: str) -> Optional[dict[str, Any]]:
        location = self._candidate_location(doc_id)
        if location is None:
            return None
        return await self._read_base_item(location, [*_REPAIR_ATTRIBUTES, *self._ttl_attribute_names()])

    def _is_manifest_current(self, manifest: _Manifest, base: dict[str, Any]) -> bool:
        return (
            _string_attr(base, _INDEXED_REVISION_ATTR) == manifest.revision
            and _string_attr(base, _INDEXED_SCOPE_ATTR) == self._scope_marker
            and not self._port.is_expired(base)
        )

    def _client(self) -> Any:
        return self._port.client()


def _merge_postings(
    posting_lists: Sequence[_PostingList],
    max_candidates: int,
    locate: Callable[[str], Optional[_DocumentLocation]],
) -> _CandidatePool:
    pool = _CandidatePool(max_candidates=max_candidates, locate=locate)
    for partition_index, posting_list in enumerate(posting_lists):
        if posting_list.truncated:
            pool.truncation_reasons.add(_PAGES_TRUNCATION)
        for doc_id, revision in posting_list.postings:
            pool.add(partition_index, doc_id, revision)
    return pool


def _ranking(match: _Match) -> tuple[float, str]:
    """Score descending, then key ascending (code-point order equals UTF-8 byte order)."""
    return -match.result.score, match.result.key


def _matches_filter(metadata: Optional[dict[str, Any]], expected: Optional[dict[str, _MetaValue]]) -> bool:
    """Strict equality of every filter entry against the item's current metadata.

    Values must be the same kind (bool, number or string; a bool never equals a number) and
    equal, with numbers compared numerically. A missing field or missing metadata never
    matches. Deliberately stricter than ``DynamoDBStorage``'s vector-search filter, where
    ``True == 1``.
    """
    if expected is None:
        return True
    return all(
        metadata is not None and name in metadata and _same_meta_value(metadata[name], value)
        for name, value in expected.items()
    )


def _same_meta_value(actual: Any, expected: Any) -> bool:
    kind = _meta_kind(actual)
    return kind is not None and kind is _meta_kind(expected) and bool(actual == expected)


def _meta_kind(value: Any) -> Optional[type]:
    if isinstance(value, bool):
        return bool
    if isinstance(value, (int, float)):
        return float
    if isinstance(value, str):
        return str
    return None


def _posting_entries(items: Iterable[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    """``(doc_id, revision)`` of each posting row, skipping malformed rows instead of failing the query."""
    for item in items:
        doc_id = _string_attr(item, _SK)
        revision = _string_attr(item, _REVISION_ATTR)
        if doc_id is not None and revision is not None:
            yield doc_id, revision


def _with_value(match: _Match, item: Optional[dict[str, Any]]) -> Optional[LexicalSearchResult]:
    """``match``'s result carrying the value read at its validated revision, or ``None`` if that value is gone."""
    if item is None or _string_attr(item, _INDEXED_REVISION_ATTR) != match.revision:
        return None
    value = _inline_value(item)
    return dataclasses.replace(match.result, data=value) if value is not None else None


def _inline_value(item: dict[str, Any]) -> Optional[bytes]:
    """Decoded inline value (gunzipped when ``z``), or ``None`` when the item is offloaded or has no ``data``."""
    raw = item.get(_DATA_ATTR, {}).get("B")
    if raw is None or _is_offloaded(item):
        return None
    payload = bytes(raw)
    return gzip.decompress(payload) if item.get(_Z_ATTR, {}).get("BOOL") is True else payload


def _is_offloaded(item: dict[str, Any]) -> bool:
    return bool(item.get(_S3_ATTR, {}).get("BOOL"))


def _classify_cancellation(error: Exception) -> Optional[type[_TransactionCancelled]]:
    """Retryable kind of a ``TransactionCanceledException``, or ``None`` for any other failure.

    A conflict code wins over a throttling code: the condition failure alone already means a
    concurrent writer changed the document.
    """
    codes = _cancellation_codes(error)
    if not codes.isdisjoint(_CONFLICT_CANCELLATION_CODES):
        return _WriteConflict
    if not codes.isdisjoint(_THROTTLING_CANCELLATION_CODES):
        return _WriteThrottled
    return None


def _cancellation_codes(error: Exception) -> frozenset[str]:
    response = getattr(error, "response", None)
    if not isinstance(response, dict) or response.get("Error", {}).get("Code") != "TransactionCanceledException":
        return frozenset()
    return frozenset(reason.get("Code") for reason in response.get("CancellationReasons", []))


def _require_transaction_bounds(actions: Sequence[dict[str, Any]]) -> None:
    """Guard the TransactWriteItems limits; unreachable within the configured limits."""
    if len(actions) > _MAX_TRANSACTION_ACTIONS:
        raise StorageError(
            f"Lexical index transaction has {len(actions)} actions, above the limit of {_MAX_TRANSACTION_ACTIONS}"
        )
    size = sum(_item_size(action["Put"]["Item"]) for action in actions if "Put" in action)
    if size > _MAX_TRANSACTION_BYTES:
        raise StorageError(
            f"Lexical index transaction is about {size} bytes, above the limit of {_MAX_TRANSACTION_BYTES}"
        )


def _item_size(item: dict[str, Any]) -> int:
    return sum(utf8_byte_length(name) + _attribute_value_size(value) for name, value in item.items())


def _attribute_value_size(value: dict[str, Any]) -> int:
    """Conservative size estimate of one AttributeValue (never below DynamoDB's own accounting)."""
    kind, content = next(iter(value.items()))
    if kind in ("S", "N"):
        return utf8_byte_length(content)
    if kind == "B":
        return len(content)
    if kind == "L":
        return 3 + sum(1 + _attribute_value_size(element) for element in content)
    if kind == "M":
        return 3 + sum(1 + utf8_byte_length(name) + _attribute_value_size(nested) for name, nested in content.items())
    if kind in ("SS", "NS"):
        return sum(utf8_byte_length(element) for element in content)
    if kind == "BS":
        return sum(len(element) for element in content)
    return 1


def _backoff(attempt: int, cap_seconds: float) -> float:
    return float(min(_BACKOFF_BASE_SECONDS * 2**attempt, cap_seconds))


def _primary_key(pk: str, sk: str) -> dict[str, Any]:
    return {_PK: {"S": pk}, _SK: {"S": sk}}


def _location_id(location: _DocumentLocation) -> tuple[str, str]:
    return location.pk, location.sk


def _attribute_names(attributes: Sequence[str]) -> dict[str, str]:
    """Positional ``#a<n>`` placeholders, so any attribute name (reserved words included) is safe."""
    return {f"#a{position}": name for position, name in enumerate(attributes)}


def _revision_condition(revision: str) -> dict[str, Any]:
    return {
        "ConditionExpression": "#rev = :rev",
        "ExpressionAttributeNames": {"#rev": _REVISION_ATTR},
        "ExpressionAttributeValues": {":rev": {"S": revision}},
    }


def _absent_condition() -> dict[str, Any]:
    return {"ConditionExpression": "attribute_not_exists(#pk)", "ExpressionAttributeNames": {"#pk": _PK}}


def _string_attr(item: dict[str, Any], name: str) -> Optional[str]:
    value: Optional[str] = item.get(name, {}).get("S")
    return value


def _string_list(attribute: Optional[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(element["S"] for element in (attribute or {}).get("L", []))


def _string_list_value(values: Sequence[str]) -> dict[str, Any]:
    return {"L": [{"S": value} for value in values]}


def _require_table_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise StorageError("LexicalIndex requires a non-empty index_table_name")
    return name


def _require_unreserved_ttl_attribute(ttl_attribute: Optional[str]) -> None:
    """The TTL stamp is copied onto postings and projected beside index attributes, so it must not share a name."""
    if ttl_attribute in _RESERVED_ATTRIBUTE_NAMES:
        raise StorageError(
            f"Storage TTL attribute '{ttl_attribute}' is reserved by the lexical index; configure another ttl_attribute"
        )


def _require_partition_keys(partitions: Iterable[str]) -> None:
    """Reject a partition key DynamoDB would refuse, naming only the limit so no term or identifier leaks."""
    if any(utf8_byte_length(partition) > _MAX_PARTITION_KEY_BYTES for partition in partitions):
        raise StorageError(
            f"A lexical index partition key is above the DynamoDB partition-key limit of "
            f"{_MAX_PARTITION_KEY_BYTES} bytes"
        )


def _require_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= _MAX_TOP_K:
        raise StorageError(f"top_k must be between 1 and {_MAX_TOP_K}; got {top_k!r}")


def _require_integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StorageError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _require_at_most(name: str, value: int, ceiling: int, reason: str = "") -> None:
    if value > ceiling:
        raise StorageError(f"LexicalIndexLimits.{name} must be at most {ceiling}{reason}; got {value}")


def _require_size(subject: str, size: int, limit_name: str, limit: int) -> None:
    if size > limit:
        raise StorageError(f"{subject} is {size} UTF-8 bytes, above {limit_name} ({limit})")
