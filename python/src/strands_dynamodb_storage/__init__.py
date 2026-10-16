# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon DynamoDB storage backend for the Strands Agents SDK.

Implements the SDK's unified byte ``Storage`` interface, so one DynamoDB-backed
instance persists any subsystem's data (Session Manager, Memory Manager, offloader,
transcripts) — with optional S3 offload, gzip compression, TTL, and native vector search.
An opt-in :class:`LexicalIndex` (preview) adds term and exact-identifier retrieval, and
:class:`LexicalSearchStrategy` exposes it to SDK consumers through ``search_strategy``.
"""

from .dynamodb_storage import (
    DynamoDBListQuery,
    DynamoDBStorage,
    SearchQuery,
    SearchResult,
    VectorSearchAdapter,
)
from .lexical_index import (
    IdentifierQuery,
    LexicalIndex,
    LexicalIndexLimits,
    LexicalQuery,
    LexicalSearchResponse,
    LexicalSearchResult,
    RepairReport,
    RevisionConflictError,
    SearchableDocument,
)
from .lexical_search_strategy import LexicalSearchStrategy, SearchableText, SearchableTextExtractor
from .lexical_terms import LEXICAL_TOKENIZER_VERSION

__all__ = [
    "DynamoDBStorage",
    "DynamoDBListQuery",
    "SearchQuery",
    "SearchResult",
    "VectorSearchAdapter",
    "LexicalIndex",
    "LexicalIndexLimits",
    "SearchableDocument",
    "LexicalQuery",
    "IdentifierQuery",
    "LexicalSearchResult",
    "LexicalSearchResponse",
    "RepairReport",
    "RevisionConflictError",
    "LexicalSearchStrategy",
    "SearchableText",
    "SearchableTextExtractor",
    "LEXICAL_TOKENIZER_VERSION",
]
