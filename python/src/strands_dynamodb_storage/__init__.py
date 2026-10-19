# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon DynamoDB storage backend for the Strands Agents SDK.

Implements the SDK's unified byte ``Storage`` interface, so one DynamoDB-backed
instance persists any subsystem's data (Session Manager, Memory Manager, offloader,
transcripts) — with optional S3 offload, gzip compression, TTL, and native vector search.

:class:`DynamoDBMemoryStore` implements the SDK's ``MemoryStore`` for the Memory Manager
on top of that vector search, embedding text with :class:`BedrockEmbedder` or a function
you supply.
"""

from .dynamodb_storage import (
    DynamoDBListQuery,
    DynamoDBStorage,
    SearchQuery,
    SearchResult,
    VectorSearchAdapter,
)
from .embedders import (
    DEFAULT_EMBEDDING_MODEL_ID,
    BedrockEmbedder,
    BedrockModelFamily,
    Embedder,
    EmbeddingPurpose,
    EmbedFunction,
)
from .memory_store import DynamoDBMemoryStore

__all__ = [
    "DynamoDBStorage",
    "DynamoDBListQuery",
    "SearchQuery",
    "SearchResult",
    "VectorSearchAdapter",
    "DynamoDBMemoryStore",
    "BedrockEmbedder",
    "BedrockModelFamily",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "Embedder",
    "EmbedFunction",
    "EmbeddingPurpose",
]
