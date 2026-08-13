# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon DynamoDB storage backend for the Strands Agents SDK.

Implements the SDK's unified byte ``Storage`` interface, so one DynamoDB-backed
instance persists any subsystem's data (Session Manager, Memory Manager, offloader,
transcripts) — with optional S3 offload, gzip compression, TTL, and native vector search.
"""

from .dynamodb_storage import (
    DynamoDBListQuery,
    DynamoDBStorage,
    SearchQuery,
    SearchResult,
    VectorSearchAdapter,
)

__all__ = [
    "DynamoDBStorage",
    "DynamoDBListQuery",
    "SearchQuery",
    "SearchResult",
    "VectorSearchAdapter",
]
