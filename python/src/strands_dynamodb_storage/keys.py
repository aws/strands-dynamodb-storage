# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage-key normalization helpers.

Behaviourally identical to the Strands SDK's internal ``_normalize_key`` /
``_normalize_prefix`` (which are not part of the public API), reimplemented here so
this community backend does not depend on unexported SDK internals while producing
keys that are byte-for-byte consistent with the shipped ``Storage`` implementations.
"""

from __future__ import annotations

import re

from strands.types.exceptions import StorageError


def normalize_key(key: str) -> str:
    """Validate and normalize a storage key.

    Collapses runs of ``/``, strips leading and trailing ``/``, rejects empty keys
    and any ``..`` segment.

    Raises:
        StorageError: If the key is empty or contains a ``..`` segment.
    """
    normalized = re.sub(r"/+", "/", key).strip("/")
    if not normalized:
        raise StorageError("Storage key must not be empty")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage key '{key}': '..' path segments are not allowed")
    return normalized


def normalize_prefix(prefix: str) -> str:
    """Normalize a list prefix.

    Collapses slash runs and strips leading slashes; a trailing slash is preserved
    (it is significant for prefix matching). An empty prefix is valid.

    Raises:
        StorageError: If the prefix contains a ``..`` segment.
    """
    normalized = re.sub(r"/+", "/", prefix).lstrip("/")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage prefix '{prefix}': '..' path segments are not allowed")
    return normalized
