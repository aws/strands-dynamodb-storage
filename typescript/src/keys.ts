// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'

/**
 * Validates and normalizes a storage key: collapses runs of `/`, strips leading
 * and trailing `/`, and rejects empty keys and any `..` segment.
 *
 * Behaviourally identical to the Strands SDK's internal key normalization, which
 * is not part of the public API surface. Reimplemented here so this package does
 * not depend on unexported SDK internals while still producing keys
 * that are byte-for-byte consistent with the shipped `Storage` implementations.
 *
 * @param key - The raw key to normalize
 * @returns The normalized key
 * @throws {@link StorageError} if the key is empty or contains a `..` segment
 */
export function normalizeKey(key: string): string {
  const segments = key.split('/').filter(Boolean)
  if (segments.length === 0) {
    throw new StorageError('Storage key must not be empty')
  }
  if (segments.includes('..')) {
    throw new StorageError(`Invalid storage key '${key}': '..' path segments are not allowed`)
  }
  return segments.join('/')
}

/**
 * Normalizes a list prefix: collapses slash runs, strips leading slashes, and
 * preserves a single trailing slash when the caller supplied one. Unlike a key,
 * an empty prefix is valid and matches everything.
 *
 * Matches the Strands SDK's internal prefix normalization (see {@link normalizeKey}).
 *
 * @param prefix - The raw prefix to normalize
 * @returns The normalized prefix
 * @throws {@link StorageError} if the prefix contains a `..` segment
 */
export function normalizePrefix(prefix: string): string {
  const parts = prefix.split('/')
  const segments = parts.filter(Boolean)
  if (segments.includes('..')) {
    throw new StorageError(`Invalid storage prefix '${prefix}': '..' path segments are not allowed`)
  }
  const joined = segments.join('/')
  return parts[parts.length - 1] === '' && joined ? `${joined}/` : joined
}
