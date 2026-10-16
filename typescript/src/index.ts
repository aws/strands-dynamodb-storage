// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Amazon DynamoDB {@link https://www.npmjs.com/package/@strands-agents/sdk | Strands Agents}
 * storage backend.
 *
 * Implements the SDK's unified `Storage` interface so a single `DynamoDBStorage`
 * instance can be passed to Session Manager, Memory Manager, and any other
 * subsystem that persists bytes — with optional S3 offload for values above the
 * DynamoDB item limit and an optional native vector-search hook.
 *
 * Also ships {@link LexicalIndex} (preview), an opt-in lexical document index: it writes documents
 * together with their term and exact-identifier postings (kept in a separate, user-provisioned index
 * table) in one transaction, and retrieves them by term overlap or exact identifier.
 * {@link LexicalSearchStrategy} plugs it into SDK consumers of the byte `Storage` API as a `searchStrategy`.
 *
 * @example
 * ```typescript
 * import { Agent, SessionManager } from '@strands-agents/sdk'
 * import { DynamoDBStorage } from 'strands-dynamodb-storage'
 *
 * const storage = new DynamoDBStorage('agent-data', { region: 'us-east-1' })
 * const agent = new Agent({ sessionManager: new SessionManager({ storage }) })
 * ```
 *
 * @packageDocumentation
 */

export { DynamoDBStorage } from './dynamodb-storage.js'
export type {
  DynamoDBStorageConfig,
  DynamoDBListQuery,
  DocumentItemOptions,
  SearchQuery,
  SearchResult,
  StringSearchStrategy,
  VectorSearchAdapter,
} from './dynamodb-storage.js'
export { LexicalIndex, DEFAULT_LEXICAL_INDEX_LIMITS, RevisionConflictError } from './lexical-index.js'
export type {
  SearchableDocument,
  LexicalIndexLimits,
  LexicalIndexConfig,
  LexicalQuery,
  IdentifierQuery,
  LexicalSearchResult,
  LexicalSearchResponse,
  RepairReport,
  RepairOptions,
  RevisionOptions,
} from './lexical-index.js'
export { LEXICAL_TOKENIZER_VERSION } from './lexical-terms.js'
export { LexicalSearchStrategy } from './lexical-search-strategy.js'
export type {
  SearchableText,
  SearchableTextExtractor,
  LexicalSearchStrategyConfig,
  LexicalSearchStrategyOptions,
} from './lexical-search-strategy.js'
