// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Community DynamoDB {@link https://www.npmjs.com/package/@strands-agents/sdk | Strands Agents}
 * storage backend.
 *
 * Implements the SDK's unified `Storage` interface so a single `DynamoDBStorage`
 * instance can be passed to Session Manager, Memory Manager, and any other
 * subsystem that persists bytes — with optional S3 offload for values above the
 * DynamoDB item limit and an optional native vector-search hook.
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
  SearchQuery,
  SearchResult,
  VectorSearchAdapter,
} from './dynamodb-storage.js'
