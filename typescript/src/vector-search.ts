// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'
import {
  storageRegion,
  type DynamoDBStorage,
  type EmbeddingSearchStrategy,
  type SearchResult,
} from './dynamodb-storage.js'
import {
  asVector,
  BedrockEmbedder,
  callEmbedder,
  type EmbedFunction,
  type Embedder,
  type EmbeddingPurpose,
} from './embedders.js'

/** SearchVectors returns at most this many results. */
const MAX_TOP_K = 100

/** The `DistanceFunction` of the table's vector index. */
export type DistanceFunction = 'COSINE' | 'EUCLIDEAN' | 'DOT_PRODUCT'

/** Configuration for {@link DynamoDBVectorSearchStrategy}. */
export interface DynamoDBVectorSearchStrategyConfig {
  /**
   * Any SDK `Embedder`, a {@link BedrockEmbedder} or other object with `embed(text, purpose)`,
   * or a function from text to vector. Defaults to a `BedrockEmbedder` in the storage's region,
   * created on first use.
   */
  embedder?: Embedder | EmbedFunction
  /** Most results one search returns, from 1 to 100. Defaults to 10. */
  maxResults?: number
  /** The vector index's `DistanceFunction`, so scores can be returned with higher meaning closer. Defaults to `COSINE`. */
  distanceFunction?: DistanceFunction
}

/**
 * Vector search for {@link DynamoDBStorage}, backed by the table's native vector index.
 *
 * The DynamoDB counterpart of the SDK's `S3VectorSearchStrategy`. Pass it as `searchStrategy`:
 * each `write` embeds the value and stores its vector on the same item, in the same request, and
 * a plain-string `search` embeds the query and runs `SearchVectors`. `delete` removes the vector
 * with the item. Empty and non-UTF-8 values are stored without a vector, which clears any earlier one.
 *
 * The table needs the vector index from the README, with `Dimensions` matching the embedder (1024
 * for the default, Titan Text Embeddings V2). Searches are pinned to the storage's partition, so
 * use a storage namespaced at least two segments deep, as `FileMemoryStore`'s `memory/<name>` is.
 * Every write is embedded, so give the strategy a storage of its own rather than the agent's. Each
 * value is embedded whole and must fit the embedding model's input. Values written before the
 * strategy was configured have no vector until rewritten, or passed to {@link index}.
 *
 * @example
 * ```typescript
 * const storage = new DynamoDBStorage('agent-storage', { searchStrategy: new DynamoDBVectorSearchStrategy() })
 * const knowledge = new FileMemoryStore({ name: 'facts', storage })
 * ```
 */
export class DynamoDBVectorSearchStrategy implements EmbeddingSearchStrategy {
  private _embedder: Embedder | EmbedFunction | undefined
  private readonly _maxResults: number
  private readonly _distanceFunction: DistanceFunction

  /**
   * @param config - Optional embedder, result limit, and the index's distance function
   * @throws RangeError if `maxResults` is out of range or `distanceFunction` is unknown
   */
  constructor(config: DynamoDBVectorSearchStrategyConfig = {}) {
    const maxResults = config.maxResults ?? 10
    if (!Number.isInteger(maxResults) || maxResults < 1 || maxResults > MAX_TOP_K) {
      throw new RangeError(`maxResults must be an integer from 1 to ${MAX_TOP_K}; got ${String(config.maxResults)}`)
    }
    const distanceFunction = config.distanceFunction ?? 'COSINE'
    if (!['COSINE', 'EUCLIDEAN', 'DOT_PRODUCT'].includes(distanceFunction)) {
      throw new RangeError(`distanceFunction must be COSINE, EUCLIDEAN or DOT_PRODUCT; got ${String(distanceFunction)}`)
    }
    this._embedder = config.embedder
    this._maxResults = maxResults
    this._distanceFunction = distanceFunction
  }

  /**
   * Adds the vector for `data` to the item stored under `key`.
   *
   * `write` already stores the vector in the same request and does not call this; use it to backfill a
   * value written before the strategy was configured. Only the vector is written, so the stored value,
   * its metadata and its TTL are left as they are. A no-op when the item already has a vector, when it
   * is gone, and when `data` is empty or not UTF-8.
   *
   * @param storage - The storage the value is in
   * @param key - The key, relative to `storage`
   * @param data - The value's bytes, as `read` returns them
   * @throws {@link StorageError} if embedding or the write fails
   */
  async index(storage: DynamoDBStorage, key: string, data: Uint8Array): Promise<void> {
    const vector = await this._documentVector(storage, data)
    if (vector === undefined) return
    await storage._setVector(key, vector)
  }

  /**
   * Returns the stored values nearest `query`, best first, with their bytes as `data`.
   *
   * @param storage - The storage to search
   * @param query - A natural-language string. A blank one returns `[]` without embedding.
   * @throws {@link StorageError} if embedding or the search fails
   */
  async search(storage: DynamoDBStorage, query: string): Promise<SearchResult[]> {
    if (query.trim() === '') return []
    const vector = await this._embed(storage, query, 'query')
    const pk = partition(storage)
    const matches = await storage.search({
      vector,
      topK: this._maxResults,
      includeValues: true,
      ...(pk !== undefined ? { pk } : {}),
    })
    return matches.map((match) => ({
      key: match.key,
      score: this._similarity(match.score),
      ...(match.data !== undefined ? { data: match.data } : {}),
    }))
  }

  /**
   * The vector `DynamoDBStorage.write` stores with `data`, or `undefined` for empty or non-UTF-8 data.
   *
   * @internal
   */
  async _documentVector(storage: DynamoDBStorage, data: Uint8Array): Promise<number[] | undefined> {
    let text: string
    try {
      text = new TextDecoder('utf-8', { fatal: true }).decode(data)
    } catch {
      return undefined
    }
    if (text.trim() === '') return undefined
    return this._embed(storage, text, 'document')
  }

  private async _embed(storage: DynamoDBStorage, text: string, purpose: EmbeddingPurpose): Promise<number[]> {
    if (!this._embedder) {
      const region = storageRegion(storage)
      this._embedder = new BedrockEmbedder(region !== undefined ? { region } : {})
    }
    let result: unknown
    try {
      result = await callEmbedder(this._embedder, text, purpose)
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError('Embedding failed', { cause: error })
    }
    const vector = asVector(result)
    if (!vector)
      throw new StorageError("The search strategy's embedder must return a non-empty array of finite numbers")
    return vector
  }

  /** The index score with higher meaning closer, as the SDK's search results rank. */
  private _similarity(score: number): number {
    return this._distanceFunction === 'DOT_PRODUCT' ? score : 1 / (1 + score)
  }
}

/** The partition every key under `storage` shares, or `undefined` when its prefix spans several. */
function partition(storage: DynamoDBStorage): string | undefined {
  if (storage.prefix.split('/').filter(Boolean).length < 2) return undefined
  return storage.partitionKey('_')
}
