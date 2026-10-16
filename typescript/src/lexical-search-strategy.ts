// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'
import type { Storage } from '@strands-agents/sdk/storage'

import {
  DynamoDBStorage,
  type DocumentItemOptions,
  type SearchResult,
  type StringSearchStrategy,
} from './dynamodb-storage.js'
import {
  LexicalIndex,
  assertTopK,
  resolveLimits,
  type LexicalIndexLimits,
  type LexicalSearchResult,
} from './lexical-index.js'
import { textTerms, utf8ByteLength } from './lexical-terms.js'

/** Searchable content of a stored value: text tokenized into terms, and exact-match identifiers. */
export interface SearchableText {
  /** Text tokenized into searchable terms. Default `''`. */
  readonly text?: string
  /** Exact-match identifiers (case-sensitive), e.g. ticket or invoice numbers. Default none. */
  readonly identifiers?: readonly string[]
}

/**
 * Derives the searchable content of the value written under `key` (relative to the storage prefix), or returns
 * `null`/`undefined` to leave it unindexed. Stored bytes are never decoded implicitly: the extractor decides what
 * text they carry.
 */
export type SearchableTextExtractor = (key: string, data: Uint8Array) => SearchableText | null | undefined

/** Configuration for {@link LexicalSearchStrategy}. */
export interface LexicalSearchStrategyConfig {
  /** User-provisioned index table (partition key `pk` string, sort key `sk` string). */
  indexTableName: string
  /** Extracts the text and identifiers to index from each written value. */
  extract: SearchableTextExtractor
  /** Results per search when the call does not set `topK`, 1 to 100. Default 10. */
  topK?: number
  /** When true, results carry the stored value. Default true. */
  includeValues?: boolean
  /** Overrides merged over `DEFAULT_LEXICAL_INDEX_LIMITS`. */
  limits?: Partial<LexicalIndexLimits>
}

/** Per-call options of {@link LexicalSearchStrategy.search}. */
export interface LexicalSearchStrategyOptions {
  /** Number of results, 1 to 100. Defaults to the strategy's `topK`. */
  topK?: number
}

/** A match in the SDK `StorageSearchResult` shape. */
type StrategySearchResult = Pick<SearchResult, 'key' | 'score' | 'data'>

const DEFAULT_TOP_K = 10

/**
 * SDK-style search strategy backed by a {@link LexicalIndex} (preview), so SDK consumers that only know the byte
 * `Storage` API, such as the SDK `FileMemoryStore`, can search a {@link DynamoDBStorage} by text.
 *
 * Configure it as the storage's `searchStrategy`. After every successful `write()`, {@link LexicalSearchStrategy.index}
 * extracts the value's searchable text and rewrites the item together with its postings through
 * {@link LexicalIndex.upsert}, preserving the write's vector, metadata and TTL. That costs one extra transactional
 * write per indexed write and is at-least-once: when the hook fails, the value stays stored but unsearchable and
 * `write()` throws, so retrying the write re-indexes it. {@link LexicalIndex.upsert} remains the single-write, atomic
 * path. Plain-string `search()` calls run {@link LexicalSearchStrategy.search}.
 *
 * Each call builds a `LexicalIndex` over the storage it receives, so every namespaced view is its own lexical scope.
 *
 * @example
 * ```typescript
 * const storage = new DynamoDBStorage('agent-data', {
 *   region: 'us-east-1',
 *   searchStrategy: new LexicalSearchStrategy({
 *     indexTableName: 'agent-lexical-index',
 *     extract: (_key, data) => ({ text: new TextDecoder().decode(data) }),
 *   }),
 * })
 * const memory = new FileMemoryStore({ name: 'agent-memory', storage })
 * ```
 */
export class LexicalSearchStrategy implements StringSearchStrategy {
  private readonly _indexTableName: string
  private readonly _extract: SearchableTextExtractor
  private readonly _topK: number
  private readonly _includeValues: boolean
  private readonly _limits: LexicalIndexLimits

  /**
   * @param config - Index table, extractor, result defaults and optional limit overrides
   * @throws {@link StorageError} if the table name is empty, `extract` is not a function, `topK` is not an integer
   *   from 1 to 100, or a limit is out of range
   */
  constructor(config: LexicalSearchStrategyConfig) {
    if (!config.indexTableName) throw new StorageError('LexicalSearchStrategy requires a non-empty indexTableName')
    if (typeof config.extract !== 'function') {
      throw new StorageError('LexicalSearchStrategy requires an extract function')
    }
    this._topK = config.topK ?? DEFAULT_TOP_K
    assertTopK(this._topK)
    this._limits = resolveLimits(config.limits)
    this._indexTableName = config.indexTableName
    this._extract = config.extract
    this._includeValues = config.includeValues ?? true
  }

  /**
   * Indexes the value `storage` has just written under `key` with the text and identifiers the extractor returns. When
   * it returns nothing, the value stays unindexed; a previous indexed version is then excluded at read time and its
   * postings are removed by {@link LexicalIndex.repair}. Extracted text is never truncated: over-limit text (for
   * example more than `maxPostingsPerDocument` distinct terms and identifiers) is rejected.
   *
   * @throws {@link StorageError} if `storage` is not a `DynamoDBStorage` or the upsert fails
   */
  async index(storage: Storage, key: string, data: Uint8Array, options?: DocumentItemOptions): Promise<void> {
    const index = this._indexFor(storage)
    const searchable = this._extract(key, data)
    if (!searchable) return
    await index.upsert({
      key,
      data,
      text: searchable.text,
      identifiers: searchable.identifiers,
      metadata: options?.metadata,
      vector: options?.vector,
      ttlSeconds: options?.ttlSeconds,
    })
  }

  /**
   * Ranks documents by the fraction of query terms they match, like {@link LexicalIndex.search}, but tolerant of
   * natural-language queries: terms above `maxTermBytes` are skipped (no indexed document can contain them), only the
   * first `maxQueryTerms` distinct remaining terms are used, and a query left without terms returns no results.
   * Truncation flags cannot be represented in the SDK result shape and are dropped.
   *
   * @throws {@link StorageError} if `storage` is not a `DynamoDBStorage`, `topK` is out of range, the query is not
   *   well-formed Unicode, or a read fails
   */
  async search(
    storage: Storage,
    query: string,
    options?: LexicalSearchStrategyOptions
  ): Promise<StrategySearchResult[]> {
    const index = this._indexFor(storage)
    const topK = options?.topK ?? this._topK
    assertTopK(topK)
    const terms = this._queryTerms(query)
    if (terms.length === 0) return []
    const response = await index._searchTerms(terms, { topK, includeValues: this._includeValues })
    return response.results.map(toStrategySearchResult)
  }

  private _queryTerms(query: string): string[] {
    const indexable = textTerms(query).filter((term) => utf8ByteLength(term) <= this._limits.maxTermBytes)
    return indexable.slice(0, this._limits.maxQueryTerms)
  }

  private _indexFor(storage: Storage): LexicalIndex {
    if (!(storage instanceof DynamoDBStorage)) {
      throw new StorageError(`LexicalSearchStrategy requires a DynamoDBStorage; got ${storage.constructor.name}`)
    }
    return new LexicalIndex(storage, { indexTableName: this._indexTableName, limits: this._limits })
  }
}

function toStrategySearchResult({ key, score, data }: LexicalSearchResult): StrategySearchResult {
  return data === undefined ? { key, score } : { key, score, data }
}
