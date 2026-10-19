// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { StorageError } from '@strands-agents/sdk'
import type { ExtractionConfig, JSONValue, MemoryEntry, MemoryStore, SearchOptions } from '@strands-agents/sdk'
import { DynamoDBStorage, storageRegion } from './dynamodb-storage.js'
import {
  BedrockEmbedder,
  describeError,
  type EmbedFunction,
  type Embedder,
  type EmbeddingPurpose,
} from './embedders.js'

/** SearchVectors returns at most this many results. */
const MAX_TOP_K = 100
const DEFAULT_MAX_SEARCH_RESULTS = 10
const RESERVED_METADATA_KEYS = new Set(['_key', '_score'])
/** A code point from U+D800 to U+DFFF, which UTF-8 cannot encode. */
const LONE_SURROGATE = /\p{Cs}/u
// marshall loses `__proto__` and refuses an object whose `constructor` is not a function.
const UNWRITABLE_KEYS = new Set(['__proto__', 'constructor'])
const DEFAULT_DESCRIPTION = 'Long-term memories stored in Amazon DynamoDB, searched by meaning'

/** Configuration for {@link DynamoDBMemoryStore}. */
export interface DynamoDBMemoryStoreConfig {
  /**
   * An {@link Embedder} such as {@link BedrockEmbedder}, or a plain function from text to
   * vector. Defaults to `new BedrockEmbedder()` (Amazon Titan Text Embeddings V2, 1024
   * dimensions) in the storage's Region.
   */
  embedder?: Embedder | EmbedFunction
  /**
   * Store name, unique within a `MemoryManager`; also the key segment `memory/<name>/`. Must not be `.` or
   * `..` or contain `/`, control characters, lone surrogates, or leading or trailing whitespace. Default
   * `'dynamodb'`.
   */
  name?: string
  /**
   * Description shown to the model in the memory tools, or `null` for none. Default
   * `'Long-term memories stored in Amazon DynamoDB, searched by meaning'`.
   */
  description?: string | null
  /**
   * Default number of results per search, 1 to 100. When unset, a `MemoryManager` search
   * asks for its own default (3), and a direct {@link DynamoDBMemoryStore.search} call
   * without options returns up to 10.
   */
  maxSearchResults?: number
  /** Whether the `MemoryManager` may write to this store. Default `true`. */
  writable?: boolean
  /** Automatic extraction configuration, passed through to the `MemoryManager`. */
  extraction?: boolean | ExtractionConfig
}

/**
 * A Strands `MemoryStore` that keeps memories in DynamoDB and finds them by meaning.
 *
 * Each memory is one item in the storage's table, written with its text, its embedding,
 * and optional metadata, under the key `memory/<name>/<id>` inside the storage namespace.
 * `<id>` is derived from the text, so adding the same text again overwrites the item
 * instead of creating a duplicate. A search embeds the query and runs one `SearchVectors`
 * call, pinned to the partition that holds this store's memories. Memories stored inline
 * come back from the index projection with no base-table read; a memory offloaded to S3,
 * or a match from an index that does not project the data attribute, costs one point read
 * (GetItem, plus the S3 read for an offloaded memory).
 *
 * Embedding happens here, not in the storage: session snapshots and other values written
 * through the same {@link DynamoDBStorage} are never embedded.
 *
 * The partition is the first two segments of the full key, so every namespace that shares
 * its first two segments with this one (`user/u1/agent-a` and `user/u1/agent-b` both use
 * `user/u1`) shares the partition. Every item there that carries a vector competes for the
 * same `SearchVectors` results, and this store keeps only its own, so a search can return
 * fewer than requested. Give each memory store a partition of its own, for example a
 * two-segment namespace such as `storage.namespace('user/u1')`.
 *
 * @example
 * ```typescript
 * const storage = new DynamoDBStorage('agent-data', { region: 'us-east-1' }).namespace('user/u1')
 * const memory = new DynamoDBMemoryStore(storage)
 * const agent = new Agent({ memoryManager: new MemoryManager({ stores: [memory] }) })
 * ```
 */
export class DynamoDBMemoryStore implements MemoryStore {
  readonly name: string
  readonly description?: string
  readonly maxSearchResults?: number
  readonly writable: boolean
  readonly extraction?: boolean | ExtractionConfig
  private readonly _storage: DynamoDBStorage
  private readonly _embedder: Embedder | EmbedFunction
  private readonly _keyPrefix: string
  private readonly _partition: string

  /**
   * No AWS call is made until the first add or search.
   *
   * @param storage - The {@link DynamoDBStorage} (usually a namespaced view) to keep
   *   memories in. Its table needs the vector index described in the package README.
   * @throws TypeError if `storage` is not a {@link DynamoDBStorage}, `name` is not a string, `description` is
   *   not a string or `null`, `writable` is not a boolean, or `embedder` is neither a function nor an object with
   *   an `embed` method
   * @throws RangeError if `name` or `maxSearchResults` is invalid
   */
  constructor(storage: DynamoDBStorage, config?: DynamoDBMemoryStoreConfig) {
    if (!(storage instanceof DynamoDBStorage)) {
      throw new TypeError('DynamoDBMemoryStore requires a DynamoDBStorage')
    }
    const name = config?.name === undefined ? 'dynamodb' : config.name
    if (typeof name !== 'string') throw new TypeError('Memory store name must be a string')
    if (
      name.trim() === '' ||
      name !== name.trim() ||
      name.includes('/') ||
      name === '.' ||
      name === '..' ||
      [...name].some(isControl) ||
      LONE_SURROGATE.test(name)
    ) {
      throw new RangeError(
        "Memory store name must be non-empty, not '.' or '..', and without '/', control characters, " +
          `lone surrogates, or leading or trailing whitespace; got ${JSON.stringify(name)}`
      )
    }
    // null means what None means for the same Python parameter: unset, or no description.
    const description = config?.description === undefined ? DEFAULT_DESCRIPTION : config.description
    if (description !== null && typeof description !== 'string') {
      throw new TypeError('description must be a string or null')
    }
    const writable = config?.writable === undefined ? true : config.writable
    if (typeof writable !== 'boolean') throw new TypeError('writable must be a boolean')
    const embedder = config?.embedder
    if (
      embedder != null &&
      typeof embedder !== 'function' &&
      typeof (embedder as Partial<Embedder>).embed !== 'function'
    ) {
      throw new TypeError('embedder must be a function or an object with an embed method')
    }
    if (config?.maxSearchResults != null) checkLimit(config.maxSearchResults, 'maxSearchResults')
    this.name = name
    if (description !== null) this.description = description
    if (config?.maxSearchResults != null) this.maxSearchResults = config.maxSearchResults
    this.writable = writable
    if (config?.extraction != null) this.extraction = config.extraction
    this._storage = storage
    this._keyPrefix = `memory/${name}/`
    // Every memory key shares this partition: the id is at least the third key segment.
    this._partition = storage.partitionKey(`${this._keyPrefix}0`)
    const region = storageRegion(storage)
    this._embedder = embedder ?? new BedrockEmbedder(region !== undefined ? { region } : {})
  }

  /**
   * Returns the stored memories closest in meaning to `query`, closest first.
   *
   * Each entry's `metadata` holds the metadata stored with the memory, plus `_key` (the
   * storage key, which {@link DynamoDBStorage.delete} accepts) and `_score` (the raw
   * vector index score: for COSINE and EUCLIDEAN, lower is closer). A per-call
   * `maxSearchResults` above 100 is capped at 100, the `SearchVectors` limit. Memories
   * stored inline come back from the index projection; a memory large enough to be
   * offloaded to S3 is read back from S3. A query that is empty or whitespace returns an
   * empty array without calling the embedder or DynamoDB.
   *
   * @throws TypeError if `query` is not a string
   * @throws RangeError if `maxSearchResults` is less than 1 or not an integer, or `query`
   *   contains a lone surrogate
   * @throws {@link StorageError} if embedding or the search fails
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    if (typeof query !== 'string') throw new TypeError('query must be a string')
    const limit = this._resolveLimit(options)
    if (isBlank(query)) return []
    // Bedrock would embed it as "?", a different query from the one asked.
    if (LONE_SURROGATE.test(query)) {
      throw new RangeError('query must be valid Unicode text; it contains a lone surrogate')
    }
    const vector = await this._embed(query, 'query')
    const results = await this._storage.search({ vector, topK: limit, pk: this._partition, includeValues: true })
    // ignoreBOM keeps a leading U+FEFF, so content reads back exactly as it was stored.
    const decoder = new TextDecoder('utf-8', { ignoreBOM: true })
    const entries: MemoryEntry[] = []
    for (const result of results) {
      // The partition can hold other vector items (another store's memories, or the
      // application's own); keep only this store's.
      if (!result.key.startsWith(this._keyPrefix) || result.data === undefined) continue
      const metadata: Record<string, JSONValue> = {
        ...((result.metadata ?? {}) as Record<string, JSONValue>),
        _key: result.key,
        _score: result.score,
      }
      entries.push({ content: decoder.decode(result.data), metadata })
    }
    return entries
  }

  /**
   * Embeds `content` and stores it as a memory; returns its storage key.
   *
   * Adding text that is already stored overwrites that item, replacing its metadata and
   * restarting its TTL, so retried writes do not create duplicates.
   *
   * `metadata` values that are strings, booleans, or numbers from 1e-130 to
   * `Number.MAX_SAFE_INTEGER` in magnitude (or zero) are stored. Anything else is dropped
   * with a warning, so a memory whose metadata came from a model is still stored: arrays,
   * nested objects, `null`, other numbers, keys or strings with a lone surrogate, empty
   * keys, the keys `__proto__` and `constructor`, and the keys `_key` and `_score`, which
   * are reserved for search results.
   *
   * @throws TypeError if `content` is not a string
   * @throws RangeError if `content` is empty or whitespace, or contains a lone surrogate
   * @throws {@link StorageError} if embedding or the write fails
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<string> {
    if (typeof content !== 'string') throw new TypeError('content must be a string')
    if (isBlank(content)) throw new RangeError('content must not be empty')
    if (LONE_SURROGATE.test(content))
      throw new RangeError('content must be valid Unicode text; it contains a lone surrogate')
    const checked = this._storableMetadata(metadata)
    const payload = new TextEncoder().encode(content)
    const key = `${this._keyPrefix}${createHash('sha256').update(payload).digest('hex').slice(0, 32)}`
    const vector = await this._embed(content, 'document')
    await this._storage.write(key, payload, { vector, ...(checked ? { metadata: checked } : {}) })
    return key
  }

  private _resolveLimit(options: SearchOptions | undefined): number {
    const requested = options?.maxSearchResults ?? this.maxSearchResults
    if (requested === undefined) return DEFAULT_MAX_SEARCH_RESULTS
    if (!Number.isInteger(requested) || requested < 1) {
      throw new RangeError(`maxSearchResults must be an integer >= 1; got ${String(requested)}`)
    }
    return Math.min(requested, MAX_TOP_K)
  }

  private async _embed(text: string, purpose: EmbeddingPurpose): Promise<number[]> {
    const embedder = this._embedder
    let result: unknown
    try {
      // An object with an embed method is an Embedder even when it is also callable, as in Python.
      const asEmbedder = embedder as Partial<Embedder>
      result =
        typeof asEmbedder.embed === 'function'
          ? await asEmbedder.embed(text, purpose)
          : await (embedder as EmbedFunction)(text)
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError(`Embedding failed in memory store '${this.name}': ${describeError(error)}`, {
        cause: error,
      })
    }
    if (
      !Array.isArray(result) ||
      result.length === 0 ||
      // Array.from turns holes in a sparse array into undefined, which fails the check.
      !Array.from(result as unknown[]).every((v) => typeof v === 'number' && Number.isFinite(v))
    ) {
      throw new StorageError(
        `The embedder for memory store '${this.name}' must return a non-empty array of finite numbers`
      )
    }
    return result as number[]
  }

  /** Keeps the metadata entries the storage can store; drops the rest with a warning. */
  private _storableMetadata(metadata: unknown): Record<string, string | number | boolean> | undefined {
    if (metadata === undefined || metadata === null) return undefined
    const prototype = typeof metadata === 'object' ? Object.getPrototypeOf(metadata) : undefined
    if (typeof metadata !== 'object' || (prototype !== Object.prototype && prototype !== null)) {
      console.warn(`store=<${this.name}> | dropping metadata that is not a plain object`)
      return undefined
    }
    const kept: Record<string, string | number | boolean> = {}
    for (const [key, value] of Object.entries(metadata)) {
      let reason: string | undefined
      if (RESERVED_METADATA_KEYS.has(key)) reason = 'is reserved for search results'
      else if (key === '') reason = 'is empty, which DynamoDB rejects'
      else if (UNWRITABLE_KEYS.has(key)) reason = 'cannot be written by the AWS SDK for JavaScript'
      else if (LONE_SURROGATE.test(key) || (typeof value === 'string' && LONE_SURROGATE.test(value))) {
        reason = "contains a lone surrogate, which DynamoDB stores as '?'"
      } else if (typeof value === 'number' && !isDynamoDBNumber(value)) {
        reason = 'is out of range (finite, zero or 1e-130 to 2^53 - 1 in magnitude)'
      } else if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
        reason = `has type ${value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value}; values must be strings, numbers, or booleans`
      }
      if (reason === undefined) kept[key] = value as string | number | boolean
      else console.warn(`store=<${this.name}>, key=<${key}> | dropping metadata entry: ${reason}`)
    }
    return Object.keys(kept).length > 0 ? kept : undefined
  }
}

/**
 * Whether `value` can be written as a DynamoDB number: finite, zero or at least 1e-130 in
 * magnitude (DynamoDB's lower bound), and at most `Number.MAX_SAFE_INTEGER`, above which
 * the AWS SDK refuses to marshall a plain number.
 */
function isDynamoDBNumber(value: number): boolean {
  const magnitude = Math.abs(value)
  return Number.isFinite(value) && magnitude <= Number.MAX_SAFE_INTEGER && (value === 0 || magnitude >= 1e-130)
}

/** Whether `text` is only whitespace, by the union of JavaScript's and Python's definitions. */
function isBlank(text: string): boolean {
  return [...text].every((c) => c.trim() === '' || c === '\u0085' || (c >= '\u001c' && c <= '\u001f'))
}

/** C0 and C1 control characters and DEL, the set both packages reject in store names. */
function isControl(character: string): boolean {
  const code = character.charCodeAt(0)
  return code < 0x20 || (code >= 0x7f && code <= 0x9f)
}

function checkLimit(value: unknown, label: string): void {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > MAX_TOP_K) {
    throw new RangeError(`${label} must be an integer from 1 to ${MAX_TOP_K}; got ${String(value)}`)
  }
}
