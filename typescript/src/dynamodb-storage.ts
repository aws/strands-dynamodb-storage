// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'
import type { Storage } from '@strands-agents/sdk/storage'

import { gunzip, gzip } from 'node:zlib'
import { promisify } from 'node:util'

import { normalizeKey, normalizePrefix } from './keys.js'

const gzipAsync = promisify(gzip)
const gunzipAsync = promisify(gunzip)

/**
 * Structured list query for the DynamoDB backend.
 *
 * The {@link Storage} `list` argument is generic precisely so backends like
 * DynamoDB and RDS can accept a richer query than a plain string prefix. A
 * `DynamoDBListQuery` names the partition (`pk`) directly, so listing is a native
 * `Query` against one partition instead of an inferred prefix scan.
 *
 * SDK-internal callers still pass a plain string prefix; see {@link DynamoDBStorage.list}.
 */
export interface DynamoDBListQuery {
  /** Partition key to query. */
  pk: string
  /** Optional sort-key prefix (`begins_with`). Mutually exclusive with `skBetween`. */
  skPrefix?: string
  /** Optional inclusive sort-key range `[from, to]`. Mutually exclusive with `skPrefix`. */
  skBetween?: [string, string]
  /** Max number of keys to return. */
  limit?: number
  /** Exclusive start cursor (a key returned by a prior call). */
  startAfter?: string
}

/**
 * Vector similarity query for backends that support native ANN search (e.g. the
 * DynamoDB vector index). Optional on {@link Storage}; consumers feature-detect
 * `if (storage.search)` and fall back to client-side KNN when absent.
 */
export interface SearchQuery {
  /** Query embedding. */
  vector: number[]
  /** Number of nearest neighbours to return. */
  topK: number
  /** Partition/scope to search within. Required when the index's SearchSchema declares a HASH element; omit otherwise. */
  pk?: string
  /** Optional metadata pre-filter (equality). */
  filter?: Record<string, string | number | boolean>
  /** When true, hydrate each result's stored bytes via `read`. Default false. */
  includeValues?: boolean
}

/** A single similarity match. */
export interface SearchResult {
  /** Storage key of the matched item (with any constructor prefix stripped). */
  key: string
  /** Raw index score; direction depends on the index's distance function: lower is nearer for COSINE/EUCLIDEAN, higher for DOT_PRODUCT. */
  score: number
  /** Stored bytes, present only when `includeValues` was requested. */
  data?: Uint8Array
  /** Metadata stored alongside the item, when available. */
  metadata?: Record<string, unknown>
}

/**
 * Optional override for the vector-search call. By default {@link DynamoDBStorage.search}
 * issues DynamoDB `SearchVectors` natively (requires `@aws-sdk/client-dynamodb` >= 3.1103.0);
 * supply an adapter to replace the call (e.g. for testing or custom routing). An adapter
 * opts in to TTL filtering by returning each item's epoch-seconds expiry as `expireAt`.
 */
export type VectorSearchAdapter = (params: {
  tableName: string
  indexName: string
  vectorAttribute: string
  pk?: string
  vector: number[]
  topK: number
  filter?: Record<string, string | number | boolean>
}) => Promise<Array<{ key: string; score: number; metadata?: Record<string, unknown>; expireAt?: number }>>

/**
 * Text search strategy that {@link DynamoDBStorage} delegates plain-string `search()` calls to, following the SDK
 * backends' pattern. Structurally compatible with the SDK `SearchStrategy` (declared here so this package does not
 * depend on SDK versions that ship it), so an SDK strategy can be configured as is.
 *
 * `index`, when present, runs after every successful `write()` with the normalized key (relative to the storage
 * prefix), the written bytes and the write's options. It is a second step after the write, not atomic with it.
 */
export interface StringSearchStrategy {
  /** Returns the best matches for a natural-language `query` over `storage`, best first. */
  search(storage: Storage, query: string): Promise<Array<Pick<SearchResult, 'key' | 'score' | 'data'>>>
  /** Indexes the value `storage` has just written under `key`. */
  index?(storage: Storage, key: string, data: Uint8Array, options?: DocumentItemOptions): Promise<void>
}

/** Configuration for {@link DynamoDBStorage}. */
export interface DynamoDBStorageConfig {
  /** AWS region override. Ignored when `client` is supplied. */
  region?: string
  /** Pre-configured DynamoDB Document client. Cannot be combined with `region`. */
  client?: import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient
  /** Optional key prefix prepended to every key (a namespace within the table). */
  prefix?: string
  /**
   * Optional S3 bucket for offloading values larger than the DynamoDB item limit.
   * When unset, oversized values throw rather than silently truncating.
   */
  s3Bucket?: string
  /** Optional key prefix for offloaded S3 objects. */
  s3Prefix?: string
  /** Pre-configured S3 client for offload. Ignored unless `s3Bucket` is set. */
  s3Client?: import('@aws-sdk/client-s3').S3Client
  /**
   * Optional transparent gzip compression of stored values. Applied before the
   * S3-offload size check, so compressible values are more likely to stay inline in
   * DynamoDB — reducing item size, storage cost, and S3 spillover. Each item records
   * whether it was compressed, so reads are correct regardless of this setting, and
   * values that do not shrink are stored uncompressed. Default `'none'`.
   */
  compression?: 'gzip' | 'none'
  /**
   * Optional time-to-live in seconds. Setting it opts in to TTL: each written item
   * carries an epoch-seconds expiry attribute (see `ttlAttribute`) so DynamoDB native
   * TTL reaps it, and `read`/`list`/`search` filter out items whose expiry has already passed —
   * covering the up-to-~48h lag before DynamoDB physically deletes them. A per-write
   * `ttlSeconds` tunes the duration for that write.
   * On an instance that did not opt in here, a per-write `ttlSeconds` is ignored.
   *
   * Physical cleanup requires TTL enabled on that attribute at the table level. When
   * combined with S3 offload, expiry removes only the DynamoDB pointer item — configure
   * an S3 lifecycle rule to reclaim offloaded objects.
   */
  ttlSeconds?: number
  /** Item attribute holding the epoch-seconds TTL. Default `'expireAt'`. */
  ttlAttribute?: string
  /** Name of the vector index on the table (for `search`). Default `'vector_index'`. */
  indexName?: string
  /** Item attribute holding the embedding vector. Default `'vector'`. */
  vectorAttribute?: string
  /** Optional override of the native `SearchVectors` call (testing, custom routing). When unset, `search()` calls DynamoDB natively. */
  vectorSearch?: VectorSearchAdapter
  /**
   * Optional text search strategy, such as `LexicalSearchStrategy`: plain-string `search()` calls delegate to it,
   * and its `index` hook, when present, runs after every successful `write()`. Namespaced views inherit it.
   */
  searchStrategy?: StringSearchStrategy
}

/**
 * Where a document lives: its normalized key relative to the storage prefix, its canonical id
 * (prefix + key), and the base-table partition and sort keys derived from that id.
 *
 * @internal
 */
export interface DocumentLocation {
  key: string
  docId: string
  pk: string
  sk: string
}

/**
 * Optional attributes of a document item, as accepted by {@link DynamoDBStorage.write} and forwarded to a
 * {@link StringSearchStrategy.index} hook.
 */
export interface DocumentItemOptions {
  /** Embedding for the native vector index; stored inline even when the value is offloaded. */
  vector?: number[]
  /** Metadata stored on the item; returned by and filterable in `search()`. */
  metadata?: Record<string, string | number | boolean>
  /** Per-write TTL; honoured only when the storage opted in to TTL. */
  ttlSeconds?: number
}

/**
 * Narrow view of a {@link DynamoDBStorage} used by the lexical index: the base-table facts and item
 * helpers it needs, so the index never reaches into other storage internals. Not part of the public API.
 *
 * @internal
 */
export interface DocumentItemPort {
  /** Base table name. */
  readonly tableName: string
  /** Storage prefix exactly as held (`'tenant/a/'` or `''`). */
  readonly scope: string
  /** TTL attribute name; set only when the storage opted in to TTL. */
  readonly ttlAttribute: string | undefined
  /** The storage's (lazily built) Document client. */
  client(): Promise<import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient>
  /** Normalizes `key` and derives its canonical id and base-table keys. */
  locate(key: string): DocumentLocation
  /** Strips the storage prefix from a canonical id, or returns `null` when the id is outside this storage. */
  relativeKey(docId: string): string | null
  /** True when this storage opted in to TTL and the item carries a TTL stamp at or before now. */
  isExpired(item: Record<string, unknown>): boolean
  /**
   * Builds the inline base item exactly as `write()` would.
   *
   * @throws {@link StorageError} when the encoded value would need S3 offload, which indexed documents do not support
   */
  inlineItem(
    location: DocumentLocation,
    data: Uint8Array,
    options?: DocumentItemOptions
  ): Promise<Record<string, unknown>>
  /** Best-effort delete of the offloaded S3 object of `docId`; a no-op without a bucket; never throws. */
  deleteOffloaded(docId: string): Promise<void>
}

/** User-Agent marker attributing this package's AWS traffic (self-built clients only). */
const USER_AGENT_MARKER = 'strands-dynamodb-storage'

/**
 * Attribute names for the single-table layout. Shared with the lexical index, which writes the same
 * base items; not part of the public API.
 *
 * @internal
 */
export const PK = 'pk'
export const SK = 'sk'
export const KEY_ATTR = 'k'
export const DATA_ATTR = 'data'
export const S3_ATTR = 's3'
export const META_ATTR = 'meta'
export const Z_ATTR = 'z'
/** Service maximum for SearchVectors TopK ("must be between 1 and 100 inclusive"). */
const MAX_TOP_K = 100
const TEXT_QUERY_WITHOUT_STRATEGY_MESSAGE =
  'DynamoDBStorage.search requires a SearchQuery with a pre-computed embedding vector. ' +
  'Plain-string queries are not supported without a searchStrategy: embed the text first and pass { vector, topK }, ' +
  'or configure a searchStrategy (for example LexicalSearchStrategy).'
/** Over-fetch factor for client-side metadata filtering, capped at MAX_TOP_K. */
const FILTER_OVERFETCH = 10

/**
 * DynamoDB {@link Storage} backend (single-table design).
 *
 * Persists each key as one item: the `/`-separated key is split into a partition
 * key (leading scope) and a sort key (remainder), with the raw bytes in a binary
 * `data` attribute. Point operations are single-item `PutItem`/`GetItem`/`DeleteItem`.
 * Listing is a native `Query` — a string prefix is resolved to a partition plus a
 * `begins_with` sort condition, while a {@link DynamoDBListQuery} names the partition
 * directly (the intended DDB extension point of the generic `Storage` interface).
 *
 * Values above the 400 KB item limit are offloaded to S3 when an `s3Bucket` is
 * configured; a small pointer item stays in DynamoDB. The AWS SDK modules are
 * lazy-imported and declared as peer dependencies (S3 optional), so consumers that
 * never construct a `DynamoDBStorage` are not forced to install them.
 *
 * Optional transparent gzip `compression` shrinks stored values (and keeps more of
 * them inline, out of S3), and an optional `ttlSeconds` stamps a DynamoDB-native TTL
 * attribute so ephemeral session/memory data self-expires.
 *
 * @example
 * ```typescript
 * import { Agent, SessionManager } from '@strands-agents/sdk'
 * import { DynamoDBStorage } from 'strands-dynamodb-storage'
 *
 * const storage = new DynamoDBStorage('agent-data', { region: 'us-east-1' })
 * const agent = new Agent({ sessionManager: new SessionManager({ storage }) })
 * ```
 */
export class DynamoDBStorage implements Storage<string | DynamoDBListQuery> {
  /** Leave margin below DynamoDB's 400 KB item limit for keys and metadata. */
  private static readonly OFFLOAD_THRESHOLD_BYTES = 380_000

  private readonly _tableName: string
  private readonly _prefix: string
  private readonly _region: string | undefined
  private readonly _s3Bucket: string | undefined
  private readonly _s3Prefix: string
  private readonly _indexName: string
  private readonly _vectorAttribute: string
  private readonly _vectorSearch: VectorSearchAdapter | undefined
  private readonly _searchStrategy: StringSearchStrategy | undefined
  private readonly _compress: boolean
  private readonly _ttlSeconds: number | undefined
  private readonly _ttlEnabled: boolean
  private readonly _ttlAttribute: string
  private _client: import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient | undefined
  private _s3Client: import('@aws-sdk/client-s3').S3Client | undefined

  /**
   * @param tableName - Target DynamoDB table (partition key `pk` string, sort key `sk` string)
   * @param config - Optional region/client, key prefix, and S3-offload settings
   * @throws {@link StorageError} if both `region` and `client` are provided
   */
  constructor(tableName: string, config?: DynamoDBStorageConfig) {
    if (config?.client && config.region) {
      throw new StorageError('Cannot specify both client and region. Configure the region on the client instead.')
    }
    this._tableName = tableName
    this._prefix = config?.prefix ? config.prefix.split('/').filter(Boolean).join('/') + '/' : ''
    this._region = config?.region
    this._client = config?.client
    this._s3Bucket = config?.s3Bucket
    this._s3Prefix = config?.s3Prefix ? config.s3Prefix.split('/').filter(Boolean).join('/') + '/' : ''
    this._s3Client = config?.s3Client
    this._indexName = config?.indexName ?? 'vector_index'
    this._vectorAttribute = config?.vectorAttribute ?? 'vector'
    this._vectorSearch = config?.vectorSearch
    this._searchStrategy = config?.searchStrategy
    this._compress = config?.compression === 'gzip'
    this._ttlSeconds = config?.ttlSeconds
    this._ttlEnabled = config?.ttlSeconds !== undefined
    this._ttlAttribute = config?.ttlAttribute ?? 'expireAt'
  }

  /**
   * Stores `data` under `key`, overwriting any existing value. Values above the
   * item-size limit are offloaded to S3 when an `s3Bucket` is configured. An optional
   * `vector` (kept inline for the native index) and `metadata` enable `search`. When the
   * configured `searchStrategy` has an `index` hook, it runs after the write succeeds.
   *
   * @throws {@link StorageError} if the key is invalid, the value is oversized with
   *   no S3 bucket configured, or the write fails; or, with the value already stored,
   *   `Wrote '<key>' but indexing failed` (the hook's error is the cause)
   */
  async write(key: string, data: Uint8Array, options?: DocumentItemOptions): Promise<void> {
    const normalized = normalizeKey(key)
    const full = `${this._prefix}${normalized}`
    const { pk, sk } = this._split(full)
    const extra = this._itemExtras(options)
    const { payload, compressed } = await this._encodeValue(data)
    if (compressed) extra[Z_ATTR] = true
    try {
      if (payload.byteLength > DynamoDBStorage.OFFLOAD_THRESHOLD_BYTES) {
        if (!this._s3Bucket) {
          throw new StorageError(
            `Value for '${normalized}' is ${payload.byteLength} bytes, above the ${DynamoDBStorage.OFFLOAD_THRESHOLD_BYTES}-byte limit; configure s3Bucket to offload large values`
          )
        }
        await this._s3Put(full, payload)
        await this._put({ [PK]: pk, [SK]: sk, [KEY_ATTR]: full, [S3_ATTR]: true, ...extra })
      } else {
        // An overwrite can shrink a previously offloaded value back inline. Ask for
        // the replaced item so the now-unreferenced S3 object can be reclaimed: the
        // new item carries no s3 flag, so without this the object is orphaned
        // forever (a later delete() never reaches it).
        const old = await this._put(
          { [PK]: pk, [SK]: sk, [KEY_ATTR]: full, [DATA_ATTR]: payload, ...extra },
          { returnOld: Boolean(this._s3Bucket) }
        )
        if (old?.[S3_ATTR]) await this._reclaimOffloaded(full)
      }
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError(`Failed to write '${normalized}' to DynamoDB table '${this._tableName}'`, { cause: error })
    }
    await this._indexWritten(normalized, data, options)
  }

  /**
   * Retrieves the bytes previously stored under `key`.
   *
   * @returns The stored bytes, or `null` if no value exists for `key`
   * @throws {@link StorageError} if the key is invalid or the read fails
   */
  async read(key: string): Promise<Uint8Array | null> {
    const normalized = normalizeKey(key)
    const full = `${this._prefix}${normalized}`
    const { pk, sk } = this._split(full)
    try {
      const { GetCommand } = await import('@aws-sdk/lib-dynamodb')
      const client = await this._getClient()
      const response = await client.send(new GetCommand({ TableName: this._tableName, Key: { [PK]: pk, [SK]: sk } }))
      const item = response.Item
      if (!item) return null
      // Hide items whose TTL has passed but DynamoDB has not yet physically reaped
      // (native TTL deletion lags up to ~48h). Only when TTL is opted into. Runs
      // before any S3 fetch.
      if (this._ttlEnabled && this._isExpired(item)) return null
      const compressed = item[Z_ATTR] === true
      let raw: Uint8Array | null
      if (item[S3_ATTR]) {
        raw = await this._s3Get(full)
      } else {
        const stored = item[DATA_ATTR] as Uint8Array | undefined
        raw = stored ? new Uint8Array(stored) : null
      }
      if (raw === null) return null
      return compressed ? new Uint8Array(await gunzipAsync(raw)) : raw
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError(`Failed to read '${normalized}' from DynamoDB table '${this._tableName}'`, {
        cause: error,
      })
    }
  }

  /**
   * Deletes the value stored under `key`, including any offloaded S3 object.
   * A no-op if the key does not exist.
   *
   * @throws {@link StorageError} if the key is invalid or the delete fails
   */
  async delete(key: string): Promise<void> {
    const normalized = normalizeKey(key)
    const full = `${this._prefix}${normalized}`
    const { pk, sk } = this._split(full)
    try {
      const { DeleteCommand } = await import('@aws-sdk/lib-dynamodb')
      const client = await this._getClient()
      // ReturnValues=ALL_OLD lets us clean up an offloaded S3 object without a prior read.
      const response = await client.send(
        new DeleteCommand({ TableName: this._tableName, Key: { [PK]: pk, [SK]: sk }, ReturnValues: 'ALL_OLD' })
      )
      if (response.Attributes?.[S3_ATTR]) await this._s3Delete(full)
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError(`Failed to delete '${normalized}' from DynamoDB table '${this._tableName}'`, {
        cause: error,
      })
    }
  }

  /**
   * Lists keys, sorted ascending.
   *
   * Accepts either:
   * - a **string prefix** (SDK-internal path): resolved to a partition plus a
   *   `begins_with` sort condition. The prefix must include at least the scope and
   *   its identifier (e.g. `sessions/<id>/...`) so a partition can be determined.
   * - a **{@link DynamoDBListQuery}**: a native query that names the partition directly.
   *
   * @returns The matching full keys, sorted ascending
   * @throws {@link StorageError} if the query is invalid or the list fails
   */
  async list(query: string | DynamoDBListQuery): Promise<string[]> {
    try {
      if (typeof query === 'string') {
        return await this._listByPrefix(query)
      }
      return await this._listByQuery(query)
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      const label = typeof query === 'string' ? `'${query}'` : `pk='${query.pk}'`
      throw new StorageError(`Failed to list DynamoDB table '${this._tableName}' under ${label}`, { cause: error })
    }
  }

  /**
   * Returns a view of this storage with all keys prefixed by `prefix`, without
   * mutating the original.
   *
   * Unlike the SDK's generic string-only namespace helper, this returns a real
   * `DynamoDBStorage` sharing the same client and settings, so structured
   * {@link DynamoDBListQuery} listing and {@link DynamoDBStorage.search} survive
   * namespacing. Nested calls compose.
   */
  namespace(prefix: string): DynamoDBStorage {
    const sub = normalizePrefix(prefix)
    const config: DynamoDBStorageConfig = {
      prefix: `${this._prefix}${sub}`,
      indexName: this._indexName,
      vectorAttribute: this._vectorAttribute,
    }
    if (this._client) config.client = this._client
    else if (this._region !== undefined) config.region = this._region
    if (this._s3Bucket !== undefined) config.s3Bucket = this._s3Bucket
    if (this._s3Prefix) config.s3Prefix = this._s3Prefix
    if (this._s3Client) config.s3Client = this._s3Client
    if (this._vectorSearch) config.vectorSearch = this._vectorSearch
    if (this._searchStrategy) config.searchStrategy = this._searchStrategy
    if (this._compress) config.compression = 'gzip'
    if (this._ttlSeconds !== undefined) config.ttlSeconds = this._ttlSeconds
    config.ttlAttribute = this._ttlAttribute
    return new DynamoDBStorage(this._tableName, config)
  }

  /**
   * Nearest-neighbour vector search over items written with a `vector`.
   *
   * Optional part of the {@link Storage} contract — consumers feature-detect
   * `if (storage.search)` and fall back to client-side KNN when absent. On DynamoDB
   * this runs against the native vector index, so scoring happens in the database.
   *
   * Issues DynamoDB `SearchVectors` natively; a `vectorSearch` adapter, when
   * configured, overrides the call. The score is the raw index `Score` and its
   * direction depends on the index's distance function: for COSINE and EUCLIDEAN
   * lower is closer; for DOT_PRODUCT higher is more similar. Results arrive in
   * the service's most-similar-first order. Like a global secondary index, the
   * vector index is eventually consistent.
   *
   * With TTL enabled, matches whose TTL attribute has passed are dropped, the same
   * check `read` applies, without any extra reads: the native path projects the TTL
   * attribute from the vector index, and an adapter opts in by returning `expireAt`
   * on each match. A match that carries no expiry (an index that does not project the
   * TTL attribute, or an adapter that does not return it) is kept. Expired matches are
   * not back-filled, so a search can return fewer than `topK` results.
   *
   * A plain-string query (the SDK's natural-language form) is delegated to the configured
   * `searchStrategy`; its matches come back as `{ key, score, data? }`, without metadata.
   *
   * @throws {@link StorageError} if the search fails, `topK` is out of range, or the
   *   query is a plain string and no `searchStrategy` is configured — this store searches
   *   pre-computed embedding vectors and does not embed text
   */
  async search(query: SearchQuery | string): Promise<SearchResult[]> {
    if (typeof query === 'string') return this._searchText(query)
    if (query.pk !== undefined) this._assertPkInScope(query.pk)
    try {
      const matches: Array<{
        key: string
        score: number
        metadata?: Record<string, unknown>
        data?: Uint8Array
        expireAt?: number
      }> = this._vectorSearch
        ? await this._vectorSearch({
            tableName: this._tableName,
            indexName: this._indexName,
            vectorAttribute: this._vectorAttribute,
            vector: query.vector,
            topK: query.topK,
            ...(query.pk !== undefined && { pk: query.pk }),
            ...(query.filter !== undefined && { filter: query.filter }),
          })
        : await this._nativeVectorSearch(query)
      const results: SearchResult[] = []
      for (const match of matches) {
        // Defence in depth: pk is optional, so the index may return matches from
        // outside this namespace. Drop them rather than de-prefixing blindly — a
        // foreign key would otherwise be handed back looking like one of ours.
        const key = this._stripPrefix(match.key)
        if (key === null) continue
        if (this._ttlEnabled && this._hasPassed(match.expireAt)) continue
        const result: SearchResult = { key, score: match.score }
        if (match.metadata !== undefined) result.metadata = match.metadata
        if (query.includeValues) {
          // The native path decodes the projected payload into the match; offloaded
          // values, narrow projections, and adapter matches fall back to a point
          // read (which also fetches from S3 and decompresses).
          const data = match.data !== undefined ? match.data : await this.read(key)
          if (data) result.data = data
        }
        results.push(result)
      }
      return results
    } catch (error: unknown) {
      throw this._searchError(error)
    }
  }

  /**
   * Delegates a natural-language query to the configured search strategy. This store has no embedding model, so
   * without a strategy it cannot honour one and rejects it.
   */
  private async _searchText(query: string): Promise<SearchResult[]> {
    const strategy = this._searchStrategy
    if (!strategy) throw new StorageError(TEXT_QUERY_WITHOUT_STRATEGY_MESSAGE)
    try {
      const matches = await strategy.search(this, query)
      return matches.map(({ key, score, data }) => (data === undefined ? { key, score } : { key, score, data }))
    } catch (error: unknown) {
      throw this._searchError(error)
    }
  }

  private _searchError(error: unknown): StorageError {
    if (error instanceof StorageError) return error
    return new StorageError(`Failed to search DynamoDB table '${this._tableName}'`, { cause: error })
  }

  /**
   * Runs the search strategy's `index` hook after a successful write, as the SDK backends do. The value is already
   * stored by then, so a failed hook is reported as an indexing failure (its error as the cause), not a failed write.
   */
  private async _indexWritten(key: string, data: Uint8Array, options: DocumentItemOptions | undefined): Promise<void> {
    const strategy = this._searchStrategy
    if (!strategy?.index) return
    try {
      await strategy.index(this, key, data, {
        vector: options?.vector,
        metadata: options?.metadata,
        ttlSeconds: options?.ttlSeconds,
      })
    } catch (error: unknown) {
      throw new StorageError(`Wrote '${key}' but indexing failed`, { cause: error })
    }
  }

  /**
   * Issues DynamoDB `SearchVectors` directly (GA path, no adapter).
   *
   * `query.pk` pins `SearchConditionExpression` to one partition (already
   * namespace-validated by the caller). `query.filter` is applied client-side
   * after the search, with the request over-fetching (capped at the service
   * TopK limit of 100) so filtering doesn't silently drop results below
   * `topK`; a filtered search can still return fewer than `topK` matches once
   * the cap truncates the over-fetch. With TTL enabled, the TTL attribute is
   * projected and expired matches are dropped before counting towards `topK`.
   */
  private async _nativeVectorSearch(
    query: SearchQuery
  ): Promise<Array<{ key: string; score: number; metadata?: Record<string, unknown>; data?: Uint8Array }>> {
    if (query.topK < 1 || query.topK > MAX_TOP_K) {
      throw new StorageError(`topK must be between 1 and ${MAX_TOP_K} (SearchVectors TopK limit); got ${query.topK}`)
    }
    if (!query.vector.every((v) => Number.isFinite(v))) {
      throw new StorageError(
        'Query vector contains non-finite values (NaN/Infinity); the DynamoDB N type rejects them.'
      )
    }
    const topK = query.filter !== undefined ? Math.min(query.topK * FILTER_OVERFETCH, MAX_TOP_K) : query.topK

    const { SearchVectorsCommand } = await import('@aws-sdk/client-dynamodb')
    const { unmarshall } = await import('@aws-sdk/util-dynamodb')
    const client = await this._getClient()
    // Project only what the response is for: keys (to rebuild the storage key)
    // and metadata (results + client-side filter), adding the payload and its
    // s3/gzip flags only when the caller asked for values. Everything else the
    // index projects (ProjectionType ALL projects the whole item) is billed
    // response bytes for data nobody reads. Every name is aliased: `data` is a
    // DynamoDB reserved word.
    const names: Record<string, string> = { '#pk': PK, '#sk': SK, '#m': META_ATTR }
    let projection = '#pk, #sk, #m'
    if (query.includeValues) {
      Object.assign(names, { '#d': DATA_ATTR, '#s3': S3_ATTR, '#z': Z_ATTR })
      projection += ', #d, #s3, #z'
    }
    if (this._ttlEnabled) {
      names['#ttl'] = this._ttlAttribute
      projection += ', #ttl'
    }
    const response = await client.send(
      new SearchVectorsCommand({
        TableName: this._tableName,
        IndexName: this._indexName,
        SearchVector: query.vector.map((v) => ({ N: String(v) })),
        TopK: topK,
        ProjectionExpression: projection,
        ExpressionAttributeNames: names,
        ...(query.pk !== undefined && {
          SearchConditionExpression: '#pk = :pk',
          ExpressionAttributeValues: { ':pk': { S: query.pk } },
        }),
      })
    )

    const matches: Array<{ key: string; score: number; metadata?: Record<string, unknown>; data?: Uint8Array }> = []
    for (const result of response.SearchResults ?? []) {
      if (!result.Item || result.Score === undefined) continue
      const item = unmarshall(result.Item)
      const pk = item[PK] as string
      const sk = (item[SK] as string | undefined) ?? ''
      const fullKey = sk === '' || sk === '\u0000' ? pk : `${pk}/${sk}`
      if (this._ttlEnabled && this._isExpired(item)) continue
      const metadata = item[META_ATTR] as Record<string, unknown> | undefined
      if (query.filter !== undefined && !DynamoDBStorage._matchesSearchFilter(metadata, query.filter)) continue
      const match: { key: string; score: number; metadata?: Record<string, unknown>; data?: Uint8Array } = {
        key: fullKey,
        score: result.Score,
        ...(metadata !== undefined && { metadata }),
      }
      if (query.includeValues && item[S3_ATTR] !== true) {
        const stored = item[DATA_ATTR] as Uint8Array | undefined
        if (stored !== undefined) {
          const raw = new Uint8Array(stored)
          match.data = item[Z_ATTR] === true ? new Uint8Array(await gunzipAsync(raw)) : raw
        }
      }
      matches.push(match)
      if (matches.length >= query.topK) break
    }
    return matches
  }

  /** Exact-equality match of every filter entry against item metadata. */
  private static _matchesSearchFilter(
    metadata: Record<string, unknown> | undefined,
    filter: Record<string, string | number | boolean>
  ): boolean {
    if (metadata === undefined) return false
    return Object.entries(filter).every(([k, v]) => metadata[k] === v)
  }

  /**
   * Returns the document item port the lexical index builds on. Internal: not part of the
   * public API and may change without notice.
   *
   * @internal
   */
  _documentItemPort(): DocumentItemPort {
    return {
      tableName: this._tableName,
      scope: this._prefix,
      ttlAttribute: this._ttlEnabled ? this._ttlAttribute : undefined,
      client: () => this._getClient(),
      locate: (key) => this._locate(key),
      relativeKey: (docId) => this._stripPrefix(docId),
      isExpired: (item) => this._ttlEnabled && this._isExpired(item),
      inlineItem: (location, data, options) => this._inlineItem(location, data, options),
      deleteOffloaded: (docId) => this._reclaimOffloaded(docId),
    }
  }

  // ---------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------

  /**
   * Optional attributes shared by every write path: the embedding, metadata, and TTL stamp.
   *
   * The embedding stays inline even when the payload is offloaded, because the native vector index
   * can only index an on-item attribute; non-finite values are rejected up front (mirroring the
   * query-side check in `search()`) instead of surfacing as an opaque write failure. The TTL stamp is
   * written only when this instance opted in to TTL, and the whole stamp is floored so a fractional
   * `ttlSeconds` cannot emit a fractional value.
   */
  private _itemExtras(options?: DocumentItemOptions): Record<string, unknown> {
    const extra: Record<string, unknown> = {}
    if (options?.vector) {
      if (!options.vector.every((v) => Number.isFinite(v))) {
        throw new StorageError('Vector contains non-finite values (nan/inf); the DynamoDB N type rejects them.')
      }
      extra[this._vectorAttribute] = options.vector
    }
    if (options?.metadata) extra[META_ATTR] = options.metadata
    const ttlSeconds = options?.ttlSeconds ?? this._ttlSeconds
    if (this._ttlEnabled && ttlSeconds !== undefined) {
      extra[this._ttlAttribute] = Math.floor(Date.now() / 1000 + ttlSeconds)
    }
    return extra
  }

  /**
   * Applies the configured compression to a value. Runs before the offload size check so
   * compressible values can stay inline (and out of S3). The compressed form is kept only when it
   * actually shrinks, and callers record the choice per item (`z`) so reads decompress correctly
   * regardless of the current setting.
   */
  private async _encodeValue(data: Uint8Array): Promise<{ payload: Uint8Array; compressed: boolean }> {
    if (!this._compress) return { payload: data, compressed: false }
    const gz = await gzipAsync(data)
    if (gz.byteLength < data.byteLength) return { payload: gz, compressed: true }
    return { payload: data, compressed: false }
  }

  /** Normalizes a caller key and derives its canonical id and base-table keys. */
  private _locate(key: string): DocumentLocation {
    const normalized = normalizeKey(key)
    const docId = `${this._prefix}${normalized}`
    return { key: normalized, docId, ...this._split(docId) }
  }

  /**
   * Builds the inline item `write()` would store for `data`, with the same extras and encoding.
   * Indexed documents must stay inline so the lexical transaction can write them atomically, so a
   * value that would need S3 offload is rejected even when a bucket is configured.
   */
  private async _inlineItem(
    location: DocumentLocation,
    data: Uint8Array,
    options?: DocumentItemOptions
  ): Promise<Record<string, unknown>> {
    const extra = this._itemExtras(options)
    const { payload, compressed } = await this._encodeValue(data)
    if (compressed) extra[Z_ATTR] = true
    if (payload.byteLength > DynamoDBStorage.OFFLOAD_THRESHOLD_BYTES) {
      throw new StorageError(
        `Value for '${location.key}' is ${payload.byteLength} bytes, above the ${DynamoDBStorage.OFFLOAD_THRESHOLD_BYTES}-byte inline limit; S3 offload is not supported for indexed documents`
      )
    }
    return { [PK]: location.pk, [SK]: location.sk, [KEY_ATTR]: location.docId, [DATA_ATTR]: payload, ...extra }
  }

  /**
   * Best-effort delete of the offloaded S3 object for `fullKey`; a no-op without a bucket. Never
   * throws: the DynamoDB write it follows is already durable, so a failed cleanup must not fail it.
   * An S3 lifecycle rule is the backstop for missed reclamations.
   */
  private async _reclaimOffloaded(fullKey: string): Promise<void> {
    if (!this._s3Bucket) return
    await this._s3Delete(fullKey).catch(() => undefined)
  }

  /**
   * Splits a full key into partition and sort keys. The partition is the leading
   * scope and its identifier (first two segments, e.g. `sessions/<id>`); the sort
   * key is the remainder. Single-segment keys use the segment as the partition and
   * a sentinel sort key so point operations round-trip.
   */
  private _split(fullKey: string): { pk: string; sk: string } {
    const segments = fullKey.split('/').filter(Boolean)
    if (segments.length <= 1) return { pk: segments[0] ?? fullKey, sk: '\u0000' }
    return { pk: segments.slice(0, 2).join('/'), sk: segments.slice(2).join('/') || '\u0000' }
  }

  /** True when the item carries a TTL attribute whose epoch-seconds value has passed. */
  private _isExpired(item: Record<string, unknown>): boolean {
    return this._hasPassed(item[this._ttlAttribute])
  }

  /** True when `epochSeconds` is a number at or before now. */
  private _hasPassed(epochSeconds: unknown): boolean {
    return typeof epochSeconds === 'number' && epochSeconds <= Math.floor(Date.now() / 1000)
  }

  /**
   * Rejects a caller-supplied partition key that falls outside this namespace.
   *
   * `write`/`read`/`delete` and the string-prefix `list` all derive the physical
   * partition from `prefix + key`, so the namespace is enforced by construction. A
   * {@link DynamoDBListQuery} and {@link SearchQuery} name the partition directly, so
   * the same boundary is enforced here instead. The partition is rejected rather than
   * silently rewritten: a caller asking for another namespace's partition has a bug or
   * is probing, and neither should be answered with quietly different data.
   *
   * A partition at or below the namespace is in scope. So is an ancestor partition:
   * because the physical partition is only the first two key segments, a deeply
   * namespaced view legitimately lives inside a shallower partition, and any rows
   * outside the namespace are dropped by {@link _stripPrefix}. A partition on a
   * different branch (another tenant) is rejected.
   */
  private _assertPkInScope(pk: string): void {
    if (!this._prefix) return
    const candidate = pk.endsWith('/') ? pk : `${pk}/`
    if (pk.startsWith(this._prefix) || this._prefix.startsWith(candidate)) return
    throw new StorageError(
      `Partition key '${pk}' is outside this storage namespace '${this._prefix}'. ` +
        `Prefix the partition key with '${this._prefix}' (for example '${this._prefix}sessions') ` +
        `or use the storage instance that owns it.`
    )
  }

  /**
   * Strips the namespace prefix from a stored key, or returns `null` when the key is
   * outside this namespace. Guards the de-prefix: slicing blindly would turn another
   * namespace's key into something that looks like one of ours.
   */
  private _stripPrefix(stored: string): string | null {
    if (!this._prefix) return stored
    if (!stored.startsWith(this._prefix)) return null
    return stored.slice(this._prefix.length)
  }

  private async _listByPrefix(prefix: string): Promise<string[]> {
    const normalized = normalizePrefix(prefix)
    const full = `${this._prefix}${normalized}`
    const segments = full.split('/').filter(Boolean)
    if (segments.length < 2) {
      throw new StorageError(
        `DynamoDB list prefix '${prefix}' is too broad; include at least the scope and identifier (e.g. 'sessions/<id>/'). ` +
          `Cross-partition listing requires a DynamoDBListQuery or the optional query() extension.`
      )
    }
    const pk = segments.slice(0, 2).join('/')
    let skPrefix = segments.slice(2).join('/')
    // Restore the trailing slash the segment split drops:
    // 'a/b/turns/' must not match 'a/b/turnstile'.
    if (skPrefix && full.endsWith('/')) skPrefix += '/'
    const keys = await this._listByQuery(skPrefix ? { pk, skPrefix } : { pk })
    // begins_with on the sort key can't exclude the partition's sentinel row;
    // enforce true string-prefix semantics on the full key.
    return keys.filter((key) => key.startsWith(normalized))
  }

  private async _listByQuery(query: DynamoDBListQuery): Promise<string[]> {
    if (query.skPrefix && query.skBetween) {
      throw new StorageError('DynamoDBListQuery accepts skPrefix or skBetween, not both')
    }
    this._assertPkInScope(query.pk)
    const { QueryCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._getClient()
    const keys: string[] = []
    let exclusiveStartKey: Record<string, unknown> | undefined
    const names: Record<string, string> = { '#pk': PK, '#k': KEY_ATTR }
    const values: Record<string, unknown> = { ':pk': query.pk }
    let condition = '#pk = :pk'
    if (query.skPrefix !== undefined) {
      names['#sk'] = SK
      values[':sk'] = query.skPrefix
      condition += ' AND begins_with(#sk, :sk)'
    } else if (query.skBetween !== undefined) {
      names['#sk'] = SK
      values[':from'] = query.skBetween[0]
      values[':to'] = query.skBetween[1]
      condition += ' AND #sk BETWEEN :from AND :to'
    }
    // Hide items past their TTL — only when TTL is opted into. FilterExpression
    // evaluates the full item server-side (independent of ProjectionExpression) and
    // adds no RCU cost — the partition is scanned regardless. The attribute_not_exists
    // clause keeps rows written before TTL was enabled from being dropped.
    let filterExpression: string | undefined
    if (this._ttlEnabled) {
      names['#ttl'] = this._ttlAttribute
      values[':now'] = Math.floor(Date.now() / 1000)
      filterExpression = '(attribute_not_exists(#ttl) OR #ttl > :now)'
    }
    // DynamoDB's Limit caps items EVALUATED before the FilterExpression, and startAfter
    // is applied client-side — so pushing Limit while either is active can under-return.
    // Only push it down when neither is in play; otherwise the client-side count is authoritative.
    const pushLimit = query.limit && !filterExpression && query.startAfter === undefined ? query.limit : undefined
    do {
      const response = await client.send(
        new QueryCommand({
          TableName: this._tableName,
          KeyConditionExpression: condition,
          FilterExpression: filterExpression,
          ExpressionAttributeNames: names,
          ExpressionAttributeValues: values,
          ProjectionExpression: '#k',
          ExclusiveStartKey: exclusiveStartKey,
          Limit: pushLimit,
        })
      )
      for (const item of response.Items ?? []) {
        const k = item[KEY_ATTR] as string | undefined
        if (k === undefined) continue
        const key = this._stripPrefix(k)
        if (key === null) continue
        // Apply startAfter during collection (before the limit check) so a limit is
        // filled with post-cursor keys rather than truncated after the fact.
        if (query.startAfter !== undefined && key <= query.startAfter) continue
        keys.push(key)
        if (query.limit && keys.length >= query.limit) return keys.sort()
      }
      exclusiveStartKey = response.LastEvaluatedKey
    } while (exclusiveStartKey)
    return keys.sort()
  }

  private async _put(
    item: Record<string, unknown>,
    options?: { returnOld?: boolean }
  ): Promise<Record<string, unknown> | undefined> {
    const { PutCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._getClient()
    const response = await client.send(
      new PutCommand({
        TableName: this._tableName,
        Item: item,
        ...(options?.returnOld ? { ReturnValues: 'ALL_OLD' as const } : {}),
      })
    )
    return response.Attributes
  }

  private async _getClient(): Promise<import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient> {
    if (this._client) return this._client
    const { DynamoDBClient } = await import('@aws-sdk/client-dynamodb')
    const { DynamoDBDocumentClient } = await import('@aws-sdk/lib-dynamodb')
    // Attribute this package's traffic in the User-Agent, mirroring the Python
    // package's user_agent_extra. Injected clients are consumer-owned and are
    // deliberately left untouched.
    const base = new DynamoDBClient({
      ...(this._region ? { region: this._region } : {}),
      customUserAgent: USER_AGENT_MARKER,
    })
    this._client = DynamoDBDocumentClient.from(base)
    return this._client
  }

  private _s3Key(fullKey: string): string {
    return `${this._s3Prefix}${fullKey}`
  }

  private async _getS3Client(): Promise<import('@aws-sdk/client-s3').S3Client> {
    if (this._s3Client) return this._s3Client
    const { S3Client } = await import('@aws-sdk/client-s3')
    this._s3Client = new S3Client({
      ...(this._region ? { region: this._region } : {}),
      customUserAgent: USER_AGENT_MARKER,
    })
    return this._s3Client
  }

  private async _s3Put(fullKey: string, data: Uint8Array): Promise<void> {
    const { PutObjectCommand } = await import('@aws-sdk/client-s3')
    const client = await this._getS3Client()
    await client.send(new PutObjectCommand({ Bucket: this._s3Bucket, Key: this._s3Key(fullKey), Body: data }))
  }

  private async _s3Get(fullKey: string): Promise<Uint8Array | null> {
    const { GetObjectCommand } = await import('@aws-sdk/client-s3')
    const client = await this._getS3Client()
    const response = await client.send(new GetObjectCommand({ Bucket: this._s3Bucket, Key: this._s3Key(fullKey) }))
    const body = await response.Body?.transformToByteArray()
    return body ? new Uint8Array(body) : null
  }

  private async _s3Delete(fullKey: string): Promise<void> {
    const { DeleteObjectCommand } = await import('@aws-sdk/client-s3')
    const client = await this._getS3Client()
    await client.send(new DeleteObjectCommand({ Bucket: this._s3Bucket, Key: this._s3Key(fullKey) }))
  }
}
