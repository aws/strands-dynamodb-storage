// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'
import type { TransactWriteCommandInput } from '@aws-sdk/lib-dynamodb'

import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'
import { promisify } from 'node:util'
import { gunzip } from 'node:zlib'

import {
  DATA_ATTR,
  KEY_ATTR,
  META_ATTR,
  PK,
  S3_ATTR,
  SK,
  Z_ATTR,
  type DocumentItemPort,
  type DocumentLocation,
  type DynamoDBStorage,
} from './dynamodb-storage.js'
import {
  LEXICAL_TOKENIZER_VERSION,
  compareUtf8,
  identifierPostingPk,
  manifestPk,
  normalizeIdentifier,
  normalizeIdentifiers,
  postingPartitionKeys,
  scopeSegment,
  termPostingPk,
  textTerms,
  utf8ByteLength,
} from './lexical-terms.js'

const gunzipAsync = promisify(gunzip)

/** A document to index: the stored value plus the text and identifiers it is retrievable by. */
export interface SearchableDocument {
  /** Storage key, relative to the storage prefix. */
  key: string
  /** Stored value. Must fit inline: S3 offload is not supported for indexed documents. */
  data: Uint8Array
  /** Text tokenized into searchable terms. Default `''`: the document is stored but not searchable by text. */
  text?: string
  /** Exact-match identifiers (case-sensitive), e.g. ticket or invoice numbers. */
  identifiers?: readonly string[]
  /** Metadata stored on the item; usable as a retrieval filter. */
  metadata?: Record<string, string | number | boolean>
  /** Optional embedding, stored exactly as `DynamoDBStorage.write` stores it. */
  vector?: number[]
  /** Per-document TTL; honoured only when the storage opted in to TTL. */
  ttlSeconds?: number
}

/** Bounds on index writes and retrieval. Every limit is a positive integer unless noted. */
export interface LexicalIndexLimits {
  /**
   * Distinct terms plus identifiers per document; at most 49, because an overwrite with disjoint
   * postings needs `2 + old + new <= 100` transaction actions.
   */
  maxPostingsPerDocument: number
  /** UTF-8 bytes per term. */
  maxTermBytes: number
  /** UTF-8 bytes per identifier. */
  maxIdentifierBytes: number
  /** UTF-8 bytes of a document's text. */
  maxTextBytes: number
  /** Distinct terms per query; at most `maxPostingsPerDocument`. */
  maxQueryTerms: number
  /** Postings per `Query` page; at most 1000. */
  pageSize: number
  /** `Query` pages read per term or identifier. */
  maxPagesPerTerm: number
  /** Distinct documents validated per retrieval; at most 1000. */
  maxCandidates: number
  /** Concurrent posting queries and validation reads per retrieval; at most 16. */
  maxConcurrency: number
  /** Retries after a transaction cancelled by a concurrent modification or by throttling; may be 0. */
  maxConflictRetries: number
  /** Consecutive `BatchGetItem` rounds without progress tolerated before failing; may be 0. */
  maxUnprocessedRetries: number
}

/** DynamoDB limit on the actions of one `TransactWriteItems`. */
const MAX_TRANSACTION_ACTIONS = 100
/**
 * Ceiling of `maxPostingsPerDocument`: an overwrite with disjoint postings puts the base item and the
 * manifest, puts every new posting and deletes every old one, all in one transaction.
 */
const MAX_POSTINGS_PER_DOCUMENT = Math.floor((MAX_TRANSACTION_ACTIONS - 2) / 2)

/** Default {@link LexicalIndexLimits}. */
export const DEFAULT_LEXICAL_INDEX_LIMITS: Readonly<LexicalIndexLimits> = Object.freeze({
  maxPostingsPerDocument: MAX_POSTINGS_PER_DOCUMENT,
  maxTermBytes: 64,
  maxIdentifierBytes: 128,
  maxTextBytes: 65_536,
  maxQueryTerms: 16,
  pageSize: 100,
  maxPagesPerTerm: 5,
  maxCandidates: 300,
  maxConcurrency: 4,
  maxConflictRetries: 3,
  maxUnprocessedRetries: 5,
})

/** Configuration for {@link LexicalIndex}. */
export interface LexicalIndexConfig {
  /** User-provisioned index table (partition key `pk` string, sort key `sk` string). */
  indexTableName: string
  /** Overrides merged over {@link DEFAULT_LEXICAL_INDEX_LIMITS}. */
  limits?: Partial<LexicalIndexLimits>
}

/** Metadata equality filter: every entry must match a value of the same kind (string, number or boolean). */
type MetadataFilter = Record<string, string | number | boolean>

/** Full-text query over document terms. */
export interface LexicalQuery {
  /** Query text, tokenized with the same rules as document text. */
  text: string
  /** Number of results, 1 to 100. */
  topK: number
  /** When true, only documents matching every query term are returned. Default false. */
  requireAllTerms?: boolean
  /** Optional strict metadata equality filter (`1` matches `1.0`, never `true`). */
  filter?: MetadataFilter
  /**
   * When true, each result carries the stored value, read by a second strongly consistent read of the
   * kept results only. A result whose document changed between the two reads is dropped, so fewer than
   * `topK` results may be returned. Default false.
   */
  includeValues?: boolean
}

/** A {@link LexicalQuery} whose text is already tokenized into terms. */
type TermQuery = Omit<LexicalQuery, 'text'>

/** Exact-match query over document identifiers. */
export interface IdentifierQuery {
  /** Identifier to match exactly (case-sensitive). */
  identifier: string
  /** Number of results, 1 to 100. */
  topK: number
  /** Optional strict metadata equality filter (`1` matches `1.0`, never `true`). */
  filter?: MetadataFilter
  /** When true, each result carries the stored value; see {@link LexicalQuery.includeValues}. Default false. */
  includeValues?: boolean
}

/** One retrieved document. */
export interface LexicalSearchResult {
  /** Storage key, relative to the storage prefix. */
  key: string
  /** `matchedTerms / query terms`, in `(0, 1]`; always 1 for identifier lookups. */
  score: number
  /** Number of query terms the current revision of the document matches. */
  matchedTerms: number
  /** Stored value, present only when `includeValues` was requested. */
  data?: Uint8Array
  /** Current metadata of the document, when it has any. */
  metadata?: Record<string, unknown>
}

/** Retrieval response. `truncated` is true when a limit cut the candidate set short. */
export interface LexicalSearchResponse {
  results: LexicalSearchResult[]
  truncated: boolean
  /** Limits that truncated the candidate set, sorted ascending without duplicates. */
  truncationReasons: Array<'max_candidates' | 'max_pages_per_term'>
  /** Distinct in-scope documents validated against the base table. */
  candidatesExamined: number
}

/** Outcome of one {@link LexicalIndex.repair} batch. */
export interface RepairReport {
  documentsChecked: number
  documentsRemoved: number
  postingsRebuilt: number
  /** Pass to the next `repair` call to resume; `null` when no manifests remain. */
  cursor: string | null
}

/** Options of {@link LexicalIndex.repair}. */
export interface RepairOptions {
  /** Manifests to examine in this batch. Default 100. */
  maxDocuments?: number
  /** Resume point returned by a previous batch; `null`, `undefined` or `''` start from the beginning. */
  cursor?: string | null
  /** Also re-put the postings of valid documents. Default false. */
  rebuildPostings?: boolean
}

/** Optimistic-concurrency options of {@link LexicalIndex.upsert} and {@link LexicalIndex.delete}. */
export interface RevisionOptions {
  /** Proceed only if the document is currently at this revision. */
  expectedRevision?: string
}

/** A write rejected because the document is not at the `expectedRevision` the caller supplied. */
export class RevisionConflictError extends StorageError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'RevisionConflictError'
  }
}

type Item = Record<string, unknown>
type TransactAction = NonNullable<TransactWriteCommandInput['TransactItems']>[number]
type TruncationReason = LexicalSearchResponse['truncationReasons'][number]
type RepairOutcome = 'kept' | 'removed' | 'rebuilt' | 'skipped'
type CancellationKind = 'conflict' | 'throttled'

interface WriteCondition {
  ConditionExpression: string
  ExpressionAttributeNames: Record<string, string>
  ExpressionAttributeValues?: Record<string, unknown>
}

interface Manifest {
  docId: string
  revision: string
  terms: string[]
  identifiers: string[]
}

interface BaseState {
  scopeTag: string | undefined
  offloaded: boolean
}

interface DocumentState {
  manifest: Manifest | undefined
  base: BaseState | undefined
}

interface UpsertPlan {
  location: DocumentLocation
  baseItem: Item
  terms: string[]
  identifiers: string[]
  postingKeys: string[]
}

interface Posting {
  docId: string
  rev: string
}

interface PostingList {
  postings: Posting[]
  truncated: boolean
}

interface Candidate {
  location: DocumentLocation
  partitionsByRevision: Map<string, Set<number>>
}

interface CandidateMerge {
  candidates: Candidate[]
  truncationReasons: TruncationReason[]
}

interface RetrievalOptions {
  topK: number
  filter: MetadataFilter | undefined
  includeValues: boolean
  requireAllTerms: boolean
}

interface Match {
  location: DocumentLocation
  revision: string
  score: number
  matchedTerms: number
  metadata: unknown
}

const REV_ATTR = 'rev'
const TOKENIZER_VERSION_ATTR = 'tv'
const TERMS_ATTR = 'terms'
const IDS_ATTR = 'ids'
const LEXICAL_REV_ATTR = 'lxrev'
const LEXICAL_SCOPE_ATTR = 'lxscope'

/** A base item that passed candidate validation, so it carries the revision it was indexed at. */
type IndexedItem = Item & { [LEXICAL_REV_ATTR]: string }

/** Attributes the index reads or writes on base items, manifests and postings; a TTL attribute must not alias them. */
const INDEX_RESERVED_ATTRIBUTES: ReadonlySet<string> = new Set([
  PK,
  SK,
  KEY_ATTR,
  DATA_ATTR,
  S3_ATTR,
  META_ATTR,
  Z_ATTR,
  REV_ATTR,
  TOKENIZER_VERSION_ATTR,
  TERMS_ATTR,
  IDS_ATTR,
  LEXICAL_REV_ATTR,
  LEXICAL_SCOPE_ATTR,
])
const MANIFEST_ATTRIBUTES = [PK, REV_ATTR, TERMS_ATTR, IDS_ATTR]
const VALUE_ATTRIBUTES = [PK, SK, LEXICAL_REV_ATTR, DATA_ATTR, Z_ATTR, S3_ATTR]

const LIMIT_NAMES = Object.keys(DEFAULT_LEXICAL_INDEX_LIMITS) as Array<keyof LexicalIndexLimits>
const RETRY_LIMIT_NAMES: ReadonlySet<keyof LexicalIndexLimits> = new Set([
  'maxConflictRetries',
  'maxUnprocessedRetries',
])
const LIMIT_CEILINGS: Partial<Record<keyof LexicalIndexLimits, number>> = {
  maxPostingsPerDocument: MAX_POSTINGS_PER_DOCUMENT,
  pageSize: 1000,
  maxCandidates: 1000,
  maxConcurrency: 16,
}

/** DynamoDB maximum size of a sort key, which bounds document ids and scopes. */
const MAX_SORT_KEY_BYTES = 1024
/** DynamoDB maximum size of a partition key, which bounds posting and manifest partition keys. */
const MAX_PARTITION_KEY_BYTES = 2048
/** Conservative bound below the 4 MB `TransactWriteItems` aggregate size limit. */
const MAX_TRANSACTION_BYTES = 4_000_000
const MAX_BATCH_GET_KEYS = 100
const MAX_TOP_K = 100
const DEFAULT_REPAIR_BATCH = 100
/** Upper bound of the stored size of a DynamoDB number (38 significant digits). */
const MAX_NUMBER_BYTES = 21
const CONTAINER_OVERHEAD_BYTES = 3
const BACKOFF_BASE_SECONDS = 0.05
const CONFLICT_BACKOFF_CAP_SECONDS = 0.5
const UNPROCESSED_BACKOFF_CAP_SECONDS = 1
const CONFLICT_CANCELLATION_CODES: ReadonlySet<string> = new Set(['ConditionalCheckFailed', 'TransactionConflict'])
const THROTTLING_CANCELLATION_CODES: ReadonlySet<string> = new Set([
  'ThrottlingError',
  'ProvisionedThroughputExceeded',
  'RequestLimitExceeded',
])
const EXHAUSTED_RETRY_REASONS: Readonly<Record<CancellationKind, string>> = {
  conflict: 'concurrent modification',
  throttled: 'transaction throttled',
}
const MAX_CANDIDATES_REASON: TruncationReason = 'max_candidates'
const MAX_PAGES_PER_TERM_REASON: TruncationReason = 'max_pages_per_term'

/**
 * Opt-in lexical document index over a {@link DynamoDBStorage} (preview).
 *
 * Owns the write path of indexed documents: each {@link LexicalIndex.upsert} stores the document in the
 * storage's base table and replaces its term and identifier postings (in a separate, user-provisioned
 * index table) in one `TransactWriteItems`, so the document and its postings never disagree.
 * {@link LexicalIndex.search} ranks documents by the fraction of query terms they match and
 * {@link LexicalIndex.lookup} matches identifiers exactly; both validate every candidate against the
 * base table with a strongly consistent read, so stale, bypassed or expired documents are never
 * returned. This is not BM25, not hybrid retrieval, and not phrase or substring search.
 *
 * The index is scoped to the storage prefix: build it over `storage.namespace('tenant-a')` to isolate
 * a tenant. Writes that bypass the index (a plain `storage.write()` of an indexed key) make the document
 * invisible to lexical retrieval until it is upserted again; {@link LexicalIndex.repair} removes the
 * stale manifests and postings they leave behind.
 *
 * @example
 * ```typescript
 * const index = new LexicalIndex(storage.namespace('tenant-a'), { indexTableName: 'agent-lexical' })
 * await index.upsert({ key: 'tickets/42', data, text: 'Login fails with ERR_AUTH_403', identifiers: ['T-42'] })
 * const { results } = await index.search({ text: 'login ERR_AUTH_403', topK: 5 })
 * ```
 */
export class LexicalIndex {
  private readonly _port: DocumentItemPort
  private readonly _indexTableName: string
  private readonly _limits: LexicalIndexLimits
  private readonly _scopeTag: string

  /**
   * @param storage - Storage holding the documents; its prefix is the lexical scope
   * @param config - Index table name and optional limit overrides
   * @throws {@link StorageError} if the table name is empty, a limit is out of range, the scope exceeds
   *   1024 UTF-8 bytes, or the storage's TTL attribute is one of the attributes the index writes
   */
  constructor(storage: DynamoDBStorage, config: LexicalIndexConfig) {
    if (!config.indexTableName) throw new StorageError('LexicalIndex requires a non-empty indexTableName')
    this._limits = resolveLimits(config.limits)
    this._port = storage._documentItemPort()
    this._indexTableName = config.indexTableName
    assertScopeFits(this._port.scope)
    assertPartitionKeysFit([manifestPk(this._port.scope)])
    assertTtlAttributeUnreserved(this._port.ttlAttribute)
    this._scopeTag = scopeSegment(this._port.scope)
  }

  /** The storage prefix this index is scoped to (`''` for an unprefixed storage). */
  get scope(): string {
    return this._port.scope
  }

  /**
   * Stores `document` and atomically replaces its postings. Validation happens before any I/O and
   * never truncates: a document with too many distinct terms is rejected.
   *
   * @returns The new revision of the document
   * @throws {@link RevisionConflictError} if `expectedRevision` does not match the current revision
   * @throws {@link StorageError} if validation fails, the key belongs to another lexical scope, the
   *   manifest is malformed, retries are exhausted, or the write fails
   */
  async upsert(document: SearchableDocument, options?: RevisionOptions): Promise<string> {
    return this._guard('upsert', document.key, async () => {
      const plan = await this._planUpsert(document)
      return this._retryOnCancellation('upsert', plan.location.key, () =>
        this._attemptUpsert(plan, options?.expectedRevision)
      )
    })
  }

  /**
   * Deletes the document and its postings in one transaction.
   *
   * @returns `true` if anything was deleted, `false` if neither the document nor its manifest existed
   * @throws {@link RevisionConflictError} if `expectedRevision` does not match the current revision
   * @throws {@link StorageError} if the key is invalid or belongs to another lexical scope, the manifest
   *   is malformed, retries are exhausted, or the delete fails
   */
  async delete(key: string, options?: RevisionOptions): Promise<boolean> {
    return this._guard('delete', key, async () => {
      const location = this._locate(key)
      return this._retryOnCancellation('delete', location.key, () =>
        this._attemptDelete(location, options?.expectedRevision)
      )
    })
  }

  /**
   * Current revision of an indexed document (strongly consistent), or `null` when it has no manifest.
   *
   * @throws {@link StorageError} if the key is invalid, the manifest is malformed, or the read fails
   */
  async revision(key: string): Promise<string | null> {
    return this._guard('revision', key, async () => {
      const manifest = await this._readManifest(this._locate(key).docId)
      return manifest?.revision ?? null
    })
  }

  /**
   * Ranks documents by the fraction of query terms their current revision contains, then by key
   * (UTF-8 byte order).
   *
   * @throws {@link StorageError} if the query is invalid, base-table keys stay unprocessed after
   *   `maxUnprocessedRetries` rounds without progress, or a read fails
   */
  async search(query: LexicalQuery): Promise<LexicalSearchResponse> {
    return this._guard('search', undefined, async () => {
      assertTopK(query.topK)
      return this._retrieveTerms(textTerms(query.text), query)
    })
  }

  /**
   * {@link LexicalIndex.search} over already tokenized, distinct query `terms`, for `LexicalSearchStrategy`, which
   * selects the terms itself. Internal: not part of the public API and may change without notice.
   *
   * @internal
   */
  async _searchTerms(terms: readonly string[], query: TermQuery): Promise<LexicalSearchResponse> {
    return this._guard('search', undefined, async () => {
      assertTopK(query.topK)
      return this._retrieveTerms(terms, query)
    })
  }

  /**
   * Finds documents carrying `identifier` exactly (case-sensitive), ordered by key (UTF-8 byte order).
   *
   * @throws {@link StorageError} if the query is invalid, base-table keys stay unprocessed after
   *   `maxUnprocessedRetries` rounds without progress, or a read fails
   */
  async lookup(query: IdentifierQuery): Promise<LexicalSearchResponse> {
    return this._guard('lookup', undefined, async () => {
      assertTopK(query.topK)
      const identifier = normalizeIdentifier(query.identifier)
      assertEachFits([identifier], this._limits.maxIdentifierBytes, 'The identifier', 'maxIdentifierBytes')
      return this._retrieve([identifierPostingPk(this.scope, identifier)], retrievalOptions(query, false))
    })
  }

  /**
   * Offline-safe, resumable maintenance over this scope's manifests (never a table scan). Removes the
   * manifest and postings of every document that is missing, expired, or was overwritten outside the
   * index, and optionally re-puts the postings of valid documents. Documents modified concurrently are
   * skipped. Source text is not stored, so a tokenizer change requires re-upserting documents.
   *
   * @throws {@link StorageError} if `maxDocuments` is not a positive integer, a manifest is malformed,
   *   a transaction is throttled, or a read or write fails
   */
  async repair(options?: RepairOptions): Promise<RepairReport> {
    return this._guard('repair', undefined, async () => {
      const maxDocuments = options?.maxDocuments ?? DEFAULT_REPAIR_BATCH
      assertPositiveInteger('maxDocuments', maxDocuments)
      const page = await this._readManifests(maxDocuments, options?.cursor ?? undefined)
      const outcomes = await this._repairDocuments(page.manifests, options?.rebuildPostings ?? false)
      return {
        documentsChecked: page.manifests.length,
        documentsRemoved: outcomes.filter((outcome) => outcome === 'removed').length,
        postingsRebuilt: outcomes.filter((outcome) => outcome === 'rebuilt').length,
        cursor: page.hasMore ? (page.manifests.at(-1)?.docId ?? null) : null,
      }
    })
  }

  private async _planUpsert(document: SearchableDocument): Promise<UpsertPlan> {
    const location = this._locate(document.key)
    const terms = this._documentTerms(document.text ?? '')
    const identifiers = this._documentIdentifiers(document.identifiers ?? [])
    this._assertPostingBudget(location, terms.length + identifiers.length)
    const postingKeys = postingPartitionKeys(this.scope, terms, identifiers)
    assertPartitionKeysFit(postingKeys)
    const baseItem = await this._port.inlineItem(location, document.data, {
      vector: document.vector,
      metadata: document.metadata,
      ttlSeconds: document.ttlSeconds,
    })
    return { location, baseItem, terms, identifiers, postingKeys }
  }

  private _documentTerms(text: string): string[] {
    const textBytes = utf8ByteLength(text)
    if (textBytes > this._limits.maxTextBytes) {
      throw new StorageError(`Document text is ${textBytes} bytes, above maxTextBytes (${this._limits.maxTextBytes})`)
    }
    const terms = textTerms(text)
    assertEachFits(terms, this._limits.maxTermBytes, 'A document term', 'maxTermBytes')
    return terms
  }

  private _documentIdentifiers(identifiers: readonly string[]): string[] {
    const normalized = normalizeIdentifiers(identifiers)
    assertEachFits(normalized, this._limits.maxIdentifierBytes, 'A document identifier', 'maxIdentifierBytes')
    return normalized
  }

  private _assertPostingBudget(location: DocumentLocation, postings: number): void {
    const limit = this._limits.maxPostingsPerDocument
    if (postings <= limit) return
    throw new StorageError(
      `Document '${location.key}' has ${postings} distinct terms and identifiers, above maxPostingsPerDocument (${limit}); split the document or reduce distinct terms`
    )
  }

  private async _attemptUpsert(plan: UpsertPlan, expectedRevision: string | undefined): Promise<string> {
    const state = await this._readDocumentState(plan.location)
    assertExpectedRevision(plan.location, state.manifest, expectedRevision)
    this._assertOwnedByScope(plan.location, state.base)
    const rev = newRevision()
    const docId = plan.location.docId
    const ttl = this._postingTtl(plan.baseItem)
    await this._transact(
      [
        this._putBaseAction(plan.baseItem, rev),
        this._putManifestAction(plan, rev, state.manifest),
        ...plan.postingKeys.map((postingKey) => this._putPostingAction(postingKey, docId, rev, ttl)),
        ...this._stalePostingKeys(state.manifest, plan.postingKeys).map((postingKey) =>
          this._deletePostingAction(postingKey, docId)
        ),
      ],
      rev
    )
    if (state.base?.offloaded) await this._port.deleteOffloaded(docId)
    return rev
  }

  /**
   * Deletes whatever the reads found. When no manifest was found, the transaction also checks that none
   * appeared since: otherwise a concurrent first upsert could commit between the reads and this delete,
   * which would remove its base item and orphan its manifest and postings.
   */
  private async _attemptDelete(location: DocumentLocation, expectedRevision: string | undefined): Promise<boolean> {
    const state = await this._readDocumentState(location)
    assertExpectedRevision(location, state.manifest, expectedRevision)
    this._assertOwnedByScope(location, state.base)
    if (!state.manifest && !state.base) return false
    await this._transact([
      ...(state.base ? [this._deleteBaseAction(location)] : []),
      ...(state.manifest ? this._removeManifestActions(state.manifest) : [this._manifestAbsentCheck(location.docId)]),
    ])
    if (state.base?.offloaded) await this._port.deleteOffloaded(location.docId)
    return true
  }

  private async _readDocumentState(location: DocumentLocation): Promise<DocumentState> {
    const [manifest, baseItem] = await Promise.all([
      this._readManifest(location.docId),
      this._getItem(this._port.tableName, baseKey(location), [PK, LEXICAL_SCOPE_ATTR, S3_ATTR]),
    ])
    return {
      manifest,
      base: baseItem && {
        scopeTag: stringAttribute(baseItem, LEXICAL_SCOPE_ATTR),
        offloaded: Boolean(baseItem[S3_ATTR]),
      },
    }
  }

  /** Strongly consistent manifest read; `pk` is projected so a manifest missing its revision still reads as present. */
  private async _readManifest(docId: string): Promise<Manifest | undefined> {
    const item = await this._getItem(this._indexTableName, this._manifestKey(docId), MANIFEST_ATTRIBUTES)
    return item && this._parseManifest(docId, item)
  }

  /**
   * A manifest without a string revision cannot be matched by any condition, so it is reported instead
   * of being treated as absent, which would let an upsert or delete orphan its postings.
   */
  private _parseManifest(docId: string, item: Item): Manifest {
    const revision = stringAttribute(item, REV_ATTR)
    if (revision === undefined) {
      throw new StorageError(
        `Lexical index manifest for '${docId}' in index table '${this._indexTableName}' has no string revision`
      )
    }
    return { docId, revision, terms: stringList(item[TERMS_ATTR]), identifiers: stringList(item[IDS_ATTR]) }
  }

  private _assertOwnedByScope(location: DocumentLocation, base: BaseState | undefined): void {
    if (base?.scopeTag === undefined || base.scopeTag === this._scopeTag) return
    throw new StorageError(`Document '${location.key}' is owned by another lexical scope`)
  }

  /**
   * Retries `attempt` while its transaction is cancelled by a concurrent writer or by throttling, backing
   * off exponentially between attempts (never after the last one). Once retries are exhausted, the
   * error chains the last cancellation as its cause.
   */
  private async _retryOnCancellation<T>(operation: string, key: string, attempt: () => Promise<T>): Promise<T> {
    for (let attemptNumber = 0; ; attemptNumber++) {
      try {
        return await attempt()
      } catch (error: unknown) {
        const kind = cancellationKind(error)
        if (kind === undefined) throw error
        if (attemptNumber >= this._limits.maxConflictRetries) {
          throw new StorageError(
            `${this._failureMessage(operation, key)}: ${EXHAUSTED_RETRY_REASONS[kind]}; retries exhausted`,
            { cause: error }
          )
        }
      }
      await delay(backoffMs(attemptNumber, CONFLICT_BACKOFF_CAP_SECONDS))
    }
  }

  private async _transact(actions: TransactAction[], clientRequestToken?: string): Promise<void> {
    assertTransactionFits(actions)
    const { TransactWriteCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._port.client()
    await client.send(
      new TransactWriteCommand({
        TransactItems: actions,
        ...(clientRequestToken === undefined ? {} : { ClientRequestToken: clientRequestToken }),
      })
    )
  }

  /**
   * Runs a repair transaction, reporting `false` instead of throwing when a concurrent writer cancelled
   * it. Throttling is not skipped: a skipped document would look repaired.
   */
  private async _tryTransact(actions: TransactAction[]): Promise<boolean> {
    try {
      await this._transact(actions)
      return true
    } catch (error: unknown) {
      if (cancellationKind(error) === 'conflict') return false
      throw error
    }
  }

  private _putBaseAction(baseItem: Item, rev: string): TransactAction {
    return {
      Put: {
        TableName: this._port.tableName,
        Item: { ...baseItem, [LEXICAL_REV_ATTR]: rev, [LEXICAL_SCOPE_ATTR]: this._scopeTag },
        ...this._ownedByScopeCondition(),
      },
    }
  }

  private _deleteBaseAction(location: DocumentLocation): TransactAction {
    return {
      Delete: { TableName: this._port.tableName, Key: baseKey(location), ...this._ownedByScopeCondition() },
    }
  }

  private _putManifestAction(plan: UpsertPlan, rev: string, previous: Manifest | undefined): TransactAction {
    return {
      Put: {
        TableName: this._indexTableName,
        Item: {
          ...this._manifestKey(plan.location.docId),
          [REV_ATTR]: rev,
          [TOKENIZER_VERSION_ATTR]: LEXICAL_TOKENIZER_VERSION,
          [TERMS_ATTR]: plan.terms,
          [IDS_ATTR]: plan.identifiers,
        },
        ...(previous ? revisionCondition(previous.revision) : absentCondition()),
      },
    }
  }

  private _manifestAbsentCheck(docId: string): TransactAction {
    return {
      ConditionCheck: { TableName: this._indexTableName, Key: this._manifestKey(docId), ...absentCondition() },
    }
  }

  private _putPostingAction(postingKey: string, docId: string, rev: string, ttl: Item): TransactAction {
    return {
      Put: { TableName: this._indexTableName, Item: { [PK]: postingKey, [SK]: docId, [REV_ATTR]: rev, ...ttl } },
    }
  }

  private _deletePostingAction(postingKey: string, docId: string): TransactAction {
    return { Delete: { TableName: this._indexTableName, Key: { [PK]: postingKey, [SK]: docId } } }
  }

  private _removeManifestActions(manifest: Manifest): TransactAction[] {
    return [
      {
        Delete: {
          TableName: this._indexTableName,
          Key: this._manifestKey(manifest.docId),
          ...revisionCondition(manifest.revision),
        },
      },
      ...this._manifestPostingKeys(manifest).map((postingKey) => this._deletePostingAction(postingKey, manifest.docId)),
    ]
  }

  private _rebuildPostingActions(manifest: Manifest, baseItem: Item): TransactAction[] {
    const ttl = this._postingTtl(baseItem)
    return [
      {
        ConditionCheck: {
          TableName: this._indexTableName,
          Key: this._manifestKey(manifest.docId),
          ...revisionCondition(manifest.revision),
        },
      },
      ...this._manifestPostingKeys(manifest).map((postingKey) =>
        this._putPostingAction(postingKey, manifest.docId, manifest.revision, ttl)
      ),
    ]
  }

  private _ownedByScopeCondition(): WriteCondition {
    return {
      ConditionExpression: 'attribute_not_exists(#lxscope) OR #lxscope = :scope',
      ExpressionAttributeNames: { '#lxscope': LEXICAL_SCOPE_ATTR },
      ExpressionAttributeValues: { ':scope': this._scopeTag },
    }
  }

  private _manifestKey(docId: string): Item {
    return { [PK]: manifestPk(this.scope), [SK]: docId }
  }

  private _manifestPostingKeys(manifest: Manifest): string[] {
    return postingPartitionKeys(this.scope, manifest.terms, manifest.identifiers)
  }

  private _stalePostingKeys(manifest: Manifest | undefined, retainedKeys: readonly string[]): string[] {
    if (!manifest) return []
    const retained = new Set(retainedKeys)
    return this._manifestPostingKeys(manifest).filter((postingKey) => !retained.has(postingKey))
  }

  /** Postings expire with their document, so they copy its TTL stamp when it has one. */
  private _postingTtl(baseItem: Item): Item {
    const attribute = this._port.ttlAttribute
    if (attribute === undefined || typeof baseItem[attribute] !== 'number') return {}
    return { [attribute]: baseItem[attribute] }
  }

  private _ttlAttributes(): string[] {
    const attribute = this._port.ttlAttribute
    return attribute === undefined ? [] : [attribute]
  }

  private async _retrieveTerms(terms: readonly string[], query: TermQuery): Promise<LexicalSearchResponse> {
    this._assertQueryTerms(terms)
    return this._retrieve(
      terms.map((term) => termPostingPk(this.scope, term)),
      retrievalOptions(query, query.requireAllTerms ?? false)
    )
  }

  private _assertQueryTerms(terms: readonly string[]): void {
    assertEachFits(terms, this._limits.maxTermBytes, 'A query term', 'maxTermBytes')
    if (terms.length === 0) throw new StorageError('Lexical query contains no searchable terms')
    if (terms.length > this._limits.maxQueryTerms) {
      throw new StorageError(
        `Lexical query has ${terms.length} distinct terms, above maxQueryTerms (${this._limits.maxQueryTerms})`
      )
    }
  }

  private async _retrieve(partitionKeys: string[], options: RetrievalOptions): Promise<LexicalSearchResponse> {
    assertPartitionKeysFit(partitionKeys)
    const postingLists = await mapWithConcurrency(partitionKeys, this._limits.maxConcurrency, (partitionKey) =>
      this._readPostings(partitionKey)
    )
    const { candidates, truncationReasons } = mergeCandidates(postingLists, this._limits.maxCandidates, (docId) =>
      this._candidateLocation(docId)
    )
    const matches = await this._rankCandidates(candidates, partitionKeys.length, options)
    return {
      results: options.includeValues ? await this._withValues(matches) : matches.map(toResult),
      truncated: truncationReasons.length > 0,
      truncationReasons,
      candidatesExamined: candidates.length,
    }
  }

  private async _readPostings(partitionKey: string): Promise<PostingList> {
    const { QueryCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._port.client()
    const postings: Posting[] = []
    let startKey: Item | undefined
    for (let page = 0; page < this._limits.maxPagesPerTerm; page++) {
      const response = await client.send(
        new QueryCommand({
          TableName: this._indexTableName,
          ...partitionQuery(partitionKey, [SK, REV_ATTR]),
          Limit: this._limits.pageSize,
          ConsistentRead: false,
          ExclusiveStartKey: startKey,
        })
      )
      postings.push(...parsePostings(response.Items))
      startKey = response.LastEvaluatedKey
      if (!startKey) return { postings, truncated: false }
    }
    return { postings, truncated: true }
  }

  /**
   * Base-table location of an index doc id, or `undefined` when the id lies outside this scope or does
   * not map back to the same id. Resolved before any base-table read, so a foreign or malformed posting
   * can never cause a read of another scope's item.
   */
  private _candidateLocation(docId: string): DocumentLocation | undefined {
    const relativeKey = this._port.relativeKey(docId)
    if (relativeKey === null) return undefined
    try {
      const location = this._port.locate(relativeKey)
      return location.docId === docId ? location : undefined
    } catch {
      return undefined
    }
  }

  /**
   * Validates candidates against one strongly consistent read of their base items, which never projects
   * stored values, then scores, sorts and keeps the best `topK`.
   */
  private async _rankCandidates(
    candidates: readonly Candidate[],
    termCount: number,
    options: RetrievalOptions
  ): Promise<Match[]> {
    const items = await this._readBaseItems(
      candidates.map(({ location }) => location),
      this._validationAttributes()
    )
    return candidates
      .map((candidate) => this._scoreCandidate(candidate, items.get(baseKeyId(candidate.location)), termCount, options))
      .filter((match): match is Match => match !== undefined)
      .sort((left, right) => right.score - left.score || compareUtf8(left.location.key, right.location.key))
      .slice(0, options.topK)
  }

  private _validationAttributes(): string[] {
    return [PK, SK, KEY_ATTR, LEXICAL_REV_ATTR, LEXICAL_SCOPE_ATTR, META_ATTR, S3_ATTR, ...this._ttlAttributes()]
  }

  private _scoreCandidate(
    candidate: Candidate,
    item: Item | undefined,
    termCount: number,
    options: RetrievalOptions
  ): Match | undefined {
    if (!item || !this._isValidCandidate(candidate, item)) return undefined
    if (!matchesMetadataFilter(item[META_ATTR], options.filter)) return undefined
    const revision = item[LEXICAL_REV_ATTR]
    const matchedTerms = candidate.partitionsByRevision.get(revision)?.size ?? 0
    if (options.requireAllTerms && matchedTerms < termCount) return undefined
    return {
      location: candidate.location,
      revision,
      score: matchedTerms / termCount,
      matchedTerms,
      metadata: item[META_ATTR],
    }
  }

  /**
   * The base item is this candidate's document, owned by this scope, at a revision one of its postings
   * named, stored inline (indexed documents never are offloaded) and not expired.
   */
  private _isValidCandidate(candidate: Candidate, item: Item): item is IndexedItem {
    const revision = stringAttribute(item, LEXICAL_REV_ATTR)
    return (
      revision !== undefined &&
      candidate.partitionsByRevision.has(revision) &&
      item[KEY_ATTR] === candidate.location.docId &&
      item[LEXICAL_SCOPE_ATTR] === this._scopeTag &&
      !item[S3_ATTR] &&
      !this._port.isExpired(item)
    )
  }

  /**
   * Reads the stored values of the kept results with a second strongly consistent read, so values of
   * candidates that were filtered out or ranked below `topK` are never read. A result whose document
   * changed revision, disappeared, was offloaded or lost its value in between is dropped.
   */
  private async _withValues(matches: readonly Match[]): Promise<LexicalSearchResult[]> {
    const items = await this._readBaseItems(
      matches.map(({ location }) => location),
      VALUE_ATTRIBUTES
    )
    const results = await Promise.all(
      matches.map((match) => resultWithValue(match, items.get(baseKeyId(match.location))))
    )
    return results.filter((result): result is LexicalSearchResult => result !== undefined)
  }

  private async _readBaseItems(
    locations: readonly DocumentLocation[],
    attributes: readonly string[]
  ): Promise<Map<string, Item>> {
    const keys = [...new Map(locations.map((location) => [baseKeyId(location), baseKey(location)])).values()]
    const pages = await mapWithConcurrency(chunk(keys, MAX_BATCH_GET_KEYS), this._limits.maxConcurrency, (batch) =>
      this._batchGet(batch, attributes)
    )
    return new Map(pages.flat().map((item) => [baseKeyId(item), item]))
  }

  /**
   * Strongly consistent `BatchGetItem` of up to 100 base items. Unprocessed keys are re-requested with
   * exponential backoff that restarts after every round returning an item; `maxUnprocessedRetries` bounds
   * only consecutive rounds without progress, so a throttled but progressing read is never abandoned.
   * If keys remain once that bound is hit, the read fails rather than returning a silently partial answer.
   */
  private async _batchGet(keys: Item[], attributes: readonly string[]): Promise<Item[]> {
    const items: Item[] = []
    let pending = keys
    let stalledRounds = 0
    for (;;) {
      const round = await this._batchGetRound(pending, attributes)
      items.push(...round.items)
      pending = round.unprocessed
      if (pending.length === 0) return items
      stalledRounds = round.items.length > 0 ? 0 : stalledRounds + 1
      if (stalledRounds > this._limits.maxUnprocessedRetries) {
        throw new StorageError(
          `Lexical index read left ${pending.length} keys unprocessed in base table '${this._port.tableName}' after ${this._limits.maxUnprocessedRetries} retries without progress`
        )
      }
      await delay(backoffMs(Math.max(stalledRounds - 1, 0), UNPROCESSED_BACKOFF_CAP_SECONDS))
    }
  }

  private async _batchGetRound(
    keys: Item[],
    attributes: readonly string[]
  ): Promise<{ items: Item[]; unprocessed: Item[] }> {
    const { BatchGetCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._port.client()
    const tableName = this._port.tableName
    const response = await client.send(
      new BatchGetCommand({
        RequestItems: { [tableName]: { Keys: keys, ConsistentRead: true, ...projection(attributes) } },
      })
    )
    return {
      items: response.Responses?.[tableName] ?? [],
      unprocessed: response.UnprocessedKeys?.[tableName]?.Keys ?? [],
    }
  }

  private async _readManifests(
    maxDocuments: number,
    cursor: string | undefined
  ): Promise<{ manifests: Manifest[]; hasMore: boolean }> {
    const { QueryCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._port.client()
    const manifests: Manifest[] = []
    let startKey: Item | undefined = cursor ? this._manifestKey(cursor) : undefined
    do {
      const response = await client.send(
        new QueryCommand({
          TableName: this._indexTableName,
          ...partitionQuery(manifestPk(this.scope), [SK, REV_ATTR, TERMS_ATTR, IDS_ATTR]),
          Limit: maxDocuments - manifests.length,
          ConsistentRead: true,
          ExclusiveStartKey: startKey,
        })
      )
      manifests.push(...(response.Items ?? []).map((item) => this._parseManifest(String(item[SK]), item)))
      startKey = response.LastEvaluatedKey
    } while (startKey && manifests.length < maxDocuments)
    return { manifests, hasMore: startKey !== undefined }
  }

  /** Repairs manifests one at a time, so the first failure other than a concurrent modification stops the batch. */
  private async _repairDocuments(manifests: readonly Manifest[], rebuildPostings: boolean): Promise<RepairOutcome[]> {
    const outcomes: RepairOutcome[] = []
    for (const manifest of manifests) outcomes.push(await this._repairDocument(manifest, rebuildPostings))
    return outcomes
  }

  private async _repairDocument(manifest: Manifest, rebuildPostings: boolean): Promise<RepairOutcome> {
    const baseItem = await this._readRepairBase(manifest.docId)
    if (!baseItem || !this._isManifestCurrent(manifest, baseItem)) {
      return (await this._tryTransact(this._removeManifestActions(manifest))) ? 'removed' : 'skipped'
    }
    if (!rebuildPostings) return 'kept'
    return (await this._tryTransact(this._rebuildPostingActions(manifest, baseItem))) ? 'rebuilt' : 'skipped'
  }

  private async _readRepairBase(docId: string): Promise<Item | undefined> {
    const location = this._candidateLocation(docId)
    if (!location) return undefined
    return this._getItem(this._port.tableName, baseKey(location), [
      PK,
      LEXICAL_REV_ATTR,
      LEXICAL_SCOPE_ATTR,
      ...this._ttlAttributes(),
    ])
  }

  private _isManifestCurrent(manifest: Manifest, baseItem: Item): boolean {
    return (
      baseItem[LEXICAL_REV_ATTR] === manifest.revision &&
      baseItem[LEXICAL_SCOPE_ATTR] === this._scopeTag &&
      !this._port.isExpired(baseItem)
    )
  }

  private async _getItem(tableName: string, key: Item, attributes: readonly string[]): Promise<Item | undefined> {
    const { GetCommand } = await import('@aws-sdk/lib-dynamodb')
    const client = await this._port.client()
    const response = await client.send(
      new GetCommand({ TableName: tableName, Key: key, ConsistentRead: true, ...projection(attributes) })
    )
    return response.Item
  }

  private _locate(key: string): DocumentLocation {
    const location = this._port.locate(key)
    const bytes = utf8ByteLength(location.docId)
    if (bytes > MAX_SORT_KEY_BYTES) {
      throw new StorageError(
        `Document id is ${bytes} bytes, above the DynamoDB sort-key limit of ${MAX_SORT_KEY_BYTES} bytes`
      )
    }
    return location
  }

  /** Runs an operation, wrapping any unexpected error in a {@link StorageError} naming it and the tables. */
  private async _guard<T>(operation: string, key: string | undefined, run: () => Promise<T>): Promise<T> {
    try {
      return await run()
    } catch (error: unknown) {
      if (error instanceof StorageError) throw error
      throw new StorageError(this._failureMessage(operation, key), { cause: error })
    }
  }

  /** Failure message naming the operation, the key when there is one, and both tables; never document content. */
  private _failureMessage(operation: string, key: string | undefined): string {
    const subject = key === undefined ? '' : ` for '${key}'`
    const tables = `index table '${this._indexTableName}', base table '${this._port.tableName}'`
    return `Lexical index ${operation} failed${subject} (${tables})`
  }
}

/**
 * Merges `overrides` over {@link DEFAULT_LEXICAL_INDEX_LIMITS} and validates the result. Internal: shared with
 * `LexicalSearchStrategy` so its limits fail at construction rather than on the first write hook.
 *
 * @internal
 * @throws {@link StorageError} if a limit is out of range
 */
export function resolveLimits(overrides: Partial<LexicalIndexLimits> | undefined): LexicalIndexLimits {
  const limits: LexicalIndexLimits = { ...DEFAULT_LEXICAL_INDEX_LIMITS }
  for (const name of LIMIT_NAMES) {
    const value = overrides?.[name]
    if (value !== undefined) limits[name] = value
  }
  assertLimitsValid(limits)
  return limits
}

function assertLimitsValid(limits: LexicalIndexLimits): void {
  for (const name of LIMIT_NAMES) {
    const value = limits[name]
    const minimum = RETRY_LIMIT_NAMES.has(name) ? 0 : 1
    if (!Number.isInteger(value) || value < minimum) {
      throw new StorageError(`LexicalIndex limit '${name}' must be an integer >= ${minimum}; got ${value}`)
    }
    const ceiling = LIMIT_CEILINGS[name]
    if (ceiling !== undefined && value > ceiling) {
      throw new StorageError(`LexicalIndex limit '${name}' must be <= ${ceiling}; got ${value}`)
    }
  }
  if (limits.maxQueryTerms > limits.maxPostingsPerDocument) {
    throw new StorageError(
      `LexicalIndex limit 'maxQueryTerms' (${limits.maxQueryTerms}) must be <= maxPostingsPerDocument (${limits.maxPostingsPerDocument})`
    )
  }
}

function assertScopeFits(scope: string): void {
  const bytes = utf8ByteLength(scope)
  if (bytes <= MAX_SORT_KEY_BYTES) return
  throw new StorageError(`Lexical scope is ${bytes} bytes, above the ${MAX_SORT_KEY_BYTES}-byte limit`)
}

/**
 * A TTL attribute that aliases an index attribute would be overwritten by the TTL stamp copied onto
 * postings (or would expire items on unrelated values), so such a storage is rejected up front.
 */
function assertTtlAttributeUnreserved(ttlAttribute: string | undefined): void {
  if (ttlAttribute === undefined || !INDEX_RESERVED_ATTRIBUTES.has(ttlAttribute)) return
  throw new StorageError(
    `Storage TTL attribute '${ttlAttribute}' is reserved by the lexical index; configure another ttlAttribute`
  )
}

/** Rejects a partition key DynamoDB would refuse, naming only the limit so no term or identifier leaks. */
function assertPartitionKeysFit(partitionKeys: readonly string[]): void {
  if (partitionKeys.every((partitionKey) => utf8ByteLength(partitionKey) <= MAX_PARTITION_KEY_BYTES)) return
  throw new StorageError(
    `A lexical index partition key is above the DynamoDB partition-key limit of ${MAX_PARTITION_KEY_BYTES} bytes`
  )
}

function assertPositiveInteger(name: string, value: number): void {
  if (Number.isInteger(value) && value >= 1) return
  throw new StorageError(`${name} must be a positive integer; got ${value}`)
}

/**
 * Rejects a `topK` outside 1 to 100. Internal: shared with `LexicalSearchStrategy`.
 *
 * @internal
 */
export function assertTopK(topK: number): void {
  if (Number.isInteger(topK) && topK >= 1 && topK <= MAX_TOP_K) return
  throw new StorageError(`topK must be an integer between 1 and ${MAX_TOP_K}; got ${topK}`)
}

/** Rejects the first value above `maxBytes`, naming only its size so no text leaks into the message. */
function assertEachFits(values: readonly string[], maxBytes: number, subject: string, limitName: string): void {
  for (const value of values) {
    const bytes = utf8ByteLength(value)
    if (bytes > maxBytes) throw new StorageError(`${subject} is ${bytes} bytes, above ${limitName} (${maxBytes})`)
  }
}

function assertExpectedRevision(
  location: DocumentLocation,
  manifest: Manifest | undefined,
  expectedRevision: string | undefined
): void {
  if (expectedRevision === undefined || manifest?.revision === expectedRevision) return
  throw new RevisionConflictError(`Document '${location.key}' is not at the expected revision`)
}

/**
 * Rejects a transaction DynamoDB would refuse: more than 100 actions, or an aggregate size over the
 * 4 MB limit, estimated conservatively from the Put items. Unreachable within the validated limits.
 */
function assertTransactionFits(actions: readonly TransactAction[]): void {
  if (actions.length > MAX_TRANSACTION_ACTIONS) {
    throw new StorageError(
      `Lexical index transaction needs ${actions.length} actions, above the limit of ${MAX_TRANSACTION_ACTIONS}`
    )
  }
  const bytes = actions.reduce((total, action) => total + (action.Put ? estimateAttributeBytes(action.Put.Item) : 0), 0)
  if (bytes > MAX_TRANSACTION_BYTES) {
    throw new StorageError(
      `Lexical index transaction is about ${bytes} bytes, above the limit of ${MAX_TRANSACTION_BYTES}`
    )
  }
}

function estimateAttributeBytes(value: unknown): number {
  if (typeof value === 'string') return utf8ByteLength(value)
  if (value instanceof Uint8Array) return value.byteLength
  if (typeof value === 'number') return MAX_NUMBER_BYTES
  if (Array.isArray(value)) {
    return value.reduce<number>(
      (total, element) => total + 1 + estimateAttributeBytes(element),
      CONTAINER_OVERHEAD_BYTES
    )
  }
  if (isRecord(value)) {
    return Object.entries(value).reduce(
      (total, [name, element]) => total + 1 + utf8ByteLength(name) + estimateAttributeBytes(element),
      CONTAINER_OVERHEAD_BYTES
    )
  }
  return 1
}

/**
 * Classifies a `TransactionCanceledException` by its cancellation reasons: a failed condition or a
 * concurrent transaction is a conflict (it wins over any other reason), a throttled action is
 * throttling; anything else is not retryable.
 */
function cancellationKind(error: unknown): CancellationKind | undefined {
  const codes = cancellationCodes(error)
  if (codes.some((code) => CONFLICT_CANCELLATION_CODES.has(code))) return 'conflict'
  if (codes.some((code) => THROTTLING_CANCELLATION_CODES.has(code))) return 'throttled'
  return undefined
}

function cancellationCodes(error: unknown): string[] {
  if (!(error instanceof Error) || error.name !== 'TransactionCanceledException') return []
  const reasons = 'CancellationReasons' in error ? error.CancellationReasons : undefined
  return Array.isArray(reasons) ? reasons.filter(isRecord).map((reason) => String(reason.Code)) : []
}

function newRevision(): string {
  return randomUUID().replaceAll('-', '')
}

function backoffMs(attempt: number, capSeconds: number): number {
  return Math.min(BACKOFF_BASE_SECONDS * 2 ** attempt, capSeconds) * 1000
}

function revisionCondition(rev: string): WriteCondition {
  return {
    ConditionExpression: '#rev = :rev',
    ExpressionAttributeNames: { '#rev': REV_ATTR },
    ExpressionAttributeValues: { ':rev': rev },
  }
}

function absentCondition(): WriteCondition {
  return { ConditionExpression: 'attribute_not_exists(#pk)', ExpressionAttributeNames: { '#pk': PK } }
}

function projection(attributes: readonly string[]): {
  ProjectionExpression: string
  ExpressionAttributeNames: Record<string, string>
} {
  return {
    ProjectionExpression: attributes.map((_, index) => attributeAlias(index)).join(', '),
    ExpressionAttributeNames: Object.fromEntries(
      attributes.map((attribute, index) => [attributeAlias(index), attribute])
    ),
  }
}

/** Positional placeholders keep any attribute name (reserved words, user TTL names) expression-safe. */
function attributeAlias(index: number): string {
  return `#a${index}`
}

function partitionQuery(partitionKey: string, attributes: readonly string[]) {
  const { ProjectionExpression, ExpressionAttributeNames } = projection(attributes)
  return {
    KeyConditionExpression: '#pk = :pk',
    ProjectionExpression,
    ExpressionAttributeNames: { ...ExpressionAttributeNames, '#pk': PK },
    ExpressionAttributeValues: { ':pk': partitionKey },
  }
}

function baseKey(location: DocumentLocation): Item {
  return { [PK]: location.pk, [SK]: location.sk }
}

function baseKeyId(keyed: { [PK]?: unknown; [SK]?: unknown }): string {
  return JSON.stringify([keyed[PK], keyed[SK]])
}

function chunk<T>(values: readonly T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let start = 0; start < values.length; start += size) chunks.push(values.slice(start, start + size))
  return chunks
}

/** Runs `task` over `values` with at most `concurrency` in flight, preserving input order in the result. */
async function mapWithConcurrency<T, R>(
  values: readonly T[],
  concurrency: number,
  task: (value: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = []
  const queue = values.entries()
  const worker = async (): Promise<void> => {
    for (const [index, value] of queue) results[index] = await task(value)
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, worker))
  return results
}

/**
 * Deterministic candidate merge, independent of fetch timing: posting lists are consumed in query-term
 * order and each list in sort-key order. A doc id that does not resolve to this scope is dropped
 * without taking a candidate slot; once `maxCandidates` is reached, new doc ids are skipped (recorded
 * as `max_candidates`) while existing candidates keep accumulating matches.
 */
function mergeCandidates(
  postingLists: readonly PostingList[],
  maxCandidates: number,
  resolveLocation: (docId: string) => DocumentLocation | undefined
): CandidateMerge {
  const candidates = new Map<string, Candidate>()
  const unresolved = new Set<string>()
  const reasons = new Set<TruncationReason>()
  const admit = (docId: string): Candidate | undefined => {
    if (unresolved.has(docId)) return undefined
    const location = resolveLocation(docId)
    if (!location) {
      unresolved.add(docId)
      return undefined
    }
    if (candidates.size >= maxCandidates) {
      reasons.add(MAX_CANDIDATES_REASON)
      return undefined
    }
    const candidate: Candidate = { location, partitionsByRevision: new Map() }
    candidates.set(docId, candidate)
    return candidate
  }
  postingLists.forEach((list, partitionIndex) => {
    if (list.truncated) reasons.add(MAX_PAGES_PER_TERM_REASON)
    for (const posting of list.postings) {
      const candidate = candidates.get(posting.docId) ?? admit(posting.docId)
      if (candidate) addPartitionMatch(candidate, posting.rev, partitionIndex)
    }
  })
  return { candidates: [...candidates.values()], truncationReasons: [...reasons].sort() }
}

function addPartitionMatch(candidate: Candidate, rev: string, partitionIndex: number): void {
  const partitions = candidate.partitionsByRevision.get(rev) ?? new Set<number>()
  partitions.add(partitionIndex)
  candidate.partitionsByRevision.set(rev, partitions)
}

/**
 * Strict metadata equality: every filter entry must exist in the item's current metadata with a value
 * of the same kind (a boolean never equals a number) that is equal to it. An empty filter matches all.
 */
function matchesMetadataFilter(metadata: unknown, filter: MetadataFilter | undefined): boolean {
  if (filter === undefined) return true
  return Object.entries(filter).every(
    ([name, expected]) =>
      isRecord(metadata) &&
      Object.hasOwn(metadata, name) &&
      typeof metadata[name] === typeof expected &&
      metadata[name] === expected
  )
}

function retrievalOptions(
  query: Pick<LexicalQuery, 'topK' | 'filter' | 'includeValues'>,
  requireAllTerms: boolean
): RetrievalOptions {
  return { topK: query.topK, filter: query.filter, includeValues: query.includeValues ?? false, requireAllTerms }
}

function toResult(match: Match): LexicalSearchResult {
  return {
    key: match.location.key,
    score: match.score,
    matchedTerms: match.matchedTerms,
    ...(isRecord(match.metadata) ? { metadata: match.metadata } : {}),
  }
}

async function resultWithValue(match: Match, item: Item | undefined): Promise<LexicalSearchResult | undefined> {
  if (item?.[LEXICAL_REV_ATTR] !== match.revision) return undefined
  const value = inlineValue(item)
  if (value === undefined) return undefined
  return { ...toResult(match), data: await decodeValue(value, item[Z_ATTR] === true) }
}

/** Stored bytes of an inline item; `undefined` for an offloaded item, which can never be an indexed document. */
function inlineValue(item: Item): Uint8Array | undefined {
  const data = item[DATA_ATTR]
  return !item[S3_ATTR] && data instanceof Uint8Array ? data : undefined
}

async function decodeValue(stored: Uint8Array, compressed: boolean): Promise<Uint8Array> {
  const raw = new Uint8Array(stored)
  return compressed ? new Uint8Array(await gunzipAsync(raw)) : raw
}

function parsePostings(items: Item[] | undefined): Posting[] {
  return (items ?? []).flatMap((item) => {
    const docId = stringAttribute(item, SK)
    const rev = stringAttribute(item, REV_ATTR)
    return docId === undefined || rev === undefined ? [] : [{ docId, rev }]
  })
}

function stringAttribute(item: Item | undefined, name: string): string | undefined {
  const value = item?.[name]
  return typeof value === 'string' ? value : undefined
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((element): element is string => typeof element === 'string') : []
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && !(value instanceof Uint8Array)
}
