// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  BatchGetCommand,
  GetCommand,
  PutCommand,
  QueryCommand,
  TransactWriteCommand,
  type BatchGetCommandInput,
  type DynamoDBDocumentClient,
  type GetCommandInput,
  type PutCommandInput,
  type QueryCommandInput,
  type TransactWriteCommandInput,
} from '@aws-sdk/lib-dynamodb'
import { StorageError } from '@strands-agents/sdk'
import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { KeywordSearchStrategy, type SearchStrategy } from '@strands-agents/sdk/storage/search'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'

import { Buffer } from 'node:buffer'

import {
  DynamoDBStorage,
  type DynamoDBStorageConfig,
  type StringSearchStrategy,
  type VectorSearchAdapter,
} from './dynamodb-storage.js'
import { DEFAULT_LEXICAL_INDEX_LIMITS, LexicalIndex } from './lexical-index.js'
import {
  LexicalSearchStrategy,
  type LexicalSearchStrategyConfig,
  type SearchableTextExtractor,
} from './lexical-search-strategy.js'
import { manifestPk, termPostingPk } from './lexical-terms.js'
import * as pkgIndex from './index.js'

type Item = Record<string, unknown>
type TransactItem = NonNullable<TransactWriteCommandInput['TransactItems']>[number]

const BASE_TABLE = 'base-table'
const INDEX_TABLE = 'lexical-index'
const FROZEN_NOW_MS = 1_750_000_000_000
const FROZEN_NOW_SECONDS = FROZEN_NOW_MS / 1000
const TEXT_QUERY_WITHOUT_STRATEGY_MESSAGE =
  'DynamoDBStorage.search requires a SearchQuery with a pre-computed embedding vector. ' +
  'Plain-string queries are not supported without a searchStrategy: embed the text first and pass { vector, topK }, ' +
  'or configure a searchStrategy (for example LexicalSearchStrategy).'

/** Expression fields shared by conditional transaction actions. */
interface Conditional {
  ConditionExpression?: string
  ExpressionAttributeNames?: Record<string, string>
  ExpressionAttributeValues?: Record<string, unknown>
}

/** One resolved `TransactWriteItems` action: its target item, condition and effect. */
interface TransactOperation {
  table: string
  pk: unknown
  sk: unknown
  condition: Conditional
  apply: () => void
}

function serviceError(name: string, message: string, fields: Item = {}): Error {
  return Object.assign(new Error(message), { name, ...fields })
}

function itemId(pk: unknown, sk: unknown): string {
  return JSON.stringify([pk, sk])
}

function compareBytes(left: unknown, right: unknown): number {
  return Buffer.compare(Buffer.from(String(left)), Buffer.from(String(right)))
}

function project(item: Item, expression: string | undefined, names: Record<string, string> | undefined): Item {
  const copy = structuredClone(item)
  if (!expression) return copy
  const wanted = expression.split(',').map((token) => names?.[token.trim()] ?? token.trim())
  return Object.fromEntries(Object.entries(copy).filter(([name]) => wanted.includes(name)))
}

function attributeName(condition: Conditional, alias: string): string {
  const name = condition.ExpressionAttributeNames?.[alias]
  if (name === undefined) throw new Error(`undefined attribute alias ${alias}`)
  return name
}

/** Evaluates exactly the grammar the index emits: `attribute_not_exists(#x)`, `#x = :v`, joined by `OR`. */
function conditionHolds(condition: Conditional, current: Item | undefined): boolean {
  const expression = condition.ConditionExpression
  if (expression === undefined) return true
  return expression.split(' OR ').some((clause) => clauseHolds(clause.trim(), condition, current))
}

function clauseHolds(clause: string, condition: Conditional, current: Item | undefined): boolean {
  const absent = /^attribute_not_exists\((#\w+)\)$/.exec(clause)
  if (absent?.[1]) return current === undefined || !(attributeName(condition, absent[1]) in current)
  const equality = /^(#\w+) = (:\w+)$/.exec(clause)
  if (equality?.[1] && equality[2]) {
    const expected = condition.ExpressionAttributeValues?.[equality[2]]
    return current !== undefined && current[attributeName(condition, equality[1])] === expected
  }
  throw new Error(`unsupported condition clause: ${clause}`)
}

/**
 * Table-aware in-memory stand-in for a DynamoDBDocumentClient, reduced from the one in `lexical-index.test.ts` to
 * the commands a storage write/read and the lexical index send: projections, pagination, and all-or-nothing
 * conditional TransactWriteItems with an injectable failure.
 */
class FakeDocumentClient {
  readonly sent: unknown[] = []
  transactionFailure: Error | undefined
  private readonly _tables = new Map<string, Map<string, Item>>()

  clearLog(): void {
    this.sent.length = 0
  }

  sentOf<T>(type: abstract new (...args: never[]) => T): T[] {
    return this.sent.filter((command): command is T => command instanceof type)
  }

  commandNames(): string[] {
    return this.sent.map((command) => (command as object).constructor.name)
  }

  rows(tableName: string): Item[] {
    return [...this._table(tableName).values()]
  }

  row(tableName: string, pk: unknown, sk: unknown): Item | undefined {
    return this._table(tableName).get(itemId(pk, sk))
  }

  async send(command: unknown): Promise<unknown> {
    this.sent.push(command)
    if (command instanceof GetCommand) return this._get(command.input)
    if (command instanceof PutCommand) return this._put(command.input)
    if (command instanceof QueryCommand) return this._query(command.input)
    if (command instanceof BatchGetCommand) return this._batchGet(command.input)
    if (command instanceof TransactWriteCommand) return this._transactWrite(command.input)
    throw new Error(`unexpected command ${(command as object | undefined)?.constructor?.name}`)
  }

  private _table(tableName: string | undefined): Map<string, Item> {
    if (!tableName) throw new Error('missing TableName')
    const table = this._tables.get(tableName) ?? new Map<string, Item>()
    this._tables.set(tableName, table)
    return table
  }

  private _putRow(tableName: string, item: Item): void {
    this._table(tableName).set(itemId(item.pk, item.sk), structuredClone(item))
  }

  private _get(input: GetCommandInput): unknown {
    const item = this.row(input.TableName ?? '', input.Key?.pk, input.Key?.sk)
    const projected = item && project(item, input.ProjectionExpression, input.ExpressionAttributeNames)
    return projected && Object.keys(projected).length > 0 ? { Item: projected } : {}
  }

  private _put(input: PutCommandInput): unknown {
    const item = input.Item as Item
    const old = this.row(input.TableName ?? '', item.pk, item.sk)
    this._putRow(input.TableName ?? '', item)
    return input.ReturnValues === 'ALL_OLD' && old ? { Attributes: old } : {}
  }

  private _query(input: QueryCommandInput): unknown {
    if (input.KeyConditionExpression !== '#pk = :pk') throw new Error('unsupported key condition')
    const partition = input.ExpressionAttributeValues?.[':pk']
    const startSk = input.ExclusiveStartKey?.sk
    const rows = this.rows(input.TableName ?? '')
      .filter((item) => item.pk === partition && (startSk === undefined || compareBytes(item.sk, startSk) > 0))
      .sort((left, right) => compareBytes(left.sk, right.sk))
    const page = rows.slice(0, input.Limit ?? rows.length)
    const last = page.at(-1)
    return {
      Items: page.map((item) => project(item, input.ProjectionExpression, input.ExpressionAttributeNames)),
      LastEvaluatedKey: rows.length > page.length && last ? { pk: last.pk, sk: last.sk } : undefined,
    }
  }

  private _batchGet(input: BatchGetCommandInput): unknown {
    const responses: Record<string, Item[]> = {}
    for (const [tableName, request] of Object.entries(input.RequestItems ?? {})) {
      responses[tableName] = (request.Keys ?? []).flatMap((key) => {
        const item = this.row(tableName, key.pk, key.sk)
        return item ? [project(item, request.ProjectionExpression, request.ExpressionAttributeNames)] : []
      })
    }
    return { Responses: responses, UnprocessedKeys: {} }
  }

  private async _transactWrite(input: TransactWriteCommandInput): Promise<unknown> {
    if (this.transactionFailure) throw this.transactionFailure
    const operations = (input.TransactItems ?? []).map((action) => this._operation(action))
    const codes = operations.map(({ table, pk, sk, condition }) =>
      conditionHolds(condition, this.row(table, pk, sk)) ? 'None' : 'ConditionalCheckFailed'
    )
    if (codes.includes('ConditionalCheckFailed')) {
      throw serviceError('TransactionCanceledException', 'Transaction cancelled', {
        CancellationReasons: codes.map((Code) => ({ Code })),
      })
    }
    for (const operation of operations) operation.apply()
    return {}
  }

  private _operation(action: TransactItem): TransactOperation {
    if (action.Put) {
      const tableName = action.Put.TableName ?? ''
      const item = action.Put.Item as Item
      return {
        table: tableName,
        pk: item.pk,
        sk: item.sk,
        condition: action.Put,
        apply: () => this._putRow(tableName, item),
      }
    }
    if (action.Delete) {
      const tableName = action.Delete.TableName ?? ''
      const key = action.Delete.Key as Item
      return {
        table: tableName,
        pk: key.pk,
        sk: key.sk,
        condition: action.Delete,
        apply: () => this._table(tableName).delete(itemId(key.pk, key.sk)),
      }
    }
    if (action.ConditionCheck) {
      const key = action.ConditionCheck.Key as Item
      const tableName = action.ConditionCheck.TableName ?? ''
      return { table: tableName, pk: key.pk, sk: key.sk, condition: action.ConditionCheck, apply: () => undefined }
    }
    throw new Error('unsupported transaction action')
  }
}

interface SetupOptions {
  strategy?: Partial<LexicalSearchStrategyConfig>
  storage?: Partial<DynamoDBStorageConfig>
}

/** The fake satisfies the client contract structurally; the double cast is the seam. */
const asDocClient = (fake: FakeDocumentClient) => fake as unknown as DynamoDBDocumentClient

const bytes = (value: string) => new TextEncoder().encode(value)
const decode = (value: Uint8Array) => new TextDecoder().decode(value)
const extractText: SearchableTextExtractor = (_key, data) => ({ text: decode(data) })

function setup(options: SetupOptions = {}) {
  const client = new FakeDocumentClient()
  const strategy = new LexicalSearchStrategy({ indexTableName: INDEX_TABLE, extract: extractText, ...options.strategy })
  const storage = new DynamoDBStorage(BASE_TABLE, {
    client: asDocClient(client),
    searchStrategy: strategy,
    ...options.storage,
  })
  return { client, strategy, storage }
}

function storageWith(searchStrategy: StringSearchStrategy | undefined, client = new FakeDocumentClient()) {
  return new DynamoDBStorage(BASE_TABLE, { client: asDocClient(client), ...(searchStrategy && { searchStrategy }) })
}

function distinctWords(count: number, stem = 'w'): string[] {
  return Array.from({ length: count }, (_, index) => `${stem}${index}`)
}

function baseRow(client: FakeDocumentClient, docId: string): Item | undefined {
  return client.rows(BASE_TABLE).find((row) => row.k === docId)
}

async function rejection(promise: Promise<unknown>): Promise<Error> {
  const error: unknown = await promise.then(
    () => undefined,
    (reason: unknown) => reason
  )
  if (!(error instanceof Error)) throw new Error('expected the promise to reject with an Error')
  return error
}

beforeEach(() => {
  vi.spyOn(Date, 'now').mockReturnValue(FROZEN_NOW_MS)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('LexicalSearchStrategy — write hook and text search through DynamoDBStorage', () => {
  it('indexes every write and answers plain-string searches with SDK-shaped results', async () => {
    const { storage } = setup()
    await storage.write('notes/a', bytes('Login fails with ERR_AUTH_403'))
    await storage.write('notes/b', bytes('Deploy succeeded'))
    expect(await storage.search('login err_auth_403')).toEqual([
      { key: 'notes/a', score: 1, data: bytes('Login fails with ERR_AUTH_403') },
    ])
    expect(await storage.search('deploy failed')).toEqual([
      { key: 'notes/b', score: 0.5, data: bytes('Deploy succeeded') },
    ])
  })

  it('costs one storage write plus one indexing transaction per write', async () => {
    const { client, storage } = setup()
    await storage.write('notes/a', bytes('alpha'))
    expect(client.commandNames()).toEqual(['PutCommand', 'GetCommand', 'GetCommand', 'TransactWriteCommand'])
  })

  it('passes the normalized key, the bytes and the write options to the index hook', async () => {
    const index = vi.fn<NonNullable<StringSearchStrategy['index']>>(async () => undefined)
    const storage = storageWith({ search: async () => [], index })
    const data = bytes('value')
    await storage.write('//notes//a/', data, { vector: [0.5], metadata: { kind: 'note' }, ttlSeconds: 60 })
    expect(index).toHaveBeenCalledExactlyOnceWith(storage, 'notes/a', data, {
      vector: [0.5],
      metadata: { kind: 'note' },
      ttlSeconds: 60,
    })
  })

  it('does not run the index hook when the write itself fails', async () => {
    const index = vi.fn<NonNullable<StringSearchStrategy['index']>>(async () => undefined)
    const client = new FakeDocumentClient()
    vi.spyOn(client, 'send').mockRejectedValue(new Error('boom'))
    const storage = storageWith({ search: async () => [], index }, client)
    const error = await rejection(storage.write('notes/a', bytes('value')))
    expect(error.message).toBe(`Failed to write 'notes/a' to DynamoDB table '${BASE_TABLE}'`)
    expect(index).not.toHaveBeenCalled()
  })

  it('preserves the vector, metadata and TTL of the write on the re-indexed item and its postings', async () => {
    const { client, storage } = setup({ storage: { ttlSeconds: 3600 } })
    await storage.write('notes/a', bytes('alpha'), { vector: [0.25, 0.5], metadata: { kind: 'note' }, ttlSeconds: 60 })
    const row = baseRow(client, 'notes/a')
    expect(row).toMatchObject({
      k: 'notes/a',
      data: bytes('alpha'),
      vector: [0.25, 0.5],
      meta: { kind: 'note' },
      expireAt: FROZEN_NOW_SECONDS + 60,
      lxscope: '0:',
    })
    expect(client.row(INDEX_TABLE, termPostingPk('', 'alpha'), 'notes/a')).toEqual({
      pk: termPostingPk('', 'alpha'),
      sk: 'notes/a',
      rev: row?.lxrev,
      expireAt: FROZEN_NOW_SECONDS + 60,
    })
  })

  it('propagates the strategy to namespaced views, each searched in its own scope', async () => {
    const { client, storage } = setup()
    const tenantA = storage.namespace('tenant-a')
    const tenantB = storage.namespace('tenant-b')
    await tenantA.write('notes/n1', bytes('shared alpha'))
    await tenantB.write('notes/n1', bytes('shared beta'))
    expect(await tenantA.search('shared')).toEqual([{ key: 'notes/n1', score: 1, data: bytes('shared alpha') }])
    expect(await tenantB.search('shared')).toEqual([{ key: 'notes/n1', score: 1, data: bytes('shared beta') }])
    expect(await tenantA.search('beta')).toEqual([])
    expect(await storage.search('shared')).toEqual([])
    expect(client.row(INDEX_TABLE, manifestPk('tenant-a/'), 'tenant-a/notes/n1')).toBeDefined()
    expect(client.row(INDEX_TABLE, manifestPk('tenant-b/'), 'tenant-b/notes/n1')).toBeDefined()
  })

  it('stores without indexing when the extractor returns null', async () => {
    const { client, storage } = setup({ strategy: { extract: () => null } })
    await storage.write('notes/a', bytes('alpha'))
    expect(decode((await storage.read('notes/a')) ?? new Uint8Array())).toBe('alpha')
    expect(client.sentOf(TransactWriteCommand)).toEqual([])
    expect(client.rows(INDEX_TABLE)).toEqual([])
    expect(await storage.search('alpha')).toEqual([])
  })

  it('excludes a previously indexed version the extractor now skips, and repair removes its index entries', async () => {
    let indexing = true
    const { client, storage } = setup({
      strategy: { extract: (key, data) => (indexing ? extractText(key, data) : null) },
    })
    await storage.write('notes/a', bytes('alpha'))
    expect(await storage.search('alpha')).toHaveLength(1)
    indexing = false
    await storage.write('notes/a', bytes('alpha again'))
    expect(await storage.search('alpha')).toEqual([])
    const report = await new LexicalIndex(storage, { indexTableName: INDEX_TABLE }).repair()
    expect(report).toMatchObject({ documentsChecked: 1, documentsRemoved: 1 })
    expect(client.rows(INDEX_TABLE)).toEqual([])
  })
})

describe('LexicalSearchStrategy — indexing failures', () => {
  it('reports over-limit extracted text as an indexing failure, leaving the value stored but unsearchable', async () => {
    const { client, storage } = setup()
    const body = distinctWords(DEFAULT_LEXICAL_INDEX_LIMITS.maxPostingsPerDocument + 1).join(' ')
    const error = await rejection(storage.write('notes/big', bytes(body)))
    expect(error).toBeInstanceOf(StorageError)
    expect(error.message).toBe("Wrote 'notes/big' but indexing failed")
    expect(error.cause).toBeInstanceOf(StorageError)
    expect((error.cause as Error).message).toMatch(/above maxPostingsPerDocument \(49\)/)
    expect(decode((await storage.read('notes/big')) ?? new Uint8Array())).toBe(body)
    expect(client.rows(INDEX_TABLE)).toEqual([])
    expect(await storage.search('w0')).toEqual([])
  })

  it('chains a failed indexing transaction as the cause, and a retried write makes the value searchable', async () => {
    const { client, storage } = setup()
    const serviceFailure = serviceError('InternalServerError', 'internal failure')
    client.transactionFailure = serviceFailure
    const error = await rejection(storage.write('notes/a', bytes('alpha')))
    expect(error.message).toBe("Wrote 'notes/a' but indexing failed")
    expect(error.cause).toBeInstanceOf(StorageError)
    expect((error.cause as Error).cause).toBe(serviceFailure)
    expect(decode((await storage.read('notes/a')) ?? new Uint8Array())).toBe('alpha')
    expect(await storage.search('alpha')).toEqual([])
    client.transactionFailure = undefined
    await storage.write('notes/a', bytes('alpha'))
    expect(await storage.search('alpha')).toEqual([{ key: 'notes/a', score: 1, data: bytes('alpha') }])
  })

  it('reports an extractor error as an indexing failure', async () => {
    const extractorFailure = new Error('cannot extract')
    const { storage } = setup({
      strategy: {
        extract: () => {
          throw extractorFailure
        },
      },
    })
    const error = await rejection(storage.write('notes/a', bytes('alpha')))
    expect(error.message).toBe("Wrote 'notes/a' but indexing failed")
    expect(error.cause).toBe(extractorFailure)
  })
})

describe('LexicalSearchStrategy — queries', () => {
  it('uses only the first maxQueryTerms distinct terms of a long query, without error', async () => {
    const { client, storage } = setup()
    await storage.write('notes/a', bytes('needle'))
    const filler = distinctWords(20, 'filler')
    const maxQueryTerms = DEFAULT_LEXICAL_INDEX_LIMITS.maxQueryTerms
    const needleLast = [...filler.slice(0, maxQueryTerms - 1), 'needle', ...filler.slice(maxQueryTerms - 1)]
    client.clearLog()
    expect(await storage.search(needleLast.join(' '))).toEqual([
      { key: 'notes/a', score: 1 / maxQueryTerms, data: bytes('needle') },
    ])
    expect(client.sentOf(QueryCommand)).toHaveLength(maxQueryTerms)
    const needleDropped = [...filler.slice(0, maxQueryTerms), 'needle']
    expect(await storage.search(needleDropped.join(' '))).toEqual([])
  })

  it('skips query terms above maxTermBytes, which no indexed document can contain', async () => {
    const { client, storage } = setup()
    await storage.write('notes/a', bytes('needle'))
    const oversized = 'x'.repeat(DEFAULT_LEXICAL_INDEX_LIMITS.maxTermBytes + 1)
    client.clearLog()
    expect(await storage.search(`${oversized} needle`)).toEqual([{ key: 'notes/a', score: 1, data: bytes('needle') }])
    expect(client.sentOf(QueryCommand)).toHaveLength(1)
    expect(await storage.search(oversized)).toEqual([])
  })

  it('returns no results for a query without terms, without any read', async () => {
    const { client, storage } = setup()
    await storage.write('notes/a', bytes('alpha'))
    client.clearLog()
    expect(await storage.search('')).toEqual([])
    expect(await storage.search(' ?! ... ')).toEqual([])
    expect(client.sent).toEqual([])
  })

  it('validates topK before answering a query without terms', async () => {
    const { storage, strategy } = setup()
    await expect(strategy.search(storage, '', { topK: 0 })).rejects.toThrow(/topK must be an integer between 1 and 100/)
  })

  it('rejects a query that is not well-formed Unicode', async () => {
    const { storage } = setup()
    await expect(storage.search(`alpha ${String.fromCharCode(0xd800)}`)).rejects.toThrow(StorageError)
  })

  it('honours the configured topK and includeValues, and a per-call topK', async () => {
    const { storage, strategy } = setup({ strategy: { topK: 1, includeValues: false } })
    await storage.write('notes/a', bytes('alpha beta'))
    await storage.write('notes/b', bytes('alpha'))
    expect(await storage.search('alpha beta')).toEqual([{ key: 'notes/a', score: 1 }])
    expect(await strategy.search(storage, 'alpha beta', { topK: 2 })).toEqual([
      { key: 'notes/a', score: 1 },
      { key: 'notes/b', score: 0.5 },
    ])
    await expect(strategy.search(storage, 'alpha', { topK: 0 })).rejects.toThrow(/topK/)
  })

  it('keeps the native vector path for a SearchQuery when a strategy is configured', async () => {
    const vectorSearch = vi.fn<VectorSearchAdapter>(async () => [
      { key: 'notes/a', score: 0.1, metadata: { kind: 'note' } },
    ])
    const { storage, strategy } = setup({ storage: { vectorSearch } })
    const strategySearch = vi.spyOn(strategy, 'search')
    expect(await storage.search({ vector: [1, 0], topK: 3 })).toEqual([
      { key: 'notes/a', score: 0.1, metadata: { kind: 'note' } },
    ])
    expect(vectorSearch).toHaveBeenCalledOnce()
    expect(strategySearch).not.toHaveBeenCalled()
  })
})

describe('DynamoDBStorage — plain-string search delegation', () => {
  it('rejects a plain-string query without a strategy, naming both ways out', async () => {
    const error = await rejection(storageWith(undefined).search('dark mode'))
    expect(error).toBeInstanceOf(StorageError)
    expect(error.message).toBe(TEXT_QUERY_WITHOUT_STRATEGY_MESSAGE)
  })

  it('maps strategy matches to { key, score, data }, dropping any other field', async () => {
    const search = vi.fn(async () => [{ key: 'notes/a', score: 2, metadata: { kind: 'note' } }])
    const storage = storageWith({ search })
    expect(await storage.search('query text')).toEqual([{ key: 'notes/a', score: 2 }])
    expect(search).toHaveBeenCalledExactlyOnceWith(storage, 'query text')
  })

  it('wraps a non-StorageError strategy failure as a search failure', async () => {
    const strategyFailure = new Error('boom')
    const storage = storageWith({
      search: async () => {
        throw strategyFailure
      },
    })
    const error = await rejection(storage.search('query text'))
    expect(error).toBeInstanceOf(StorageError)
    expect(error.message).toBe(`Failed to search DynamoDB table '${BASE_TABLE}'`)
    expect(error.cause).toBe(strategyFailure)
  })

  it('writes without a hook when the strategy has no index method', async () => {
    const client = new FakeDocumentClient()
    const storage = storageWith(KeywordSearchStrategy, client)
    await storage.write('notes/a', bytes('alpha'))
    expect(client.commandNames()).toEqual(['PutCommand'])
  })
})

describe('LexicalSearchStrategy — SDK integration', () => {
  it('lets the SDK FileMemoryStore add and search entries over DynamoDBStorage', async () => {
    const { storage } = setup()
    const memory = new FileMemoryStore({ name: 'agent-memory', storage })
    const key = await memory.add('# User preferences\nPrefers dark mode')
    await memory.add('# User preferences\nLikes window seats')
    expect(key).toBe('user-preferences.md')
    expect(await memory.search('window seats')).toEqual([
      {
        content: '# User preferences\nPrefers dark mode\nLikes window seats',
        metadata: { path: 'user-preferences.md', score: 1 },
      },
    ])
    expect(await memory.search('unrelated topic')).toEqual([])
  })

  it('is structurally compatible with the SDK SearchStrategy in both directions', () => {
    const sdkStrategy: SearchStrategy = new LexicalSearchStrategy({ indexTableName: INDEX_TABLE, extract: extractText })
    const packageStrategy: StringSearchStrategy = KeywordSearchStrategy
    expect(typeof sdkStrategy.search).toBe('function')
    expect(typeof packageStrategy.search).toBe('function')
  })

  it('rejects a storage that is not a DynamoDBStorage', async () => {
    const strategy = new LexicalSearchStrategy({ indexTableName: INDEX_TABLE, extract: extractText })
    const storage = new InMemoryStorage()
    const expected = new StorageError('LexicalSearchStrategy requires a DynamoDBStorage; got InMemoryStorage')
    await expect(strategy.search(storage, 'alpha')).rejects.toThrow(expected)
    await expect(strategy.index(storage, 'notes/a', bytes('alpha'))).rejects.toThrow(expected)
  })

  it.each<[string, Partial<LexicalSearchStrategyConfig>, RegExp]>([
    ['an empty index table name', { indexTableName: '' }, /non-empty indexTableName/],
    ['a missing extractor', { extract: undefined as unknown as SearchableTextExtractor }, /extract function/],
    ['topK 0', { topK: 0 }, /topK/],
    ['topK 101', { topK: 101 }, /topK/],
    ['an out-of-range limit', { limits: { maxQueryTerms: 0 } }, /'maxQueryTerms' must be an integer >= 1/],
  ])('rejects %s at construction', (_label, overrides, message) => {
    const config: LexicalSearchStrategyConfig = { indexTableName: INDEX_TABLE, extract: extractText, ...overrides }
    expect(() => new LexicalSearchStrategy(config)).toThrow(StorageError)
    expect(() => new LexicalSearchStrategy(config)).toThrow(message)
  })

  it('is re-exported from the package index barrel', () => {
    expect(pkgIndex.LexicalSearchStrategy).toBe(LexicalSearchStrategy)
  })
})
