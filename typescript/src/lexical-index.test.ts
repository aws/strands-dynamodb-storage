// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  BatchGetCommand,
  DeleteCommand,
  GetCommand,
  PutCommand,
  QueryCommand,
  TransactWriteCommand,
  type BatchGetCommandInput,
  type DeleteCommandInput,
  type DynamoDBDocumentClient,
  type GetCommandInput,
  type PutCommandInput,
  type QueryCommandInput,
  type TransactWriteCommandInput,
} from '@aws-sdk/lib-dynamodb'
import { DeleteObjectCommand, GetObjectCommand, PutObjectCommand, type S3Client } from '@aws-sdk/client-s3'
import { StorageError } from '@strands-agents/sdk'

import { Buffer } from 'node:buffer'

import { DynamoDBStorage } from './dynamodb-storage.js'
import {
  DEFAULT_LEXICAL_INDEX_LIMITS,
  LexicalIndex,
  RevisionConflictError,
  type LexicalIndexLimits,
  type LexicalSearchResponse,
  type SearchableDocument,
} from './lexical-index.js'
import { identifierPostingPk, keySegment, manifestPk, postingPartitionKeys, termPostingPk } from './lexical-terms.js'
import * as pkgIndex from './index.js'

type Item = Record<string, unknown>
type TransactItem = NonNullable<TransactWriteCommandInput['TransactItems']>[number]

const BASE_TABLE = 'base-table'
const INDEX_TABLE = 'lexical-index'
const FROZEN_NOW_MS = 1_750_000_000_000
const LONE_SURROGATE = String.fromCharCode(0xd800)
const E_ACUTE = String.fromCodePoint(0xe9)
const LAST_BMP_CHARACTER = String.fromCodePoint(0xffff)
const GRINNING_FACE = String.fromCodePoint(0x1f600)
const RESERVED_INDEX_ATTRIBUTES = [
  'pk',
  'sk',
  'k',
  'data',
  's3',
  'meta',
  'z',
  'rev',
  'tv',
  'terms',
  'ids',
  'lxrev',
  'lxscope',
]

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

function cancellation(...codes: string[]): Error {
  return serviceError('TransactionCanceledException', 'Transaction cancelled', {
    CancellationReasons: codes.map((Code) => ({ Code })),
  })
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
 * Table-aware in-memory stand-in for a DynamoDBDocumentClient, keyed by table + pk + sk. Models the
 * commands the storage and the lexical index send, including projections, pagination, injectable
 * unprocessed BatchGet keys, and all-or-nothing TransactWriteItems with cancellation reasons.
 */
class FakeDocumentClient {
  readonly sent: unknown[] = []
  readonly transactionFailures: Error[] = []
  unprocessedRounds = 0
  transactionFailure: Error | undefined
  cancelledTransactions = 0
  private readonly _tables = new Map<string, Map<string, Item>>()
  private _beforeNextTransaction: (() => Promise<void>) | undefined

  beforeNextTransaction(hook: () => Promise<void>): void {
    this._beforeNextTransaction = hook
  }

  clearLog(): void {
    this.sent.length = 0
  }

  sentOf<T>(type: abstract new (...args: never[]) => T): T[] {
    return this.sent.filter((command): command is T => command instanceof type)
  }

  rows(tableName: string): Item[] {
    return [...this._table(tableName).values()]
  }

  row(tableName: string, pk: unknown, sk: unknown): Item | undefined {
    return this._table(tableName).get(itemId(pk, sk))
  }

  putRow(tableName: string, item: Item): void {
    this._table(tableName).set(itemId(item.pk, item.sk), structuredClone(item))
  }

  deleteRow(tableName: string, pk: unknown, sk: unknown): void {
    this._table(tableName).delete(itemId(pk, sk))
  }

  async send(command: unknown): Promise<unknown> {
    this.sent.push(command)
    if (command instanceof GetCommand) return this._get(command.input)
    if (command instanceof PutCommand) return this._put(command.input)
    if (command instanceof DeleteCommand) return this._delete(command.input)
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

  /** Models the pessimistic case: an item without any projected attribute comes back as no item. */
  private _get(input: GetCommandInput): unknown {
    const item = this.row(input.TableName ?? '', input.Key?.pk, input.Key?.sk)
    const projected = item && project(item, input.ProjectionExpression, input.ExpressionAttributeNames)
    return projected && Object.keys(projected).length > 0 ? { Item: projected } : {}
  }

  private _put(input: PutCommandInput): unknown {
    const item = input.Item as Item
    const old = this.row(input.TableName ?? '', item.pk, item.sk)
    this.putRow(input.TableName ?? '', item)
    return input.ReturnValues === 'ALL_OLD' && old ? { Attributes: old } : {}
  }

  private _delete(input: DeleteCommandInput): unknown {
    const old = this.row(input.TableName ?? '', input.Key?.pk, input.Key?.sk)
    this.deleteRow(input.TableName ?? '', input.Key?.pk, input.Key?.sk)
    return input.ReturnValues === 'ALL_OLD' ? { Attributes: old } : {}
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
    const unprocessed: Record<string, { Keys: Item[] }> = {}
    for (const [tableName, request] of Object.entries(input.RequestItems ?? {})) {
      const keys: Item[] = request.Keys ?? []
      if (keys.length > 100) throw serviceError('ValidationException', 'Too many items requested')
      if (new Set(keys.map((key) => itemId(key.pk, key.sk))).size !== keys.length) {
        throw serviceError('ValidationException', 'Provided list of item keys contains duplicates')
      }
      const withheld = this._withheldKeys(keys)
      responses[tableName] = keys.slice(withheld.length).flatMap((key) => {
        const item = this.row(tableName, key.pk, key.sk)
        return item ? [project(item, request.ProjectionExpression, request.ExpressionAttributeNames)] : []
      })
      if (withheld.length > 0) unprocessed[tableName] = { Keys: withheld }
    }
    return { Responses: responses, UnprocessedKeys: unprocessed }
  }

  private _withheldKeys(keys: Item[]): Item[] {
    if (this.unprocessedRounds <= 0) return []
    this.unprocessedRounds -= 1
    return keys.slice(0, 1)
  }

  private async _transactWrite(input: TransactWriteCommandInput): Promise<unknown> {
    const hook = this._beforeNextTransaction
    this._beforeNextTransaction = undefined
    await hook?.()
    const failure = this.transactionFailures.shift() ?? this.transactionFailure
    if (failure) throw failure
    const operations = (input.TransactItems ?? []).map((action) => this._operation(action))
    if (operations.length > 100) throw serviceError('ValidationException', 'Too many transaction items')
    if (new Set(operations.map(({ table, pk, sk }) => `${table} ${itemId(pk, sk)}`)).size !== operations.length) {
      throw serviceError('ValidationException', 'Transaction request cannot include multiple operations on one item')
    }
    const codes = operations.map(({ table, pk, sk, condition }) =>
      conditionHolds(condition, this.row(table, pk, sk)) ? 'None' : 'ConditionalCheckFailed'
    )
    if (codes.includes('ConditionalCheckFailed')) {
      this.cancelledTransactions += 1
      throw cancellation(...codes)
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
        apply: () => this.putRow(tableName, item),
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
        apply: () => this.deleteRow(tableName, key.pk, key.sk),
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

/** In-memory stand-in for an S3Client covering Put/Get/Delete of object bytes. */
class FakeS3Client {
  readonly objects = new Map<string, Uint8Array>()
  async send(command: unknown): Promise<unknown> {
    if (command instanceof PutObjectCommand) {
      this.objects.set(command.input.Key ?? '', command.input.Body as Uint8Array)
      return {}
    }
    if (command instanceof GetObjectCommand) {
      const body = this.objects.get(command.input.Key ?? '')
      if (!body) throw serviceError('NoSuchKey', 'NoSuchKey')
      return { Body: { transformToByteArray: async () => body } }
    }
    if (command instanceof DeleteObjectCommand) {
      this.objects.delete(command.input.Key ?? '')
      return {}
    }
    throw new Error(`unexpected S3 command ${(command as object | undefined)?.constructor?.name}`)
  }
}

interface SetupOptions {
  prefix?: string
  compression?: 'gzip'
  ttlSeconds?: number
  s3?: boolean
  limits?: Partial<LexicalIndexLimits>
}

/** The fakes satisfy the client contracts structurally; the double cast is the seam. */
const asDocClient = (fake: FakeDocumentClient) => fake as unknown as DynamoDBDocumentClient
const asS3Client = (fake: FakeS3Client) => fake as unknown as S3Client

function setup(options: SetupOptions = {}) {
  const client = new FakeDocumentClient()
  const s3 = new FakeS3Client()
  const storage = new DynamoDBStorage(BASE_TABLE, {
    client: asDocClient(client),
    ...(options.prefix ? { prefix: options.prefix } : {}),
    ...(options.compression ? { compression: options.compression } : {}),
    ...(options.ttlSeconds !== undefined ? { ttlSeconds: options.ttlSeconds } : {}),
    ...(options.s3 ? { s3Bucket: 'offload-bucket', s3Client: asS3Client(s3) } : {}),
  })
  const index = new LexicalIndex(storage, { indexTableName: INDEX_TABLE, limits: options.limits })
  return { client, s3, storage, index }
}

function scopedIndex(storage: DynamoDBStorage, namespace?: string): LexicalIndex {
  return new LexicalIndex(namespace ? storage.namespace(namespace) : storage, { indexTableName: INDEX_TABLE })
}

const bytes = (text: string) => new TextEncoder().encode(text)
const text = (value: Uint8Array | undefined) => (value === undefined ? undefined : new TextDecoder().decode(value))

function randomBytes(length: number): Uint8Array {
  return Uint8Array.from({ length }, () => Math.floor(Math.random() * 256))
}

function distinctWords(count: number): string {
  return Array.from({ length: count }, (_, index) => `w${index}`).join(' ')
}

function doc(key: string, body: string, fields: Partial<SearchableDocument> = {}): SearchableDocument {
  return { key, data: bytes(`value of ${key}`), text: body, ...fields }
}

function keysOf(response: LexicalSearchResponse): string[] {
  return response.results.map((result) => result.key)
}

function baseRow(client: FakeDocumentClient, docId: string): Item | undefined {
  return client.rows(BASE_TABLE).find((row) => row.k === docId)
}

function setNow(milliseconds: number): void {
  vi.spyOn(Date, 'now').mockReturnValue(milliseconds)
}

/** Attribute names a request projects, resolving its `#alias` placeholders. */
function projectedAttributes(
  request: { ProjectionExpression?: string; ExpressionAttributeNames?: Record<string, string> } | undefined
): string[] {
  const aliases = request?.ProjectionExpression?.split(', ') ?? []
  return aliases.map((alias) => request?.ExpressionAttributeNames?.[alias] ?? alias)
}

/** Base-table request of the `position`-th BatchGet sent. */
function batchGetRequest(client: FakeDocumentClient, position: number) {
  return client.sentOf(BatchGetCommand)[position]?.input.RequestItems?.[BASE_TABLE]
}

function describeAction(action: TransactItem): string {
  if (action.Put) return `Put ${action.Put.TableName} ${String(action.Put.Item?.pk)}`
  if (action.Delete) return `Delete ${action.Delete.TableName} ${String(action.Delete.Key?.pk)}`
  return `ConditionCheck ${action.ConditionCheck?.TableName} ${String(action.ConditionCheck?.Key?.pk)}`
}

/** Asserts the manifest ⇔ posting invariant of `scope`: same set, same revision, no orphan postings. */
function expectIndexConsistent(client: FakeDocumentClient, scope: string): void {
  const postingPrefixes = ['t|', 'i|'].map((tag) => `${tag}${keySegment(scope)}`)
  const rows = client.rows(INDEX_TABLE)
  const expected = rows
    .filter((row) => row.pk === manifestPk(scope))
    .flatMap((manifest) =>
      postingPartitionKeys(scope, manifest.terms as string[], manifest.ids as string[]).map(
        (pk) => `${pk} ${String(manifest.sk)} ${String(manifest.rev)}`
      )
    )
  const actual = rows
    .filter((row) => postingPrefixes.some((prefix) => String(row.pk).startsWith(prefix)))
    .map((row) => `${String(row.pk)} ${String(row.sk)} ${String(row.rev)}`)
  expect(actual.sort()).toEqual(expected.sort())
}

beforeEach(() => {
  setNow(FROZEN_NOW_MS)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('LexicalIndex — upsert, search and lookup', () => {
  it('upserts a document and finds it by text and by identifier', async () => {
    const { index } = setup()
    await index.upsert(
      doc('tickets/t1', 'Login fails with ERR_AUTH_403', { identifiers: ['T-1'], metadata: { status: 'open' } })
    )
    const search = await index.search({ text: 'login err_auth_403', topK: 5 })
    expect(search).toEqual({
      results: [{ key: 'tickets/t1', score: 1, matchedTerms: 5, metadata: { status: 'open' } }],
      truncated: false,
      truncationReasons: [],
      candidatesExamined: 1,
    })
    const lookup = await index.lookup({ identifier: 'T-1', topK: 5, includeValues: true })
    expect(lookup.results).toHaveLength(1)
    expect(lookup.results[0]).toMatchObject({ key: 'tickets/t1', score: 1, matchedTerms: 1 })
    expect(text(lookup.results[0]?.data)).toBe('value of tickets/t1')
  })

  it('writes the base item, manifest and postings in one transaction keyed by the revision', async () => {
    const { client, index } = setup({ prefix: 'tenant/a' })
    const rev = await index.upsert(doc('tickets/t1', 'Alpha beta', { identifiers: ['T-1'] }))
    const docId = 'tenant/a/tickets/t1'
    expect(rev).toMatch(/^[0-9a-f]{32}$/)
    expect(client.row(BASE_TABLE, 'tenant/a', 'tickets/t1')).toMatchObject({
      k: docId,
      lxrev: rev,
      lxscope: '9:tenant/a/',
    })
    expect(client.row(INDEX_TABLE, 'm|9:tenant/a/', docId)).toEqual({
      pk: 'm|9:tenant/a/',
      sk: docId,
      rev,
      tv: 'lexical-v1',
      terms: ['alpha', 'beta'],
      ids: ['T-1'],
    })
    expect(client.row(INDEX_TABLE, identifierPostingPk('tenant/a/', 'T-1'), docId)).toEqual({
      pk: 'i|9:tenant/a/3:T-1',
      sk: docId,
      rev,
    })
    const [transaction] = client.sentOf(TransactWriteCommand)
    expect(transaction?.input.ClientRequestToken).toBe(rev)
    expect(transaction?.input.TransactItems?.map(describeAction)).toEqual([
      `Put ${BASE_TABLE} tenant/a`,
      `Put ${INDEX_TABLE} m|9:tenant/a/`,
      `Put ${INDEX_TABLE} t|9:tenant/a/5:alpha`,
      `Put ${INDEX_TABLE} t|9:tenant/a/4:beta`,
      `Put ${INDEX_TABLE} i|9:tenant/a/3:T-1`,
    ])
    expectIndexConsistent(client, 'tenant/a/')
  })

  it('reads state strongly consistently and pages postings eventually consistently', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    expect(client.sentOf(GetCommand).map((command) => command.input.ConsistentRead)).toEqual([true, true])
    client.clearLog()
    await index.search({ text: 'alpha', topK: 1 })
    expect(client.sentOf(QueryCommand).map((command) => command.input)).toEqual([
      expect.objectContaining({ ConsistentRead: false, Limit: DEFAULT_LEXICAL_INDEX_LIMITS.pageSize }),
    ])
    expect(client.sentOf(BatchGetCommand)[0]?.input.RequestItems?.[BASE_TABLE]?.ConsistentRead).toBe(true)
  })

  it('ranks by matched-term fraction, then by key in UTF-8 byte order', async () => {
    const { index } = setup()
    await index.upsert(doc('docs/\u{1F600}', 'beta'))
    await index.upsert(doc('docs/\uFFFF', 'beta'))
    await index.upsert(doc('docs/one', 'alpha'))
    await index.upsert(doc('docs/both', 'alpha beta'))
    const ranked = await index.search({ text: 'alpha beta', topK: 10 })
    expect(keysOf(ranked)).toEqual(['docs/both', 'docs/one', 'docs/\uFFFF', 'docs/\u{1F600}'])
    expect(ranked.results.map((result) => [result.score, result.matchedTerms])).toEqual([
      [1, 2],
      [0.5, 1],
      [0.5, 1],
      [0.5, 1],
    ])
    expect(keysOf(await index.search({ text: 'alpha beta', topK: 2 }))).toEqual(['docs/both', 'docs/one'])
    expect(keysOf(await index.search({ text: 'alpha beta', topK: 10, requireAllTerms: true }))).toEqual(['docs/both'])
  })

  it('orders lookup results by key in UTF-8 byte order', async () => {
    const { index } = setup()
    const keys = [GRINNING_FACE, E_ACUTE, 'z', LAST_BMP_CHARACTER, 'b', 'a'].map((suffix) => `docs/${suffix}`)
    for (const key of keys) await index.upsert(doc(key, '', { identifiers: ['ID-1'] }))
    const expected = ['a', 'b', 'z', E_ACUTE, LAST_BMP_CHARACTER, GRINNING_FACE].map((suffix) => `docs/${suffix}`)
    expect(keysOf(await index.lookup({ identifier: 'ID-1', topK: 10 }))).toEqual(expected)
    expect(keysOf(await index.lookup({ identifier: 'ID-1', topK: 2 }))).toEqual(['docs/a', 'docs/b'])
  })

  it('distinguishes FC-00123 from FC-123 and ERR_AUTH_403 from err_auth_403', async () => {
    const { index } = setup()
    await index.upsert(doc('invoices/a', 'Invoice FC-00123', { identifiers: ['FC-00123'] }))
    await index.upsert(doc('invoices/b', 'Invoice FC-123', { identifiers: ['FC-123'] }))
    await index.upsert(doc('errors/upper', 'auth failure', { identifiers: ['ERR_AUTH_403'] }))
    await index.upsert(doc('errors/lower', 'auth failure', { identifiers: ['err_auth_403'] }))
    const lookup = async (identifier: string) => keysOf(await index.lookup({ identifier, topK: 10 }))
    expect(await lookup('FC-00123')).toEqual(['invoices/a'])
    expect(await lookup('FC-123')).toEqual(['invoices/b'])
    expect(await lookup('fc-123')).toEqual([])
    expect(await lookup('ERR_AUTH_403')).toEqual(['errors/upper'])
    expect(await lookup('err_auth_403')).toEqual(['errors/lower'])
    const exact = await index.search({ text: 'FC-123', topK: 10, requireAllTerms: true })
    expect(keysOf(exact)).toEqual(['invoices/b'])
  })

  it('preserves an empty value with includeValues', async () => {
    const { index } = setup()
    await index.upsert({ key: 'docs/empty', data: new Uint8Array(0), text: 'alpha' })
    const [result] = (await index.search({ text: 'alpha', topK: 1, includeValues: true })).results
    expect(result?.data).toBeInstanceOf(Uint8Array)
    expect(result?.data?.byteLength).toBe(0)
  })

  it('round-trips a gzip-compressed document value', async () => {
    const { client, index } = setup({ compression: 'gzip' })
    const original = 'compressible text '.repeat(500)
    await index.upsert({ key: 'docs/big', data: bytes(original), text: 'alpha', identifiers: ['BIG-1'] })
    expect(baseRow(client, 'docs/big')?.z).toBe(true)
    const [searched] = (await index.search({ text: 'alpha', topK: 1, includeValues: true })).results
    const [looked] = (await index.lookup({ identifier: 'BIG-1', topK: 1, includeValues: true })).results
    expect(text(searched?.data)).toBe(original)
    expect(text(looked?.data)).toBe(original)
  })

  it('stores a document without text or identifiers without making it searchable', async () => {
    const { client, index, storage } = setup()
    await index.upsert({ key: 'docs/plain', data: bytes('plain') })
    expect(text((await storage.read('docs/plain')) ?? undefined)).toBe('plain')
    expect(client.row(INDEX_TABLE, manifestPk(''), 'docs/plain')).toMatchObject({ terms: [], ids: [] })
  })

  it('reclaims the S3 object of an offloaded value it overwrites, best-effort', async () => {
    const { s3, storage, index } = setup({ s3: true })
    await storage.write('docs/a', randomBytes(400_001))
    await storage.write('docs/b', randomBytes(400_001))
    expect(s3.objects.size).toBe(2)
    await index.upsert(doc('docs/a', 'alpha'))
    expect(s3.objects.size).toBe(1)
    const realSend = s3.send.bind(s3)
    s3.send = async (command: unknown) => {
      if (command instanceof DeleteObjectCommand) throw new Error('s3 down')
      return realSend(command)
    }
    await expect(index.upsert(doc('docs/b', 'beta'))).resolves.toMatch(/^[0-9a-f]{32}$/)
    const [result] = (await index.search({ text: 'beta', topK: 1, includeValues: true })).results
    expect(text(result?.data)).toBe('value of docs/b')
  })
})

describe('LexicalIndex — overwrite and delete', () => {
  it('overwrite A→B removes the old postings', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha beta', { identifiers: ['ID-1'] }))
    await index.upsert(doc('docs/a', 'gamma', { identifiers: ['ID-2'] }))
    expect(keysOf(await index.search({ text: 'alpha', topK: 5 }))).toEqual([])
    expect(keysOf(await index.lookup({ identifier: 'ID-1', topK: 5 }))).toEqual([])
    expect(keysOf(await index.search({ text: 'gamma', topK: 5 }))).toEqual(['docs/a'])
    expect(client.rows(INDEX_TABLE)).toHaveLength(3)
    const deletes = client
      .sentOf(TransactWriteCommand)
      .at(-1)
      ?.input.TransactItems?.map(describeAction)
      .filter((action) => action.startsWith('Delete'))
    expect(deletes).toEqual([
      `Delete ${INDEX_TABLE} ${termPostingPk('', 'alpha')}`,
      `Delete ${INDEX_TABLE} ${termPostingPk('', 'beta')}`,
      `Delete ${INDEX_TABLE} ${identifierPostingPk('', 'ID-1')}`,
    ])
    expectIndexConsistent(client, '')
  })

  it('examines no candidate for a term an overwrite removed', async () => {
    const { index } = setup()
    await index.upsert(doc('docs/a', 'alpha beta'))
    await index.upsert(doc('docs/a', 'gamma'))
    expect(await index.search({ text: 'alpha', topK: 5 })).toEqual({
      results: [],
      truncated: false,
      truncationReasons: [],
      candidatesExamined: 0,
    })
  })

  it('guards a delete that found no manifest against a concurrent first upsert', async () => {
    const { client, storage, index } = setup()
    await storage.write('docs/a', bytes('plain'))
    client.beforeNextTransaction(async () => {
      await index.upsert(doc('docs/a', 'alpha', { identifiers: ['A-1'] }))
    })
    expect(await index.delete('docs/a')).toBe(true)
    const [firstDelete, , retriedDelete] = client.sentOf(TransactWriteCommand)
    expect(firstDelete?.input.TransactItems?.map(describeAction)).toEqual([
      `Delete ${BASE_TABLE} docs/a`,
      `ConditionCheck ${INDEX_TABLE} ${manifestPk('')}`,
    ])
    expect(client.cancelledTransactions).toBe(1)
    expect(retriedDelete?.input.TransactItems?.map(describeAction)).toEqual([
      `Delete ${BASE_TABLE} docs/a`,
      `Delete ${INDEX_TABLE} ${manifestPk('')}`,
      `Delete ${INDEX_TABLE} ${termPostingPk('', 'alpha')}`,
      `Delete ${INDEX_TABLE} ${identifierPostingPk('', 'A-1')}`,
    ])
    expect(client.rows(BASE_TABLE)).toEqual([])
    expect(client.rows(INDEX_TABLE)).toEqual([])
  })

  it('touches each posting once when a hand-written manifest repeats terms and identifiers', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha beta', { identifiers: ['ID-1'] }))
    const repeatTerms = (terms: string[], ids: string[]) =>
      client.putRow(INDEX_TABLE, { ...client.row(INDEX_TABLE, manifestPk(''), 'docs/a'), terms, ids })
    repeatTerms(['alpha', 'beta', 'alpha'], ['ID-1', 'ID-1'])
    expect(await index.repair({ rebuildPostings: true })).toMatchObject({ documentsRemoved: 0, postingsRebuilt: 1 })
    await index.upsert(doc('docs/a', 'gamma'))
    expectIndexConsistent(client, '')
    repeatTerms(['gamma', 'gamma'], [])
    expect(await index.delete('docs/a')).toBe(true)
    expect(client.rows(INDEX_TABLE)).toEqual([])
    expect(client.rows(BASE_TABLE)).toEqual([])
  })

  it('overwrite A→A advances the revision of shared postings and serves the new value', async () => {
    const { client, index } = setup()
    const first = await index.upsert(doc('docs/a', 'alpha beta', { data: bytes('v1') }))
    const second = await index.upsert(doc('docs/a', 'alpha beta', { data: bytes('v2') }))
    expect(second).not.toBe(first)
    expect(client.row(INDEX_TABLE, termPostingPk('', 'alpha'), 'docs/a')?.rev).toBe(second)
    expect(client.row(INDEX_TABLE, termPostingPk('', 'beta'), 'docs/a')?.rev).toBe(second)
    expect(client.rows(INDEX_TABLE)).toHaveLength(3)
    const [result] = (await index.search({ text: 'alpha', topK: 1, includeValues: true })).results
    expect(text(result?.data)).toBe('v2')
    expectIndexConsistent(client, '')
  })

  it('delete then recreate', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha', { identifiers: ['A-1'] }))
    expect(await index.delete('docs/a')).toBe(true)
    expect(client.rows(INDEX_TABLE)).toEqual([])
    expect(client.rows(BASE_TABLE)).toEqual([])
    expect(await index.revision('docs/a')).toBeNull()
    expect(keysOf(await index.search({ text: 'alpha', topK: 5 }))).toEqual([])
    client.clearLog()
    expect(await index.delete('docs/a')).toBe(false)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(0)
    const rev = await index.upsert(doc('docs/a', 'beta'))
    expect(await index.revision('docs/a')).toBe(rev)
    expect(keysOf(await index.search({ text: 'beta', topK: 5 }))).toEqual(['docs/a'])
    expect(keysOf(await index.search({ text: 'alpha', topK: 5 }))).toEqual([])
    expectIndexConsistent(client, '')
  })

  it('delete sends no client request token', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    client.clearLog()
    await index.delete('docs/a')
    expect(client.sentOf(TransactWriteCommand)[0]?.input).not.toHaveProperty('ClientRequestToken')
  })

  it('deletes a document written directly with storage.write(), which has no lexical attributes', async () => {
    const { client, storage, index } = setup()
    await storage.write('docs/plain', bytes('plain'))
    expect(await index.delete('docs/plain')).toBe(true)
    expect(client.rows(BASE_TABLE)).toEqual([])
    expect(await storage.read('docs/plain')).toBeNull()
  })
})

describe('LexicalIndex — concurrency', () => {
  it('retries once when a competing upsert commits between the reads and the transaction', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha beta'))
    client.clearLog()
    client.beforeNextTransaction(async () => {
      await index.upsert(doc('docs/a', 'gamma delta', { identifiers: ['G-1'] }))
    })
    const rev = await index.upsert(doc('docs/a', 'alpha epsilon', { data: bytes('winner') }))
    expect(client.cancelledTransactions).toBe(1)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(3)
    expect(await index.revision('docs/a')).toBe(rev)
    expect(baseRow(client, 'docs/a')?.lxrev).toBe(rev)
    expect(client.row(INDEX_TABLE, manifestPk(''), 'docs/a')).toMatchObject({ terms: ['alpha', 'epsilon'], ids: [] })
    expectIndexConsistent(client, '')
    const [result] = (await index.search({ text: 'epsilon', topK: 1, includeValues: true })).results
    expect(text(result?.data)).toBe('winner')
  })

  it('sends a fresh ClientRequestToken on every retry', async () => {
    const { client, index } = setup()
    client.transactionFailures.push(cancellation('None', 'ConditionalCheckFailed'))
    const rev = await index.upsert(doc('docs/a', 'alpha'))
    const tokens = client.sentOf(TransactWriteCommand).map((command) => command.input.ClientRequestToken)
    expect(tokens).toHaveLength(2)
    expect(tokens[0]).toMatch(/^[0-9a-f]{32}$/)
    expect(tokens[0]).not.toBe(tokens[1])
    expect(tokens[1]).toBe(rev)
  })

  it('retries throttled transactions of upsert and delete within maxConflictRetries', async () => {
    const { client, index } = setup()
    client.transactionFailures.push(
      cancellation('ThrottlingError'),
      cancellation('None', 'ProvisionedThroughputExceeded')
    )
    const rev = await index.upsert(doc('docs/a', 'alpha'))
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(3)
    expect(await index.revision('docs/a')).toBe(rev)
    client.clearLog()
    client.transactionFailures.push(cancellation('RequestLimitExceeded'))
    expect(await index.delete('docs/a')).toBe(true)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(2)
    expect(client.rows(INDEX_TABLE)).toEqual([])
  })

  it('chains the last cancellation as the cause once retries are exhausted', async () => {
    const { client, index } = setup({ limits: { maxConflictRetries: 1 } })
    const throttled = cancellation('None', 'ThrottlingError')
    client.transactionFailures.push(cancellation('TransactionConflict'))
    client.transactionFailure = throttled
    const error = await index.upsert(doc('docs/a', 'alpha')).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(StorageError)
    expect(error).not.toBeInstanceOf(RevisionConflictError)
    expect((error as Error).message).toBe(
      `Lexical index upsert failed for 'docs/a' (index table '${INDEX_TABLE}', base table '${BASE_TABLE}'): transaction throttled; retries exhausted`
    )
    expect((error as Error).cause).toBe(throttled)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(2)
  })

  it.each<[string, Item]>([
    ['without a revision', { terms: ['alpha'], ids: [] }],
    ['with a non-string revision', { rev: 7, terms: ['alpha'], ids: [] }],
  ])('fails on a manifest %s instead of treating it as absent', async (_label, manifest) => {
    const { client, storage, index } = setup()
    await storage.write('docs/a', bytes('plain'))
    client.putRow(INDEX_TABLE, { pk: manifestPk(''), sk: 'docs/a', ...manifest })
    client.clearLog()
    const malformed = /manifest for 'docs\/a' in index table 'lexical-index' has no string revision/
    await expect(index.revision('docs/a')).rejects.toThrow(malformed)
    await expect(index.upsert(doc('docs/a', 'beta'))).rejects.toThrow(malformed)
    await expect(index.delete('docs/a')).rejects.toThrow(malformed)
    await expect(index.repair()).rejects.toThrow(malformed)
    await expect(index.revision('docs/a')).rejects.toBeInstanceOf(StorageError)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(0)
  })

  it('rejects a stale expectedRevision without writing', async () => {
    const { client, index } = setup()
    const first = await index.upsert(doc('docs/a', 'alpha'))
    const second = await index.upsert(doc('docs/a', 'beta'), { expectedRevision: first })
    client.clearLog()
    const stale = await index
      .upsert(doc('docs/a', 'gamma'), { expectedRevision: first })
      .catch((error: unknown) => error)
    expect(stale).toBeInstanceOf(RevisionConflictError)
    expect(stale).toBeInstanceOf(StorageError)
    expect((stale as Error).name).toBe('RevisionConflictError')
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(0)
    await expect(index.upsert(doc('docs/new', 'alpha'), { expectedRevision: first })).rejects.toBeInstanceOf(
      RevisionConflictError
    )
    await expect(index.delete('docs/a', { expectedRevision: first })).rejects.toBeInstanceOf(RevisionConflictError)
    expect(await index.delete('docs/a', { expectedRevision: second })).toBe(true)
  })

  it('fails with StorageError once conflict retries are exhausted', async () => {
    const { client, index } = setup({ limits: { maxConflictRetries: 1 } })
    client.transactionFailure = cancellation('None', 'ConditionalCheckFailed')
    const error = await index.upsert(doc('docs/a', 'alpha')).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(StorageError)
    expect(error).not.toBeInstanceOf(RevisionConflictError)
    expect((error as Error).message).toMatch(/concurrent modification; retries exhausted/)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(2)
  })

  it('treats a TransactionConflict cancellation as retryable', async () => {
    const { client, index } = setup({ limits: { maxConflictRetries: 0 } })
    client.transactionFailure = cancellation('TransactionConflict')
    await expect(index.upsert(doc('docs/a', 'alpha'))).rejects.toThrow(/retries exhausted/)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(1)
  })

  it('wraps a non-conflict transaction failure in StorageError without retrying', async () => {
    const { client, index } = setup()
    const throttled = serviceError('ProvisionedThroughputExceededException', 'slow down')
    client.transactionFailure = throttled
    const error = await index.upsert(doc('docs/a', 'alpha')).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(StorageError)
    expect((error as Error).message).toBe(
      `Lexical index upsert failed for 'docs/a' (index table '${INDEX_TABLE}', base table '${BASE_TABLE}')`
    )
    expect((error as Error).cause).toBe(throttled)
    client.transactionFailure = cancellation('None', 'ValidationError')
    await expect(index.upsert(doc('docs/a', 'alpha'))).rejects.toBeInstanceOf(StorageError)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(2)
    expect(client.rows(BASE_TABLE)).toEqual([])
  })

  it('wraps an unexpected read failure naming the operation and table', async () => {
    const { client, index } = setup()
    const realSend = client.send.bind(client)
    client.send = async (command: unknown) => {
      if (command instanceof QueryCommand) throw new Error('network down')
      return realSend(command)
    }
    await expect(index.search({ text: 'alpha', topK: 1 })).rejects.toThrow(
      `Lexical index search failed (index table '${INDEX_TABLE}', base table '${BASE_TABLE}')`
    )
  })
})

describe('LexicalIndex — validation against the base item', () => {
  it('excludes documents expiring at or before now, and repair removes them', async () => {
    const { client, index } = setup({ ttlSeconds: 60 })
    await index.upsert(doc('docs/a', 'alpha'))
    const expireAt = Math.floor(FROZEN_NOW_MS / 1000) + 60
    expect(baseRow(client, 'docs/a')?.expireAt).toBe(expireAt)
    expect(client.row(INDEX_TABLE, termPostingPk('', 'alpha'), 'docs/a')?.expireAt).toBe(expireAt)
    expect(client.row(INDEX_TABLE, manifestPk(''), 'docs/a')).not.toHaveProperty('expireAt')
    setNow(expireAt * 1000 - 1)
    expect(keysOf(await index.search({ text: 'alpha', topK: 1 }))).toEqual(['docs/a'])
    setNow(expireAt * 1000)
    expect(keysOf(await index.search({ text: 'alpha', topK: 1 }))).toEqual([])
    expect(await index.repair()).toEqual({ documentsChecked: 1, documentsRemoved: 1, postingsRebuilt: 0, cursor: null })
    expect(client.rows(INDEX_TABLE)).toEqual([])
  })

  it('copies the document TTL onto term and identifier postings but not onto the manifest', async () => {
    const { client, index } = setup({ ttlSeconds: 60 })
    await index.upsert(doc('docs/a', 'alpha', { identifiers: ['A-1'] }))
    const expireAt = baseRow(client, 'docs/a')?.expireAt
    expect(expireAt).toBe(Math.floor(FROZEN_NOW_MS / 1000) + 60)
    expect(client.row(INDEX_TABLE, termPostingPk('', 'alpha'), 'docs/a')?.expireAt).toBe(expireAt)
    expect(client.row(INDEX_TABLE, identifierPostingPk('', 'A-1'), 'docs/a')?.expireAt).toBe(expireAt)
    expect(client.row(INDEX_TABLE, manifestPk(''), 'docs/a')).not.toHaveProperty('expireAt')
  })

  it('never projects stored values in the validation read and reads values only for the kept results', async () => {
    const { client, index } = setup({ ttlSeconds: 60 })
    for (const key of ['docs/a', 'docs/b', 'docs/c']) await index.upsert(doc(key, 'alpha'))
    client.clearLog()
    const response = await index.search({ text: 'alpha', topK: 2, includeValues: true })
    expect(response.results.map((result) => [result.key, text(result.data)])).toEqual([
      ['docs/a', 'value of docs/a'],
      ['docs/b', 'value of docs/b'],
    ])
    expect(response.candidatesExamined).toBe(3)
    const validation = batchGetRequest(client, 0)
    const values = batchGetRequest(client, 1)
    expect(projectedAttributes(validation)).toEqual(['pk', 'sk', 'k', 'lxrev', 'lxscope', 'meta', 's3', 'expireAt'])
    expect(validation?.Keys?.map((key) => key.pk)).toEqual(['docs/a', 'docs/b', 'docs/c'])
    expect(projectedAttributes(values)).toEqual(['pk', 'sk', 'lxrev', 'data', 'z', 's3'])
    expect(values?.Keys?.map((key) => key.pk)).toEqual(['docs/a', 'docs/b'])
    expect(values?.ConsistentRead).toBe(true)
    expect(client.sentOf(BatchGetCommand)).toHaveLength(2)
  })

  it('reads no values when includeValues is off or nothing matched', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    client.clearLog()
    await index.search({ text: 'alpha', topK: 5 })
    await index.search({ text: 'alpha', topK: 5, includeValues: true, filter: { missing: 'x' } })
    expect(client.sentOf(BatchGetCommand)).toHaveLength(2)
    expect(projectedAttributes(batchGetRequest(client, 0))).not.toContain('data')
    expect(projectedAttributes(batchGetRequest(client, 1))).not.toContain('data')
  })

  it('drops a result whose document changes between the validation and the value read', async () => {
    const { client, index } = setup()
    for (const key of ['docs/a', 'docs/b', 'docs/c', 'docs/d', 'docs/e']) await index.upsert(doc(key, 'alpha'))
    const changeDocumentsBeforeValueRead = () => {
      const row = (docId: string): Item => ({ ...baseRow(client, docId) })
      client.putRow(BASE_TABLE, { ...row('docs/a'), lxrev: 'rewritten' })
      client.deleteRow(BASE_TABLE, row('docs/b').pk, row('docs/b').sk)
      const { data: _data, ...withoutValue } = row('docs/c')
      client.putRow(BASE_TABLE, withoutValue)
      client.putRow(BASE_TABLE, { ...row('docs/e'), s3: true })
    }
    const realSend = client.send.bind(client)
    let batchGets = 0
    client.send = async (command: unknown) => {
      if (command instanceof BatchGetCommand) batchGets += 1
      if (command instanceof BatchGetCommand && batchGets === 2) changeDocumentsBeforeValueRead()
      return realSend(command)
    }
    const response = await index.search({ text: 'alpha', topK: 10, includeValues: true })
    expect(response.results.map((result) => [result.key, text(result.data)])).toEqual([['docs/d', 'value of docs/d']])
    expect(response.candidatesExamined).toBe(5)
    expect(batchGets).toBe(2)
  })

  it('writes no TTL on any record when the storage did not opt in to TTL', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha', { identifiers: ['A-1'], ttlSeconds: 60 }))
    const records = [...client.rows(BASE_TABLE), ...client.rows(INDEX_TABLE)]
    expect(records).toHaveLength(4)
    expect(records.filter((record) => 'expireAt' in record)).toEqual([])
  })

  it('excludes a document overwritten by a direct storage.write(), and repair removes its manifest', async () => {
    const { client, storage, index } = setup()
    const rev = await index.upsert(doc('docs/a', 'alpha', { identifiers: ['A-1'] }))
    await storage.write('docs/a', bytes('rewritten directly'))
    expect(keysOf(await index.search({ text: 'alpha', topK: 1 }))).toEqual([])
    expect(keysOf(await index.lookup({ identifier: 'A-1', topK: 1 }))).toEqual([])
    expect(await index.revision('docs/a')).toBe(rev)
    expect(await index.repair()).toEqual({ documentsChecked: 1, documentsRemoved: 1, postingsRebuilt: 0, cursor: null })
    expect(client.rows(INDEX_TABLE)).toEqual([])
    expect(text((await storage.read('docs/a')) ?? undefined)).toBe('rewritten directly')
  })

  it('excludes postings whose revision is stale', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha beta'))
    client.putRow(INDEX_TABLE, { pk: termPostingPk('', 'alpha'), sk: 'docs/a', rev: 'stale' })
    client.putRow(INDEX_TABLE, { pk: termPostingPk('', 'alpha'), sk: 'docs/ghost', rev: 'stale' })
    const alpha = await index.search({ text: 'alpha', topK: 5 })
    expect(alpha.results).toEqual([])
    expect(alpha.candidatesExamined).toBe(2)
    const both = await index.search({ text: 'alpha beta', topK: 5 })
    expect(both.results).toEqual([{ key: 'docs/a', score: 0.5, matchedTerms: 1 }])
    expect((await index.search({ text: 'alpha beta', topK: 5, requireAllTerms: true })).results).toEqual([])
  })

  it('drops a posting outside the scope before any base-table read', async () => {
    const { client, storage } = setup()
    const tenantA = scopedIndex(storage, 'tenant-a')
    const tenantB = scopedIndex(storage, 'tenant-b')
    await tenantB.upsert(doc('docs/x', 'secret'))
    const foreignRev = baseRow(client, 'tenant-b/docs/x')?.lxrev
    client.putRow(INDEX_TABLE, { pk: termPostingPk('tenant-a/', 'secret'), sk: 'tenant-b/docs/x', rev: foreignRev })
    client.clearLog()
    const response = await tenantA.search({ text: 'secret', topK: 5 })
    expect(response).toEqual({ results: [], truncated: false, truncationReasons: [], candidatesExamined: 0 })
    expect(client.sentOf(BatchGetCommand)).toHaveLength(0)
  })

  it('reads a base item once when two doc ids share its key', async () => {
    const { client, index } = setup()
    const rev = await index.upsert(doc('x/y', 'alpha'))
    client.putRow(INDEX_TABLE, { pk: termPostingPk('', 'alpha'), sk: 'x/y/\u0000', rev })
    client.clearLog()
    const response = await index.search({ text: 'alpha', topK: 5 })
    expect(keysOf(response)).toEqual(['x/y'])
    expect(response.candidatesExamined).toBe(2)
    expect(client.sentOf(BatchGetCommand)[0]?.input.RequestItems?.[BASE_TABLE]?.Keys).toEqual([
      { pk: 'x/y', sk: '\u0000' },
    ])
  })

  it('filters on the current metadata', async () => {
    const { index } = setup()
    await index.upsert(doc('docs/a', 'alpha', { metadata: { status: 'open' } }))
    const matching = async (status: string) =>
      keysOf(await index.search({ text: 'alpha', topK: 5, filter: { status } }))
    expect(await matching('open')).toEqual(['docs/a'])
    await index.upsert(doc('docs/a', 'alpha', { metadata: { status: 'closed' } }))
    expect(await matching('open')).toEqual([])
    expect(await matching('closed')).toEqual(['docs/a'])
  })

  it('never equates a boolean with a number in filters', async () => {
    const { index } = setup()
    await index.upsert(doc('docs/a', 'alpha', { metadata: { flag: true, count: 1, ratio: 2.5, label: '' } }))
    await index.upsert(doc('docs/b', 'alpha'))
    const matching = async (filter: Record<string, string | number | boolean>) =>
      keysOf(await index.search({ text: 'alpha', topK: 5, filter }))
    expect(await matching({ flag: true })).toEqual(['docs/a'])
    expect(await matching({ flag: 1 })).toEqual([])
    expect(await matching({ count: true })).toEqual([])
    expect(await matching({ count: 1 })).toEqual(['docs/a'])
    expect(await matching({ count: '1' })).toEqual([])
    expect(await matching({ ratio: 2.5 })).toEqual(['docs/a'])
    expect(await matching({ label: '' })).toEqual(['docs/a'])
    expect(await matching({ label: 0 })).toEqual([])
    expect(await matching({ missing: 'x' })).toEqual([])
    expect(await matching({})).toEqual(['docs/a', 'docs/b'])
  })
})

describe('LexicalIndex — scopes', () => {
  it('isolates namespaced scopes that share keys and terms', async () => {
    const { client, storage } = setup()
    const tenantA = scopedIndex(storage, 'tenant-a')
    const tenantB = scopedIndex(storage, 'tenant-b')
    expect(tenantA.scope).toBe('tenant-a/')
    await tenantA.upsert(doc('docs/x', 'shared words', { identifiers: ['ID-1'], data: bytes('A') }))
    await tenantB.upsert(doc('docs/x', 'shared words', { identifiers: ['ID-1'], data: bytes('B') }))
    const searchA = await tenantA.search({ text: 'shared', topK: 5, includeValues: true })
    const lookupB = await tenantB.lookup({ identifier: 'ID-1', topK: 5, includeValues: true })
    expect(searchA.results.map((result) => [result.key, text(result.data)])).toEqual([['docs/x', 'A']])
    expect(lookupB.results.map((result) => [result.key, text(result.data)])).toEqual([['docs/x', 'B']])
    await tenantA.delete('docs/x')
    expect(keysOf(await tenantA.search({ text: 'shared', topK: 5 }))).toEqual([])
    expect(keysOf(await tenantB.search({ text: 'shared', topK: 5 }))).toEqual(['docs/x'])
    expectIndexConsistent(client, 'tenant-a/')
    expectIndexConsistent(client, 'tenant-b/')
  })

  it('gives a root-scope index no view of namespaced scopes', async () => {
    const { storage } = setup()
    await scopedIndex(storage, 'tenant-a').upsert(doc('docs/x', 'shared', { identifiers: ['ID-1'] }))
    const root = scopedIndex(storage)
    const empty = { results: [], truncated: false, truncationReasons: [], candidatesExamined: 0 }
    expect(await root.search({ text: 'shared', topK: 5 })).toEqual(empty)
    expect(await root.lookup({ identifier: 'ID-1', topK: 5 })).toEqual(empty)
    expect(await root.repair()).toEqual({ documentsChecked: 0, documentsRemoved: 0, postingsRebuilt: 0, cursor: null })
  })

  it('rejects overwriting or deleting a document owned by another lexical scope', async () => {
    const { client, storage } = setup()
    const root = scopedIndex(storage)
    const tenant = scopedIndex(storage, 'tenant-a')
    await tenant.upsert(doc('docs/x', 'alpha'))
    client.clearLog()
    await expect(root.upsert(doc('tenant-a/docs/x', 'beta'))).rejects.toThrow(/owned by another lexical scope/)
    await expect(root.delete('tenant-a/docs/x')).rejects.toThrow(/owned by another lexical scope/)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(0)
    expect(keysOf(await tenant.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/x'])
    expect(keysOf(await root.search({ text: 'beta', topK: 5 }))).toEqual([])
  })

  it('rejects a claim that races with another scope through the transaction condition', async () => {
    const { client, storage } = setup()
    const root = scopedIndex(storage)
    const tenant = scopedIndex(storage, 'tenant-a')
    client.beforeNextTransaction(async () => {
      await tenant.upsert(doc('docs/x', 'alpha'))
    })
    await expect(root.upsert(doc('tenant-a/docs/x', 'beta'))).rejects.toThrow(/owned by another lexical scope/)
    expect(client.cancelledTransactions).toBe(1)
    expect(keysOf(await tenant.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/x'])
    expect(client.rows(INDEX_TABLE).filter((row) => row.pk === manifestPk(''))).toEqual([])
  })
})

describe('LexicalIndex — limits', () => {
  const invalidDocuments: Array<{ label: string; document: SearchableDocument; error: RegExp; secret?: string }> = [
    {
      label: 'more postings than maxPostingsPerDocument',
      document: doc('docs/a', distinctWords(50)),
      error:
        /has 50 distinct terms and identifiers, above maxPostingsPerDocument \(49\); split the document or reduce distinct terms/,
    },
    {
      label: 'postings counted across terms and identifiers',
      document: doc('docs/a', distinctWords(48), { identifiers: ['ID-1', 'ID-2'] }),
      error: /has 50 distinct terms and identifiers/,
    },
    {
      label: 'a term above maxTermBytes',
      document: doc('docs/a', `ok ${'secretterm'.repeat(7)}`),
      error: /A document term is 70 bytes, above maxTermBytes \(64\)/,
      secret: 'secretterm',
    },
    {
      label: 'an identifier above maxIdentifierBytes',
      document: doc('docs/a', 'ok', { identifiers: ['SECRETID'.repeat(17)] }),
      error: /A document identifier is 136 bytes, above maxIdentifierBytes \(128\)/,
      secret: 'SECRETID',
    },
    {
      label: 'text above maxTextBytes',
      document: doc('docs/a', 'secret '.repeat(9363)),
      error: /Document text is 65541 bytes, above maxTextBytes \(65536\)/,
      secret: 'secret',
    },
    {
      label: 'an invalid identifier',
      document: doc('docs/a', 'ok', { identifiers: [' SECRET-1'] }),
      error: /whitespace or separator/,
      secret: 'SECRET',
    },
    {
      label: 'a value that would need S3 offload',
      document: { key: 'docs/a', data: randomBytes(380_001), text: 'ok' },
      error: /S3 offload is not supported for indexed documents/,
    },
    {
      label: 'a document id above 1024 bytes',
      document: doc(`docs/${'k'.repeat(1020)}`, 'ok'),
      error: /sort-key limit of 1024 bytes/,
    },
    {
      label: 'a non-finite vector',
      document: doc('docs/a', 'ok', { vector: [Number.NaN] }),
      error: /non-finite/,
    },
    {
      label: 'a key with a .. segment',
      document: doc('docs/../a', 'ok'),
      error: /'\.\.' path segments are not allowed/,
    },
    {
      label: 'text with a lone surrogate',
      document: doc('docs/a', `secret ${LONE_SURROGATE} text`),
      error: /^Text must be well-formed Unicode; it contains a lone surrogate$/,
      secret: 'secret',
    },
    {
      label: 'an identifier with a lone surrogate',
      document: doc('docs/a', 'ok', { identifiers: [`SECRET-${LONE_SURROGATE}`] }),
      error: /^Identifier must be well-formed Unicode; it contains a lone surrogate$/,
      secret: 'SECRET',
    },
  ]

  it.each(invalidDocuments)('rejects $label before any I/O', async ({ document, error, secret }) => {
    const { client, index } = setup()
    const caught = await index.upsert(document).catch((rejection: unknown) => rejection)
    expect(caught).toBeInstanceOf(StorageError)
    expect((caught as Error).message).toMatch(error)
    if (secret) expect((caught as Error).message).not.toContain(secret)
    expect(client.sent).toEqual([])
    expect([...client.rows(BASE_TABLE), ...client.rows(INDEX_TABLE)]).toEqual([])
  })

  it('rejects an S3-size document even when an offload bucket is configured', async () => {
    const { client, s3, index } = setup({ s3: true })
    await expect(index.upsert({ key: 'docs/a', data: randomBytes(380_001), text: 'ok' })).rejects.toThrow(
      /S3 offload is not supported for indexed documents/
    )
    expect(client.sent).toEqual([])
    expect(s3.objects.size).toBe(0)
  })

  const invalidQueries: Array<{ label: string; run: (index: LexicalIndex) => Promise<unknown>; error: RegExp }> = [
    {
      label: 'a query without terms',
      run: (index) => index.search({ text: '-- !! __', topK: 1 }),
      error: /no searchable terms/,
    },
    {
      label: 'more terms than maxQueryTerms',
      run: (index) => index.search({ text: distinctWords(17), topK: 1 }),
      error: /17 distinct terms, above maxQueryTerms \(16\)/,
    },
    {
      label: 'a query term above maxTermBytes',
      run: (index) => index.search({ text: 'q'.repeat(65), topK: 1 }),
      error: /A query term is 65 bytes/,
    },
    { label: 'topK 0', run: (index) => index.search({ text: 'alpha', topK: 0 }), error: /topK/ },
    { label: 'topK 101', run: (index) => index.search({ text: 'alpha', topK: 101 }), error: /topK/ },
    { label: 'a fractional topK', run: (index) => index.search({ text: 'alpha', topK: 1.5 }), error: /topK/ },
    { label: 'lookup topK 101', run: (index) => index.lookup({ identifier: 'A', topK: 101 }), error: /topK/ },
    {
      label: 'an invalid lookup identifier',
      run: (index) => index.lookup({ identifier: 'A-1 ', topK: 1 }),
      error: /whitespace or separator/,
    },
    {
      label: 'a lookup identifier above maxIdentifierBytes',
      run: (index) => index.lookup({ identifier: 'I'.repeat(129), topK: 1 }),
      error: /The identifier is 129 bytes, above maxIdentifierBytes \(128\)/,
    },
    { label: 'a non-positive repair batch', run: (index) => index.repair({ maxDocuments: 0 }), error: /maxDocuments/ },
    {
      label: 'a lookup identifier with a control character',
      run: (index) => index.lookup({ identifier: `ID${String.fromCharCode(9)}1`, topK: 1 }),
      error: /control characters/,
    },
    {
      label: 'a query with a lone surrogate',
      run: (index) => index.search({ text: `alpha ${LONE_SURROGATE}`, topK: 1 }),
      error: /^Text must be well-formed Unicode; it contains a lone surrogate$/,
    },
    {
      label: 'a lookup identifier with a lone surrogate',
      run: (index) => index.lookup({ identifier: `ID-${LONE_SURROGATE}`, topK: 1 }),
      error: /^Identifier must be well-formed Unicode; it contains a lone surrogate$/,
    },
    {
      label: 'a delete of a key above 1024 bytes',
      run: (index) => index.delete(`docs/${'k'.repeat(1020)}`),
      error: /sort-key limit of 1024 bytes/,
    },
    {
      label: 'a revision of a key above 1024 bytes',
      run: (index) => index.revision(`docs/${'k'.repeat(1020)}`),
      error: /sort-key limit of 1024 bytes/,
    },
  ]

  it.each(invalidQueries)('rejects $label before any I/O', async ({ run, error }) => {
    const { client, index } = setup()
    await expect(run(index)).rejects.toThrow(error)
    expect(client.sent).toEqual([])
  })

  it.each<[Partial<LexicalIndexLimits>, RegExp]>([
    [{ maxPostingsPerDocument: 50 }, /'maxPostingsPerDocument' must be <= 49/],
    [{ pageSize: 1001 }, /'pageSize' must be <= 1000/],
    [{ maxCandidates: 1001 }, /'maxCandidates' must be <= 1000/],
    [{ maxConcurrency: 17 }, /'maxConcurrency' must be <= 16/],
    [{ maxTermBytes: 0 }, /'maxTermBytes' must be an integer >= 1/],
    [{ maxTextBytes: 1.5 }, /'maxTextBytes' must be an integer >= 1/],
    [{ maxConflictRetries: -1 }, /'maxConflictRetries' must be an integer >= 0/],
    [{ pageSize: true as unknown as number }, /'pageSize' must be an integer >= 1; got true/],
    [{ maxUnprocessedRetries: false as unknown as number }, /'maxUnprocessedRetries' must be an integer >= 0/],
    [{ maxPostingsPerDocument: 10, maxQueryTerms: 11 }, /'maxQueryTerms' \(11\) must be <= maxPostingsPerDocument/],
  ])('rejects the limits %j at construction', (limits, error) => {
    expect(() => setup({ limits })).toThrow(error)
  })

  it('accepts zero retries and merges partial limits over the defaults', () => {
    expect(() => setup({ limits: { maxConflictRetries: 0, maxUnprocessedRetries: 0 } })).not.toThrow()
    expect(Object.isFrozen(DEFAULT_LEXICAL_INDEX_LIMITS)).toBe(true)
  })

  it('rejects an empty index table name at construction', () => {
    const { client, storage } = setup()
    const construct = () => new LexicalIndex(storage, { indexTableName: '' })
    expect(construct).toThrow(StorageError)
    expect(construct).toThrow('LexicalIndex requires a non-empty indexTableName')
    expect(client.sent).toEqual([])
  })

  it.each(RESERVED_INDEX_ATTRIBUTES)('rejects a storage whose TTL attribute is the index attribute %j', (name) => {
    const storage = new DynamoDBStorage(BASE_TABLE, {
      client: asDocClient(new FakeDocumentClient()),
      ttlSeconds: 60,
      ttlAttribute: name,
    })
    const construct = () => new LexicalIndex(storage, { indexTableName: INDEX_TABLE })
    expect(construct).toThrow(StorageError)
    expect(construct).toThrow(`Storage TTL attribute '${name}' is reserved by the lexical index`)
  })

  it('accepts an index attribute name as the TTL attribute of a storage without TTL', () => {
    const storage = new DynamoDBStorage(BASE_TABLE, {
      client: asDocClient(new FakeDocumentClient()),
      ttlAttribute: 'rev',
    })
    expect(() => new LexicalIndex(storage, { indexTableName: INDEX_TABLE })).not.toThrow()
  })

  it('rejects a posting partition key above 2048 bytes before any I/O', async () => {
    const { client, index } = setup({ limits: { maxTermBytes: 4096, maxIdentifierBytes: 4096 } })
    const oversized = 'q'.repeat(2040)
    expect(termPostingPk('', oversized)).toHaveLength(2049)
    const attempts: Array<() => Promise<unknown>> = [
      () => index.upsert(doc('docs/a', oversized)),
      () => index.upsert(doc('docs/a', 'ok', { identifiers: [oversized] })),
      () => index.search({ text: oversized, topK: 1 }),
      () => index.lookup({ identifier: oversized, topK: 1 }),
    ]
    for (const attempt of attempts) {
      const error = await attempt().catch((caught: unknown) => caught)
      expect(error).toBeInstanceOf(StorageError)
      expect((error as Error).message).toBe(
        'A lexical index partition key is above the DynamoDB partition-key limit of 2048 bytes'
      )
    }
    expect(client.sent).toEqual([])
  })

  it('accepts a posting partition key of exactly 2048 bytes', async () => {
    const { index } = setup({ limits: { maxTermBytes: 4096 } })
    const term = 'q'.repeat(2039)
    expect(termPostingPk('', term)).toHaveLength(2048)
    await index.upsert(doc('docs/a', term))
    expect(keysOf(await index.search({ text: term, topK: 1 }))).toEqual(['docs/a'])
  })

  it('rejects a missing index table name and a scope above 1024 bytes', () => {
    const { storage } = setup()
    expect(() => new LexicalIndex(storage, { indexTableName: '' })).toThrow(/indexTableName/)
    expect(() => scopedIndex(storage, 's'.repeat(1024))).toThrow(/1025 bytes, above the 1024-byte limit/)
  })
})

describe('LexicalIndex — retrieval bounds', () => {
  it('reports max_candidates while existing candidates keep accumulating matches', async () => {
    const { index } = setup({ limits: { maxCandidates: 2 } })
    await index.upsert(doc('docs/a', 'alpha beta'))
    await index.upsert(doc('docs/b', 'alpha'))
    await index.upsert(doc('docs/c', 'alpha beta'))
    const response = await index.search({ text: 'alpha beta', topK: 10 })
    expect(response.results.map((result) => [result.key, result.matchedTerms])).toEqual([
      ['docs/a', 2],
      ['docs/b', 1],
    ])
    expect(response).toMatchObject({ truncated: true, truncationReasons: ['max_candidates'], candidatesExamined: 2 })
  })

  it('reports max_pages_per_term when postings remain after the last page', async () => {
    const { client, index } = setup({ limits: { pageSize: 1, maxPagesPerTerm: 2 } })
    for (const key of ['docs/a', 'docs/b', 'docs/c']) await index.upsert(doc(key, 'alpha'))
    client.clearLog()
    const response = await index.search({ text: 'alpha', topK: 10 })
    expect(keysOf(response)).toEqual(['docs/a', 'docs/b'])
    expect(response).toMatchObject({
      truncated: true,
      truncationReasons: ['max_pages_per_term'],
      candidatesExamined: 2,
    })
    expect(client.sentOf(QueryCommand)).toHaveLength(2)
  })

  it('sorts and de-duplicates truncation reasons', async () => {
    const { index } = setup({ limits: { pageSize: 1, maxPagesPerTerm: 2, maxCandidates: 1 } })
    for (const key of ['docs/a', 'docs/b', 'docs/c']) await index.upsert(doc(key, 'alpha beta'))
    const response = await index.search({ text: 'alpha beta', topK: 10 })
    expect(response.truncationReasons).toEqual(['max_candidates', 'max_pages_per_term'])
    expect(response.candidatesExamined).toBe(1)
  })

  it('validates candidates in BatchGet chunks of at most 100 keys', async () => {
    const { client, index } = setup()
    for (let number = 0; number < 101; number++) await index.upsert(doc(`docs/${number}`, 'alpha'))
    client.clearLog()
    const response = await index.search({ text: 'alpha', topK: 100 })
    expect(response.results).toHaveLength(100)
    expect(response.candidatesExamined).toBe(101)
    const chunkSizes = client
      .sentOf(BatchGetCommand)
      .map((command) => command.input.RequestItems?.[BASE_TABLE]?.Keys?.length)
      .sort()
    expect(chunkSizes).toEqual([1, 100])
  })

  it('retries unprocessed keys until the validation read completes', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    await index.upsert(doc('docs/b', 'alpha'))
    client.unprocessedRounds = 1
    client.clearLog()
    expect(keysOf(await index.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/a', 'docs/b'])
    expect(client.sentOf(BatchGetCommand)).toHaveLength(2)
  })

  it('counts only consecutive rounds without progress against maxUnprocessedRetries', async () => {
    const { client, index } = setup({ limits: { maxUnprocessedRetries: 1 } })
    for (const key of ['docs/a', 'docs/b', 'docs/c']) await index.upsert(doc(key, 'alpha'))
    client.unprocessedRounds = 2
    client.clearLog()
    expect(keysOf(await index.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/a', 'docs/b', 'docs/c'])
    expect(
      client.sentOf(BatchGetCommand).map((command) => command.input.RequestItems?.[BASE_TABLE]?.Keys?.length)
    ).toEqual([3, 1, 1])
    client.unprocessedRounds = 3
    client.clearLog()
    await expect(index.search({ text: 'alpha', topK: 5 })).rejects.toThrow(
      `Lexical index read left 1 keys unprocessed in base table '${BASE_TABLE}' after 1 retries without progress`
    )
    expect(client.sentOf(BatchGetCommand)).toHaveLength(3)
  })

  it('fails instead of answering partially when unprocessed keys remain', async () => {
    const { client, index } = setup({ limits: { maxUnprocessedRetries: 2 } })
    await index.upsert(doc('docs/a', 'alpha'))
    client.unprocessedRounds = Number.POSITIVE_INFINITY
    client.clearLog()
    await expect(index.search({ text: 'alpha', topK: 5 })).rejects.toThrow(/1 keys unprocessed .* after 2 retries/)
    expect(client.sentOf(BatchGetCommand)).toHaveLength(3)
  })
})

describe('LexicalIndex — repair', () => {
  it('resumes from the cursor and reads manifests strongly consistently', async () => {
    const { client, index } = setup()
    for (const key of ['docs/a', 'docs/b', 'docs/c']) await index.upsert(doc(key, 'alpha'))
    client.clearLog()
    const first = await index.repair({ maxDocuments: 2 })
    expect(first).toEqual({ documentsChecked: 2, documentsRemoved: 0, postingsRebuilt: 0, cursor: 'docs/b' })
    const second = await index.repair({ maxDocuments: 2, cursor: first.cursor })
    expect(second).toEqual({ documentsChecked: 1, documentsRemoved: 0, postingsRebuilt: 0, cursor: null })
    expect(client.sentOf(QueryCommand).map((command) => command.input.ConsistentRead)).toEqual([true, true])
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(0)
  })

  it('rebuilds lost postings with the manifest revision and the base TTL', async () => {
    const { client, index } = setup({ ttlSeconds: 60 })
    const rev = await index.upsert(doc('docs/a', 'alpha beta'))
    client.deleteRow(INDEX_TABLE, termPostingPk('', 'beta'), 'docs/a')
    expect(keysOf(await index.search({ text: 'beta', topK: 1 }))).toEqual([])
    client.clearLog()
    expect(await index.repair({ rebuildPostings: true })).toEqual({
      documentsChecked: 1,
      documentsRemoved: 0,
      postingsRebuilt: 1,
      cursor: null,
    })
    const [transaction] = client.sentOf(TransactWriteCommand)
    expect(transaction?.input.TransactItems?.map(describeAction)[0]).toBe(`ConditionCheck ${INDEX_TABLE} m|0:`)
    expect(transaction?.input).not.toHaveProperty('ClientRequestToken')
    expect(client.row(INDEX_TABLE, termPostingPk('', 'beta'), 'docs/a')).toEqual({
      pk: termPostingPk('', 'beta'),
      sk: 'docs/a',
      rev,
      expireAt: baseRow(client, 'docs/a')?.expireAt,
    })
    expect(keysOf(await index.search({ text: 'beta', topK: 1 }))).toEqual(['docs/a'])
    expectIndexConsistent(client, '')
  })

  it('skips a document modified concurrently and still counts it as checked', async () => {
    const { client, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    client.beforeNextTransaction(async () => {
      await index.upsert(doc('docs/a', 'beta'))
    })
    expect(await index.repair({ rebuildPostings: true })).toEqual({
      documentsChecked: 1,
      documentsRemoved: 0,
      postingsRebuilt: 0,
      cursor: null,
    })
    expect(keysOf(await index.search({ text: 'beta', topK: 1 }))).toEqual(['docs/a'])
    expectIndexConsistent(client, '')
  })

  it('treats an empty cursor as the start of the manifests', async () => {
    const { index } = setup()
    for (const key of ['docs/a', 'docs/b']) await index.upsert(doc(key, 'alpha'))
    expect(await index.repair({ maxDocuments: 1, cursor: '' })).toEqual({
      documentsChecked: 1,
      documentsRemoved: 0,
      postingsRebuilt: 0,
      cursor: 'docs/a',
    })
  })

  it('skips a document whose removal is cancelled by a concurrent writer', async () => {
    const { client, storage, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    await storage.write('docs/a', bytes('bypass'))
    client.transactionFailure = cancellation('None', 'ConditionalCheckFailed')
    client.clearLog()
    expect(await index.repair()).toEqual({ documentsChecked: 1, documentsRemoved: 0, postingsRebuilt: 0, cursor: null })
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(1)
    expect(client.row(INDEX_TABLE, manifestPk(''), 'docs/a')).toBeDefined()
  })

  it('repairs a namespaced scope without touching valid documents or other scopes', async () => {
    const { client, storage } = setup()
    const tenantStorage = storage.namespace('tenant-a')
    const tenant = scopedIndex(storage, 'tenant-a')
    const other = scopedIndex(storage, 'tenant-b')
    await tenant.upsert(doc('docs/kept', 'alpha'))
    await tenant.upsert(doc('docs/bypassed', 'alpha'))
    await other.upsert(doc('docs/bypassed', 'alpha'))
    await tenantStorage.write('docs/bypassed', bytes('bypass'))
    client.clearLog()
    expect(await tenant.repair()).toEqual({
      documentsChecked: 2,
      documentsRemoved: 1,
      postingsRebuilt: 0,
      cursor: null,
    })
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(1)
    expect(keysOf(await tenant.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/kept'])
    expect(keysOf(await other.search({ text: 'alpha', topK: 5 }))).toEqual(['docs/bypassed'])
    expectIndexConsistent(client, 'tenant-a/')
    expectIndexConsistent(client, 'tenant-b/')
  })

  it('processes manifests sequentially and stops at a throttled transaction without retrying it', async () => {
    const { client, storage, index } = setup()
    for (const key of ['docs/a', 'docs/b']) await index.upsert(doc(key, 'alpha'))
    for (const key of ['docs/a', 'docs/b']) await storage.write(key, bytes('bypass'))
    client.transactionFailure = cancellation('None', 'ProvisionedThroughputExceeded')
    client.clearLog()
    const error = await index.repair().catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(StorageError)
    expect((error as Error).message).toBe(
      `Lexical index repair failed (index table '${INDEX_TABLE}', base table '${BASE_TABLE}')`
    )
    expect(client.sentOf(QueryCommand)).toHaveLength(1)
    expect(client.sentOf(GetCommand)).toHaveLength(1)
    expect(client.sentOf(TransactWriteCommand)).toHaveLength(1)
  })

  it('fails with StorageError on a non-conflict transaction failure', async () => {
    const { client, storage, index } = setup()
    await index.upsert(doc('docs/a', 'alpha'))
    await storage.write('docs/a', bytes('bypass'))
    client.transactionFailure = cancellation('ThrottlingError')
    await expect(index.repair()).rejects.toThrow(
      `Lexical index repair failed (index table '${INDEX_TABLE}', base table '${BASE_TABLE}')`
    )
  })
})

describe('DynamoDBStorage — document item port', () => {
  it('builds the same inline item write() stores', async () => {
    const { client, storage } = setup({ compression: 'gzip', ttlSeconds: 60 })
    const options = { vector: [0.5, 1], metadata: { kind: 'note' }, ttlSeconds: 30 }
    const data = bytes('x'.repeat(2048))
    await storage.write('docs/a', data, options)
    const port = storage._documentItemPort()
    const item = await port.inlineItem(port.locate('docs/a'), data, options)
    expect(structuredClone(item)).toEqual(baseRow(client, 'docs/a'))
    expect(item.z).toBe(true)
  })

  it('exposes the base table, scope, TTL attribute and key mapping', () => {
    const { storage } = setup({ prefix: 'tenant/a', ttlSeconds: 60 })
    const port = storage._documentItemPort()
    expect(port).toMatchObject({ tableName: BASE_TABLE, scope: 'tenant/a/', ttlAttribute: 'expireAt' })
    expect(port.locate('//docs//x/')).toEqual({ key: 'docs/x', docId: 'tenant/a/docs/x', pk: 'tenant/a', sk: 'docs/x' })
    expect(port.relativeKey('tenant/a/docs/x')).toBe('docs/x')
    expect(port.relativeKey('tenant/b/docs/x')).toBeNull()
    expect(setup().storage._documentItemPort().ttlAttribute).toBeUndefined()
  })

  it('reports expiry only when the storage opted in to TTL', () => {
    const expired = { expireAt: Math.floor(FROZEN_NOW_MS / 1000) }
    expect(setup({ ttlSeconds: 60 }).storage._documentItemPort().isExpired(expired)).toBe(true)
    expect(setup().storage._documentItemPort().isExpired(expired)).toBe(false)
  })

  it('keeps the port and tokenizer internals out of the package barrel', () => {
    expect(Object.keys(pkgIndex).sort()).toEqual([
      'DEFAULT_LEXICAL_INDEX_LIMITS',
      'DynamoDBStorage',
      'LEXICAL_TOKENIZER_VERSION',
      'LexicalIndex',
      'LexicalSearchStrategy',
      'RevisionConflictError',
    ])
  })
})
