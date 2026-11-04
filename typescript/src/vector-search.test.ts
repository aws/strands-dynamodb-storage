// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'
import { SearchVectorsCommand, type SearchVectorsCommandInput } from '@aws-sdk/client-dynamodb'
import { GetCommand, PutCommand, DeleteCommand, type DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb'
import { marshall } from '@aws-sdk/util-dynamodb'
import { StorageError } from '@strands-agents/sdk'
import { Embedder as SdkEmbedder } from '@strands-agents/sdk/storage/search'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { DynamoDBStorage } from './dynamodb-storage.js'
import { BedrockEmbedder, type EmbeddingPurpose } from './embedders.js'
import { DynamoDBVectorSearchStrategy } from './vector-search.js'

interface StoredItem {
  pk: string
  sk: string
  k?: string
  vector?: number[]
  [attribute: string]: unknown
}

/** In-memory Document client covering Put/Get/Delete and SearchVectors with COSINE distance, as the service scores it. */
class FakeDocumentClient {
  readonly items = new Map<string, StoredItem>()
  readonly searches: SearchVectorsCommandInput[] = []
  async send(command: unknown): Promise<unknown> {
    if (command instanceof SearchVectorsCommand) {
      const input = command.input
      this.searches.push(input)
      const query = input.SearchVector!.map((av) => Number(av.N))
      const pk = input.ExpressionAttributeValues?.[':pk']?.S
      const wanted = new Set(
        (input.ProjectionExpression ?? '').split(',').map((alias) => input.ExpressionAttributeNames![alias.trim()]!)
      )
      const results = [...this.items.values()]
        .filter((item): item is StoredItem & { vector: number[] } => Array.isArray(item.vector))
        .filter((item) => pk === undefined || item.pk === pk)
        .map((item) => {
          const dot = item.vector.reduce((acc, x, i) => acc + x * (query[i] ?? 0), 0)
          const norm =
            Math.sqrt(item.vector.reduce((a, x) => a + x * x, 0)) * Math.sqrt(query.reduce((a, x) => a + x * x, 0))
          const projected = Object.fromEntries(Object.entries(item).filter(([k]) => wanted.has(k)))
          return { Item: marshall(projected), Score: 1 - (norm === 0 ? 0 : dot / norm) }
        })
        .sort((a, b) => a.Score - b.Score)
        .slice(0, input.TopK)
      return { SearchResults: results }
    }
    if (command instanceof PutCommand) {
      const item = command.input.Item as StoredItem
      marshall(item)
      this.items.set(`${item.pk}\u0000${item.sk}`, item)
      return {}
    }
    if (command instanceof GetCommand) {
      const k = command.input.Key as { pk: string; sk: string }
      return { Item: this.items.get(`${k.pk}\u0000${k.sk}`) }
    }
    if (command instanceof DeleteCommand) {
      const k = command.input.Key as { pk: string; sk: string }
      this.items.delete(`${k.pk}\u0000${k.sk}`)
      return {}
    }
    throw new Error(`unexpected command ${(command as object | undefined)?.constructor?.name}`)
  }
}

const VOCABULARY = ['dark', 'mode', 'coffee', 'tea']

/** One dimension per vocabulary word, plus a floor so no vector is zero. */
function words(text: string): number[] {
  const tokens = text.toLowerCase().split(/\s+/)
  return VOCABULARY.map((word) => 0.01 + tokens.filter((token) => token === word).length)
}

/** Records `[text, purpose]` for each call, the package's own embedder shape. */
class PurposeEmbedder {
  readonly calls: Array<[string, EmbeddingPurpose]> = []
  async embed(text: string, purpose: EmbeddingPurpose): Promise<number[]> {
    this.calls.push([text, purpose])
    return words(text)
  }
}

const bytes = (s: string) => new TextEncoder().encode(s)

function setup(strategy: DynamoDBVectorSearchStrategy) {
  const client = new FakeDocumentClient()
  const storage = new DynamoDBStorage('test-table', {
    client: client as unknown as DynamoDBDocumentClient,
    searchStrategy: strategy,
  })
  return { client, storage, item: (pk: string, sk: string) => client.items.get(`${pk}\u0000${sk}`) }
}

describe('DynamoDBVectorSearchStrategy: construction', () => {
  it.each([0, 101, 1.5])('rejects maxResults %s', (maxResults) => {
    expect(() => new DynamoDBVectorSearchStrategy({ maxResults })).toThrow(RangeError)
  })

  it('rejects an unknown distance function', () => {
    expect(() => new DynamoDBVectorSearchStrategy({ distanceFunction: 'MANHATTAN' as never })).toThrow(
      /distanceFunction/
    )
  })
})

describe('DynamoDBVectorSearchStrategy: write', () => {
  it('stores the vector on the same item', async () => {
    const embedder = new PurposeEmbedder()
    const { storage, item } = setup(new DynamoDBVectorSearchStrategy({ embedder }))
    await storage.namespace('memory/facts').write('prefs.md', bytes('dark mode'))
    expect(item('memory/facts', 'prefs.md')?.vector).toEqual(words('dark mode'))
    expect(embedder.calls).toEqual([['dark mode', 'document']])
  })

  it('stores a caller vector as given', async () => {
    const embedder = new PurposeEmbedder()
    const { storage, item } = setup(new DynamoDBVectorSearchStrategy({ embedder }))
    await storage.write('memory/facts/a', bytes('dark mode'), { vector: [1, 0, 0, 0] })
    expect(item('memory/facts', 'a')?.vector).toEqual([1, 0, 0, 0])
    expect(embedder.calls).toEqual([])
  })

  it.each([new Uint8Array(), bytes('   '), new Uint8Array([0xff, 0xfe])])(
    'stores an empty or binary value without a vector, clearing the old one',
    async (data) => {
      const embedder = new PurposeEmbedder()
      const { storage, item } = setup(new DynamoDBVectorSearchStrategy({ embedder }))
      await storage.write('memory/facts/a', bytes('dark mode'))
      await storage.write('memory/facts/a', data)
      expect(item('memory/facts', 'a')?.vector).toBeUndefined()
      expect(await storage.read('memory/facts/a')).toEqual(data)
      expect(embedder.calls).toEqual([['dark mode', 'document']])
    }
  )

  it('writes nothing when embedding fails', async () => {
    const cause = new Error('model down')
    const { storage, item } = setup(
      new DynamoDBVectorSearchStrategy({
        embedder: async () => {
          throw cause
        },
      })
    )
    const write = storage.write('memory/facts/a', bytes('dark mode'))
    await expect(write).rejects.toThrow("Failed to embed 'memory/facts/a' for search, so nothing was written")
    await expect(write).rejects.toBeInstanceOf(StorageError)
    expect(item('memory/facts', 'a')).toBeUndefined()
  })

  it('writes nothing when the embedder returns no vector', async () => {
    const { storage, item } = setup(new DynamoDBVectorSearchStrategy({ embedder: () => [] }))
    await expect(storage.write('memory/facts/a', bytes('dark mode'))).rejects.toThrow(/nothing was written/)
    expect(item('memory/facts', 'a')).toBeUndefined()
  })
})

describe('DynamoDBVectorSearchStrategy: index', () => {
  it('is not called by write, which embeds once', async () => {
    const embedder = new PurposeEmbedder()
    const strategy = new DynamoDBVectorSearchStrategy({ embedder })
    strategy.index = async () => {
      throw new Error('write must not call index for this strategy')
    }
    await setup(strategy).storage.write('memory/facts/a', bytes('dark mode'))
    expect(embedder.calls).toEqual([['dark mode', 'document']])
  })

  it('backfills a value written before the strategy', async () => {
    const client = new FakeDocumentClient()
    const asClient = client as unknown as DynamoDBDocumentClient
    await new DynamoDBStorage('test-table', { client: asClient }).write('memory/facts/a', bytes('dark mode'))
    expect(client.items.get('memory/facts\u0000a')?.vector).toBeUndefined()
    const strategy = new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder() })
    const view = new DynamoDBStorage('test-table', { client: asClient, searchStrategy: strategy }).namespace(
      'memory/facts'
    )
    await strategy.index(view, 'a', (await view.read('a'))!)
    expect(client.items.get('memory/facts\u0000a')?.vector).toEqual(words('dark mode'))
  })
})

describe('DynamoDBVectorSearchStrategy: search', () => {
  it('ranks by meaning with higher scores closer and returns the bytes', async () => {
    const embedder = new PurposeEmbedder()
    const view = setup(new DynamoDBVectorSearchStrategy({ embedder })).storage.namespace('memory/facts')
    await view.write('prefs.md', bytes('dark mode'))
    await view.write('drinks.md', bytes('coffee tea'))
    const found = await view.search('dark')
    expect(found.map((r) => [r.key, new TextDecoder().decode(r.data)])).toEqual([
      ['prefs.md', 'dark mode'],
      ['drinks.md', 'coffee tea'],
    ])
    expect(found[0]!.score).toBeGreaterThan(found[1]!.score)
    expect(found.every((r) => r.score > 0 && r.score <= 1)).toBe(true)
    expect(embedder.calls.at(-1)).toEqual(['dark', 'query'])
  })

  it('is pinned to the storage partition', async () => {
    const { client, storage } = setup(new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder() }))
    await storage.write('memory/facts/a', bytes('dark mode'))
    await storage.write('memory/other/a', bytes('dark mode'))
    const found = await storage.namespace('memory/facts').search('dark')
    expect(client.searches.at(-1)?.ExpressionAttributeValues?.[':pk']?.S).toBe('memory/facts')
    expect(found.map((r) => r.key)).toEqual(['a'])
  })

  it('searches unpinned on a storage shallower than a partition', async () => {
    const { client, storage } = setup(new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder() }))
    await storage.write('memory/facts/a', bytes('dark mode'))
    await storage.namespace('memory').search('dark')
    expect(client.searches.at(-1)?.ExpressionAttributeValues?.[':pk']).toBeUndefined()
  })

  it('bounds the search by maxResults', async () => {
    const { client, storage } = setup(
      new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder(), maxResults: 1 })
    )
    const view = storage.namespace('memory/facts')
    await view.write('a', bytes('dark mode'))
    await view.write('b', bytes('dark'))
    expect(await view.search('dark')).toHaveLength(1)
    expect(client.searches.at(-1)?.TopK).toBe(1)
  })

  it('returns nothing for a blank query without embedding', async () => {
    const embedder = new PurposeEmbedder()
    const view = setup(new DynamoDBVectorSearchStrategy({ embedder })).storage.namespace('memory/facts')
    expect(await view.search('   ')).toEqual([])
    expect(embedder.calls).toEqual([])
  })

  it('stops matching a deleted value', async () => {
    const view = setup(new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder() })).storage.namespace(
      'memory/facts'
    )
    await view.write('a', bytes('dark mode'))
    await view.delete('a')
    expect(await view.search('dark')).toEqual([])
  })

  it('returns DOT_PRODUCT scores as the index gives them', async () => {
    const storage = new DynamoDBStorage('test-table', {
      client: new FakeDocumentClient() as unknown as DynamoDBDocumentClient,
      searchStrategy: new DynamoDBVectorSearchStrategy({
        embedder: new PurposeEmbedder(),
        distanceFunction: 'DOT_PRODUCT',
      }),
      vectorSearch: async () => [{ key: 'memory/facts/a', score: 7.5 }],
    })
    expect((await storage.namespace('memory/facts').search('dark')).map((r) => r.score)).toEqual([7.5])
  })
})

describe('DynamoDBVectorSearchStrategy: embedder shapes', () => {
  it('calls an SDK Embedder with the text', async () => {
    const calls: string[] = []
    class Sdk extends SdkEmbedder {
      async embed(text: string): Promise<number[]> {
        calls.push(text)
        return words(text)
      }
    }
    const view = setup(new DynamoDBVectorSearchStrategy({ embedder: new Sdk() })).storage.namespace('memory/facts')
    await view.write('a', bytes('dark mode'))
    expect((await view.search('dark')).map((r) => r.key)).toEqual(['a'])
    expect(calls).toEqual(['dark mode', 'dark'])
  })

  it('embeds values and queries with a plain function', async () => {
    const view = setup(new DynamoDBVectorSearchStrategy({ embedder: words })).storage.namespace('memory/facts')
    await view.write('a', bytes('dark mode'))
    expect((await view.search('dark')).map((r) => r.key)).toEqual(['a'])
  })

  it('accepts a BedrockEmbedder where an SDK Embedder is expected', () => {
    const sdkShaped: SdkEmbedder = new BedrockEmbedder({ region: 'us-east-1' })
    expect(typeof sdkShaped.embed).toBe('function')
  })
})

describe('DynamoDBVectorSearchStrategy: default embedder', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('is a BedrockEmbedder, created on first use', async () => {
    const embed = vi.spyOn(BedrockEmbedder.prototype, 'embed').mockImplementation(async (text) => words(text))
    const view = setup(new DynamoDBVectorSearchStrategy()).storage.namespace('memory/facts')
    await view.write('a', bytes('dark mode'))
    expect((await view.search('dark')).map((r) => r.key)).toEqual(['a'])
    expect(embed.mock.calls).toEqual([
      ['dark mode', 'document'],
      ['dark', 'query'],
    ])
  })
})

describe('DynamoDBVectorSearchStrategy: FileMemoryStore', () => {
  it('recalls by meaning', async () => {
    const { storage } = setup(new DynamoDBVectorSearchStrategy({ embedder: new PurposeEmbedder() }))
    const knowledge = new FileMemoryStore({ name: 'facts', storage, writable: true })
    await knowledge.add('# Display\nprefers dark mode')
    await knowledge.add('# Drinks\nlikes coffee and tea')
    const found = await knowledge.search('dark')
    expect(found[0]?.content.startsWith('# Display')).toBe(true)
  })
})
