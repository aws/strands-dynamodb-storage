// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime'
import { StorageError } from '@strands-agents/sdk'
import { BedrockEmbedder, DEFAULT_EMBEDDING_MODEL_ID, type BedrockModelFamily } from './embedders.js'

/** Records InvokeModelCommand inputs and replies with a canned JSON body. */
class FakeBedrock {
  readonly calls: Array<Record<string, unknown>> = []
  constructor(private readonly reply: unknown) {}
  async send(command: unknown): Promise<unknown> {
    if (!(command instanceof InvokeModelCommand)) throw new Error('unexpected command')
    this.calls.push(command.input as Record<string, unknown>)
    if (this.reply instanceof Error) throw this.reply
    return { body: new TextEncoder().encode(JSON.stringify(this.reply)) }
  }
  body(index = -1): unknown {
    return JSON.parse(this.calls.at(index)!.body as string)
  }
}

const asClient = (fake: FakeBedrock) => fake as unknown as BedrockRuntimeClient
const titanReply = (dims = 1024) => ({
  embedding: Array(dims).fill(0.5),
  embeddingsByType: { float: Array(dims).fill(0.5) },
  inputTextTokenCount: 3,
})
const cohereReply = (dims = 1024) => ({
  id: 'x',
  texts: ['t'],
  embeddings: { float: [Array(dims).fill(0.25)] },
  response_type: 'embeddings_by_type',
})

describe('BedrockEmbedder: request formats', () => {
  it('calls Titan Text Embeddings V2 by default', async () => {
    const fake = new FakeBedrock(titanReply())
    const vector = await new BedrockEmbedder({ client: asClient(fake) }).embed('Prefers a window seat', 'query')
    expect(vector).toEqual(Array(1024).fill(0.5))
    expect(fake.calls[0]).toMatchObject({
      modelId: 'amazon.titan-embed-text-v2:0',
      contentType: 'application/json',
      accept: 'application/json',
    })
    expect(fake.body()).toEqual({ inputText: 'Prefers a window seat', dimensions: 1024, normalize: true })
  })

  it.each([[256], [512], [1024]])('requests %i Titan dimensions', async (dims) => {
    const fake = new FakeBedrock(titanReply(dims))
    expect(await new BedrockEmbedder({ dimensions: dims, client: asClient(fake) }).embed('text')).toHaveLength(dims)
    expect((fake.body() as { dimensions: number }).dimensions).toBe(dims)
  })

  it.each([['cohere.embed-english-v3'], ['cohere.embed-multilingual-v3']])(
    'uses Cohere input types and no output dimension for %s',
    async (modelId) => {
      const fake = new FakeBedrock(cohereReply())
      const embedder = new BedrockEmbedder({ modelId, client: asClient(fake) })
      expect(await embedder.embed('Prefers a window seat', 'document')).toEqual(Array(1024).fill(0.25))
      expect(fake.body()).toEqual({
        texts: ['Prefers a window seat'],
        input_type: 'search_document',
        embedding_types: ['float'],
      })
      await embedder.embed('which seat?', 'query')
      expect((fake.body() as { input_type: string }).input_type).toBe('search_query')
    }
  )

  it.each([['cohere.embed-v4:0'], ['us.cohere.embed-v4:0'], ['global.cohere.embed-v4:0']])(
    'sends output_dimension for %s',
    async (modelId) => {
      const fake = new FakeBedrock(cohereReply(1536))
      const embedder = new BedrockEmbedder({ modelId, dimensions: 1536, client: asClient(fake) })
      expect(await embedder.embed('text', 'query')).toHaveLength(1536)
      expect(fake.calls.at(-1)!.modelId).toBe(modelId)
      expect(fake.body()).toEqual({
        texts: ['text'],
        input_type: 'search_query',
        embedding_types: ['float'],
        output_dimension: 1536,
      })
    }
  )

  it('defaults to the document purpose', async () => {
    const fake = new FakeBedrock(cohereReply())
    await new BedrockEmbedder({ modelId: 'cohere.embed-english-v3', client: asClient(fake) }).embed('text')
    expect((fake.body() as { input_type: string }).input_type).toBe('search_document')
  })
})

describe('BedrockEmbedder: model IDs and dimensions', () => {
  it.each([
    ['amazon.titan-embed-text-v2:0'],
    ['amazon.titan-embed-text-v2:0:8k'],
    ['arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0'],
    ['cohere.embed-english-v3'],
    ['cohere.embed-multilingual-v3'],
    ['cohere.embed-v4:0'],
    ['us.cohere.embed-v4:0'],
    ['eu.cohere.embed-v4:0'],
    ['arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.cohere.embed-v4:0'],
  ])('accepts %s', (modelId) => {
    expect(() => new BedrockEmbedder({ modelId })).not.toThrow()
  })

  it.each([
    ['amazon.titan-embed-text-v1'],
    ['amazon.titan-embed-image-v1'],
    ['amazon.nova-2-multimodal-embeddings-v1:0'],
    ['cohere.command-r-v1:0'],
    ['arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123'],
    ['arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123'],
    [''],
    ['.cohere.embed-v4:0'],
    ['cohere.embed-v5:0'],
    ['amazon.titan-embed-text-v20'],
    ['cohere.embed-english-v30'],
    ['amazon.titan-embed-text-v2-foo'],
    ['Amazon.Titan-Embed-Text-V2:0'],
    ['x/amazon.titan-embed-text-v2:0'],
  ])('rejects %j', (modelId) => {
    expect(() => new BedrockEmbedder({ modelId })).toThrow(/Unsupported Bedrock embedding model/)
  })

  it.each([
    ['amazon.titan-embed-text-v2:0', 1536],
    ['amazon.titan-embed-text-v2:0', 300],
    ['cohere.embed-english-v3', 512],
    ['cohere.embed-v4:0', 2048],
    ['amazon.titan-embed-text-v2:0', 1024.5],
    ['amazon.titan-embed-text-v2:0', true],
  ])('rejects %s with %s dimensions', (modelId, dimensions) => {
    expect(() => new BedrockEmbedder({ modelId, dimensions: dimensions as number })).toThrow(/supports dimensions/)
  })

  it('rejects a client together with a region', () => {
    const create = () => new BedrockEmbedder({ client: asClient(new FakeBedrock(titanReply())), region: 'us-east-1' })
    expect(create).toThrow(StorageError)
    expect(create).toThrow(/both client and region/)
    expect(() => new BedrockEmbedder({ client: asClient(new FakeBedrock(titanReply())), region: '' })).toThrow(
      /both client and region/
    )
  })

  it.each([[3], [null]])('rejects the model ID %j, which is not a string', (modelId) => {
    const create = () => new BedrockEmbedder({ modelId: modelId as unknown as string })
    expect(create).toThrow(TypeError)
    expect(create).toThrow('modelId must be a string')
  })

  it('exports the default model ID', () => {
    expect(DEFAULT_EMBEDDING_MODEL_ID).toBe('amazon.titan-embed-text-v2:0')
    expect(new BedrockEmbedder().modelId).toBe(DEFAULT_EMBEDDING_MODEL_ID)
  })
})

const APP_PROFILE = 'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123'

describe('BedrockEmbedder: modelFamily', () => {
  it.each<[BedrockModelFamily, unknown, unknown]>([
    ['amazon.titan-embed-text-v2', titanReply(), { inputText: 'text', dimensions: 1024, normalize: true }],
    ['cohere.embed-v3', cohereReply(), { texts: ['text'], input_type: 'search_query', embedding_types: ['float'] }],
    [
      'cohere.embed-v4',
      cohereReply(),
      { texts: ['text'], input_type: 'search_query', embedding_types: ['float'], output_dimension: 1024 },
    ],
  ])('makes an application inference profile usable as %s', async (modelFamily, reply, body) => {
    const fake = new FakeBedrock(reply)
    const embedder = new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily, client: asClient(fake) })
    expect(await embedder.embed('text', 'query')).toHaveLength(1024)
    expect(fake.calls.at(-1)!.modelId).toBe(APP_PROFILE)
    expect(fake.body()).toEqual(body)
  })

  it('needs a family for a custom-model ARN', () => {
    const custom = 'arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.titan-embed-text-v2:0/abc123'
    expect(() => new BedrockEmbedder({ modelId: custom })).toThrow(/Unsupported Bedrock embedding model/)
    expect(new BedrockEmbedder({ modelId: custom, modelFamily: 'amazon.titan-embed-text-v2' }).modelId).toBe(custom)
  })

  it('must be known and agree with the model ID', () => {
    expect(() => new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily: 'titan' as never })).toThrow(/must be one of/)
    expect(() => new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily: 'toString' as never })).toThrow(
      /must be one of/
    )
    expect(() => new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily: ['cohere.embed-v4'] as never })).toThrow(
      /must be one of/
    )
    expect(
      () => new BedrockEmbedder({ modelId: 'amazon.titan-embed-text-v2:0', modelFamily: 'cohere.embed-v4' })
    ).toThrow(/does not match/)
    expect(
      () => new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily: 'cohere.embed-v3', dimensions: 512 })
    ).toThrow(/supports dimensions/)
    expect(new BedrockEmbedder({ modelId: 'cohere.embed-v4:0', modelFamily: 'cohere.embed-v4' }).modelId).toBe(
      'cohere.embed-v4:0'
    )
  })

  it('treats null like undefined, as Python treats None', () => {
    const family = null as unknown as BedrockModelFamily
    expect(new BedrockEmbedder({ modelId: 'cohere.embed-v4:0', modelFamily: family }).modelId).toBe('cohere.embed-v4:0')
    const fake = asClient(new FakeBedrock(titanReply()))
    expect(new BedrockEmbedder({ client: fake, region: null as unknown as string }).modelId).toBeDefined()
    expect(() => new BedrockEmbedder({ dimensions: null as unknown as number })).toThrow(/supports dimensions/)
  })

  it('is suggested by the unsupported-model error', () => {
    expect(() => new BedrockEmbedder({ modelId: APP_PROFILE })).toThrow(/pass modelFamily/)
  })
})

describe('BedrockEmbedder: input validation', () => {
  it.each([
    ['', RangeError],
    [undefined, TypeError],
    [42, TypeError],
  ])('rejects the text %j', async (text, error) => {
    const fake = new FakeBedrock(titanReply())
    await expect(
      new BedrockEmbedder({ client: asClient(fake) }).embed(text as unknown as string)
    ).rejects.toBeInstanceOf(error)
    expect(fake.calls).toEqual([])
  })

  it('sends whitespace text to the model', async () => {
    // Titan V2 embeds whitespace-only input; the memory store rejects it before calling.
    const fake = new FakeBedrock(titanReply())
    expect(await new BedrockEmbedder({ client: asClient(fake) }).embed(' ')).toHaveLength(1024)
    expect((fake.body() as { inputText: string }).inputText).toBe(' ')
  })

  it.each(
    ['amazon.titan-embed-text-v2:0', 'cohere.embed-v4:0'].flatMap((modelId) =>
      [['search_query'], ['DOCUMENT'], [null], [['document']], [{ a: 1 }]].map(([purpose]) => [modelId, purpose])
    )
  )('rejects for %s the purpose %j', async (modelId, purpose) => {
    const fake = new FakeBedrock(titanReply())
    const embed = new BedrockEmbedder({ modelId: modelId as string, client: asClient(fake) }).embed(
      'text',
      purpose as never
    )
    await expect(embed).rejects.toBeInstanceOf(RangeError)
    await expect(embed).rejects.toThrow("purpose must be 'document' or 'query'")
    expect(fake.calls).toEqual([])
  })
})

describe('BedrockEmbedder: response handling', () => {
  it('wraps Bedrock errors in StorageError naming the cause', async () => {
    const failure = Object.assign(new Error('Too many input tokens'), { name: 'ValidationException' })
    const error = await new BedrockEmbedder({ client: asClient(new FakeBedrock(failure)) })
      .embed('text')
      .catch((e: unknown) => e)
    expect(error).toBeInstanceOf(StorageError)
    expect((error as Error).message).toBe(
      "Bedrock embedding request to model 'amazon.titan-embed-text-v2:0' failed: ValidationException: Too many input tokens"
    )
    expect((error as Error).cause).toBe(failure)
  })

  const WITHOUT = 'a response without an embedding'
  const NOT_1024 = 'an embedding that is not 1024 finite numbers'

  it.each([
    [{}, WITHOUT],
    [null, WITHOUT],
    [{ embeddingsByType: { float: Array(1024).fill(0.5) } }, WITHOUT],
    [{ embedding: null }, NOT_1024],
    [{ embedding: Array(512).fill(0.5) }, NOT_1024],
    [{ embedding: Array(1025).fill(0.5) }, NOT_1024],
    [{ embedding: [...Array(1023).fill(0.5), 'x'] }, NOT_1024],
    [{ embedding: [...Array(1023).fill(0.5), null] }, NOT_1024],
    [{ embedding: Array(1024).fill(true) }, NOT_1024],
  ])('rejects the Titan response %#', async (reply, message) => {
    await expect(new BedrockEmbedder({ client: asClient(new FakeBedrock(reply)) }).embed('text')).rejects.toThrow(
      `Bedrock model 'amazon.titan-embed-text-v2:0' returned ${message}`
    )
  })

  it.each([
    [{ embeddings: [Array(1024).fill(0.25)] }, WITHOUT],
    [{ embeddings: { float: [] } }, WITHOUT],
    [{ embeddings: { float: 'x' } }, WITHOUT],
    [{ embeddings: { float: { 0: Array(1024).fill(0.25) } } }, WITHOUT],
    [{ embeddings: null }, WITHOUT],
    [{ embeddings: { int8: [Array(1024).fill(1)] } }, WITHOUT],
    [{ embeddings: { float: [null] } }, NOT_1024],
    [{ embeddings: { float: [Array(256).fill(0.25)] } }, NOT_1024],
    [{ embeddings: { float: [Array(1025).fill(0.25)] } }, NOT_1024],
  ])('rejects the Cohere response %#', async (reply, message) => {
    const embedder = new BedrockEmbedder({
      modelId: 'cohere.embed-english-v3',
      client: asClient(new FakeBedrock(reply)),
    })
    await expect(embedder.embed('text')).rejects.toThrow(`Bedrock model 'cohere.embed-english-v3' returned ${message}`)
  })
})

describe('BedrockEmbedder: client creation', () => {
  it('creates a Bedrock runtime client in the given region with the user-agent marker', async () => {
    const embedder = new BedrockEmbedder({ region: 'eu-west-1' })
    const sdk = await import('@aws-sdk/client-bedrock-runtime')
    const client = embedder['_getClient'](sdk)
    expect(client).toBeInstanceOf(BedrockRuntimeClient)
    expect(await client.config.region()).toBe('eu-west-1')
    expect(JSON.stringify(client.config.customUserAgent)).toContain('strands-dynamodb-storage')
    expect(embedder['_getClient'](sdk)).toBe(client)
  })

  it('accepts a region provider', async () => {
    const embedder = new BedrockEmbedder({ region: async () => 'ap-southeast-2' })
    const client = embedder['_getClient'](await import('@aws-sdk/client-bedrock-runtime'))
    expect(await client.config.region()).toBe('ap-southeast-2')
  })
})

describe('BedrockEmbedder: error messages', () => {
  const messageOf = async (run: () => unknown): Promise<string> => {
    try {
      await run()
    } catch (error: unknown) {
      return `${(error as Error).name}: ${(error as Error).message}`
    }
    throw new Error('expected an error')
  }
  const fake = () => asClient(new FakeBedrock(titanReply()))

  it.each<[string, () => unknown, string]>([
    ['modelId type', () => new BedrockEmbedder({ modelId: 3 as never }), 'TypeError: modelId must be a string'],
    [
      'unknown family',
      () => new BedrockEmbedder({ modelId: APP_PROFILE, modelFamily: 'titan' as never }),
      'RangeError: modelFamily must be one of amazon.titan-embed-text-v2, cohere.embed-v3, cohere.embed-v4; got titan',
    ],
    [
      'family mismatch',
      () => new BedrockEmbedder({ modelId: 'amazon.titan-embed-text-v2:0', modelFamily: 'cohere.embed-v4' }),
      "RangeError: modelFamily 'cohere.embed-v4' does not match model 'amazon.titan-embed-text-v2:0' " +
        '(amazon.titan-embed-text-v2)',
    ],
    [
      'unsupported model',
      () => new BedrockEmbedder({ modelId: 'amazon.titan-embed-text-v1' }),
      "RangeError: Unsupported Bedrock embedding model 'amazon.titan-embed-text-v1': BedrockEmbedder supports " +
        'amazon.titan-embed-text-v2, cohere.embed-english-v3, cohere.embed-multilingual-v3, and cohere.embed-v4. ' +
        'For an application inference profile or provisioned model ARN, pass modelFamily; for other models, ' +
        'pass an embed function to DynamoDBMemoryStore.',
    ],
    [
      'dimensions',
      () => new BedrockEmbedder({ dimensions: 300 }),
      "RangeError: Model 'amazon.titan-embed-text-v2:0' supports dimensions [256, 512, 1024]; got 300",
    ],
    [
      'client and region',
      () => new BedrockEmbedder({ client: fake(), region: 'us-east-1' }),
      'StorageError: Cannot specify both client and region. Configure the region on the client instead.',
    ],
    [
      'embed type',
      () => new BedrockEmbedder({ client: fake() }).embed(3 as never),
      'TypeError: BedrockEmbedder.embed requires a string',
    ],
    [
      'embed empty',
      () => new BedrockEmbedder({ client: fake() }).embed(''),
      'RangeError: BedrockEmbedder.embed requires a non-empty string',
    ],
    [
      'purpose',
      () => new BedrockEmbedder({ client: fake() }).embed('x', 'search_query' as never),
      "RangeError: purpose must be 'document' or 'query'; got search_query",
    ],
  ])('%s', async (_label, run, message) => {
    expect(await messageOf(run)).toBe(message)
  })
})
