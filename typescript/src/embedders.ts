// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'
import { USER_AGENT_MARKER } from './dynamodb-storage.js'

/** What a text is embedded for: `'document'` for a memory being stored, `'query'` for a search. */
export type EmbeddingPurpose = 'document' | 'query'

/**
 * Turns text into an embedding vector.
 *
 * `purpose` lets a model that embeds stored text and search queries differently
 * (Cohere Embed) use the matching input type. A model that does not distinguish the
 * two ignores it.
 */
export interface Embedder {
  /** Returns the embedding vector of `text`, directly or as a promise. */
  embed(text: string, purpose: EmbeddingPurpose): number[] | Promise<number[]>
}

/**
 * A plain function from text to vector, sync or async, used for both memories and queries.
 * It returns an array of numbers; convert a `Float32Array` with `Array.from`.
 */
export type EmbedFunction = (text: string) => number[] | Promise<number[]>

/** Default {@link BedrockEmbedder} model: Amazon Titan Text Embeddings V2. */
export const DEFAULT_EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'

/** Request format of a Bedrock embedding model, for identifiers that do not name the model. */
export type BedrockModelFamily = 'amazon.titan-embed-text-v2' | 'cohere.embed-v3' | 'cohere.embed-v4'

const SUPPORTED_DIMENSIONS: Record<BedrockModelFamily, readonly number[]> = {
  'amazon.titan-embed-text-v2': [256, 512, 1024],
  'cohere.embed-v3': [1024],
  'cohere.embed-v4': [256, 512, 1024, 1536],
}

const COHERE_INPUT_TYPE: Record<EmbeddingPurpose, string> = {
  document: 'search_document',
  query: 'search_query',
}

/**
 * Maps a Bedrock model ID, inference profile ID, or ARN to its request format, or
 * `undefined` for a model this embedder does not recognize (including application
 * inference profiles and provisioned models, whose ARNs do not name the underlying
 * model).
 */
function modelFamily(modelId: string): BedrockModelFamily | undefined {
  let base = modelId.startsWith('arn:') ? modelId.slice(modelId.lastIndexOf('/') + 1) : modelId
  // A cross-Region inference profile ID prefixes the model ID with a geography,
  // for example "us.cohere.embed-v4:0".
  const dot = base.indexOf('.')
  if (dot > 0) {
    const geography = base.slice(0, dot)
    if (geography !== 'amazon' && geography !== 'cohere') base = base.slice(dot + 1)
  }
  for (const [prefix, family] of MODEL_PREFIXES) {
    // The model name ends at the end of the ID or at a ":" before the version.
    if (base === prefix || base.startsWith(`${prefix}:`)) return family
  }
  return undefined
}

const MODEL_PREFIXES: ReadonlyArray<readonly [string, BedrockModelFamily]> = [
  ['amazon.titan-embed-text-v2', 'amazon.titan-embed-text-v2'],
  ['cohere.embed-english-v3', 'cohere.embed-v3'],
  ['cohere.embed-multilingual-v3', 'cohere.embed-v3'],
  ['cohere.embed-v4', 'cohere.embed-v4'],
]

type BedrockRuntimeModule = typeof import('@aws-sdk/client-bedrock-runtime')

/** Loads the optional peer dependency, with an install hint when it is missing. */
async function loadBedrockRuntime(): Promise<BedrockRuntimeModule> {
  try {
    return await import('@aws-sdk/client-bedrock-runtime')
  } catch (error: unknown) {
    throw new StorageError(
      'BedrockEmbedder requires the optional peer dependency @aws-sdk/client-bedrock-runtime; ' +
        'install it with `npm install @aws-sdk/client-bedrock-runtime`.',
      { cause: error }
    )
  }
}

/** Configuration for {@link BedrockEmbedder}. */
export interface BedrockEmbedderConfig {
  /** Bedrock model ID, cross-Region inference profile ID, or ARN. Default {@link DEFAULT_EMBEDDING_MODEL_ID}. */
  modelId?: string
  /**
   * The model's request format, required when `modelId` is an application inference profile
   * or provisioned model ARN, which does not name the model.
   */
  modelFamily?: BedrockModelFamily
  /**
   * Length of the returned vectors; must equal the vector index's `Dimensions`. Titan
   * Text Embeddings V2 supports 256, 512, and 1024; Cohere Embed v3 supports 1024;
   * Cohere Embed v4 supports 256, 512, 1024, and 1536. Default 1024.
   */
  dimensions?: number
  /** AWS Region of the Bedrock runtime endpoint, or a provider of it. Cannot be combined with `client`. */
  region?: string | (() => Promise<string>)
  /** Pre-configured Bedrock runtime client. */
  client?: import('@aws-sdk/client-bedrock-runtime').BedrockRuntimeClient
}

/**
 * Embeds text with an Amazon Bedrock embedding model.
 *
 * Supports Amazon Titan Text Embeddings V2 (the default) and Cohere Embed v3 (English
 * and multilingual) and v4, by model ID, cross-Region inference profile ID, or ARN. For
 * an application inference profile or a provisioned model, whose ARN does not name the
 * model, pass `modelFamily`. Cohere models embed stored memories and search queries with
 * separate input types; Titan does not distinguish them. For any other model, pass an
 * embed function to {@link DynamoDBMemoryStore} instead.
 *
 * Requires the optional peer dependency `@aws-sdk/client-bedrock-runtime`, and
 * `bedrock:InvokeModel` on the model (and, for an inference profile, on the profile).
 * No AWS call is made until the first {@link embed}.
 */
export class BedrockEmbedder implements Embedder {
  readonly modelId: string
  readonly dimensions: number
  private readonly _family: BedrockModelFamily
  private readonly _region: string | (() => Promise<string>) | undefined
  private _client: import('@aws-sdk/client-bedrock-runtime').BedrockRuntimeClient | undefined

  /**
   * @throws TypeError if `modelId` is not a string
   * @throws RangeError if the model is not supported, `modelFamily` is unknown or contradicts
   *   `modelId`, or `dimensions` is not one the model supports
   * @throws {@link StorageError} if both `client` and `region` are set
   */
  constructor(config?: BedrockEmbedderConfig) {
    const modelId = config?.modelId === undefined ? DEFAULT_EMBEDDING_MODEL_ID : config.modelId
    if (typeof modelId !== 'string') throw new TypeError('modelId must be a string')
    const detected = modelFamily(modelId)
    const requested = config?.modelFamily ?? undefined
    if (requested !== undefined && (typeof requested !== 'string' || !Object.hasOwn(SUPPORTED_DIMENSIONS, requested))) {
      throw new RangeError(
        `modelFamily must be one of ${Object.keys(SUPPORTED_DIMENSIONS).join(', ')}; got ${String(requested)}`
      )
    }
    if (requested !== undefined && detected !== undefined && requested !== detected) {
      throw new RangeError(`modelFamily '${requested}' does not match model '${modelId}' (${detected})`)
    }
    const family = requested ?? detected
    if (family === undefined) {
      throw new RangeError(
        `Unsupported Bedrock embedding model '${modelId}': BedrockEmbedder supports ` +
          'amazon.titan-embed-text-v2, cohere.embed-english-v3, cohere.embed-multilingual-v3, ' +
          'and cohere.embed-v4. For an application inference profile or provisioned model ARN, ' +
          'pass modelFamily; for other models, pass an embed function to DynamoDBMemoryStore.'
      )
    }
    const dimensions = config?.dimensions === undefined ? 1024 : config.dimensions
    const supported = SUPPORTED_DIMENSIONS[family]
    if (!Number.isInteger(dimensions) || !supported.includes(dimensions)) {
      throw new RangeError(`Model '${modelId}' supports dimensions [${supported.join(', ')}]; got ${dimensions}`)
    }
    if (config?.client && config.region != null) {
      throw new StorageError('Cannot specify both client and region. Configure the region on the client instead.')
    }
    this.modelId = modelId
    this.dimensions = dimensions
    this._family = family
    this._region = config?.region ?? undefined
    this._client = config?.client
  }

  /**
   * Returns the embedding vector of `text`.
   *
   * @throws TypeError if `text` is not a string
   * @throws RangeError if `text` is empty or `purpose` is not `'document'` or `'query'`
   * @throws {@link StorageError} if the Bedrock call fails or returns an unexpected response
   */
  async embed(text: string, purpose: EmbeddingPurpose = 'document'): Promise<number[]> {
    if (typeof text !== 'string') throw new TypeError('BedrockEmbedder.embed requires a string')
    if (text === '') throw new RangeError('BedrockEmbedder.embed requires a non-empty string')
    if (purpose !== 'document' && purpose !== 'query') {
      throw new RangeError(`purpose must be 'document' or 'query'; got ${String(purpose)}`)
    }
    const sdk = await loadBedrockRuntime()
    let payload: unknown
    try {
      const { InvokeModelCommand } = sdk
      const client = this._getClient(sdk)
      const response = await client.send(
        new InvokeModelCommand({
          modelId: this.modelId,
          body: JSON.stringify(this._requestBody(text, purpose)),
          contentType: 'application/json',
          accept: 'application/json',
        })
      )
      payload = JSON.parse(new TextDecoder().decode(response.body))
    } catch (error: unknown) {
      // The service message (input too long, access denied, throttled) is the actionable part,
      // and the Memory Manager logs only the error's text, so it goes in the message.
      throw new StorageError(`Bedrock embedding request to model '${this.modelId}' failed: ${describeError(error)}`, {
        cause: error,
      })
    }
    return this._parse(payload)
  }

  private _requestBody(text: string, purpose: EmbeddingPurpose): Record<string, unknown> {
    if (this._family === 'amazon.titan-embed-text-v2') {
      return { inputText: text, dimensions: this.dimensions, normalize: true }
    }
    const body: Record<string, unknown> = {
      texts: [text],
      input_type: COHERE_INPUT_TYPE[purpose],
      embedding_types: ['float'],
    }
    if (this._family === 'cohere.embed-v4') body.output_dimension = this.dimensions
    return body
  }

  private _parse(payload: unknown): number[] {
    const reply = payload as {
      embedding?: unknown
      embeddings?: { float?: unknown[] }
    } | null
    const vector =
      this._family === 'amazon.titan-embed-text-v2'
        ? reply?.embedding
        : Array.isArray(reply?.embeddings?.float)
          ? reply.embeddings.float[0]
          : undefined
    if (vector === undefined) {
      throw new StorageError(`Bedrock model '${this.modelId}' returned a response without an embedding`)
    }
    if (
      !Array.isArray(vector) ||
      vector.length !== this.dimensions ||
      !vector.every((v) => typeof v === 'number' && Number.isFinite(v))
    ) {
      throw new StorageError(
        `Bedrock model '${this.modelId}' returned an embedding that is not ${this.dimensions} finite numbers`
      )
    }
    return vector as number[]
  }

  private _getClient(sdk: BedrockRuntimeModule): import('@aws-sdk/client-bedrock-runtime').BedrockRuntimeClient {
    if (this._client) return this._client
    this._client = new sdk.BedrockRuntimeClient({
      ...(this._region !== undefined ? { region: this._region } : {}),
      customUserAgent: USER_AGENT_MARKER,
    })
    return this._client
  }
}

/** The text of an error as `Name: message`, or its string form when it is not an `Error`. */
export function describeError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error)
}
