// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, it, vi } from 'vitest'

afterEach(() => {
  vi.doUnmock('@aws-sdk/client-bedrock-runtime')
  vi.resetModules()
})

it('raises a StorageError with an install hint when the optional peer dependency is missing', async () => {
  vi.resetModules()
  vi.doMock('@aws-sdk/client-bedrock-runtime', () => {
    throw new Error("Cannot find package '@aws-sdk/client-bedrock-runtime'")
  })
  const { BedrockEmbedder } = await import('./embedders.js')
  let caught: unknown
  try {
    await new BedrockEmbedder().embed('text')
  } catch (error: unknown) {
    caught = error
  }
  expect((caught as Error).name).toBe('StorageError')
  expect((caught as Error).message).toBe(
    'BedrockEmbedder requires the optional peer dependency @aws-sdk/client-bedrock-runtime; ' +
      'install it with `npm install @aws-sdk/client-bedrock-runtime`.'
  )
})
