// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, it, vi } from 'vitest'

afterEach(() => {
  vi.doUnmock('@strands-agents/sdk/storage/search')
  vi.resetModules()
})

it('raises a StorageError naming the SDK release a plain-string search needs when keyword search is missing', async () => {
  vi.resetModules()
  vi.doMock('@strands-agents/sdk/storage/search', () => {
    throw new Error('Package subpath \'./storage/search\' is not defined by "exports"')
  })
  const { DynamoDBStorage } = await import('./dynamodb-storage.js')
  const storage = new DynamoDBStorage('test-table', { client: { send: vi.fn() } as never })
  let caught: unknown
  try {
    await storage.namespace('user/u1').search('window')
  } catch (error: unknown) {
    caught = error
  }
  expect((caught as Error).name).toBe('StorageError')
  expect((caught as Error).message).toBe(
    'Plain-string DynamoDBStorage.search needs @strands-agents/sdk 1.14.0 or later for keyword search. ' +
      'Upgrade @strands-agents/sdk, set searchStrategy, or embed the text and pass { vector, topK }.'
  )
})
