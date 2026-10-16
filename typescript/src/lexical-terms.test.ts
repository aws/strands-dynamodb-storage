// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { StorageError } from '@strands-agents/sdk'

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import {
  LEXICAL_TOKENIZER_VERSION,
  compareUtf8,
  identifierPostingPk,
  keySegment,
  manifestPk,
  normalizeIdentifier,
  normalizeIdentifiers,
  postingPartitionKeys,
  scopeSegment,
  termPostingPk,
  textTerms,
  utf8ByteLength,
} from './lexical-terms.js'

/** Shape of the cross-language fixtures shared with the Python package. */
interface LexicalFixtures {
  tokenizerVersion: string
  textTerms: Array<{ input: string; terms: string[] }>
  identifiers: { valid: Array<{ input: string; normalized: string }>; invalid: string[] }
  keys: Array<{ kind: 'term' | 'identifier' | 'manifest'; scope: string; value: string | null; pk: string }>
}

const fixtures: LexicalFixtures = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../testdata/lexical-v1.json', import.meta.url)), 'utf8')
)

const LONE_HIGH_SURROGATE = String.fromCharCode(0xd800)
const LONE_LOW_SURROGATE = String.fromCharCode(0xdc00)

function encodeFixtureKey({ kind, scope, value }: LexicalFixtures['keys'][number]): string {
  if (kind === 'manifest') return manifestPk(scope)
  if (kind === 'term') return termPostingPk(scope, value ?? '')
  return identifierPostingPk(scope, value ?? '')
}

describe('lexical-v1 shared fixtures', () => {
  it('pins the tokenizer version', () => {
    expect(LEXICAL_TOKENIZER_VERSION).toBe(fixtures.tokenizerVersion)
  })

  it.each(fixtures.textTerms.map(({ input, terms }) => [input, terms] as const))('textTerms(%j)', (input, terms) => {
    expect(textTerms(input)).toEqual(terms)
  })

  it.each(fixtures.identifiers.valid.map(({ input, normalized }) => [input, normalized] as const))(
    'accepts identifier %j',
    (input, normalized) => {
      expect(normalizeIdentifier(input)).toBe(normalized)
    }
  )

  it.each(fixtures.identifiers.invalid.map((input) => [input] as const))('rejects identifier %j', (input) => {
    expect(() => normalizeIdentifier(input)).toThrow(StorageError)
  })

  it.each(fixtures.keys.map((fixture) => [fixture.kind, fixture.pk, fixture] as const))(
    'encodes the %s key %j',
    (_kind, pk, fixture) => {
      expect(encodeFixtureKey(fixture)).toBe(pk)
    }
  )
})

describe('textTerms', () => {
  it('iterates by code point, keeping astral letters inside one term', () => {
    expect(textTerms('x\u{1D49C}y')).toEqual(['x\u{1D49C}y'])
  })

  it('composes combining marks with NFC before splitting', () => {
    expect(textTerms('cafe\u0301')).toEqual(['café'])
  })

  it('preserves leading zeroes so distinct identifiers stay distinct terms', () => {
    expect(textTerms('fc-00123')).not.toContain('123')
  })

  it.each([
    ['a high', `alpha ${LONE_HIGH_SURROGATE} beta`],
    ['a low', `alpha${LONE_LOW_SURROGATE}`],
    ['a reversed pair of', `${LONE_LOW_SURROGATE}${LONE_HIGH_SURROGATE}`],
  ])('rejects text containing %s lone surrogate', (_label, text) => {
    expect(() => textTerms(text)).toThrow(StorageError)
    expect(() => textTerms(text)).toThrow(/^Text must be well-formed Unicode; it contains a lone surrogate$/)
  })

  it('accepts a well-formed surrogate pair', () => {
    expect(textTerms(`x${String.fromCodePoint(0x1d49c)}y`)).toHaveLength(1)
  })
})

describe('normalizeIdentifier', () => {
  it('returns the NFC form of a decomposed identifier', () => {
    expect(normalizeIdentifier('A\u0308-1')).toBe('Ä-1')
  })

  it('never echoes the rejected identifier in the error message', () => {
    expect(() => normalizeIdentifier(' SECRET-ID')).toThrow(/^(?!.*SECRET).*$/)
  })

  it.each([`ID-${LONE_HIGH_SURROGATE}`, `${LONE_LOW_SURROGATE}ID`])(
    'rejects an identifier with a lone surrogate',
    (id) => {
      expect(() => normalizeIdentifier(id)).toThrow(
        /^Identifier must be well-formed Unicode; it contains a lone surrogate$/
      )
    }
  )
})

describe('normalizeIdentifiers', () => {
  it('normalizes each identifier and keeps the first occurrence of duplicates', () => {
    expect(normalizeIdentifiers(['B-2', 'A\u0308-1', 'B-2', '\u00c4-1'])).toEqual(['B-2', '\u00c4-1'])
  })

  it('rejects the whole list when one identifier is invalid', () => {
    expect(() => normalizeIdentifiers(['A-1', 'bad\tid'])).toThrow(/control characters/)
  })
})

describe('UTF-8 helpers', () => {
  it('counts UTF-8 bytes, not UTF-16 units', () => {
    expect(utf8ByteLength('a')).toBe(1)
    expect(utf8ByteLength('é')).toBe(2)
    expect(utf8ByteLength('日')).toBe(3)
    expect(utf8ByteLength('\u{1F600}')).toBe(4)
  })

  it('orders by UTF-8 bytes where UTF-16 comparison disagrees', () => {
    expect('\u{1F600}' < '\uFFFF').toBe(true)
    expect(compareUtf8('\uFFFF', '\u{1F600}')).toBeLessThan(0)
    expect(compareUtf8('\u{1F600}', '\uFFFF')).toBeGreaterThan(0)
    expect(compareUtf8('abc', 'abc')).toBe(0)
    expect(compareUtf8('ab', 'abc')).toBeLessThan(0)
  })

  it('length-prefixes key segments with their UTF-8 byte count', () => {
    expect(keySegment('')).toBe('0:')
    expect(keySegment('tenant/a/')).toBe('9:tenant/a/')
    expect(keySegment('日本')).toBe('6:日本')
  })

  it('tags a scope with its length-prefixed segment, never an empty string', () => {
    expect(scopeSegment('')).toBe('0:')
    expect(scopeSegment('tenant/a/')).toBe('9:tenant/a/')
  })
})

describe('postingPartitionKeys', () => {
  it('lists term postings before identifier postings, without duplicates', () => {
    expect(postingPartitionKeys('s/', ['fc-1', 'fc', 'fc'], ['FC-1'])).toEqual([
      termPostingPk('s/', 'fc-1'),
      termPostingPk('s/', 'fc'),
      identifierPostingPk('s/', 'FC-1'),
    ])
  })

  it('keeps the first occurrence of terms and identifiers repeated in a hand-written manifest', () => {
    expect(postingPartitionKeys('', ['b', 'a', 'b', 'a'], ['X', 'X'])).toEqual([
      termPostingPk('', 'b'),
      termPostingPk('', 'a'),
      identifierPostingPk('', 'X'),
    ])
  })
})
