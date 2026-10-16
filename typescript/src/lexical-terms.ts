// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StorageError } from '@strands-agents/sdk'

import { Buffer } from 'node:buffer'

/** Version of the tokenizer and key encoding (`lexical-v1`), recorded on every manifest. */
export const LEXICAL_TOKENIZER_VERSION = 'lexical-v1'

const WORD_CHARACTER = /^[\p{L}\p{M}\p{N}]$/u
const JOINERS = new Set(['-', '_'])
const JOINER = /[-_]/
const JOINER_RUNS = /[-_]+/
const EDGE_JOINERS = /^[-_]+|[-_]+$/g
const ASCII_UPPERCASE_RUNS = /[A-Z]+/g
const CONTROL_CHARACTER = /\p{Cc}/u
const EDGE_SEPARATOR = /^\p{Z}|\p{Z}$/u
const LONE_SURROGATE = /\p{Surrogate}/u

const TERM_POSTING_TAG = 't|'
const IDENTIFIER_POSTING_TAG = 'i|'
const MANIFEST_TAG = 'm|'

const utf8 = new TextEncoder()

/**
 * Splits `text` into searchable terms (`lexical-v1`).
 *
 * The text is NFC-normalized and split by code point into runs of letters, marks, numbers and the
 * ASCII joiners `-` and `_`; everything else separates runs. Each run loses its leading and trailing
 * joiners and has ASCII `A`–`Z` lowercased (non-ASCII case is kept because JavaScript and Python fold
 * it differently). A run is emitted as a compound term followed, when it contains joiners, by its
 * joiner-separated parts. Terms are de-duplicated keeping the first occurrence.
 *
 * @throws {@link StorageError} if the text is not well-formed Unicode (it contains a lone surrogate)
 */
export function textTerms(text: string): string[] {
  assertWellFormed(text, 'Text')
  const terms = new Set<string>()
  for (const run of wordRuns(text.normalize('NFC'))) {
    for (const term of runTerms(run)) terms.add(term)
  }
  return [...terms]
}

/**
 * Validates an exact-match identifier and returns its NFC form. Identifiers are otherwise kept as
 * given: case-sensitive, with punctuation, leading zeroes and internal spaces preserved.
 *
 * @throws {@link StorageError} if the identifier is not well-formed Unicode, is empty, contains a control
 *   character, or starts or ends with a separator (whitespace) character
 */
export function normalizeIdentifier(identifier: string): string {
  assertWellFormed(identifier, 'Identifier')
  const normalized = identifier.normalize('NFC')
  if (normalized === '') throw new StorageError('Identifier must not be empty')
  if (CONTROL_CHARACTER.test(normalized)) throw new StorageError('Identifier must not contain control characters')
  if (EDGE_SEPARATOR.test(normalized)) {
    throw new StorageError('Identifier must not start or end with a whitespace or separator character')
  }
  return normalized
}

/**
 * Normalizes each identifier and de-duplicates the result, keeping the first occurrence.
 *
 * @throws {@link StorageError} if any identifier is invalid (see {@link normalizeIdentifier})
 */
export function normalizeIdentifiers(identifiers: readonly string[]): string[] {
  return [...new Set(identifiers.map((identifier) => normalizeIdentifier(identifier)))]
}

/** Number of bytes in the UTF-8 encoding of `value`. */
export function utf8ByteLength(value: string): number {
  return utf8.encode(value).byteLength
}

/**
 * Orders strings by their UTF-8 bytes, which is the order DynamoDB sorts string keys in. The default
 * JavaScript comparison orders UTF-16 code units and disagrees for characters outside the BMP.
 */
export function compareUtf8(left: string, right: string): number {
  return Buffer.compare(utf8.encode(left), utf8.encode(right))
}

/** Length-prefixed key segment `"<utf8 bytes>:<value>"`, so adjacent segments can never be confused. */
export function keySegment(value: string): string {
  return `${utf8ByteLength(value)}:${value}`
}

/**
 * Scope tag stored as `lxscope` on indexed base items. Length-prefixed rather than raw so the root
 * scope (`''`) never becomes an empty-string expression value.
 */
export function scopeSegment(scope: string): string {
  return keySegment(scope)
}

/** Index partition key of the postings of `term` within `scope`. */
export function termPostingPk(scope: string, term: string): string {
  return `${TERM_POSTING_TAG}${keySegment(scope)}${keySegment(term)}`
}

/** Index partition key of the postings of `identifier` within `scope`. */
export function identifierPostingPk(scope: string, identifier: string): string {
  return `${IDENTIFIER_POSTING_TAG}${keySegment(scope)}${keySegment(identifier)}`
}

/** Index partition key holding one manifest per document of `scope`. */
export function manifestPk(scope: string): string {
  return `${MANIFEST_TAG}${keySegment(scope)}`
}

/**
 * Distinct posting partition keys of a document's terms followed by its identifiers, keeping the first
 * occurrence of each: a manifest written outside this library may repeat a term, and one transaction
 * cannot touch the same posting twice.
 */
export function postingPartitionKeys(
  scope: string,
  terms: readonly string[],
  identifiers: readonly string[]
): string[] {
  const termKeys = terms.map((term) => termPostingPk(scope, term))
  const identifierKeys = identifiers.map((identifier) => identifierPostingPk(scope, identifier))
  return [...new Set([...termKeys, ...identifierKeys])]
}

/**
 * Rejects a lone UTF-16 surrogate. It has no UTF-8 encoding, so its byte size and index keys would
 * silently differ from the Python package (which cannot encode it at all).
 */
function assertWellFormed(value: string, subject: string): void {
  if (LONE_SURROGATE.test(value)) {
    throw new StorageError(`${subject} must be well-formed Unicode; it contains a lone surrogate`)
  }
}

function wordRuns(text: string): string[] {
  const runs: string[] = []
  let current = ''
  for (const character of text) {
    if (isWordCharacter(character)) {
      current += character
    } else if (current) {
      runs.push(current)
      current = ''
    }
  }
  if (current) runs.push(current)
  return runs
}

function isWordCharacter(character: string): boolean {
  return JOINERS.has(character) || WORD_CHARACTER.test(character)
}

function runTerms(run: string): string[] {
  const compound = lowercaseAscii(run.replace(EDGE_JOINERS, ''))
  if (!compound) return []
  if (!JOINER.test(compound)) return [compound]
  return [compound, ...compound.split(JOINER_RUNS).filter(Boolean)]
}

function lowercaseAscii(value: string): string {
  return value.replace(ASCII_UPPERCASE_RUNS, (letters) => letters.toLowerCase())
}
