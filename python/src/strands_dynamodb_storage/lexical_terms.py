# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure ``lexical-v1`` rules: tokenizer, identifier normalization, index key encoding.

No I/O. Shared with the TypeScript package through ``testdata/lexical-v1.json`` so both
languages emit identical terms and index keys. Python ``str`` ordering is code-point order,
which equals UTF-8 byte order, so plain string sorting already matches DynamoDB sort keys.

Character classes come from the runtime's Unicode character database: code points assigned
after Unicode 15.0 may tokenize differently between runtimes (the fixtures pin assigned
characters only).
"""

from __future__ import annotations

import itertools
import re
import string
import unicodedata
from collections.abc import Iterable, Iterator, Sequence

from strands.types.exceptions import StorageError

LEXICAL_TOKENIZER_VERSION = "lexical-v1"

_JOINERS = "-_"
_JOINER_RUNS = re.compile(r"[-_]+")
_WORD_CATEGORY_CLASSES = frozenset("LMN")
_ASCII_LOWERCASE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_FIRST_SURROGATE = "\ud800"
_LAST_SURROGATE = "\udfff"

_TERM_POSTING_TAG = "t"
_IDENTIFIER_POSTING_TAG = "i"
_MANIFEST_TAG = "m"


def utf8_byte_length(value: str) -> int:
    """Return the UTF-8 encoded size of ``value`` in bytes.

    Never raises: a lone surrogate (not well-formed, so rejected by :func:`text_terms` and
    :func:`normalize_identifier`) counts 3 bytes, as the TypeScript package's ``TextEncoder``
    counts its U+FFFD replacement, so size checks behave identically in both languages.
    """
    return len(value.encode("utf-8", "surrogatepass"))


def text_terms(text: str) -> list[str]:
    """Split ``text`` into ``lexical-v1`` terms, de-duplicated in emission order.

    Runs of letters, marks, digits and the ASCII joiners ``-``/``_`` form words; everything
    else separates. Each word is emitted as a compound (joiners trimmed from the ends, ASCII
    letters lowercased, non-ASCII case kept) followed by its joiner-separated parts.

    Raises:
        StorageError: If the text is not well-formed Unicode (it contains a lone surrogate).
    """
    _require_well_formed(text, "Text")
    normalized = unicodedata.normalize("NFC", text)
    terms = itertools.chain.from_iterable(_word_terms(word) for word in _words(normalized))
    return list(dict.fromkeys(terms))


def normalize_identifier(identifier: str) -> str:
    """Return the NFC form of an exact-match identifier, otherwise unchanged.

    Identifiers stay case-sensitive and keep punctuation, leading zeroes and internal spaces.

    Raises:
        StorageError: If the identifier is not well-formed Unicode (it contains a lone surrogate),
            is empty, contains a control character, or starts or ends with a separator (whitespace).
    """
    _require_well_formed(identifier, "Identifier")
    normalized = unicodedata.normalize("NFC", identifier)
    if not normalized:
        raise StorageError("Identifier must not be empty")
    if any(unicodedata.category(char) == "Cc" for char in normalized):
        raise StorageError("Identifier must not contain control characters")
    if _is_separator(normalized[0]) or _is_separator(normalized[-1]):
        raise StorageError("Identifier must not start or end with whitespace")
    return normalized


def normalize_identifiers(identifiers: Iterable[str]) -> list[str]:
    """Normalize each identifier and de-duplicate, keeping the first occurrence.

    Raises:
        StorageError: If any identifier is invalid (see :func:`normalize_identifier`).
    """
    return list(dict.fromkeys(normalize_identifier(identifier) for identifier in identifiers))


def scope_segment(scope: str) -> str:
    """Length-prefixed form of ``scope`` (``"9:tenant/a/"``; ``"0:"`` for the root scope).

    Stored as the base item's scope marker so the root scope is never an empty string in a
    condition expression value.
    """
    return _segment(scope)


def term_posting_pk(scope: str, term: str) -> str:
    """Partition key of the posting list for ``term`` within ``scope``."""
    return f"{_TERM_POSTING_TAG}|{_segment(scope)}{_segment(term)}"


def identifier_posting_pk(scope: str, identifier: str) -> str:
    """Partition key of the posting list for an exact ``identifier`` within ``scope``."""
    return f"{_IDENTIFIER_POSTING_TAG}|{_segment(scope)}{_segment(identifier)}"


def manifest_pk(scope: str) -> str:
    """Partition key holding one manifest per indexed document of ``scope``."""
    return f"{_MANIFEST_TAG}|{_segment(scope)}"


def posting_partition_keys(scope: str, terms: Sequence[str], identifiers: Sequence[str]) -> list[str]:
    """Partition keys of every posting owned by a document with these terms and identifiers.

    Terms come first, then identifiers, de-duplicated keeping the first occurrence: a
    hand-edited manifest may repeat a term, and one transaction cannot touch an item twice.
    """
    partitions = itertools.chain(
        (term_posting_pk(scope, term) for term in terms),
        (identifier_posting_pk(scope, identifier) for identifier in identifiers),
    )
    return list(dict.fromkeys(partitions))


def _require_well_formed(value: str, subject: str) -> None:
    """Reject a lone surrogate (U+D800..U+DFFF): it has no UTF-8 encoding, so no stable term or key.

    The message names ``subject`` only, never the value.
    """
    if any(_FIRST_SURROGATE <= char <= _LAST_SURROGATE for char in value):
        raise StorageError(f"{subject} must be well-formed Unicode; it contains a lone surrogate")


def _segment(value: str) -> str:
    """Length-prefix ``value`` so concatenated segments can never collide."""
    return f"{utf8_byte_length(value)}:{value}"


def _is_word_char(char: str) -> bool:
    return char in _JOINERS or unicodedata.category(char)[0] in _WORD_CATEGORY_CLASSES


def _is_separator(char: str) -> bool:
    return unicodedata.category(char).startswith("Z")


def _words(text: str) -> Iterator[str]:
    for is_word, chars in itertools.groupby(text, key=_is_word_char):
        if is_word:
            yield "".join(chars)


def _word_terms(word: str) -> list[str]:
    """Compound term of ``word`` plus, when it contains joiners, each non-empty part."""
    compound = word.strip(_JOINERS).translate(_ASCII_LOWERCASE)
    if not compound:
        return []
    parts = _JOINER_RUNS.split(compound)
    return [compound, *parts] if len(parts) > 1 else [compound]
