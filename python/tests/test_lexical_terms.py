# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""lexical-v1 tokenizer, identifier and key-encoding tests, driven by the shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from strands.types.exceptions import StorageError

from strands_dynamodb_storage.lexical_terms import (
    LEXICAL_TOKENIZER_VERSION,
    identifier_posting_pk,
    manifest_pk,
    normalize_identifier,
    normalize_identifiers,
    posting_partition_keys,
    scope_segment,
    term_posting_pk,
    text_terms,
    utf8_byte_length,
)

FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "testdata" / "lexical-v1.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
POSTING_KEY_ENCODERS = {"term": term_posting_pk, "identifier": identifier_posting_pk}
LONE_SURROGATE_SAMPLES = [chr(0xD800), f"a{chr(0xDFFF)}b", chr(0xD83D) + chr(0xDE00)]


def test_fixtures_match_tokenizer_version():
    assert FIXTURES["tokenizerVersion"] == LEXICAL_TOKENIZER_VERSION == "lexical-v1"


@pytest.mark.parametrize("case", FIXTURES["textTerms"], ids=lambda case: ascii(case["input"]))
def test_text_terms_match_fixture(case):
    assert text_terms(case["input"]) == case["terms"]


@pytest.mark.parametrize("case", FIXTURES["identifiers"]["valid"], ids=lambda case: ascii(case["input"]))
def test_valid_identifier_normalizes_to_fixture(case):
    assert normalize_identifier(case["input"]) == case["normalized"]


@pytest.mark.parametrize("identifier", FIXTURES["identifiers"]["invalid"], ids=ascii)
def test_invalid_identifier_is_rejected_without_echoing_it(identifier):
    with pytest.raises(StorageError) as error:
        normalize_identifier(identifier)
    if identifier:
        assert identifier not in str(error.value)


@pytest.mark.parametrize("case", FIXTURES["keys"], ids=lambda case: ascii(case["pk"]))
def test_index_keys_match_fixture(case):
    if case["kind"] == "manifest":
        assert manifest_pk(case["scope"]) == case["pk"]
    else:
        assert POSTING_KEY_ENCODERS[case["kind"]](case["scope"], case["value"]) == case["pk"]


def test_leading_zeroes_and_joiners_stay_distinct():
    assert "fc-00123" in text_terms("FC-00123")
    assert "fc-00123" not in text_terms("FC-123")
    assert term_posting_pk("", "fc-00123") != term_posting_pk("", "fc-123")


def test_normalize_identifiers_deduplicates_after_nfc_keeping_first():
    decomposed, composed = "A\u0308-1", "\u00c4-1"
    assert normalize_identifiers(["FC-1", decomposed, "FC-1", composed, "fc-1"]) == ["FC-1", composed, "fc-1"]


def test_posting_partition_keys_lists_terms_then_identifiers_in_order():
    assert posting_partition_keys("s/", ["b", "a"], ["ID-1"]) == [
        term_posting_pk("s/", "b"),
        term_posting_pk("s/", "a"),
        identifier_posting_pk("s/", "ID-1"),
    ]


def test_scope_segment_is_length_prefixed_and_never_empty():
    assert scope_segment("") == "0:"
    assert scope_segment("tenant/a/") == "9:tenant/a/"
    assert scope_segment("t\u00e9/") == "4:t\u00e9/"


def test_utf8_byte_length_counts_bytes_not_code_points():
    assert utf8_byte_length("\u65e5\u672c") == 6
    assert utf8_byte_length("\U0001f680") == 4


def test_python_string_order_equals_utf8_byte_order():
    samples = ["z", "\u00e9", "\uffff", "\U0001f680", "A", "a"]
    assert sorted(samples) == sorted(samples, key=lambda value: value.encode("utf-8"))


def test_posting_partition_keys_deduplicate_keeping_the_first_occurrence():
    assert posting_partition_keys("s/", ["b", "a", "b"], ["ID-1", "ID-1"]) == [
        term_posting_pk("s/", "b"),
        term_posting_pk("s/", "a"),
        identifier_posting_pk("s/", "ID-1"),
    ]


@pytest.mark.parametrize("malformed", LONE_SURROGATE_SAMPLES, ids=["high", "low", "split-pair"])
def test_text_terms_reject_lone_surrogates_without_echoing_the_text(malformed):
    with pytest.raises(StorageError, match="Text must be well-formed Unicode; it contains a lone surrogate") as error:
        text_terms(f"secret {malformed}")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("malformed", LONE_SURROGATE_SAMPLES, ids=["high", "low", "split-pair"])
def test_normalize_identifier_rejects_lone_surrogates_without_echoing_them(malformed):
    expected = "Identifier must be well-formed Unicode; it contains a lone surrogate"
    with pytest.raises(StorageError, match=expected) as error:
        normalize_identifier(f"SECRET-{malformed}")
    assert "SECRET" not in str(error.value)


def test_utf8_byte_length_counts_a_lone_surrogate_like_text_encoder():
    assert utf8_byte_length(f"a{chr(0xD800)}") == 4
