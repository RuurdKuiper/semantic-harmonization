"""Tests for text/code preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from src.data.preprocessing import (
    deduplicate_codes,
    normalize_text,
    preprocess_corpus,
    standardize_code,
)


def test_normalize_text_basic():
    assert normalize_text("Acute Myocarditis, Unspecified!") == "acute myocarditis unspecified"


def test_normalize_text_accents():
    assert normalize_text("Café") == "cafe"


def test_normalize_text_non_string():
    assert normalize_text(None) == ""  # type: ignore[arg-type]


def test_standardize_code_icd10_adds_dot():
    assert standardize_code("I409", "ICD10") == "I40.9"


def test_standardize_code_preserves_existing_dot():
    assert standardize_code("I40.9", "ICD10") == "I40.9"


def test_standardize_code_snomed_unchanged_format():
    assert standardize_code("50920009", "SNOMED") == "50920009"


def test_standardize_code_uppercases_and_strips_whitespace():
    assert standardize_code(" i40.9 ", "icd10") == "I40.9"


def test_deduplicate_codes_removes_duplicates():
    df = pd.DataFrame(
        [
            {"code": "I40.9", "description": "A", "vocabulary": "ICD10"},
            {"code": "i40.9", "description": "A duplicate", "vocabulary": "ICD10"},
            {"code": "I40.0", "description": "B", "vocabulary": "ICD10"},
        ]
    )
    result = deduplicate_codes(df)
    assert len(result) == 2


def test_preprocess_corpus_adds_normalized_description(sample_codes):
    result = preprocess_corpus(sample_codes)
    assert "normalized_description" in result.columns
    assert result["normalized_description"].iloc[0] == normalize_text(sample_codes["description"].iloc[0])
