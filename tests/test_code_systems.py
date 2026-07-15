"""Tests for full clinical coding-system reference loaders (ICD10 etc.)."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.data.code_systems import load_code_system_corpus, load_icd10_full


def test_load_icd10_full_schema_and_size():
    codes = load_icd10_full()
    assert {"code", "description", "vocabulary"}.issubset(codes.columns)
    assert (codes["vocabulary"] == "ICD10CM").all()
    assert len(codes) > 1000
    assert not codes.duplicated(subset=["code", "vocabulary"]).any()


def test_load_icd10_full_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_icd10_full(tmp_path / "does-not-exist.xlsx")


def test_load_code_system_corpus_unknown_vocabulary_raises(tmp_path):
    with pytest.raises(ValueError):
        load_code_system_corpus(["NOT_A_REAL_VOCAB"], cache_dir=tmp_path)


def test_load_code_system_corpus_empty_list_raises(tmp_path):
    with pytest.raises(ValueError):
        load_code_system_corpus([], cache_dir=tmp_path)


def test_load_code_system_corpus_caches_to_parquet(tmp_path):
    cache_dir = tmp_path / "processed"
    codes = load_code_system_corpus(["ICD10CM"], cache_dir=cache_dir)
    cache_path = cache_dir / "codes_ICD10CM.parquet"
    assert cache_path.exists()

    # Second call should read from cache (fast path) and return identical data.
    cached_codes = load_code_system_corpus(["ICD10CM"], cache_dir=cache_dir)
    pd.testing.assert_frame_equal(codes.reset_index(drop=True), cached_codes.reset_index(drop=True))

    shutil.rmtree(cache_dir)
