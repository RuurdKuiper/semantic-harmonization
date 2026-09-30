"""Tests for full clinical coding-system reference loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.code_systems import (
    load_code_system_corpus,
    load_icd10cm_full,
    load_icd9cm_full,
    load_icpc_full,
    load_mdr_full,
    load_rcd2_full,
    load_snomedct_full,
)


@pytest.mark.parametrize(
    "loader,vocabulary",
    [
        (load_icd10cm_full, "ICD10CM"),
        (load_icd9cm_full, "ICD9CM"),
        (load_icpc_full, "ICPC"),
        (load_mdr_full, "MDR"),
        (load_rcd2_full, "RCD2"),
        (load_snomedct_full, "SNOMEDCT_US"),
    ],
)
def test_full_code_system_loaders_schema(loader, vocabulary):
    codes = loader()
    assert {"code", "description", "vocabulary"}.issubset(codes.columns)
    assert (codes["vocabulary"] == vocabulary).all()
    assert len(codes) > 0
    assert not codes.duplicated(subset=["code", "vocabulary"]).any()


def test_load_icd10cm_full_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_icd10cm_full(tmp_path / "does-not-exist.csv")


def test_load_code_system_corpus_unknown_vocabulary_raises(tmp_path):
    with pytest.raises(ValueError):
        load_code_system_corpus(["NOT_A_REAL_VOCAB"], cache_dir=tmp_path)


def test_load_code_system_corpus_empty_list_raises(tmp_path):
    with pytest.raises(ValueError):
        load_code_system_corpus([], cache_dir=tmp_path)


def test_load_code_system_corpus_caches_to_parquet(tmp_path):
    cache_dir = tmp_path / "processed"
    codes = load_code_system_corpus(["ICD10CM", "ICPC"], cache_dir=cache_dir)
    cache_path = cache_dir / "codes_ICD10CM.parquet"
    assert cache_path.exists()

    # Second call should read from cache (fast path) and return identical data.
    cached_codes = load_code_system_corpus(["ICD10CM", "ICPC"], cache_dir=cache_dir)
    pd.testing.assert_frame_equal(codes.reset_index(drop=True), cached_codes.reset_index(drop=True))
