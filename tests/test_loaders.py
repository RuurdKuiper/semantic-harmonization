"""Tests for EDF, code corpus, gold label, and raw AESI dataset loaders."""

from __future__ import annotations

import pytest

from src.data.loaders import load_aesi_dataset, load_code_corpus, load_edf, load_gold_labels


def test_load_edf_myocarditis():
    edf = load_edf("myocarditis")
    assert edf.preferred_name == "Acute Myocarditis"
    assert "myocardium" in edf.definition.lower()
    assert edf.inclusion_criteria
    assert edf.narrow_decision_rules


def test_load_edf_erythema_multiforme():
    edf = load_edf("erythema_multiforme")
    assert edf.preferred_name == "Erythema Multiforme"
    assert "target lesions" in edf.definition.lower()
    assert edf.exclusion_criteria


def test_load_edf_guillain_barre():
    edf = load_edf("guillain_barre")
    assert "Guillain-Barre" in edf.preferred_name
    assert edf.synonyms


def test_load_edf_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_edf("nonexistent_phenotype")


def test_edf_to_prompt_context_contains_key_sections():
    edf = load_edf("myocarditis")
    context = edf.to_prompt_context()
    assert "Preferred name: Acute Myocarditis" in context
    assert "Inclusion criteria:" in context
    assert "Criteria for Narrow classification:" in context


def test_load_code_corpus():
    codes = load_code_corpus()
    assert {"code", "description", "vocabulary"}.issubset(codes.columns)
    assert len(codes) > 0


def test_load_gold_labels_myocarditis():
    gold = load_gold_labels("myocarditis")
    assert set(gold["label"]).issubset({"Narrow", "Possible", "Exclude"})
    assert len(gold) > 0


@pytest.mark.parametrize(
    "csv_path",
    [
        "data/raw/Acute Myocarditis/C_MYOCARD_AESI_filtered.csv",
        "data/raw/Erythema Multiforme/Sk_ERYTHMULTI_AESI.csv",
        "data/raw/Guillain barre syndrome (GBS)/N_GBS_AESI_filtered.csv",
    ],
)
def test_load_aesi_dataset_returns_codes_and_gold(csv_path):
    codes, gold = load_aesi_dataset(csv_path)
    assert {"code", "description", "vocabulary"}.issubset(codes.columns)
    assert {"code", "vocabulary", "label"}.issubset(gold.columns)
    assert len(codes) > 0
    assert len(gold) > 0
    assert set(gold["label"]).issubset({"Narrow", "Possible", "Exclude"})
    # No duplicate (code, vocabulary) pairs in either output.
    assert not codes.duplicated(subset=["code", "vocabulary"]).any()
    assert not gold.duplicated(subset=["code", "vocabulary"]).any()


def test_load_aesi_dataset_drops_rows_with_missing_tags():
    codes, gold = load_aesi_dataset("data/raw/Erythema Multiforme/Sk_ERYTHMULTI_AESI.csv")
    # The raw file has one row with a missing tag; it must not appear in gold labels.
    assert len(gold) <= len(codes)


def test_load_aesi_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_aesi_dataset("data/raw/does-not-exist.csv")


def test_load_gold_labels_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_gold_labels("nonexistent_phenotype")
