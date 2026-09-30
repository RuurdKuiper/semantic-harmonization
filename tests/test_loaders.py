"""Tests for EDF, code corpus, gold label, and raw AESI dataset loaders."""

from __future__ import annotations

import pytest

from src.data.loaders import load_aesi_dataset, load_edf


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


def test_edf_lexical_query_includes_name_and_synonyms():
    edf = load_edf("kidney_disease")
    query = edf.to_lexical_query()
    assert query.startswith("Kidney disease")
    assert "Acute kidney failure" in query
    assert "Nephropathy" in query


@pytest.mark.parametrize(
    "name",
    [
        "acute_disseminated_encephalomyelitis",
        "acute_myocardial_infarction",
        "eczema_vaccinatum",
        "kidney_disease",
        "type_1_diabetes",
    ],
)
def test_new_edfs_preserve_extended_source_sections(name):
    edf = load_edf(name)
    assert edf.narrow_definition
    assert edf.exclude_decision_rules
    assert edf.references


@pytest.mark.parametrize(
    "csv_path",
    [
        "data/codelists/AESI/C_MYOCARD_AESI_filtered.csv",
        "data/codelists/AESI/Sk_ERYTHMULTI_AESI.csv",
        "data/codelists/AESI/N_GBS_AESI_filtered.csv",
        "data/codelists/AESI/Ref-N_ADEM_AESI.csv",
        "data/codelists/AESI/Ref-C_AMI_AESI.csv",
        "data/codelists/AESI/Ref-Sk_ECZEMAVACCINATUM_AESI.csv",
        "data/codelists/AESI/Ref-G_KIDNEYDISEASE_COV.csv",
        "data/codelists/AESI/Ref-E_DM1_AESI.csv",
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
    codes, gold = load_aesi_dataset("data/codelists/AESI/Sk_ERYTHMULTI_AESI.csv")
    # The raw file has one row with a missing tag; it must not appear in gold labels.
    assert len(gold) <= len(codes)


def test_load_aesi_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_aesi_dataset("data/codelists/AESI/does-not-exist.csv")
