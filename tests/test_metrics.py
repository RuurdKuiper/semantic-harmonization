"""Tests for retrieval and classification evaluation metrics."""

from __future__ import annotations

from src.evaluation.metrics import (
    evaluate,
    evaluate_classification,
    evaluate_retrieval,
    filter_gold_by_vocabulary,
    filter_records_by_vocabulary,
    to_predicted_codelist,
)


def test_evaluate_retrieval_perfect_match(sample_gold):
    retrieved = [
        {"code": "I40.0", "vocabulary": "ICD10"},
        {"code": "I40.9", "vocabulary": "ICD10"},
        {"code": "50920009", "vocabulary": "SNOMED"},
    ]
    metrics = evaluate_retrieval(retrieved, sample_gold)
    assert metrics.sensitivity == 1.0
    assert metrics.precision == 1.0
    assert metrics.f1 == 1.0


def test_evaluate_retrieval_false_positive_from_exclude(sample_gold):
    retrieved = [
        {"code": "I40.0", "vocabulary": "ICD10"},
        {"code": "I30.9", "vocabulary": "ICD10"},  # gold-labeled Exclude
    ]
    metrics = evaluate_retrieval(retrieved, sample_gold)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 2


def test_evaluate_retrieval_false_negative(sample_gold):
    retrieved: list[dict] = []
    metrics = evaluate_retrieval(retrieved, sample_gold)
    assert metrics.sensitivity == 0.0
    assert metrics.false_negatives == 3


def test_evaluate_classification_perfect_agreement(sample_gold):
    classified = [
        {"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow"},
        {"code": "I40.9", "vocabulary": "ICD10", "label": "Narrow"},
        {"code": "I30.9", "vocabulary": "ICD10", "label": "Exclude"},
        {"code": "50920009", "vocabulary": "SNOMED", "label": "Narrow"},
    ]
    metrics = evaluate_classification(classified, sample_gold)
    assert metrics.accuracy == 1.0
    assert metrics.cohens_kappa == 1.0


def test_evaluate_classification_partial_disagreement(sample_gold):
    classified = [
        {"code": "I40.0", "vocabulary": "ICD10", "label": "Possible"},
        {"code": "I40.9", "vocabulary": "ICD10", "label": "Narrow"},
        {"code": "I30.9", "vocabulary": "ICD10", "label": "Exclude"},
        {"code": "50920009", "vocabulary": "SNOMED", "label": "Narrow"},
    ]
    metrics = evaluate_classification(classified, sample_gold)
    assert metrics.accuracy == 0.75
    assert metrics.cohens_kappa < 1.0


def test_evaluate_combines_both(sample_gold):
    classified = [
        {"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow"},
        {"code": "I40.9", "vocabulary": "ICD10", "label": "Narrow"},
    ]
    result = evaluate(classified, sample_gold)
    assert "retrieval" in result
    assert "classification" in result
    assert result["retrieval"]["true_positives"] == 2


def test_filter_records_by_vocabulary():
    records = [
        {"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow"},
        {"code": "50920009", "vocabulary": "SNOMED", "label": "Narrow"},
    ]
    filtered = filter_records_by_vocabulary(records, ["icd10"])
    assert filtered == [{"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow"}]


def test_filter_gold_by_vocabulary(sample_gold):
    filtered = filter_gold_by_vocabulary(sample_gold, ["ICD10"])
    assert set(filtered["vocabulary"]) == {"ICD10"}
    assert len(filtered) == 3


def test_to_predicted_codelist_schema_and_values():
    classified = [
        {
            "code": "I40.0",
            "vocabulary": "ICD10CM",
            "description": "Infective myocarditis",
            "label": "Narrow",
        }
    ]
    df = to_predicted_codelist(classified)
    assert list(df.columns) == ["coding_system", "code", "code_name", "concept", "concept_name", "tags"]
    row = df.iloc[0]
    assert row["coding_system"] == "ICD10CM"
    assert row["code"] == "I40.0"
    assert row["code_name"] == "Infective myocarditis"
    assert row["concept"] == ""
    assert row["concept_name"] == "Infective myocarditis"
    assert row["tags"] == "narrow"


def test_to_predicted_codelist_empty():
    df = to_predicted_codelist([])
    assert list(df.columns) == ["coding_system", "code", "code_name", "concept", "concept_name", "tags"]
    assert len(df) == 0
