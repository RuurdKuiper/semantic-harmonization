"""Tests for retrieval and classification evaluation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import (
    evaluate,
    evaluate_classification,
    evaluate_gpt_review,
    evaluate_retrieval,
    filter_gold_by_available_codes,
    filter_gold_by_vocabulary,
    filter_records_by_vocabulary,
    narrow_loss_breakdown,
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
    assert metrics.true_negatives == 1
    assert metrics.sensitivity == 1.0
    assert metrics.precision == 1.0
    assert metrics.f1 == 1.0
    assert metrics.macro_f1 == 1.0


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
    assert {"sensitivity", "precision", "f1", "macro_f1"}.issubset(result["classification"])


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


def test_filter_gold_by_available_codes_standardizes_keys(sample_gold):
    available = pd.DataFrame(
        [
            {"code": "I400", "vocabulary": "ICD10"},
            {"code": "50920009", "vocabulary": "SNOMED"},
        ]
    )
    filtered = filter_gold_by_available_codes(sample_gold, available)
    assert set(zip(filtered["code"], filtered["vocabulary"])) == {
        ("I40.0", "ICD10"),
        ("50920009", "SNOMED"),
    }


def test_filter_gold_by_available_codes_rejects_invalid_corpus(sample_gold):
    with pytest.raises(ValueError):
        filter_gold_by_available_codes(sample_gold, pd.DataFrame({"code": ["I40.0"]}))


def test_narrow_loss_breakdown_separates_unclassified_and_misclassified():
    gold = pd.DataFrame(
        [
            {"code": "I400", "vocabulary": "ICD10CM", "label": "Narrow"},
            {"code": "I40.1", "vocabulary": "ICD10CM", "label": "Narrow"},
            {"code": "I40.2", "vocabulary": "ICD10CM", "label": "Narrow"},
            {"code": "I30.9", "vocabulary": "ICD10CM", "label": "Exclude"},
        ]
    )
    classified = [
        {"code": "I40.0", "vocabulary": "icd10cm", "label": "Narrow"},
        {"code": "I40.1", "vocabulary": "ICD10CM", "label": "Exclude"},
    ]

    losses = narrow_loss_breakdown(classified, gold)

    assert losses.gold_narrow_available == 3
    assert losses.correctly_classified == 1
    assert losses.misclassified == 1
    assert losses.not_classified == 1
    assert losses.total_missed == 2
    assert losses.end_to_end_recall == pytest.approx(1 / 3, abs=0.0001)


def test_evaluate_gpt_review_counts_corrected_and_harmful_changes():
    gold = pd.DataFrame(
        [
            {"code": "A", "vocabulary": "TEST", "label": "Narrow"},
            {"code": "B", "vocabulary": "TEST", "label": "Exclude"},
            {"code": "C", "vocabulary": "TEST", "label": "Narrow"},
        ]
    )
    stage2 = [
        {"code": "A", "vocabulary": "TEST", "label": "Exclude", "confidence": 0.1},
        {"code": "B", "vocabulary": "TEST", "label": "Exclude", "confidence": 0.2},
        {"code": "C", "vocabulary": "TEST", "label": "Narrow", "confidence": 0.3},
        {"code": "D", "vocabulary": "TEST", "label": "Exclude", "confidence": 0.9},
    ]
    final = [
        {"code": "A", "vocabulary": "TEST", "label": "Narrow"},
        {"code": "B", "vocabulary": "TEST", "label": "Narrow"},
        {"code": "C", "vocabulary": "TEST", "label": "Narrow"},
        {"code": "D", "vocabulary": "TEST", "label": "Exclude"},
    ]

    result = evaluate_gpt_review(
        stage2, final, gold, confidence_threshold=0.7, include_possible=False
    )

    assert result.candidates_reviewed == 3
    assert result.labels_changed == 2
    assert result.corrected_changes == 1
    assert result.harmful_changes == 1
    assert result.unchanged == 1
    assert result.correct_before == 2
    assert result.correct_after == 2
    assert result.net_correct_change == 0


def test_evaluate_gpt_review_respects_lowest_confidence_cap():
    stage2 = [
        {"code": "A", "vocabulary": "TEST", "label": "Exclude", "confidence": 0.6},
        {"code": "B", "vocabulary": "TEST", "label": "Exclude", "confidence": 0.1},
    ]
    final = [
        {"code": "A", "vocabulary": "TEST", "label": "Narrow"},
        {"code": "B", "vocabulary": "TEST", "label": "Narrow"},
    ]
    gold = pd.DataFrame(
        [
            {"code": "A", "vocabulary": "TEST", "label": "Narrow"},
            {"code": "B", "vocabulary": "TEST", "label": "Narrow"},
        ]
    )

    result = evaluate_gpt_review(
        stage2,
        final,
        gold,
        confidence_threshold=0.7,
        max_candidates=1,
    )

    assert result.candidates_reviewed == 1
    assert result.corrected_changes == 1


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


def test_to_predicted_codelist_includes_review_annotations_without_filtering():
    classified = [
        {
            "code": "I40.0",
            "vocabulary": "ICD10CM",
            "description": "Infective myocarditis",
            "label": "Narrow",
            "confidence": 0.5,
            "explanation": "specific",
            "manual_review": True,
            "review_reason": "low confidence",
        }
    ]
    df = to_predicted_codelist(classified)
    assert len(df) == 1
    assert bool(df.iloc[0]["manual_review"]) is True
    assert df.iloc[0]["tags"] == "narrow"


def test_binary_evaluation_collapses_possible_gold_to_exclude():
    gold = pd.DataFrame(
        [{"code": "X", "vocabulary": "TEST", "label": "Possible"}]
    )
    metrics = evaluate_classification(
        [{"code": "X", "vocabulary": "TEST", "label": "Exclude"}],
        gold,
        include_possible=False,
    )
    assert metrics.accuracy == 1.0
    assert metrics.true_negatives == 1
    assert "Possible" not in metrics.per_label


def test_multiclass_narrow_true_negative_is_one_vs_rest():
    gold = pd.DataFrame(
        [
            {"code": "N", "vocabulary": "TEST", "label": "Narrow"},
            {"code": "P", "vocabulary": "TEST", "label": "Possible"},
            {"code": "E", "vocabulary": "TEST", "label": "Exclude"},
        ]
    )
    classified = [
        {"code": "N", "vocabulary": "TEST", "label": "Narrow"},
        {"code": "P", "vocabulary": "TEST", "label": "Exclude"},
        {"code": "E", "vocabulary": "TEST", "label": "Possible"},
    ]
    metrics = evaluate_classification(classified, gold, include_possible=True)
    assert metrics.true_negatives == 2
    assert metrics.per_label["Narrow"]["tn"] == 2


def test_code_absent_from_gold_is_implicit_exclude():
    gold = pd.DataFrame(columns=["code", "vocabulary", "label"])
    metrics = evaluate_classification(
        [{"code": "X", "vocabulary": "TEST", "label": "Narrow"}],
        gold,
        include_possible=False,
    )
    assert metrics.accuracy == 0.0
    assert metrics.per_label["Narrow"]["fp"] == 1
