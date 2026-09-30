"""Tests for the deterministic experiment report generator."""

from __future__ import annotations

import json

from scripts.summarize_experiment import build_report


def test_build_report_includes_abstract_and_all_metrics(tmp_path):
    experiment = tmp_path / "experiment"
    setting = experiment / "top_k-50_lexical-0.5_embedding-0.5"
    setting.mkdir(parents=True)
    summary = [
        {
            "phenotype": "sample",
            "top_k": 50,
            "lexical_weight": 0.5,
            "embedding_weight": 0.5,
            "possible_category": False,
            "vocabularies": "ICD10CM",
            "review_rate": 0.1,
            "retrieval_sensitivity": 0.8,
            "retrieval_precision": 0.4,
            "retrieval_f1": 0.5333,
            "classification_accuracy": 0.9,
            "classification_true_negatives": 80,
            "classification_sensitivity": 0.75,
            "classification_precision": 0.6,
            "classification_f1": 0.6667,
            "classification_macro_sensitivity": 0.8,
            "classification_macro_precision": 0.75,
            "classification_macro_f1": 0.77,
            "cohens_kappa": 0.5,
        }
    ]
    (experiment / "summary.json").write_text(json.dumps(summary))
    details = {
        "metrics": {
            "retrieval": {"true_positives": 8, "false_positives": 12, "false_negatives": 2},
            "classification": {
                "per_label": {"Narrow": {"tp": 6, "tn": 80, "fp": 4, "fn": 2}}
            },
            "evaluation_scope": {
                "gold_rows_in_selected_vocabularies": 100,
                "gold_rows_available_in_loaded_sources": 90,
                "gold_rows_excluded_as_unavailable": 10,
            },
        }
    }
    (setting / "sample_metrics.json").write_text(json.dumps(details))

    report = build_report(experiment)

    assert "## Abstract" in report
    assert "TP=6, TN=80, FP=4, and FN=2" in report
    assert "80.0%" in report
    assert "Gold excluded" in report


def test_build_report_requires_summary(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        build_report(tmp_path)
