"""Tests for LLM-based ranking and classification, using mocked LLM calls."""

from __future__ import annotations

from unittest.mock import patch

from src.data.loaders import load_edf
from src.llm.classify import llm_classify
from src.llm.rank import llm_rank
from src.retrieval.hybrid import HybridCandidate


def _candidates() -> list[HybridCandidate]:
    return [
        HybridCandidate(
            rank=1, code="I40.0", description="Infective myocarditis", vocabulary="ICD10",
            lexical_score=1.0, embedding_score=0.9, score=0.95,
        ),
        HybridCandidate(
            rank=2, code="I30.9", description="Acute pericarditis unspecified", vocabulary="ICD10",
            lexical_score=0.3, embedding_score=0.4, score=0.35,
        ),
    ]


def test_llm_rank_reorders_by_relevance_score():
    edf = load_edf("myocarditis")
    fake_response = {
        "ranked_codes": [
            {"code": "I30.9", "vocabulary": "ICD10", "relevance_score": 0.9},
            {"code": "I40.0", "vocabulary": "ICD10", "relevance_score": 0.2},
        ]
    }
    with patch("src.llm.rank.call_llm_json", return_value=fake_response):
        ranked = llm_rank(_candidates(), edf, model="test-model")

    assert ranked[0].code == "I30.9"
    assert ranked[0].rank == 1
    assert ranked[1].code == "I40.0"


def test_llm_rank_appends_omitted_candidates():
    edf = load_edf("myocarditis")
    fake_response = {"ranked_codes": [{"code": "I40.0", "vocabulary": "ICD10", "relevance_score": 0.9}]}
    with patch("src.llm.rank.call_llm_json", return_value=fake_response):
        ranked = llm_rank(_candidates(), edf, model="test-model")

    assert len(ranked) == 2
    assert {r.code for r in ranked} == {"I40.0", "I30.9"}


def test_llm_rank_empty_candidates_returns_empty():
    edf = load_edf("myocarditis")
    assert llm_rank([], edf) == []


def test_llm_classify_assigns_labels():
    edf = load_edf("myocarditis")
    fake_rank_response = {
        "ranked_codes": [
            {"code": "I40.0", "vocabulary": "ICD10", "relevance_score": 0.9},
            {"code": "I30.9", "vocabulary": "ICD10", "relevance_score": 0.2},
        ]
    }
    with patch("src.llm.rank.call_llm_json", return_value=fake_rank_response):
        ranked = llm_rank(_candidates(), edf, model="test-model")

    fake_classify_response = {
        "classifications": [
            {"code": "I40.0", "vocabulary": "ICD10", "label": "Narrow", "confidence": 0.95, "explanation": "specific"},
            {"code": "I30.9", "vocabulary": "ICD10", "label": "Exclude", "confidence": 0.9, "explanation": "not relevant"},
        ]
    }
    with patch("src.llm.classify.call_llm_json", return_value=fake_classify_response):
        classified = llm_classify(ranked, edf, provider="openai", model="test-model")

    labels = {c.code: c.label for c in classified}
    assert labels["I40.0"] == "Narrow"
    assert labels["I30.9"] == "Exclude"


def test_llm_classify_invalid_label_defaults_to_exclude_when_possible_disabled():
    edf = load_edf("myocarditis")
    candidates = _candidates()
    with patch("src.llm.rank.call_llm_json", return_value={"ranked_codes": []}):
        ranked = llm_rank(candidates, edf, model="test-model")

    fake_response = {
        "classifications": [
            {"code": "I40.0", "vocabulary": "ICD10", "label": "Unknown", "confidence": 0.5, "explanation": "x"},
        ]
    }
    with patch("src.llm.classify.call_llm_json", return_value=fake_response):
        classified = llm_classify(ranked, edf, provider="openai", model="test-model")

    result = next(c for c in classified if c.code == "I40.0")
    assert result.label == "Exclude"


def test_llm_classify_missing_classification_defaults_to_exclude_low_confidence():
    edf = load_edf("myocarditis")
    candidates = _candidates()
    with patch("src.llm.rank.call_llm_json", return_value={"ranked_codes": []}):
        ranked = llm_rank(candidates, edf, model="test-model")

    with patch("src.llm.classify.call_llm_json", return_value={"classifications": []}):
        classified = llm_classify(ranked, edf, provider="openai", model="test-model")

    assert len(classified) == 2
    assert all(c.label == "Exclude" and c.confidence == 0.0 for c in classified)


def test_llm_classify_possible_can_be_enabled():
    edf = load_edf("myocarditis")
    with patch("src.llm.rank.call_llm_json", return_value={"ranked_codes": []}):
        ranked = llm_rank(_candidates(), edf, model="test-model")
    response = {
        "classifications": [
            {"code": "I40.0", "vocabulary": "ICD10", "label": "Possible", "confidence": 0.8, "explanation": "x"},
        ]
    }
    with patch("src.llm.classify.call_llm_json", return_value=response):
        classified = llm_classify(
            ranked,
            edf,
            provider="openai",
            model="test-model",
            use_possible_category=True,
        )
    assert next(c for c in classified if c.code == "I40.0").label == "Possible"


def test_llm_classify_empty_candidates_returns_empty():
    edf = load_edf("myocarditis")
    assert llm_classify([], edf) == []


def test_llm_classify_with_jev_typed_choices():
    edf = load_edf("myocarditis")
    with patch("src.llm.rank.call_llm_json", return_value={"ranked_codes": []}):
        ranked = llm_rank(_candidates(), edf, model="test-model")
    response = {
        "answers": {
            "candidate_0": {
                "type": "choice",
                "choice": "Narrow",
                "probabilities": {"Narrow": 0.97, "Exclude": 0.03},
                "confidence": 0.94,
            },
            "candidate_1": {
                "type": "choice",
                "choice": "Exclude",
                "probabilities": {"Narrow": 0.04, "Exclude": 0.96},
                "confidence": 0.92,
            },
        }
    }
    with patch("src.llm.classify.call_jev_decisions", return_value=response) as call:
        classified = llm_classify(ranked, edf, provider="jev", model="jev-test", batch_size=20)

    assert [item.label for item in classified] == ["Narrow", "Exclude"]
    assert classified[0].confidence == 0.94
    assert "0.97" in classified[0].explanation
    assert len(call.call_args.kwargs["questions"]) == 2
