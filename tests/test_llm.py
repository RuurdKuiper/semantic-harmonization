"""Tests for LLM-based ranking and classification, using mocked LLM calls."""

from __future__ import annotations

from unittest.mock import patch

from src.data.loaders import load_edf
from src.llm.classify import (
    AdaptiveStoppingConfig,
    ClassifiedCandidate,
    gpt_review_low_confidence,
    llm_classify,
)
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


def test_gpt_review_only_replaces_low_confidence_results():
    edf = load_edf("myocarditis")
    classified = [
        ClassifiedCandidate("A", "Certain", "TEST", "Narrow", 0.9, "Jev certain"),
        ClassifiedCandidate("B", "Uncertain", "TEST", "Exclude", 0.4, "Jev uncertain"),
    ]
    response = {
        "classifications": [
            {
                "code": "B",
                "vocabulary": "TEST",
                "label": "Narrow",
                "confidence": 0.95,
                "explanation": "GPT adjudication",
            }
        ]
    }

    with patch("src.llm.classify.call_llm_json", return_value=response) as call:
        reviewed, run = gpt_review_low_confidence(
            classified,
            edf,
            confidence_threshold=0.7,
            model="gpt-test",
            reasoning_effort="none",
        )

    assert call.call_count == 1
    assert reviewed[0] is classified[0]
    assert reviewed[1].label == "Narrow"
    assert reviewed[1].confidence == 0.95
    assert "previous label=Exclude" in reviewed[1].explanation
    assert run.candidates_reviewed == 1
    assert run.labels_changed == 1
    assert run.exclude_to_narrow == 1
    assert run.narrow_to_exclude == 0
    assert run.reasoning_effort == "none"
    assert call.call_args.kwargs["reasoning_effort"] == "none"


def test_gpt_review_skips_call_when_no_result_is_below_threshold():
    edf = load_edf("myocarditis")
    classified = [
        ClassifiedCandidate("A", "Certain", "TEST", "Narrow", 0.9, "Jev certain")
    ]
    with patch("src.llm.classify.call_llm_json") as call:
        reviewed, run = gpt_review_low_confidence(classified, edf, confidence_threshold=0.7)
    call.assert_not_called()
    assert reviewed == classified
    assert run.candidates_reviewed == 0


def test_gpt_review_caps_selection_at_lowest_confidence_candidates():
    edf = load_edf("myocarditis")
    classified = [
        ClassifiedCandidate("A", "A", "TEST", "Exclude", 0.6, "Jev"),
        ClassifiedCandidate("B", "B", "TEST", "Exclude", 0.1, "Jev"),
        ClassifiedCandidate("C", "C", "TEST", "Exclude", 0.4, "Jev"),
    ]
    response = {
        "classifications": [
            {"code": "B", "vocabulary": "TEST", "label": "Narrow", "confidence": 0.9},
            {"code": "C", "vocabulary": "TEST", "label": "Narrow", "confidence": 0.9},
        ]
    }

    with patch("src.llm.classify.call_llm_json", return_value=response) as call:
        reviewed, run = gpt_review_low_confidence(
            classified,
            edf,
            confidence_threshold=0.7,
            max_candidates=2,
        )

    sent_prompt = call.call_args.kwargs["user_prompt"]
    assert "B" in sent_prompt and "C" in sent_prompt
    assert reviewed[0].label == "Exclude"
    assert reviewed[1].label == "Narrow"
    assert reviewed[2].label == "Narrow"
    assert run.eligible_candidates == 3
    assert run.candidates_reviewed == 2
    assert run.max_candidates == 2


def test_llm_classify_with_jev_typed_choices():
    edf = load_edf("kidney_disease")
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
    assert "classify as Exclude because Possible is disabled" in call.call_args.kwargs["state"]


def _adaptive_candidates(count: int = 8):
    return [
        HybridCandidate(
            rank=index + 1,
            code=f"C{index}",
            description=f"Candidate {index}",
            vocabulary="TEST",
            lexical_score=1.0 - index / count,
            embedding_score=1.0 - index / count,
            score=1.0 - index / count,
        )
        for index in range(count)
    ]


def test_llm_classify_stops_after_consecutive_sparse_batches():
    edf = load_edf("myocarditis")
    candidates = _adaptive_candidates()
    responses = [
        {
            "classifications": [
                {"code": "C0", "vocabulary": "TEST", "label": "Narrow"},
                {"code": "C1", "vocabulary": "TEST", "label": "Narrow"},
            ]
        },
        {
            "classifications": [
                {"code": "C2", "vocabulary": "TEST", "label": "Narrow"},
                {"code": "C3", "vocabulary": "TEST", "label": "Exclude"},
            ]
        },
        {
            "classifications": [
                {"code": "C4", "vocabulary": "TEST", "label": "Exclude"},
                {"code": "C5", "vocabulary": "TEST", "label": "Exclude"},
            ]
        },
    ]
    run_info = []
    with patch("src.llm.classify.call_llm_json", side_effect=responses) as call:
        classified = llm_classify(
            candidates,
            edf,
            provider="openai",
            model="test-model",
            batch_size=2,
            adaptive_stopping=AdaptiveStoppingConfig(
                sparse_narrow_threshold=1,
                consecutive_sparse_batches=2,
                minimum_batches=3,
            ),
            run_info_callback=run_info.append,
        )

    assert len(classified) == 6
    assert call.call_count == 3
    assert run_info[0].stopped_early is True
    assert run_info[0].narrow_counts_by_batch == (2, 1, 0)
    assert run_info[0].candidates_classified == 6


def test_jev_classification_uses_adaptive_stopping():
    edf = load_edf("myocarditis")
    candidates = _adaptive_candidates(6)
    response = {
        "answers": {
            "candidate_0": {"choice": "Exclude", "probabilities": {"Narrow": 0.1, "Exclude": 0.9}},
            "candidate_1": {"choice": "Exclude", "probabilities": {"Narrow": 0.1, "Exclude": 0.9}},
        }
    }
    run_info = []
    with patch("src.llm.classify.call_jev_decisions", return_value=response) as call:
        classified = llm_classify(
            candidates,
            edf,
            provider="jev",
            model="jev-test",
            batch_size=2,
            adaptive_stopping=AdaptiveStoppingConfig(
                sparse_narrow_threshold=0,
                consecutive_sparse_batches=2,
                minimum_batches=2,
            ),
            run_info_callback=run_info.append,
        )

    assert len(classified) == 4
    assert call.call_count == 2
    assert run_info[0].narrow_counts_by_batch == (0, 0)
    assert run_info[0].stopped_early is True
