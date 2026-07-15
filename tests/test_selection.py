"""Tests for uncertainty-based selection of candidates for human review."""

from __future__ import annotations

from src.llm.classify import ClassifiedCandidate
from src.uncertainty.selection import review_rate, select_uncertain


def _candidate(label: str, confidence: float) -> ClassifiedCandidate:
    return ClassifiedCandidate(
        code="I40.0",
        description="Infective myocarditis",
        vocabulary="ICD10",
        label=label,
        confidence=confidence,
        explanation="test",
    )


def test_select_uncertain_flags_low_confidence():
    classified = [_candidate("Narrow", 0.5)]
    review_items = select_uncertain(classified, confidence_threshold=0.7)
    assert len(review_items) == 1
    assert "confidence" in review_items[0].reason


def test_select_uncertain_flags_possible_regardless_of_confidence():
    classified = [_candidate("Possible", 0.95)]
    review_items = select_uncertain(classified, confidence_threshold=0.7, possible_requires_review=True)
    assert len(review_items) == 1
    assert "Possible" in review_items[0].reason


def test_select_uncertain_skips_confident_narrow():
    classified = [_candidate("Narrow", 0.95)]
    review_items = select_uncertain(classified, confidence_threshold=0.7)
    assert len(review_items) == 0


def test_select_uncertain_possible_not_required():
    classified = [_candidate("Possible", 0.95)]
    review_items = select_uncertain(classified, confidence_threshold=0.7, possible_requires_review=False)
    assert len(review_items) == 0


def test_review_rate():
    classified = [_candidate("Narrow", 0.95), _candidate("Possible", 0.4)]
    review_items = select_uncertain(classified, confidence_threshold=0.7)
    assert review_rate(classified, review_items) == 0.5


def test_review_rate_empty():
    assert review_rate([], []) == 0.0
