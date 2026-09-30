"""Select uncertain classifications requiring targeted human review."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.classify import ClassifiedCandidate


@dataclass(slots=True)
class ReviewItem:
    code: str
    description: str
    vocabulary: str
    label: str
    confidence: float
    explanation: str
    reason: str


def select_uncertain(
    classified: list[ClassifiedCandidate],
    confidence_threshold: float = 0.7,
    possible_requires_review: bool = False,
) -> list[ReviewItem]:
    """Select classified candidates that require human validation.

    A candidate is flagged for review when:
    - its LLM confidence is below *confidence_threshold*, or
    - it was classified as 'Possible' and *possible_requires_review* is True.

    Parameters
    ----------
    classified : list[ClassifiedCandidate]
        Output of :func:`src.llm.classify.llm_classify`.
    confidence_threshold : float
        Minimum confidence below which a candidate is routed for review.
    possible_requires_review : bool
        Whether 'Possible' classifications are always routed for review,
        regardless of confidence.

    Returns
    -------
    list[ReviewItem]
        Items requiring human review, each annotated with the reason.
    """
    review_items: list[ReviewItem] = []
    for c in classified:
        reasons: list[str] = []
        if c.confidence < confidence_threshold:
            reasons.append(f"confidence {c.confidence:.2f} below threshold {confidence_threshold:.2f}")
        if possible_requires_review and c.label == "Possible":
            reasons.append("classified as 'Possible'")

        if reasons:
            review_items.append(
                ReviewItem(
                    code=c.code,
                    description=c.description,
                    vocabulary=c.vocabulary,
                    label=c.label,
                    confidence=c.confidence,
                    explanation=c.explanation,
                    reason="; ".join(reasons),
                )
            )
    return review_items


def review_rate(classified: list[ClassifiedCandidate], review_items: list[ReviewItem]) -> float:
    """Fraction of classified candidates flagged for human review."""
    if not classified:
        return 0.0
    return len(review_items) / len(classified)
