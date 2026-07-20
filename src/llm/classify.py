"""LLM-based classification of candidate codes into Narrow/Possible/Exclude."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from src.data.loaders import EventDefinitionForm
from src.llm.client import call_llm_json
from src.llm.prompts import CLASSIFY_SYSTEM_PROMPT, build_classify_prompt
from src.llm.rank import RankedCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"Narrow", "Possible", "Exclude"}


@dataclass(slots=True)
class ClassifiedCandidate:
    code: str
    description: str
    vocabulary: str
    label: str
    confidence: float
    explanation: str
    relevance_score: float = 0.0
    retrieval_score: float = 0.0


def llm_classify(
    candidates: list[RankedCandidate],
    edf: EventDefinitionForm,
    provider: str = "auto",
    model: str | None = None,
    max_retries: int = 3,
    batch_size: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ClassifiedCandidate]:
    """Classify ranked candidate codes as Narrow/Possible/Exclude using an LLM.

    Parameters
    ----------
    candidates : list[RankedCandidate]
        Candidates produced by :func:`src.llm.rank.llm_rank`.
    edf : EventDefinitionForm
        Structured phenotype definition providing semantic context.
    provider : str
        "auto", "anthropic", or "openai".
    model : str, optional
        Model identifier. Defaults to a provider-specific default.
    max_retries : int
        Number of LLM call attempts before failing.
    batch_size : int
        Maximum number of candidates to include in a single LLM request.
    progress_callback : callable, optional
        Callback invoked after each completed batch as ``callback(completed, total)``.

    Returns
    -------
    list[ClassifiedCandidate]
    """
    if not candidates:
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    lookup = {(c.code, c.vocabulary): c for c in candidates}
    results: list[ClassifiedCandidate] = []
    seen: set[tuple[str, str]] = set()

    total = len(candidates)
    for start in range(0, total, batch_size):
        batch = candidates[start : start + batch_size]
        payload = [
            {"code": c.code, "vocabulary": c.vocabulary, "description": c.description}
            for c in batch
        ]
        prompt = build_classify_prompt(edf, payload)
        response = call_llm_json(
            system_prompt=CLASSIFY_SYSTEM_PROMPT,
            user_prompt=prompt,
            provider=provider,
            model=model,
            max_retries=max_retries,
        )

        classifications_raw = response.get("classifications", [])
        batch_seen: set[tuple[str, str]] = set()
        for item in classifications_raw:
            key = (item.get("code", ""), item.get("vocabulary", ""))
            source = lookup.get(key)
            if source is None:
                continue
            label = item.get("label", "Possible")
            if label not in VALID_LABELS:
                logger.warning("LLM returned invalid label %r for code %s; defaulting to 'Possible'", label, key)
                label = "Possible"
            batch_seen.add(key)
            seen.add(key)
            results.append(
                ClassifiedCandidate(
                    code=source.code,
                    description=source.description,
                    vocabulary=source.vocabulary,
                    label=label,
                    confidence=float(item.get("confidence", 0.0)),
                    explanation=str(item.get("explanation", "")),
                    relevance_score=source.relevance_score,
                    retrieval_score=source.retrieval_score,
                )
            )

        # Any candidate the LLM failed to classify is conservatively marked
        # 'Possible' with zero confidence, ensuring it is routed to human review.
        for c in batch:
            key = (c.code, c.vocabulary)
            if key not in batch_seen:
                seen.add(key)
                results.append(
                    ClassifiedCandidate(
                        code=c.code,
                        description=c.description,
                        vocabulary=c.vocabulary,
                        label="Possible",
                        confidence=0.0,
                        explanation="No classification returned by LLM; routed for manual review.",
                        relevance_score=c.relevance_score,
                        retrieval_score=c.retrieval_score,
                    )
                )

        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)

    return results
