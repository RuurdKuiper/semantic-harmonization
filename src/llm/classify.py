"""LLM classification into Narrow/Exclude with an optional Possible category."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from src.data.loaders import EventDefinitionForm
from src.llm.client import call_llm_json
from src.llm.prompts import build_classify_prompt, classify_system_prompt
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
    use_possible_category: bool = False,
) -> list[ClassifiedCandidate]:
    """Classify candidates as Narrow/Exclude, optionally allowing Possible.

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
    use_possible_category : bool
        Enable three-way Narrow/Possible/Exclude labeling. Defaults to False.

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
        prompt = build_classify_prompt(edf, payload, use_possible_category=use_possible_category)
        response = call_llm_json(
            system_prompt=classify_system_prompt(use_possible_category),
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
            fallback_label = "Possible" if use_possible_category else "Exclude"
            valid_labels = VALID_LABELS if use_possible_category else {"Narrow", "Exclude"}
            label = item.get("label", fallback_label)
            if label not in valid_labels:
                logger.warning(
                    "LLM returned invalid label %r for code %s; defaulting to %r",
                    label,
                    key,
                    fallback_label,
                )
                label = fallback_label
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

        # Keep every candidate in the result. A missing response receives the
        # least-committal enabled label and zero confidence so review selection
        # can annotate it without filtering or altering any other result.
        for c in batch:
            key = (c.code, c.vocabulary)
            if key not in batch_seen:
                seen.add(key)
                results.append(
                    ClassifiedCandidate(
                        code=c.code,
                        description=c.description,
                        vocabulary=c.vocabulary,
                        label="Possible" if use_possible_category else "Exclude",
                        confidence=0.0,
                        explanation="No classification returned by LLM; routed for manual review.",
                        relevance_score=c.relevance_score,
                        retrieval_score=c.retrieval_score,
                    )
                )

        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)

    return results
