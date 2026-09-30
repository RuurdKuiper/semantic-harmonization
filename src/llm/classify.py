"""LLM classification into Narrow/Exclude with an optional Possible category."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import json

from src.data.loaders import EventDefinitionForm
from src.llm.client import call_jev_decisions, call_llm_json, resolve_provider
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

    resolved_provider = resolve_provider(provider)
    if resolved_provider == "jev":
        return _jev_classify(
            candidates,
            edf,
            model=model or "jev-1.13",
            max_retries=max_retries,
            batch_size=batch_size,
            progress_callback=progress_callback,
            use_possible_category=use_possible_category,
        )

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
            provider=resolved_provider,
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


def _jev_classify(
    candidates: list[RankedCandidate],
    edf: EventDefinitionForm,
    model: str,
    max_retries: int,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None,
    use_possible_category: bool,
) -> list[ClassifiedCandidate]:
    """Classify candidates with Jev typed choice questions."""
    effective_batch_size = min(batch_size, 20)
    labels = {
        "Narrow": "The code is sufficiently specific to the phenotype and meets the Narrow criteria.",
        "Exclude": "The code is irrelevant, insufficiently specific, historical, or explicitly excluded.",
    }
    if use_possible_category:
        labels = {
            "Narrow": labels["Narrow"],
            "Possible": "The code is related but not sufficiently specific in every context.",
            "Exclude": labels["Exclude"],
        }

    results: list[ClassifiedCandidate] = []
    total = len(candidates)
    state = "Phenotype definition:\n" + edf.to_prompt_context(
        include_possible=use_possible_category
    )
    for start in range(0, total, effective_batch_size):
        batch = candidates[start : start + effective_batch_size]
        questions = {
            f"candidate_{offset}": {
                "type": "choice",
                "instructions": (
                    "Classify this clinical terminology candidate for the phenotype: "
                    f"code={candidate.code}; vocabulary={candidate.vocabulary}; "
                    f"description={candidate.description}"
                ),
                "criteria": labels,
            }
            for offset, candidate in enumerate(batch)
        }
        response = call_jev_decisions(
            state=state,
            questions=questions,
            model=model,
            max_retries=max_retries,
        )
        answers = response.get("answers", {})
        for offset, candidate in enumerate(batch):
            answer = answers.get(f"candidate_{offset}", {})
            probabilities = answer.get("probabilities", {})
            fallback_label = "Possible" if use_possible_category else "Exclude"
            label = answer.get("choice", fallback_label)
            if label not in labels:
                label = fallback_label
            confidence = answer.get("confidence")
            if confidence is None:
                confidence = max((float(value) for value in probabilities.values()), default=0.0)
            results.append(
                ClassifiedCandidate(
                    code=candidate.code,
                    description=candidate.description,
                    vocabulary=candidate.vocabulary,
                    label=label,
                    confidence=float(confidence),
                    explanation="Jev choice probabilities: " + json.dumps(probabilities, sort_keys=True),
                    relevance_score=candidate.relevance_score,
                    retrieval_score=candidate.retrieval_score,
                )
            )
        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)
    return results
