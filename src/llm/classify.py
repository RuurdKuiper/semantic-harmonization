"""LLM classification into Narrow/Exclude with an optional Possible category."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
import json
from typing import Protocol

from src.data.loaders import EventDefinitionForm
from src.llm.client import call_jev_decisions, call_llm_json, resolve_provider
from src.llm.prompts import build_classify_prompt, classify_system_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"Narrow", "Possible", "Exclude"}


class ClassificationCandidate(Protocol):
    """Minimal candidate interface consumed by the classifier."""

    code: str
    description: str
    vocabulary: str


@dataclass(frozen=True, slots=True)
class AdaptiveStoppingConfig:
    """Stop classification after Narrow results remain sparse for several batches."""

    sparse_narrow_threshold: int = 1
    consecutive_sparse_batches: int = 3
    minimum_batches: int = 3

    def __post_init__(self) -> None:
        if self.sparse_narrow_threshold < 0:
            raise ValueError("sparse_narrow_threshold must be at least 0")
        if self.consecutive_sparse_batches < 1:
            raise ValueError("consecutive_sparse_batches must be at least 1")
        if self.minimum_batches < 1:
            raise ValueError("minimum_batches must be at least 1")


@dataclass(frozen=True, slots=True)
class ClassificationRunInfo:
    """Summary of how much of the ranked candidate list was classified."""

    candidates_available: int
    candidates_classified: int
    batches_completed: int
    narrow_counts_by_batch: tuple[int, ...]
    stopped_early: bool
    stop_reason: str


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


@dataclass(frozen=True, slots=True)
class GPTReviewRunInfo:
    """Summary of final-stage GPT adjudication of uncertain classifications."""

    confidence_threshold: float
    candidates_reviewed: int
    labels_changed: int
    narrow_to_exclude: int
    exclude_to_narrow: int
    reasoning_effort: str


def gpt_review_low_confidence(
    classified: Sequence[ClassifiedCandidate],
    edf: EventDefinitionForm,
    *,
    confidence_threshold: float = 0.7,
    model: str | None = None,
    max_retries: int = 3,
    batch_size: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    use_possible_category: bool = False,
    reasoning_effort: str = "medium",
) -> tuple[list[ClassifiedCandidate], GPTReviewRunInfo]:
    """Use GPT to adjudicate only classifications below the confidence cutoff.

    Results at or above the threshold are preserved exactly. Reviewed results
    replace the corresponding stage-2 labels while retaining their original
    list positions and recording the prior label/confidence in the explanation.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    uncertain = [item for item in classified if item.confidence < confidence_threshold]
    if not uncertain:
        return list(classified), GPTReviewRunInfo(
            confidence_threshold=confidence_threshold,
            candidates_reviewed=0,
            labels_changed=0,
            narrow_to_exclude=0,
            exclude_to_narrow=0,
            reasoning_effort=reasoning_effort,
        )

    reviewed = llm_classify(
        uncertain,
        edf,
        provider="openai",
        model=model,
        max_retries=max_retries,
        batch_size=batch_size,
        progress_callback=progress_callback,
        use_possible_category=use_possible_category,
        adaptive_stopping=None,
        reasoning_effort=reasoning_effort,
    )
    reviewed_by_key = {(item.code, item.vocabulary): item for item in reviewed}

    merged: list[ClassifiedCandidate] = []
    labels_changed = 0
    narrow_to_exclude = 0
    exclude_to_narrow = 0
    for original in classified:
        replacement = reviewed_by_key.get((original.code, original.vocabulary))
        if replacement is None:
            merged.append(original)
            continue

        if replacement.label != original.label:
            labels_changed += 1
            if original.label == "Narrow" and replacement.label == "Exclude":
                narrow_to_exclude += 1
            elif original.label == "Exclude" and replacement.label == "Narrow":
                exclude_to_narrow += 1
        merged.append(
            ClassifiedCandidate(
                code=original.code,
                description=original.description,
                vocabulary=original.vocabulary,
                label=replacement.label,
                confidence=replacement.confidence,
                explanation=(
                    "GPT review of low-confidence stage-2 result "
                    f"(previous label={original.label}, confidence={original.confidence:.2f}). "
                    + replacement.explanation
                ),
                relevance_score=original.relevance_score,
                retrieval_score=original.retrieval_score,
            )
        )

    return merged, GPTReviewRunInfo(
        confidence_threshold=confidence_threshold,
        candidates_reviewed=len(uncertain),
        labels_changed=labels_changed,
        narrow_to_exclude=narrow_to_exclude,
        exclude_to_narrow=exclude_to_narrow,
        reasoning_effort=reasoning_effort,
    )


def llm_classify(
    candidates: Sequence[ClassificationCandidate],
    edf: EventDefinitionForm,
    provider: str = "auto",
    model: str | None = None,
    max_retries: int = 3,
    batch_size: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    use_possible_category: bool = False,
    adaptive_stopping: AdaptiveStoppingConfig | None = None,
    run_info_callback: Callable[[ClassificationRunInfo], None] | None = None,
    reasoning_effort: str | None = None,
) -> list[ClassifiedCandidate]:
    """Classify candidates as Narrow/Exclude, optionally allowing Possible.

    Parameters
    ----------
    candidates : sequence
        Similarity-ranked hybrid or LLM-ranked candidates.
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
    adaptive_stopping : AdaptiveStoppingConfig, optional
        Stop after Narrow results remain at or below the configured threshold
        for the configured number of consecutive batches.
    run_info_callback : callable, optional
        Invoked once with the completed/stopped run summary.
    reasoning_effort : str, optional
        OpenAI reasoning effort, such as ``"none"`` or ``"medium"``.

    Returns
    -------
    list[ClassifiedCandidate]
    """
    if not candidates:
        if run_info_callback is not None:
            run_info_callback(
                ClassificationRunInfo(0, 0, 0, (), False, "No candidates to classify.")
            )
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    resolved_provider = resolve_provider(provider)
    if resolved_provider == "jev":
        return _jev_classify(
            candidates,
            edf,
            model=model or "jev-latest",
            max_retries=max_retries,
            batch_size=batch_size,
            progress_callback=progress_callback,
            use_possible_category=use_possible_category,
            adaptive_stopping=adaptive_stopping,
            run_info_callback=run_info_callback,
        )

    results: list[ClassifiedCandidate] = []
    narrow_counts: list[int] = []
    sparse_streak = 0

    total = len(candidates)
    for start in range(0, total, batch_size):
        batch = candidates[start : start + batch_size]
        batch_result_start = len(results)
        lookup = {(c.code, c.vocabulary): c for c in batch}
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
            reasoning_effort=reasoning_effort,
        )

        classifications_raw = response.get("classifications", [])
        batch_seen: set[tuple[str, str]] = set()
        for item in classifications_raw:
            key = (item.get("code", ""), item.get("vocabulary", ""))
            source = lookup.get(key)
            if source is None or key in batch_seen:
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
            results.append(
                ClassifiedCandidate(
                    code=source.code,
                    description=source.description,
                    vocabulary=source.vocabulary,
                    label=label,
                    confidence=float(item.get("confidence", 0.0)),
                    explanation=str(item.get("explanation", "")),
                    relevance_score=float(getattr(source, "relevance_score", 0.0)),
                    retrieval_score=float(
                        getattr(source, "retrieval_score", getattr(source, "score", 0.0))
                    ),
                )
            )

        # Keep every candidate in the result. A missing response receives the
        # least-committal enabled label and zero confidence so review selection
        # can annotate it without filtering or altering any other result.
        for c in batch:
            key = (c.code, c.vocabulary)
            if key not in batch_seen:
                results.append(
                    ClassifiedCandidate(
                        code=c.code,
                        description=c.description,
                        vocabulary=c.vocabulary,
                        label="Possible" if use_possible_category else "Exclude",
                        confidence=0.0,
                        explanation="No classification returned by LLM; routed for manual review.",
                        relevance_score=float(getattr(c, "relevance_score", 0.0)),
                        retrieval_score=float(
                            getattr(c, "retrieval_score", getattr(c, "score", 0.0))
                        ),
                    )
                )

        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)

        narrow_count = sum(
            item.label == "Narrow" for item in results[batch_result_start:]
        )
        narrow_counts.append(narrow_count)
        sparse_streak = (
            sparse_streak + 1
            if adaptive_stopping is not None
            and narrow_count <= adaptive_stopping.sparse_narrow_threshold
            else 0
        )
        if _adaptive_stop_reached(adaptive_stopping, len(narrow_counts), sparse_streak):
            break

    _report_run_info(
        run_info_callback,
        total=total,
        results=results,
        narrow_counts=narrow_counts,
        adaptive_stopping=adaptive_stopping,
    )
    return results


def _jev_classify(
    candidates: Sequence[ClassificationCandidate],
    edf: EventDefinitionForm,
    model: str,
    max_retries: int,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None,
    use_possible_category: bool,
    adaptive_stopping: AdaptiveStoppingConfig | None,
    run_info_callback: Callable[[ClassificationRunInfo], None] | None,
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
    narrow_counts: list[int] = []
    sparse_streak = 0
    total = len(candidates)
    state = "Phenotype definition:\n" + edf.to_prompt_context(
        include_possible=use_possible_category
    )
    for start in range(0, total, effective_batch_size):
        batch = candidates[start : start + effective_batch_size]
        batch_result_start = len(results)
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
                    relevance_score=float(getattr(candidate, "relevance_score", 0.0)),
                    retrieval_score=float(
                        getattr(
                            candidate,
                            "retrieval_score",
                            getattr(candidate, "score", 0.0),
                        )
                    ),
                )
            )
        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)
        narrow_count = sum(
            item.label == "Narrow" for item in results[batch_result_start:]
        )
        narrow_counts.append(narrow_count)
        sparse_streak = (
            sparse_streak + 1
            if adaptive_stopping is not None
            and narrow_count <= adaptive_stopping.sparse_narrow_threshold
            else 0
        )
        if _adaptive_stop_reached(adaptive_stopping, len(narrow_counts), sparse_streak):
            break

    _report_run_info(
        run_info_callback,
        total=total,
        results=results,
        narrow_counts=narrow_counts,
        adaptive_stopping=adaptive_stopping,
    )
    return results


def _adaptive_stop_reached(
    config: AdaptiveStoppingConfig | None,
    batches_completed: int,
    sparse_streak: int,
) -> bool:
    return bool(
        config is not None
        and batches_completed >= config.minimum_batches
        and sparse_streak >= config.consecutive_sparse_batches
    )


def _report_run_info(
    callback: Callable[[ClassificationRunInfo], None] | None,
    *,
    total: int,
    results: list[ClassifiedCandidate],
    narrow_counts: list[int],
    adaptive_stopping: AdaptiveStoppingConfig | None,
) -> None:
    if callback is None:
        return
    stopped_early = len(results) < total
    if stopped_early and adaptive_stopping is not None:
        reason = (
            f"Stopped after {adaptive_stopping.consecutive_sparse_batches} consecutive "
            f"batches with at most {adaptive_stopping.sparse_narrow_threshold} Narrow "
            "result(s) per batch."
        )
    else:
        reason = "All available candidates were classified."
    callback(
        ClassificationRunInfo(
            candidates_available=total,
            candidates_classified=len(results),
            batches_completed=len(narrow_counts),
            narrow_counts_by_batch=tuple(narrow_counts),
            stopped_early=stopped_early,
            stop_reason=reason,
        )
    )
