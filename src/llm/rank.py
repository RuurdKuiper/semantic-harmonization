"""LLM-based re-ranking of candidate codes using full EDF semantic context."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.loaders import EventDefinitionForm
from src.llm.client import call_llm_json
from src.llm.prompts import RANK_SYSTEM_PROMPT, build_rank_prompt
from src.retrieval.hybrid import HybridCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RankedCandidate:
    rank: int
    code: str
    description: str
    vocabulary: str
    relevance_score: float
    retrieval_score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RankedCandidate):
            return NotImplemented
        return self.relevance_score > other.relevance_score


def llm_rank(
    candidates: list[HybridCandidate],
    edf: EventDefinitionForm,
    provider: str = "auto",
    model: str | None = None,
    max_retries: int = 3,
) -> list[RankedCandidate]:
    """Re-rank hybrid retrieval candidates using an LLM given the full EDF context.

    Parameters
    ----------
    candidates : list[HybridCandidate]
        Candidates produced by :func:`src.retrieval.hybrid.hybrid_retrieval`.
    edf : EventDefinitionForm
        Structured phenotype definition providing semantic context.
    provider : str
        "auto", "anthropic", or "openai".
    model : str, optional
        Model identifier. Defaults to a provider-specific default.
    max_retries : int
        Number of LLM call attempts before failing.

    Returns
    -------
    list[RankedCandidate]
        Candidates ordered by descending LLM-assigned relevance score.
    """
    if not candidates:
        return []

    lookup = {(c.code, c.vocabulary): c for c in candidates}
    payload = [
        {"code": c.code, "vocabulary": c.vocabulary, "description": c.description}
        for c in candidates
    ]
    prompt = build_rank_prompt(edf, payload)
    response = call_llm_json(
        system_prompt=RANK_SYSTEM_PROMPT,
        user_prompt=prompt,
        provider=provider,
        model=model,
        max_retries=max_retries,
    )

    ranked_raw = response.get("ranked_codes", [])
    results: list[RankedCandidate] = []
    seen: set[tuple[str, str]] = set()

    for item in ranked_raw:
        key = (item.get("code", ""), item.get("vocabulary", ""))
        source = lookup.get(key)
        if source is None:
            continue
        seen.add(key)
        results.append(
            RankedCandidate(
                rank=0,
                code=source.code,
                description=source.description,
                vocabulary=source.vocabulary,
                relevance_score=float(item.get("relevance_score", 0.0)),
                retrieval_score=source.score,
            )
        )

    # Append any candidates the LLM omitted, ordered by original retrieval score,
    # so no candidate is silently dropped from downstream classification.
    for c in candidates:
        key = (c.code, c.vocabulary)
        if key not in seen:
            results.append(
                RankedCandidate(
                    rank=0,
                    code=c.code,
                    description=c.description,
                    vocabulary=c.vocabulary,
                    relevance_score=0.0,
                    retrieval_score=c.score,
                )
            )

    results.sort(key=lambda r: (-r.relevance_score, -r.retrieval_score))
    for r, item in enumerate(results):
        item.rank = r + 1
    return results
