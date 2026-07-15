"""Reusable prompt templates for LLM-assisted ranking and classification."""

from __future__ import annotations

from src.data.loaders import EventDefinitionForm

RANK_SYSTEM_PROMPT = (
    "You are a clinical terminology expert supporting semantic harmonization for "
    "real-world evidence studies. You review candidate medical codes retrieved for "
    "a phenotype and re-rank them by clinical relevance, using the structured "
    "phenotype definition (Event Definition Form) provided by the user. "
    "You must respond only with valid JSON, no additional commentary."
)

CLASSIFY_SYSTEM_PROMPT = (
    "You are a clinical terminology expert supporting semantic harmonization for "
    "real-world evidence studies. You review candidate medical codes for a phenotype "
    "and classify each one as 'Narrow' (highly specific to the phenotype), "
    "'Possible' (related but insufficiently specific, or specific in some but not "
    "all contexts), or 'Exclude' (not relevant, or explicitly an exclusion "
    "criterion). Base your decision on the structured phenotype definition (Event "
    "Definition Form) provided by the user. For every code, provide a confidence "
    "score between 0 and 1 and a brief explanation. "
    "You must respond only with valid JSON, no additional commentary."
)


def build_rank_prompt(edf: EventDefinitionForm, candidates: list[dict]) -> str:
    """Build the user prompt for LLM-based re-ranking of candidate codes."""
    candidate_lines = "\n".join(
        f"- code: {c['code']} | vocabulary: {c['vocabulary']} | description: {c['description']}"
        for c in candidates
    )
    return (
        f"Phenotype definition:\n{edf.to_prompt_context()}\n\n"
        f"Candidate codes to rank by clinical relevance to this phenotype "
        f"(most relevant first):\n{candidate_lines}\n\n"
        "Return a JSON object with a single key 'ranked_codes', containing a list of "
        "objects with keys 'code', 'vocabulary', and 'relevance_score' (float 0-1, "
        "higher = more relevant), sorted descending by relevance_score."
    )


def build_classify_prompt(edf: EventDefinitionForm, candidates: list[dict]) -> str:
    """Build the user prompt for LLM-based classification of candidate codes."""
    candidate_lines = "\n".join(
        f"- code: {c['code']} | vocabulary: {c['vocabulary']} | description: {c['description']}"
        for c in candidates
    )
    return (
        f"Phenotype definition:\n{edf.to_prompt_context()}\n\n"
        f"Classify each of the following candidate codes:\n{candidate_lines}\n\n"
        "Return a JSON object with a single key 'classifications', containing a "
        "list of objects with keys 'code', 'vocabulary', 'label' "
        "(one of 'Narrow', 'Possible', 'Exclude'), 'confidence' (float 0-1), and "
        "'explanation' (brief string)."
    )
