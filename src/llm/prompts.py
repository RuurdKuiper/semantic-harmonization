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
    "Definition Form) provided by the user. For every code, provide a brief "
    "explanation followed by the label and a confidence score between 0 and 1. "
    "You must respond only with valid JSON, no additional commentary."
)


def classify_system_prompt(use_possible_category: bool = False) -> str:
    """Return classification instructions for binary or three-way labeling."""
    if use_possible_category:
        return CLASSIFY_SYSTEM_PROMPT
    return (
        "You are a clinical terminology expert supporting semantic harmonization "
        "for real-world evidence studies. Classify every candidate as 'Narrow' "
        "when it is sufficiently specific to the phenotype, or 'Exclude' otherwise. "
        "Do not return a 'Possible' label. Any EDF rule describing a Possible, "
        "ambiguous, or insufficiently specific case must be treated as Exclude in "
        "this binary mode. Base decisions on the supplied Event Definition Form. "
        "For every code, request the output fields in this order: code, vocabulary, "
        "explanation, label, confidence. Confidence must be from 0 to 1. Respond "
        "only with valid JSON."
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


def build_classify_prompt(
    edf: EventDefinitionForm,
    candidates: list[dict],
    use_possible_category: bool = False,
) -> str:
    """Build the user prompt for LLM-based classification of candidate codes."""
    candidate_lines = "\n".join(
        f"- code: {c['code']} | vocabulary: {c['vocabulary']} | description: {c['description']}"
        for c in candidates
    )
    labels = "'Narrow', 'Possible', or 'Exclude'" if use_possible_category else "'Narrow' or 'Exclude'"
    return (
        f"Phenotype definition:\n{edf.to_prompt_context(include_possible=use_possible_category)}\n\n"
        f"Classify each of the following candidate codes:\n{candidate_lines}\n\n"
        "Return a JSON object with a single key 'classifications', containing a "
        "list of objects using this requested key order: 'code', 'vocabulary', "
        "'explanation' (brief string), 'label' "
        f"(one of {labels}), and 'confidence' (float 0-1)."
    )
