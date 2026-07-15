"""Tests for prompt-building utilities."""

from __future__ import annotations

from src.data.loaders import load_edf
from src.llm.prompts import build_classify_prompt, build_rank_prompt


def test_build_rank_prompt_includes_edf_and_candidates():
    edf = load_edf("myocarditis")
    candidates = [{"code": "I40.0", "vocabulary": "ICD10", "description": "Infective myocarditis"}]
    prompt = build_rank_prompt(edf, candidates)
    assert "Myocarditis" in prompt
    assert "I40.0" in prompt
    assert "ranked_codes" in prompt


def test_build_classify_prompt_includes_edf_and_candidates():
    edf = load_edf("myocarditis")
    candidates = [{"code": "I40.0", "vocabulary": "ICD10", "description": "Infective myocarditis"}]
    prompt = build_classify_prompt(edf, candidates)
    assert "Myocarditis" in prompt
    assert "I40.0" in prompt
    assert "classifications" in prompt
