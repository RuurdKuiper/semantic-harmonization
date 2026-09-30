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
    assert "'Narrow' or 'Exclude'" in prompt
    assert "'Possible'" not in prompt.split("Return a JSON object", 1)[1]


def test_build_classify_prompt_can_enable_possible():
    edf = load_edf("myocarditis")
    candidates = [{"code": "I40.0", "vocabulary": "ICD10", "description": "Infective myocarditis"}]
    prompt = build_classify_prompt(edf, candidates, use_possible_category=True)
    assert "'Possible'" in prompt


def test_binary_prompt_omits_possible_decision_rules():
    edf = load_edf("type_1_diabetes")
    prompt = build_classify_prompt(edf, [], use_possible_category=False)
    assert "Criteria for Possible classification" not in prompt
