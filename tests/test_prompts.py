"""Tests for prompt-building utilities."""

from __future__ import annotations

from src.data.loaders import load_edf
from src.llm.prompts import build_classify_prompt, build_rank_prompt, classify_system_prompt


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
    requested = prompt.split("requested key order:", 1)[1]
    assert requested.index("'explanation'") < requested.index("'label'")
    assert requested.index("'label'") < requested.index("'confidence'")


def test_build_classify_prompt_can_enable_possible():
    edf = load_edf("myocarditis")
    candidates = [{"code": "I40.0", "vocabulary": "ICD10", "description": "Infective myocarditis"}]
    prompt = build_classify_prompt(edf, candidates, use_possible_category=True)
    assert "'Possible'" in prompt


def test_binary_prompt_maps_possible_decision_rules_to_exclude():
    edf = load_edf("kidney_disease")
    prompt = build_classify_prompt(edf, [], use_possible_category=False)
    assert "Criteria for Possible classification" not in prompt
    assert "classify as Exclude because Possible is disabled" in prompt
    assert "Simple renal cysts or benign neoplasms" in prompt
    assert "must be treated as Exclude" in classify_system_prompt(False)
