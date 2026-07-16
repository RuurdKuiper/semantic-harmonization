"""Tests for pipeline configuration loading."""

from __future__ import annotations

from src.utils.config import load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert config.phenotype == "myocarditis"
    assert isinstance(config.retrieval.top_k, int) and config.retrieval.top_k > 0
    assert config.retrieval.vocabularies == ["ICD10CM"]
    assert config.llm.provider == "auto"
    assert config.llm.anthropic_model == "claude-opus-4-8"
    assert config.uncertainty.confidence_threshold == 0.7
    assert config.paths.code_systems.get("ICD10CM") == "data/codes/icd10.xlsx"
    assert config.evaluation.compare_vocabularies == ["ICD10CM"]
    assert "myocarditis" in config.paths.aesi_datasets
    assert "erythema_multiforme" in config.paths.aesi_datasets
    assert "guillain_barre" in config.paths.aesi_datasets
