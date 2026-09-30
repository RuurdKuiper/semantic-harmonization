"""Tests for pipeline configuration loading."""

from __future__ import annotations

from src.utils.config import load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert config.phenotype == "myocarditis"
    assert isinstance(config.retrieval.top_k, int) and config.retrieval.top_k > 0
    assert config.retrieval.vocabularies == ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    assert config.llm.provider == "auto"
    assert config.llm.anthropic_model == "claude-opus-4-8"
    assert config.uncertainty.confidence_threshold == 0.7
    assert config.llm.use_possible_category is False
    assert config.uncertainty.possible_requires_review is False
    assert config.paths.code_systems.get("ICD10CM") == "data/codes/csv/ICD10CM@2026-codes.csv"
    assert config.paths.results_dir == "results"
    assert config.evaluation.compare_vocabularies == ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    assert "myocarditis" in config.paths.aesi_datasets
    assert "erythema_multiforme" in config.paths.aesi_datasets
    assert "guillain_barre" in config.paths.aesi_datasets
    assert len(config.paths.aesi_datasets) == 8
