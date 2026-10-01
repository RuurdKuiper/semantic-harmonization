"""Tests for pipeline configuration loading."""

from __future__ import annotations

from src.utils.config import load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert config.phenotype == "myocarditis"
    assert isinstance(config.retrieval.top_k, int) and config.retrieval.top_k > 0
    assert config.retrieval.use_top_k_limit is False
    assert config.retrieval.vocabularies == ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    assert config.retrieval.lexical_weight == 0.1
    assert config.retrieval.embedding_weight == 0.9
    assert config.llm.provider == "jev"
    assert config.llm.anthropic_model == "claude-opus-4-8"
    assert config.llm.openai_model == "gpt-6-luna"
    assert config.llm.google_model == "gemini-3-flash-preview"
    assert config.llm.jev_model == "jev-latest"
    assert config.uncertainty.confidence_threshold == 0.7
    assert config.llm.use_possible_category is False
    assert config.llm.adaptive_stopping_enabled is True
    assert config.llm.sparse_narrow_threshold == 0
    assert config.llm.consecutive_sparse_batches == 10
    assert config.llm.minimum_batches == 3
    assert config.uncertainty.possible_requires_review is False
    assert config.uncertainty.gpt_review_enabled is True
    assert config.uncertainty.gpt_review_batch_size == 10
    assert config.uncertainty.gpt_review_reasoning_enabled is True
    assert config.llm.classify_effort == "medium"
    assert config.paths.code_systems.get("ICD10CM") == "data/codes/csv/ICD10CM@2026-codes.csv"
    assert config.paths.results_dir == "results"
    assert config.evaluation.compare_vocabularies == ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    assert "myocarditis" in config.paths.aesi_datasets
    assert "erythema_multiforme" in config.paths.aesi_datasets
    assert "guillain_barre" in config.paths.aesi_datasets
    assert len(config.paths.aesi_datasets) == 8
