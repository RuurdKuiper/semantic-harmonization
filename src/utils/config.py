from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    edf_dir: str = "data/edfs/yamls"
    codes_parquet_dir: str = "data/codes/parquet"
    embeddings_dir: str = "data/codes/embeddings"
    results_dir: str = "results"
    aesi_datasets: dict[str, str] = field(default_factory=dict)
    code_systems: dict[str, str] = field(
        default_factory=lambda: {
            "ICD10CM": "data/codes/csv/ICD10CM@2026-codes.csv",
            "ICD9CM": "data/codes/csv/ICD9CM@2014-codes.csv",
            "ICPC": "data/codes/csv/ICPC@1993-codes.csv",
            "MDR": "data/codes/csv/MDR@28.0-codes.csv",
            "RCD2": "data/codes/csv/RCD2@20200401-codes.csv",
            "SNOMEDCT_US": "data/codes/csv/SNOMEDCT_US@2025_09_01-codes.csv",
        }
    )


@dataclass
class RetrievalConfig:
    lexical_weight: float = 0.1
    embedding_weight: float = 0.9
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int | None = 3072
    embedding_batch_size: int = 512
    top_k: int = 50
    use_top_k_limit: bool = False
    per_vocabulary_top_k: bool = True
    vocabularies: list[str] = field(
        default_factory=lambda: ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    )


@dataclass
class LLMConfig:
    provider: str = "jev"  # Stage 2 defaults to Jev; other providers remain available.
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-6-luna"
    google_model: str = "gemini-3-flash-preview"
    jev_model: str = "jev-latest"
    rank_effort: str = "low"
    classify_effort: str = "medium"
    max_retries: int = 3
    use_possible_category: bool = False
    adaptive_stopping_enabled: bool = True
    sparse_narrow_threshold: int = 0
    consecutive_sparse_batches: int = 10
    minimum_batches: int = 3


@dataclass
class UncertaintyConfig:
    confidence_threshold: float = 0.7
    gpt_review_max_candidates: int = 200
    possible_requires_review: bool = False
    gpt_review_enabled: bool = True
    gpt_review_batch_size: int = 10
    gpt_review_reasoning_enabled: bool = True


@dataclass
class EvaluationConfig:
    narrow_only_as_positive: bool = True
    compare_vocabularies: list[str] = field(
        default_factory=lambda: ["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    )


@dataclass
class PipelineConfig:
    phenotype: str = "myocarditis"
    verbose: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file into a PipelineConfig, falling back to defaults
    for any keys not present in the file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    return PipelineConfig(
        phenotype=raw.get("phenotype", PipelineConfig.phenotype),
        verbose=raw.get("verbose", PipelineConfig.verbose),
        paths=PathsConfig(**raw.get("paths", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        uncertainty=UncertaintyConfig(**raw.get("uncertainty", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
