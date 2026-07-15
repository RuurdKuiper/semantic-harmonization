from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    edf_dir: str = "data/raw/edf"
    codes_path: str = "data/raw/codes/codes.csv"
    gold_dir: str = "data/gold"
    processed_dir: str = "data/processed"


@dataclass
class RetrievalConfig:
    lexical_weight: float = 0.5
    embedding_weight: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 25


@dataclass
class LLMConfig:
    model: str = "claude-opus-4-8"
    rank_effort: str = "low"
    classify_effort: str = "low"
    max_retries: int = 3


@dataclass
class UncertaintyConfig:
    confidence_threshold: float = 0.7
    possible_requires_review: bool = True


@dataclass
class EvaluationConfig:
    narrow_only_as_positive: bool = True


@dataclass
class PipelineConfig:
    phenotype: str = "myocarditis"
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
        paths=PathsConfig(**raw.get("paths", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        uncertainty=UncertaintyConfig(**raw.get("uncertainty", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
