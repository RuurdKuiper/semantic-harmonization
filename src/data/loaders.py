from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class EventDefinitionForm:
    """Structured phenotype definition (EDF) providing semantic context for
    LLM-assisted concept retrieval and code classification."""

    name: str
    preferred_name: str
    definition: str
    inclusion_criteria: list[str] = field(default_factory=list)
    narrow_definition: str = ""
    exclusion_criteria: list[str] = field(default_factory=list)
    narrow_decision_rules: list[str] = field(default_factory=list)
    exclude_decision_rules: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    anatomical_location: str = ""
    population_restrictions: str = ""
    temporal_characteristics: str = ""
    etiology: str = ""
    morphology: str = ""
    diagnostic_certainty: str = ""
    administrative_context: str = ""

    def to_prompt_context(self) -> str:
        """Render the EDF as structured text suitable for inclusion in an LLM prompt."""
        lines = [
            f"Preferred name: {self.preferred_name}",
            f"Definition: {self.definition}",
        ]
        if self.narrow_definition:
            lines.append(f"Narrow definition: {self.narrow_definition}")
        if self.inclusion_criteria:
            lines.append("Inclusion criteria:")
            lines += [f"  - {c}" for c in self.inclusion_criteria]
        if self.exclusion_criteria:
            lines.append("Exclusion criteria:")
            lines += [f"  - {c}" for c in self.exclusion_criteria]
        if self.narrow_decision_rules:
            lines.append("Criteria for Narrow classification:")
            lines += [f"  - {c}" for c in self.narrow_decision_rules]
        if self.exclude_decision_rules:
            lines.append("Criteria for Exclude classification:")
            lines += [f"  - {c}" for c in self.exclude_decision_rules]
        if self.synonyms:
            lines.append(f"Synonyms: {', '.join(self.synonyms)}")
        if self.anatomical_location:
            lines.append(f"Anatomical location: {self.anatomical_location}")
        if self.population_restrictions:
            lines.append(f"Population restrictions: {self.population_restrictions}")
        if self.temporal_characteristics:
            lines.append(f"Temporal characteristics: {self.temporal_characteristics}")
        if self.etiology:
            lines.append(f"Etiology: {self.etiology}")
        if self.morphology:
            lines.append(f"Morphology/pathology: {self.morphology}")
        if self.diagnostic_certainty:
            lines.append(f"Diagnostic certainty: {self.diagnostic_certainty}")
        if self.administrative_context:
            lines.append(f"Administrative/historical context: {self.administrative_context}")
        return "\n".join(lines)


def load_edf(name: str, edf_dir: str | Path = "data/raw/edf") -> EventDefinitionForm:
    """Load an Event Definition Form by phenotype name from a YAML file."""
    path = Path(edf_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"EDF not found for phenotype '{name}': {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return EventDefinitionForm(name=name, **raw)


def load_code_corpus(codes_path: str | Path = "data/raw/codes/codes.csv") -> pd.DataFrame:
    """Load the full candidate code corpus (multi-vocabulary) as a DataFrame with
    columns: code, description, vocabulary."""
    df = pd.read_csv(codes_path, dtype=str).fillna("")
    required = {"code", "description", "vocabulary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Code corpus missing required columns: {missing}")
    return df


def load_gold_labels(name: str, gold_dir: str | Path = "data/gold") -> pd.DataFrame:
    """Load expert-curated reference labels for a phenotype as a DataFrame with
    columns: code, vocabulary, label (Narrow/Possible/Exclude)."""
    path = Path(gold_dir) / f"{name}_gold.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gold labels not found for phenotype '{name}': {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"code", "vocabulary", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gold label file missing required columns: {missing}")
    valid_labels = {"Narrow", "Possible", "Exclude"}
    invalid = set(df["label"]) - valid_labels
    if invalid:
        raise ValueError(f"Gold label file contains invalid labels: {invalid}")
    return df
