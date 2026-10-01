"""Core helpers for the Matthewman et al. matched-corpus experiment."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.loaders import EventDefinitionForm


PHENOTYPES = {
    "atopic_eczema": ("Atopic eczema", "eczema codelist.csv"),
    "vascular_dementia": ("Vascular dementia", "vascular_dementia codelist.csv"),
    "eosinophilic_esophagitis": (
        "Eosinophilic esophagitis",
        "eosinophilic_esophagitis codelist.csv",
    ),
    "psoriasis": ("Psoriasis", "psoriasis codelist.csv"),
    "hidradenitis_suppurativa": (
        "Hidradentitis suppurativa",  # Preserve the paper evaluation input typo.
        "hidradenitis_suppurativa codelist.csv",
    ),
    "myocardial_infarction": ("Myocardial infarction", "myocardial_infarction codelist.csv"),
    "wrist_fracture": ("Wrist fracture", "wrist_fracture codelist.csv"),
}

CORPUS_VOCABULARY = "CPRD_AURUM_2025_09"


def build_classification_edf(
    name: str,
    preferred_name: str,
    policy: dict,
) -> EventDefinitionForm:
    """Build the experiment-only EDF used by Jev and GPT classification.

    Retrieval intentionally continues to use only ``preferred_name``. Keeping
    policy construction here makes that separation explicit and prevents the
    reference codelists themselves from entering either query or prompt.
    """
    if not isinstance(policy, dict) or not str(policy.get("definition", "")).strip():
        raise ValueError("classification_policy.definition must be a non-empty string")
    return EventDefinitionForm(
        name=name,
        preferred_name=preferred_name,
        definition=str(policy["definition"]).strip(),
        narrow_decision_rules=[str(rule) for rule in policy.get("narrow_decision_rules", [])],
        exclude_decision_rules=[str(rule) for rule in policy.get("exclude_decision_rules", [])],
    )


@dataclass(frozen=True)
class WholeCodelistScore:
    grade: str
    grade_value: float
    required_total: int
    optional_total: int
    predicted_total: int
    required_retrieved: int
    required_predicted: int
    required_not_retrieved_count: int
    retrieved_but_excluded_count: int
    required_not_classified_count: int
    classified_but_excluded_count: int
    irrelevant_predicted_count: int
    hallucinated_count: int
    missing_required: list[str]
    irrelevant_predicted: list[str]
    retrieved_but_excluded: list[str]
    required_not_classified: list[str]
    classified_but_excluded: list[str]
    retrieval_recall: float
    final_required_recall: float
    final_acceptable_precision: float


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_column(frame: pd.DataFrame, wanted: str) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    try:
        return by_lower[wanted.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Required CPRD column {wanted!r} not found; available columns: {list(frame.columns)}"
        ) from exc


def load_cprd_browser(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the paper's CPRD browser into full and Read-only retrieval corpora."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CPRD browser not found: {path}")
    if path.suffix.lower() == ".dta":
        raw = pd.read_stata(path, convert_categoricals=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        # CPRD's text export is tab-delimited, while small local fixtures and
        # alternative exports may be comma-delimited. Detect the delimiter
        # from the header instead of assuming that every .txt file is CSV.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
        try:
            delimiter = csv.Sniffer().sniff(header, delimiters=",\t;|").delimiter
        except csv.Error as exc:
            raise ValueError(f"Could not detect delimiter in CPRD browser: {path}") from exc
        raw = pd.read_csv(
            path,
            dtype=str,
            sep=delimiter,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    else:
        raise ValueError("CPRD browser must be a .dta, .csv, or .txt file")

    code_col = _find_column(raw, "originalreadcode")
    term_col = _find_column(raw, "term")
    read_col = _find_column(raw, "cleansedreadcode")
    working = raw[[code_col, term_col, read_col]].copy()
    working.columns = ["code", "description", "cleansed_read_code"]
    for column in working.columns:
        working[column] = working[column].fillna("").astype(str).str.strip()
    working = working[(working["code"] != "") & (working["description"] != "")]

    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[["code", "description"]].copy()
        result["vocabulary"] = CORPUS_VOCABULARY
        return result.drop_duplicates(subset=["code", "vocabulary"]).reset_index(drop=True)

    return canonical(working), canonical(working[working["cleansed_read_code"] != ""])


def load_reference_codelists(paper_repo: str | Path) -> dict[str, pd.DataFrame]:
    codelist_dir = Path(paper_repo) / "codelists"
    if not codelist_dir.is_dir():
        raise FileNotFoundError(f"Paper codelist directory not found: {codelist_dir}")

    references: dict[str, pd.DataFrame] = {}
    for key, (_, filename) in PHENOTYPES.items():
        path = codelist_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Paper reference codelist not found: {path}")
        raw = pd.read_csv(path, dtype=str)
        required_columns = {"OriginalReadCode", "Term", "required"}
        missing = required_columns - set(raw.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        reference = raw[["OriginalReadCode", "Term", "required"]].rename(
            columns={"OriginalReadCode": "code", "Term": "description"}
        )
        reference["code"] = reference["code"].fillna("").astype(str).str.strip()
        reference["description"] = reference["description"].fillna("").astype(str).str.strip()
        reference["required"] = pd.to_numeric(reference["required"], errors="raise").astype(int)
        if not set(reference["required"]).issubset({0, 1}):
            raise ValueError(f"{path} has required values other than 0 and 1")
        references[key] = reference.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return references


def scope_reference_to_corpus(reference: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Restrict gold codes to the selected paper corpus subset."""
    available = set(corpus["code"].astype(str))
    return reference[reference["code"].isin(available)].reset_index(drop=True)


def score_whole_codelist(
    reference: pd.DataFrame,
    retrieved_codes: Iterable[str],
    predicted_codes: Iterable[str],
    classified_codes: Iterable[str] | None = None,
) -> WholeCodelistScore:
    """Apply the paper's C/P/I rule deterministically to exact corpus codes."""
    required = set(reference.loc[reference["required"] == 1, "code"].astype(str))
    optional = set(reference.loc[reference["required"] == 0, "code"].astype(str))
    acceptable = required | optional
    retrieved = {str(code) for code in retrieved_codes}
    predicted = {str(code) for code in predicted_codes}
    classified = (
        {str(code) for code in classified_codes}
        if classified_codes is not None
        else set(retrieved)
    )

    missing_required = sorted(required - predicted)
    irrelevant = sorted(predicted - acceptable)
    retrieved_but_excluded = sorted((required & retrieved) - predicted)
    required_not_classified = sorted((required & retrieved) - classified)
    classified_but_excluded = sorted((required & classified) - predicted)
    if missing_required:
        grade, grade_value = "I", 0.0
    elif irrelevant:
        grade, grade_value = "P", 0.5
    else:
        grade, grade_value = "C", 1.0

    required_retrieved = len(required & retrieved)
    required_predicted = len(required & predicted)
    return WholeCodelistScore(
        grade=grade,
        grade_value=grade_value,
        required_total=len(required),
        optional_total=len(optional),
        predicted_total=len(predicted),
        required_retrieved=required_retrieved,
        required_predicted=required_predicted,
        required_not_retrieved_count=len(required - retrieved),
        retrieved_but_excluded_count=len(retrieved_but_excluded),
        required_not_classified_count=len(required_not_classified),
        classified_but_excluded_count=len(classified_but_excluded),
        irrelevant_predicted_count=len(irrelevant),
        hallucinated_count=0,
        missing_required=missing_required,
        irrelevant_predicted=irrelevant,
        retrieved_but_excluded=retrieved_but_excluded,
        required_not_classified=required_not_classified,
        classified_but_excluded=classified_but_excluded,
        retrieval_recall=required_retrieved / len(required) if required else 1.0,
        final_required_recall=required_predicted / len(required) if required else 1.0,
        final_acceptable_precision=len(predicted & acceptable) / len(predicted) if predicted else 0.0,
    )


def score_as_dict(score: WholeCodelistScore) -> dict:
    return asdict(score)
