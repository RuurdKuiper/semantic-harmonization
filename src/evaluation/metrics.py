"""Evaluation metrics for concept retrieval and code classification against
expert-curated gold-standard phenotype codelists."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.preprocessing import standardize_code

LABELS = ("Narrow", "Possible", "Exclude")


@dataclass(slots=True)
class RetrievalMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    precision: float
    f1: float


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    cohens_kappa: float
    per_label: dict[str, dict[str, float]]


def _standardized_key(code: str, vocabulary: str) -> tuple[str, str]:
    return standardize_code(code, vocabulary), (vocabulary or "").strip().upper()


def filter_records_by_vocabulary(records: list[dict], vocabularies: list[str]) -> list[dict]:
    """Keep only records whose ``vocabulary`` is in *vocabularies* (case-insensitive)."""
    vocab_set = {v.strip().upper() for v in vocabularies}
    return [r for r in records if (r.get("vocabulary") or "").strip().upper() in vocab_set]


def filter_gold_by_vocabulary(gold_labels: pd.DataFrame, vocabularies: list[str]) -> pd.DataFrame:
    """Keep only gold-label rows whose ``vocabulary`` is in *vocabularies* (case-insensitive)."""
    vocab_set = {v.strip().upper() for v in vocabularies}
    mask = gold_labels["vocabulary"].str.strip().str.upper().isin(vocab_set)
    return gold_labels[mask].reset_index(drop=True)


def to_predicted_codelist(classified: list[dict]) -> pd.DataFrame:
    """Render final classifications in the same schema as the ground-truth AESI
    export (``coding_system``, ``code``, ``code_name``, ``concept``,
    ``concept_name``, ``tags``) so predicted and reference codelists are
    directly comparable/exportable side by side.

    Parameters
    ----------
    classified : list[dict]
        Each dict must contain ``code``, ``vocabulary``, ``description``, and
        ``label`` keys (as produced by :func:`src.llm.classify.llm_classify`).

    Returns
    -------
    pd.DataFrame
        Columns: ``coding_system``, ``code``, ``code_name``, ``concept``,
        ``concept_name``, ``tags``. ``concept``/``concept_name`` are left blank
        since our retrieval corpus has no UMLS concept mapping.
    """
    return pd.DataFrame(
        [
            {
                "coding_system": c["vocabulary"],
                "code": c["code"],
                "code_name": c.get("description", ""),
                "concept": "",
                "concept_name": c.get("description", ""),
                "tags": (c.get("label") or "").lower(),
            }
            for c in classified
        ],
        columns=["coding_system", "code", "code_name", "concept", "concept_name", "tags"],
    )


def evaluate_retrieval(
    retrieved_codes: list[dict],
    gold_labels: pd.DataFrame,
) -> RetrievalMetrics:
    """Evaluate concept retrieval as an information-retrieval task.

    True positives are retrieved codes labeled 'Narrow' in the gold standard.
    False positives are retrieved codes that are either labeled 'Possible'/'Exclude'
    in the gold standard, or absent from it entirely.
    False negatives are gold-standard 'Narrow' codes that were not retrieved.

    Parameters
    ----------
    retrieved_codes : list[dict]
        Each dict must contain ``code`` and ``vocabulary`` keys.
    gold_labels : pd.DataFrame
        Reference standard with ``code``, ``vocabulary``, ``label`` columns.

    Returns
    -------
    RetrievalMetrics
    """
    gold_by_key = {
        _standardized_key(row["code"], row["vocabulary"]): row["label"]
        for _, row in gold_labels.iterrows()
    }
    narrow_keys = {k for k, label in gold_by_key.items() if label == "Narrow"}

    retrieved_keys = {
        _standardized_key(r["code"], r["vocabulary"]) for r in retrieved_codes
    }

    tp = len(retrieved_keys & narrow_keys)
    fp = len(retrieved_keys - narrow_keys)
    fn = len(narrow_keys - retrieved_keys)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return RetrievalMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        sensitivity=round(sensitivity, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
    )


def _cohens_kappa(confusion: dict[tuple[str, str], int], labels: tuple[str, ...], n: int) -> float:
    if n == 0:
        return 0.0
    observed_agreement = sum(confusion.get((lbl, lbl), 0) for lbl in labels) / n

    row_totals = {lbl: sum(confusion.get((lbl, o), 0) for o in labels) for lbl in labels}
    col_totals = {lbl: sum(confusion.get((o, lbl), 0) for o in labels) for lbl in labels}
    expected_agreement = sum((row_totals[lbl] * col_totals[lbl]) / n for lbl in labels) / n

    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def evaluate_classification(
    classified: list[dict],
    gold_labels: pd.DataFrame,
) -> ClassificationMetrics:
    """Evaluate code classification (Narrow/Possible/Exclude) against gold labels.

    Only codes present in both the classified set and gold standard are compared
    (codes absent from either side cannot be scored for classification agreement).

    Parameters
    ----------
    classified : list[dict]
        Each dict must contain ``code``, ``vocabulary``, ``label`` keys.
    gold_labels : pd.DataFrame
        Reference standard with ``code``, ``vocabulary``, ``label`` columns.

    Returns
    -------
    ClassificationMetrics
    """
    gold_by_key = {
        _standardized_key(row["code"], row["vocabulary"]): row["label"]
        for _, row in gold_labels.iterrows()
    }

    confusion: dict[tuple[str, str], int] = {}
    correct = 0
    total = 0
    per_label_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABELS}

    for item in classified:
        key = _standardized_key(item["code"], item["vocabulary"])
        gold_label = gold_by_key.get(key)
        if gold_label is None:
            continue
        pred_label = item["label"]
        total += 1
        confusion[(gold_label, pred_label)] = confusion.get((gold_label, pred_label), 0) + 1
        if pred_label == gold_label:
            correct += 1
            per_label_counts[gold_label]["tp"] += 1
        else:
            per_label_counts[pred_label]["fp"] += 1
            per_label_counts[gold_label]["fn"] += 1

    accuracy = correct / total if total > 0 else 0.0
    kappa = _cohens_kappa(confusion, LABELS, total)

    per_label: dict[str, dict[str, float]] = {}
    for lbl in LABELS:
        tp = per_label_counts[lbl]["tp"]
        fp = per_label_counts[lbl]["fp"]
        fn = per_label_counts[lbl]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        cohens_kappa=round(kappa, 4),
        per_label=per_label,
    )


def evaluate(classified: list[dict], gold_labels: pd.DataFrame) -> dict:
    """Compute both retrieval and classification metrics for the full pipeline output.

    Parameters
    ----------
    classified : list[dict]
        Final classified candidates, each with ``code``, ``vocabulary``, ``label`` keys.
    gold_labels : pd.DataFrame
        Reference standard with ``code``, ``vocabulary``, ``label`` columns.

    Returns
    -------
    dict
        Dictionary with 'retrieval' and 'classification' metric sub-dictionaries.
    """
    retrieval_metrics = evaluate_retrieval(classified, gold_labels)
    classification_metrics = evaluate_classification(classified, gold_labels)
    return {
        "retrieval": asdict(retrieval_metrics),
        "classification": {
            "accuracy": classification_metrics.accuracy,
            "cohens_kappa": classification_metrics.cohens_kappa,
            "per_label": classification_metrics.per_label,
        },
    }
