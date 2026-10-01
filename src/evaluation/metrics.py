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
    true_negatives: int
    sensitivity: float
    precision: float
    f1: float
    macro_sensitivity: float
    macro_precision: float
    macro_f1: float
    per_label: dict[str, dict[str, float]]


@dataclass(slots=True)
class NarrowLossBreakdown:
    """Where available gold-Narrow codes were lost end to end."""

    gold_narrow_available: int
    correctly_classified: int
    misclassified: int
    not_classified: int
    total_missed: int
    end_to_end_recall: float


@dataclass(slots=True)
class GPTReviewEvaluation:
    """Gold-standard impact of the final low-confidence GPT review."""

    candidates_reviewed: int
    labels_changed: int
    corrected_changes: int
    harmful_changes: int
    unchanged: int
    correct_before: int
    correct_after: int
    net_correct_change: int
    reviewed_accuracy_before: float
    reviewed_accuracy_after: float


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


def filter_gold_by_available_codes(
    gold_labels: pd.DataFrame,
    available_codes: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict gold labels to codes present in the loaded terminology corpus.

    This prevents codes from older or otherwise unavailable terminology
    versions from being counted as false negatives.
    """
    required = {"code", "vocabulary"}
    missing = required - set(available_codes.columns)
    if missing:
        raise ValueError(f"Available code corpus missing required columns: {missing}")

    available_keys = {
        _standardized_key(row["code"], row["vocabulary"])
        for _, row in available_codes.iterrows()
    }
    mask = gold_labels.apply(
        lambda row: _standardized_key(row["code"], row["vocabulary"]) in available_keys,
        axis=1,
    )
    return gold_labels[mask].reset_index(drop=True)


def narrow_loss_breakdown(
    classified: list[dict],
    gold_labels: pd.DataFrame,
) -> NarrowLossBreakdown:
    """Separate gold-Narrow losses before and during LLM classification.

    ``not_classified`` covers gold-Narrow codes that never reached the model
    (for example because adaptive stopping or a hard top-k limit ended the
    scan). ``misclassified`` covers codes that did reach the model but were
    assigned a label other than Narrow.
    """
    gold_narrow_keys = {
        _standardized_key(row["code"], row["vocabulary"])
        for _, row in gold_labels.iterrows()
        if str(row["label"]).strip().title() == "Narrow"
    }
    predictions_by_key = {
        _standardized_key(item["code"], item["vocabulary"]):
        str(item.get("label", "")).strip().title()
        for item in classified
    }

    correctly_classified = sum(
        predictions_by_key.get(key) == "Narrow" for key in gold_narrow_keys
    )
    misclassified = sum(
        key in predictions_by_key and predictions_by_key[key] != "Narrow"
        for key in gold_narrow_keys
    )
    not_classified = len(gold_narrow_keys - predictions_by_key.keys())
    total_missed = misclassified + not_classified
    recall = correctly_classified / len(gold_narrow_keys) if gold_narrow_keys else 0.0

    return NarrowLossBreakdown(
        gold_narrow_available=len(gold_narrow_keys),
        correctly_classified=correctly_classified,
        misclassified=misclassified,
        not_classified=not_classified,
        total_missed=total_missed,
        end_to_end_recall=round(recall, 4),
    )


def evaluate_gpt_review(
    stage2_classified: list[dict],
    final_classified: list[dict],
    gold_labels: pd.DataFrame,
    *,
    confidence_threshold: float,
    include_possible: bool = False,
) -> GPTReviewEvaluation:
    """Measure whether GPT's reviewed labels improved stage-2 decisions."""
    gold_by_key = {
        _standardized_key(row["code"], row["vocabulary"]): (
            row["label"]
            if include_possible or row["label"] != "Possible"
            else "Exclude"
        )
        for _, row in gold_labels.iterrows()
    }
    final_by_key = {
        _standardized_key(item["code"], item["vocabulary"]):
        str(item.get("label", "")).strip().title()
        for item in final_classified
    }

    reviewed = 0
    changed = 0
    corrected = 0
    harmful = 0
    correct_before = 0
    correct_after = 0
    for original in stage2_classified:
        if float(original.get("confidence", 0.0)) >= confidence_threshold:
            continue
        key = _standardized_key(original["code"], original["vocabulary"])
        if key not in final_by_key:
            continue
        reviewed += 1
        original_label = str(original.get("label", "")).strip().title()
        final_label = final_by_key[key]
        if not include_possible:
            original_label = "Exclude" if original_label == "Possible" else original_label
            final_label = "Exclude" if final_label == "Possible" else final_label
        gold_label = gold_by_key.get(key, "Exclude")
        was_correct = original_label == gold_label
        is_correct = final_label == gold_label
        correct_before += was_correct
        correct_after += is_correct
        if original_label != final_label:
            changed += 1
            corrected += is_correct and not was_correct
            harmful += was_correct and not is_correct

    return GPTReviewEvaluation(
        candidates_reviewed=reviewed,
        labels_changed=changed,
        corrected_changes=corrected,
        harmful_changes=harmful,
        unchanged=reviewed - changed,
        correct_before=correct_before,
        correct_after=correct_after,
        net_correct_change=correct_after - correct_before,
        reviewed_accuracy_before=round(correct_before / reviewed, 4) if reviewed else 0.0,
        reviewed_accuracy_after=round(correct_after / reviewed, 4) if reviewed else 0.0,
    )


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
        AESI-compatible columns. When classification annotations are supplied,
        confidence, explanation, manual-review flag, and review reason are
        appended. ``concept`` is blank because the corpus has no UMLS mapping.
    """
    base_columns = ["coding_system", "code", "code_name", "concept", "concept_name", "tags"]
    annotation_columns = ["confidence", "explanation", "manual_review", "review_reason"]
    include_annotations = any(any(column in item for column in annotation_columns) for item in classified)
    columns = base_columns + annotation_columns if include_annotations else base_columns
    return pd.DataFrame(
        [
            {
                "coding_system": c["vocabulary"],
                "code": c["code"],
                "code_name": c.get("description", ""),
                "concept": "",
                "concept_name": c.get("description", ""),
                "tags": (c.get("label") or "").lower(),
                **(
                    {
                        "confidence": c.get("confidence", ""),
                        "explanation": c.get("explanation", ""),
                        "manual_review": bool(c.get("manual_review", False)),
                        "review_reason": c.get("review_reason", ""),
                    }
                    if include_annotations
                    else {}
                ),
            }
            for c in classified
        ],
        columns=columns,
    )


def evaluate_retrieval(
    retrieved_codes: list[dict],
    gold_labels: pd.DataFrame,
    narrow_only_as_positive: bool = False,
) -> RetrievalMetrics:
    """Evaluate concept retrieval as an information-retrieval task.

    By default, true positives are retrieved codes labeled 'Narrow' or
    'Possible' in the gold standard. With ``narrow_only_as_positive=True``, only
    'Narrow' is positive.
    False positives are retrieved codes labeled 'Exclude' in the gold standard,
    or codes absent from it entirely.
    False negatives are positive gold-standard codes that were not retrieved.

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
    positive_labels = {"Narrow"} if narrow_only_as_positive else {"Narrow", "Possible"}
    positive_keys = {k for k, label in gold_by_key.items() if label in positive_labels}

    retrieved_keys = {
        _standardized_key(r["code"], r["vocabulary"]) for r in retrieved_codes
    }

    tp = len(retrieved_keys & positive_keys)
    fp = len(retrieved_keys - positive_keys)
    fn = len(positive_keys - retrieved_keys)

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
    include_possible: bool = True,
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
    labels = LABELS if include_possible else ("Narrow", "Exclude")
    gold_by_key = {
        _standardized_key(row["code"], row["vocabulary"]): (
            row["label"] if include_possible or row["label"] != "Possible" else "Exclude"
        )
        for _, row in gold_labels.iterrows()
    }

    confusion: dict[tuple[str, str], int] = {}
    correct = 0
    total = 0
    per_label_counts = {lbl: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for lbl in labels}

    for item in classified:
        key = _standardized_key(item["code"], item["vocabulary"])
        gold_label = gold_by_key.get(key)
        pred_label = item["label"]
        if not include_possible and pred_label == "Possible":
            pred_label = "Exclude"
        if gold_label is None:
            # A retrieved code absent from the curated positive codelist is an
            # implicit Exclude. Apply this consistently to both correct
            # Exclude predictions and false-positive Narrow/Possible results.
            gold_label = "Exclude"

        total += 1
        confusion[(gold_label, pred_label)] = confusion.get((gold_label, pred_label), 0) + 1
        if pred_label == gold_label:
            correct += 1
        for label in labels:
            gold_positive = gold_label == label
            predicted_positive = pred_label == label
            if gold_positive and predicted_positive:
                per_label_counts[label]["tp"] += 1
            elif not gold_positive and not predicted_positive:
                per_label_counts[label]["tn"] += 1
            elif predicted_positive:
                per_label_counts[label]["fp"] += 1
            else:
                per_label_counts[label]["fn"] += 1

    accuracy = correct / total if total > 0 else 0.0
    kappa = _cohens_kappa(confusion, labels, total)

    per_label: dict[str, dict[str, float]] = {}
    for lbl in labels:
        tp = per_label_counts[lbl]["tp"]
        tn = per_label_counts[lbl]["tn"]
        fp = per_label_counts[lbl]["fp"]
        fn = per_label_counts[lbl]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[lbl] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_labels = [
        label
        for label, counts in per_label_counts.items()
        if counts["tp"] + counts["fn"] > 0
    ] or list(labels)

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        cohens_kappa=round(kappa, 4),
        true_negatives=per_label["Narrow"]["tn"],
        sensitivity=per_label["Narrow"]["recall"],
        precision=per_label["Narrow"]["precision"],
        f1=per_label["Narrow"]["f1"],
        macro_sensitivity=round(sum(per_label[label]["recall"] for label in macro_labels) / len(macro_labels), 4),
        macro_precision=round(sum(per_label[label]["precision"] for label in macro_labels) / len(macro_labels), 4),
        macro_f1=round(sum(per_label[label]["f1"] for label in macro_labels) / len(macro_labels), 4),
        per_label=per_label,
    )


def evaluate(
    classified: list[dict],
    gold_labels: pd.DataFrame,
    narrow_only_as_positive: bool = False,
    include_possible: bool = True,
) -> dict:
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
    retrieval_metrics = evaluate_retrieval(
        classified,
        gold_labels,
        narrow_only_as_positive=narrow_only_as_positive,
    )
    classification_metrics = evaluate_classification(
        classified,
        gold_labels,
        include_possible=include_possible,
    )
    return {
        "retrieval": asdict(retrieval_metrics),
        "classification": {
            "accuracy": classification_metrics.accuracy,
            "cohens_kappa": classification_metrics.cohens_kappa,
            "true_negatives": classification_metrics.true_negatives,
            "sensitivity": classification_metrics.sensitivity,
            "precision": classification_metrics.precision,
            "f1": classification_metrics.f1,
            "macro_sensitivity": classification_metrics.macro_sensitivity,
            "macro_precision": classification_metrics.macro_precision,
            "macro_f1": classification_metrics.macro_f1,
            "per_label": classification_metrics.per_label,
        },
    }
