#!/usr/bin/env python3
"""Turn an experiment result directory into a self-contained Markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_METRICS = (
    "review_rate",
    "retrieval_sensitivity",
    "retrieval_precision",
    "retrieval_f1",
    "classification_accuracy",
    "classification_sensitivity",
    "classification_precision",
    "classification_f1",
    "classification_macro_sensitivity",
    "classification_macro_precision",
    "classification_macro_f1",
    "cohens_kappa",
)


def _format_setting_value(value: float | int) -> str:
    return f"{value:g}"


def _setting_directory(row: dict[str, Any]) -> str:
    return (
        f"top_k-{row['top_k']}_"
        f"lexical-{_format_setting_value(row['lexical_weight'])}_"
        f"embedding-{_format_setting_value(row['embedding_weight'])}"
    )


def _load_details(experiment_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    expected = (
        experiment_dir
        / _setting_directory(row)
        / f"{row['phenotype']}_metrics.json"
    )
    if expected.exists():
        return json.loads(expected.read_text())

    matches = list(experiment_dir.rglob(f"{row['phenotype']}_metrics.json"))
    if len(matches) == 1:
        return json.loads(matches[0].read_text())
    return {}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, sensitivity: float) -> float:
    return 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0


def _enrich_row(experiment_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    row = dict(source)
    details = _load_details(experiment_dir, row)
    metrics = details.get("metrics") or {}
    retrieval = metrics.get("retrieval") or {}
    classification = metrics.get("classification") or {}
    narrow = (classification.get("per_label") or {}).get("Narrow") or {}
    scope = metrics.get("evaluation_scope") or {}

    row.update(
        {
            "retrieval_tp": retrieval.get("true_positives"),
            "retrieval_fp": retrieval.get("false_positives"),
            "retrieval_fn": retrieval.get("false_negatives"),
            "classification_tp": narrow.get("tp"),
            "classification_tn": row.get(
                "classification_true_negatives",
                classification.get("true_negatives", narrow.get("tn")),
            ),
            "classification_fp": narrow.get("fp"),
            "classification_fn": narrow.get("fn"),
            "gold_rows_selected": scope.get("gold_rows_in_selected_vocabularies"),
            "gold_rows_available": scope.get("gold_rows_available_in_loaded_sources"),
            "gold_rows_excluded": scope.get("gold_rows_excluded_as_unavailable"),
        }
    )
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def _pooled_classification(rows: list[dict[str, Any]]) -> dict[str, float | int] | None:
    count_keys = ("classification_tp", "classification_tn", "classification_fp", "classification_fn")
    if not all(all(row.get(key) is not None for key in count_keys) for row in rows):
        return None
    tp = sum(int(row["classification_tp"]) for row in rows)
    tn = sum(int(row["classification_tn"]) for row in rows)
    fp = sum(int(row["classification_fp"]) for row in rows)
    fn = sum(int(row["classification_fn"]) for row in rows)
    sensitivity = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(precision, sensitivity),
    }


def build_report(experiment_dir: Path) -> str:
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment summary not found: {summary_path}")
    raw_rows = json.loads(summary_path.read_text())
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"Experiment summary must contain a non-empty JSON list: {summary_path}")
    rows = [_enrich_row(experiment_dir, row) for row in raw_rows]

    phenotype_count = len({row["phenotype"] for row in rows})
    settings = {
        (
            row["top_k"],
            row["lexical_weight"],
            row["embedding_weight"],
            row.get("possible_category", False),
        )
        for row in rows
    }
    pooled = _pooled_classification(rows)
    abstract = (
        f"This experiment evaluated semantic codelist harmonization across "
        f"{phenotype_count} phenotype samples and {len(settings)} retrieval setting(s) "
        f"({len(rows)} phenotype-setting runs). Mean retrieval sensitivity, precision, "
        f"and F1 were {_pct(_mean(rows, 'retrieval_sensitivity'))}, "
        f"{_pct(_mean(rows, 'retrieval_precision'))}, and "
        f"{_pct(_mean(rows, 'retrieval_f1'))}, respectively. Mean Narrow-class "
        f"classification sensitivity, precision, and F1 were "
        f"{_pct(_mean(rows, 'classification_sensitivity'))}, "
        f"{_pct(_mean(rows, 'classification_precision'))}, and "
        f"{_pct(_mean(rows, 'classification_f1'))}; mean accuracy was "
        f"{_pct(_mean(rows, 'classification_accuracy'))}, and mean Cohen's kappa was "
        f"{_number(_mean(rows, 'cohens_kappa'))}."
    )
    if pooled:
        abstract += (
            f" Pooling Narrow one-vs-rest counts across runs produced "
            f"TP={pooled['tp']:,}, TN={pooled['tn']:,}, FP={pooled['fp']:,}, and "
            f"FN={pooled['fn']:,}, corresponding to sensitivity "
            f"{_pct(pooled['sensitivity'])}, precision {_pct(pooled['precision'])}, "
            f"and F1 {_pct(pooled['f1'])}."
        )

    setting_rows: list[list[str]] = []
    for top_k, lexical, embedding, possible in sorted(settings):
        subset = [
            row
            for row in rows
            if (
                row["top_k"],
                row["lexical_weight"],
                row["embedding_weight"],
                row.get("possible_category", False),
            )
            == (top_k, lexical, embedding, possible)
        ]
        setting_rows.append(
            [
                str(top_k),
                _number(lexical, 2),
                _number(embedding, 2),
                "Yes" if possible else "No",
                str(len(subset)),
                _pct(_mean(subset, "review_rate")),
                _pct(_mean(subset, "retrieval_sensitivity")),
                _pct(_mean(subset, "retrieval_precision")),
                _pct(_mean(subset, "retrieval_f1")),
                _pct(_mean(subset, "classification_accuracy")),
                _pct(_mean(subset, "classification_sensitivity")),
                _pct(_mean(subset, "classification_precision")),
                _pct(_mean(subset, "classification_f1")),
                _number(_mean(subset, "cohens_kappa")),
            ]
        )

    exact_rows = [
        [
            str(row["phenotype"]),
            str(row.get("vocabularies", "")),
            str(row["top_k"]),
            _number(row["lexical_weight"], 2),
            _number(row["embedding_weight"], 2),
            "Yes" if row.get("possible_category") else "No",
            _pct(row.get("review_rate")),
            _integer(row.get("retrieval_tp")),
            _integer(row.get("retrieval_fp")),
            _integer(row.get("retrieval_fn")),
            _pct(row.get("retrieval_sensitivity")),
            _pct(row.get("retrieval_precision")),
            _pct(row.get("retrieval_f1")),
            _integer(row.get("classification_tp")),
            _integer(row.get("classification_tn")),
            _integer(row.get("classification_fp")),
            _integer(row.get("classification_fn")),
            _pct(row.get("classification_accuracy")),
            _pct(row.get("classification_sensitivity")),
            _pct(row.get("classification_precision")),
            _pct(row.get("classification_f1")),
            _pct(row.get("classification_macro_sensitivity")),
            _pct(row.get("classification_macro_precision")),
            _pct(row.get("classification_macro_f1")),
            _number(row.get("cohens_kappa")),
            _integer(row.get("gold_rows_selected")),
            _integer(row.get("gold_rows_available")),
            _integer(row.get("gold_rows_excluded")),
        ]
        for row in rows
    ]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"# Experiment report: {experiment_dir.name}",
        f"Generated {generated} from `{summary_path}`.",
        "## Abstract",
        abstract,
        "## Settings-level summary (macro average across phenotype runs)",
        _markdown_table(
            [
                "Top-K", "Lexical", "Embedding", "Possible", "Runs", "Review rate",
                "Ret sens", "Ret prec", "Ret F1", "Cls acc", "Cls sens", "Cls prec",
                "Cls F1", "Kappa",
            ],
            setting_rows,
        ),
        "## Complete results",
        _markdown_table(
            [
                "Phenotype", "Vocabularies", "K", "Lex", "Emb", "Possible", "Review",
                "Ret TP", "Ret FP", "Ret FN", "Ret sens", "Ret prec", "Ret F1",
                "Cls TP", "Cls TN", "Cls FP", "Cls FN", "Accuracy", "Cls sens",
                "Cls prec", "Cls F1", "Macro sens", "Macro prec", "Macro F1", "Kappa",
                "Gold selected", "Gold available", "Gold excluded",
            ],
            exact_rows,
        ),
        "## Interpretation notes",
        "- Retrieval metrics measure whether available Narrow gold codes were retrieved.",
        "- Classification sensitivity, precision, F1, and TP/TN/FP/FN treat Narrow as the positive class.",
        "- Macro metrics average across labels represented in the reference data.",
        "- Cohen's kappa measures agreement beyond chance and should be read alongside class-specific metrics.",
        "- Gold rows absent from the loaded terminology versions are excluded and reported in the final columns.",
        "",
    ]
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="Directory containing summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (default: <experiment_dir>/experiment_report.md)",
    )
    args = parser.parse_args()

    output_path = args.output or args.experiment_dir / "experiment_report.md"
    report = build_report(args.experiment_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Wrote experiment report to {output_path}")


if __name__ == "__main__":
    main()
