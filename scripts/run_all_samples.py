#!/usr/bin/env python3
"""Run every registered EDF/AESI sample over a small retrieval-settings grid."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_pipeline  # noqa: E402
from src.data.loaders import load_aesi_dataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _write_run(results: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    phenotype = results["phenotype"]
    results["predicted_codelist"].to_csv(output_dir / f"{phenotype}_predicted.csv", index=False)
    payload = {
        "phenotype": phenotype,
        "review_items": results["review_items"],
        "review_rate": results["review_rate"],
        "metrics": results["metrics"],
    }
    (output_dir / f"{phenotype}_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def _console_progress(prefix: str):
    """Create a callback that prints stages and a live classification bar."""

    def report(message: str, completed: int | None, total: int | None) -> None:
        if completed is None or total is None:
            print(f"{prefix} {message}", flush=True)
            return
        fraction = completed / total if total else 1.0
        width = 30
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        end = "\n" if completed >= total else "\r"
        print(
            f"{prefix} {message} [{bar}] {completed}/{total} ({fraction:.0%})",
            end=end,
            flush=True,
        )

    return report


def _ground_truth_vocabularies(config, phenotype: str) -> list[str]:
    """Return configured vocabularies that occur in a phenotype's AESI file."""
    _, gold_labels = load_aesi_dataset(config.paths.aesi_datasets[phenotype])
    present = set(gold_labels["vocabulary"].astype(str))
    return [vocabulary for vocabulary in config.retrieval.vocabularies if vocabulary in present]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=None,
        help="Phenotype keys to run. Defaults to every phenotype with a registered AESI codelist.",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=None)
    parser.add_argument(
        "--lexical-weight",
        nargs="+",
        type=float,
        default=None,
        help="One or more lexical weights; embedding weight is 1 minus each value.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run identifier. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--possible",
        action="store_true",
        help="Enable the optional Possible classification category.",
    )
    parser.add_argument(
        "--all-vocabularies",
        action="store_true",
        help="Use every configured vocabulary instead of only those present in each sample's AESI ground truth.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    phenotypes = args.phenotypes or sorted(base_config.paths.aesi_datasets)
    top_k_values = args.top_k or [base_config.retrieval.top_k]
    lexical_weights = args.lexical_weight or [base_config.retrieval.lexical_weight]
    unknown = sorted(set(phenotypes) - set(base_config.paths.aesi_datasets))
    if unknown:
        parser.error(f"Unregistered phenotype(s): {', '.join(unknown)}")
    if any(value < 1 for value in top_k_values):
        parser.error("--top-k values must be at least 1")
    if any(not 0 <= value <= 1 for value in lexical_weights):
        parser.error("--lexical-weight values must be between 0 and 1")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = Path(base_config.paths.results_dir) / "experiments" / run_id
    summary_rows: list[dict] = []
    total_runs = len(top_k_values) * len(lexical_weights) * len(phenotypes)
    run_number = 0

    for top_k in top_k_values:
        for lexical_weight in lexical_weights:
            embedding_weight = 1.0 - lexical_weight
            setting = f"top_k-{top_k}_lexical-{lexical_weight:g}_embedding-{embedding_weight:g}"
            for phenotype in phenotypes:
                run_number += 1
                config = copy.deepcopy(base_config)
                config.phenotype = phenotype
                config.retrieval.top_k = top_k
                config.retrieval.lexical_weight = lexical_weight
                config.retrieval.embedding_weight = embedding_weight
                config.llm.use_possible_category = args.possible

                if not args.all_vocabularies:
                    selected_vocabularies = _ground_truth_vocabularies(base_config, phenotype)
                    config.retrieval.vocabularies = selected_vocabularies
                    config.evaluation.compare_vocabularies = selected_vocabularies

                prefix = f"[run {run_number}/{total_runs}] [{phenotype}]"
                print(
                    f"\n{prefix} Starting {setting}; vocabularies: "
                    f"{', '.join(config.retrieval.vocabularies)}",
                    flush=True,
                )
                results = run_pipeline(config, progress_callback=_console_progress(prefix))
                payload = _write_run(results, batch_dir / setting)
                print(f"{prefix} Finished; outputs written to {batch_dir / setting}", flush=True)
                row = {
                    "phenotype": phenotype,
                    "top_k": top_k,
                    "lexical_weight": lexical_weight,
                    "embedding_weight": embedding_weight,
                    "possible_category": args.possible,
                    "vocabularies": ",".join(config.retrieval.vocabularies),
                    "review_rate": payload["review_rate"],
                }
                if payload["metrics"]:
                    row.update(
                        {
                            "retrieval_sensitivity": payload["metrics"]["retrieval"]["sensitivity"],
                            "retrieval_precision": payload["metrics"]["retrieval"]["precision"],
                            "retrieval_f1": payload["metrics"]["retrieval"]["f1"],
                            "classification_accuracy": payload["metrics"]["classification"]["accuracy"],
                            "classification_true_negatives": payload["metrics"]["classification"]["true_negatives"],
                            "classification_sensitivity": payload["metrics"]["classification"]["sensitivity"],
                            "classification_precision": payload["metrics"]["classification"]["precision"],
                            "classification_f1": payload["metrics"]["classification"]["f1"],
                            "classification_macro_sensitivity": payload["metrics"]["classification"]["macro_sensitivity"],
                            "classification_macro_precision": payload["metrics"]["classification"]["macro_precision"],
                            "classification_macro_f1": payload["metrics"]["classification"]["macro_f1"],
                            "cohens_kappa": payload["metrics"]["classification"]["cohens_kappa"],
                        }
                    )
                summary_rows.append(row)

    batch_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(batch_dir / "summary.csv", index=False)
    (batch_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"Completed {len(summary_rows)} runs. Summary: {batch_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
