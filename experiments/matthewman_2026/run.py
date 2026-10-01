#!/usr/bin/env python3
"""Run our pipeline on the exact Matthewman et al. reference design."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.matthewman_2026.experiment import (  # noqa: E402
    PHENOTYPES,
    build_classification_edf,
    file_sha256,
    load_cprd_browser,
    load_reference_codelists,
    scope_reference_to_corpus,
    score_as_dict,
    score_whole_codelist,
)
from src.llm.classify import (  # noqa: E402
    AdaptiveStoppingConfig,
    ClassificationRunInfo,
    gpt_review_low_confidence,
    llm_classify,
)
from src.llm.rank import skip_rank  # noqa: E402
from src.retrieval.embeddings import (  # noqa: E402
    EmbeddingIndex,
    create_embedding_model,
    embedding_cache_key,
)
from src.retrieval.hybrid import hybrid_retrieval  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CPRD_BROWSER = EXPERIMENT_DIR / "data" / "CPRDAurumMedical.txt"
DEFAULT_PAPER_REPO = EXPERIMENT_DIR / "data" / "llmcodelists_public"

PAPER_REPORTED_SCORES = {
    ("full", "gemini-3-pro-preview"): 0.43,
    ("full", "gemini-3-flash-preview"): 0.19,
    ("full", "gpt-5.2"): 0.14,
    ("full", "claude-sonnet-4-6"): 0.36,
    ("read", "gemini-3-pro-preview"): 0.81,
    ("read", "gemini-3-flash-preview"): 0.69,
    ("read", "gpt-5.2"): 0.29,
    ("read", "claude-sonnet-4-6"): 0.60,
}


def _summary(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(["subset", "pipeline", "jev_model", "gpt_model"], as_index=False)
        .agg(
            runs=("grade", "size"),
            correct=("grade", lambda values: int((values == "C").sum())),
            partially_correct=("grade", lambda values: int((values == "P").sum())),
            incorrect=("grade", lambda values: int((values == "I").sum())),
            paper_score=("grade_value", "mean"),
            mean_retrieval_recall=("retrieval_recall", "mean"),
            mean_final_required_recall=("final_required_recall", "mean"),
            mean_final_acceptable_precision=("final_acceptable_precision", "mean"),
            required_codes=("required_total", "sum"),
            required_not_retrieved=("required_not_retrieved_count", "sum"),
            retrieved_but_excluded=("retrieved_but_excluded_count", "sum"),
            required_not_classified=("required_not_classified_count", "sum"),
            classified_but_excluded=("classified_but_excluded_count", "sum"),
            irrelevant_predicted=("irrelevant_predicted_count", "sum"),
            hallucinated=("hallucinated_count", "sum"),
            candidates_classified=("candidates_classified", "sum"),
            gpt_candidates_reviewed=("gpt_candidates_reviewed", "sum"),
            gpt_labels_changed=("gpt_labels_changed", "sum"),
        )
    )
    return summary


def _paper_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for result in summary.itertuples():
        for (subset, paper_model), paper_score in PAPER_REPORTED_SCORES.items():
            if subset != result.subset:
                continue
            rows.append(
                {
                    "subset": subset,
                    "our_pipeline": result.pipeline,
                    "our_score": result.paper_score,
                    "paper_model": paper_model,
                    "paper_reported_score": paper_score,
                    "score_delta": result.paper_score - paper_score,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cprd-browser",
        type=Path,
        default=DEFAULT_CPRD_BROWSER,
        help=(
            "CPRD Aurum Medical Browser (.dta, .csv, or tab-delimited .txt); "
            f"default: {DEFAULT_CPRD_BROWSER}"
        ),
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=DEFAULT_PAPER_REPO,
        help=f"Authors' public repository; default: {DEFAULT_PAPER_REPO}",
    )
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional hard retrieval cap; omitted means score the complete vocabulary.",
    )
    parser.add_argument("--jev-model", default=None)
    parser.add_argument("--gpt-model", default=None)
    parser.add_argument("--phenotypes", nargs="+", choices=sorted(PHENOTYPES), default=None)
    parser.add_argument("--subsets", nargs="+", choices=("full", "read"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = yaml.safe_load(args.config.read_text())
    retrieval_config = config["retrieval"]
    configured_top_k = retrieval_config.get("top_k")
    top_k = args.top_k if args.top_k is not None else configured_top_k
    if top_k is not None and int(top_k) < 1:
        parser.error("--top-k must be at least 1")
    top_k = int(top_k) if top_k is not None else None
    design = config["design"]
    epochs = args.epochs if args.epochs is not None else int(design["epochs"])
    if epochs < 1:
        parser.error("--epochs must be at least 1")
    classification_config = config["classification"]
    review_config = config["review"]
    jev_model = args.jev_model or classification_config["model"]
    gpt_model = args.gpt_model or review_config["model"]
    phenotype_keys = args.phenotypes or list(PHENOTYPES)
    subset_names = args.subsets or design["subsets"]

    full_corpus, read_corpus = load_cprd_browser(args.cprd_browser)
    corpora = {"full": full_corpus, "read": read_corpus}
    references = load_reference_codelists(args.paper_repo)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "details").mkdir()

    manifest = {
        "paper": "10.64898/2026.04.23.26351098v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cprd_browser": str(args.cprd_browser.resolve()),
        "cprd_browser_sha256": file_sha256(args.cprd_browser),
        "paper_repo": str(args.paper_repo.resolve()),
        "reference_codelists": {
            key: {
                "path": str((args.paper_repo / "codelists" / filename).resolve()),
                "sha256": file_sha256(args.paper_repo / "codelists" / filename),
                "rows": len(references[key]),
                "required": int((references[key]["required"] == 1).sum()),
                "optional": int((references[key]["required"] == 0).sum()),
            }
            for key, (_, filename) in PHENOTYPES.items()
            if key in phenotype_keys
        },
        "config": config,
        "effective_top_k": top_k,
        "epochs": epochs,
        "pipeline": {
            "stage_2": f"jev:{jev_model}",
            "stage_3": f"openai:{gpt_model}",
        },
        "phenotypes": phenotype_keys,
        "subsets": subset_names,
        "corpus_rows": {name: len(corpora[name]) for name in subset_names},
        "dry_run": args.dry_run,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    scope_rows = []
    for subset_name in subset_names:
        corpus = corpora[subset_name]
        for phenotype_key in phenotype_keys:
            reference = references[phenotype_key]
            scoped = scope_reference_to_corpus(reference, corpus)
            scope_rows.append(
                {
                    "subset": subset_name,
                    "phenotype": phenotype_key,
                    "reference_rows_public": len(reference),
                    "reference_rows_in_corpus": len(scoped),
                    "required_in_corpus": int((scoped["required"] == 1).sum()),
                    "optional_in_corpus": int((scoped["required"] == 0).sum()),
                }
            )
    pd.DataFrame(scope_rows).to_csv(run_dir / "reference_scope.csv", index=False)
    if args.dry_run:
        print(f"Dry-run validation complete: {run_dir}")
        return

    embedding_provider = retrieval_config["embedding_provider"]
    embedding_model_name = retrieval_config["embedding_model"]
    embedding_dimensions = retrieval_config.get("embedding_dimensions")
    query_model = create_embedding_model(
        embedding_provider,
        embedding_model_name,
        dimensions=embedding_dimensions,
        batch_size=int(retrieval_config.get("embedding_batch_size", 512)),
    )
    rows: list[dict] = []
    for subset_name in subset_names:
        corpus = corpora[subset_name]
        cache_identity = embedding_cache_key(
            embedding_provider,
            embedding_model_name,
            embedding_dimensions,
        )
        cache_path = EXPERIMENT_DIR / "cache" / f"{subset_name}_{cache_identity}.npz"
        embedding_index = EmbeddingIndex.from_cache_or_build(
            corpus,
            cache_path=cache_path,
            model_name=embedding_model_name,
            model=query_model,
        )
        for phenotype_key in phenotype_keys:
            phenotype_name, _ = PHENOTYPES[phenotype_key]
            edf = build_classification_edf(
                phenotype_key,
                phenotype_name,
                config["classification_policy"],
            )
            candidates = hybrid_retrieval(
                phenotype_name,
                corpus,
                lexical_weight=float(retrieval_config["lexical_weight"]),
                embedding_weight=float(retrieval_config["embedding_weight"]),
                embedding_model=embedding_model_name,
                top_k=top_k,
                per_vocabulary_top_k=False,
                embedding_index=embedding_index,
                query_model=query_model,
                lexical_query=phenotype_name,
                embedding_query=phenotype_name,
            )
            ranked = skip_rank(candidates)
            reference = scope_reference_to_corpus(references[phenotype_key], corpus)
            for epoch in range(1, epochs + 1):
                run_info_holder: list[ClassificationRunInfo] = []
                stage2_classified = llm_classify(
                    ranked,
                    edf,
                    provider="jev",
                    model=jev_model,
                    batch_size=int(classification_config["batch_size"]),
                    use_possible_category=False,
                    adaptive_stopping=AdaptiveStoppingConfig(
                        sparse_narrow_threshold=int(
                            classification_config["sparse_narrow_threshold"]
                        ),
                        consecutive_sparse_batches=int(
                            classification_config["consecutive_sparse_batches"]
                        ),
                        minimum_batches=int(classification_config["minimum_batches"]),
                    ),
                    run_info_callback=run_info_holder.append,
                )
                classification_run = run_info_holder[0]
                final_classified, review_run = gpt_review_low_confidence(
                    stage2_classified,
                    edf,
                    confidence_threshold=float(review_config["confidence_threshold"]),
                    max_candidates=int(review_config["max_candidates"]),
                    model=gpt_model,
                    batch_size=int(review_config["batch_size"]),
                    use_possible_category=False,
                    reasoning_effort=str(review_config["reasoning_effort"]),
                )
                predicted = [
                    candidate.code
                    for candidate in final_classified
                    if candidate.label == "Narrow"
                ]
                score = score_whole_codelist(
                    reference,
                    retrieved_codes=[candidate.code for candidate in candidates],
                    classified_codes=[candidate.code for candidate in stage2_classified],
                    predicted_codes=predicted,
                )
                score_dict = score_as_dict(score)
                detail_name = f"{subset_name}__{phenotype_key}__epoch-{epoch}.json"

                def serialize(candidate):
                    return {
                        "code": candidate.code,
                        "description": candidate.description,
                        "label": candidate.label,
                        "confidence": candidate.confidence,
                        "explanation": candidate.explanation,
                        "hybrid_score": candidate.retrieval_score,
                    }

                detail = {
                    "subset": subset_name,
                    "phenotype": phenotype_key,
                    "phenotype_input": phenotype_name,
                    "pipeline": "Jev classification -> GPT-6 Luna uncertainty review",
                    "jev_model": jev_model,
                    "gpt_model": gpt_model,
                    "classification_policy": config["classification_policy"],
                    "epoch": epoch,
                    "ranked_candidates": len(candidates),
                    "classification_run": asdict(classification_run),
                    "gpt_review_run": asdict(review_run),
                    "score": score_dict,
                    "stage2_classified": [serialize(candidate) for candidate in stage2_classified],
                    "final_classified": [serialize(candidate) for candidate in final_classified],
                }
                (run_dir / "details" / detail_name).write_text(json.dumps(detail, indent=2))
                rows.append(
                    {
                        "subset": subset_name,
                        "phenotype": phenotype_key,
                        "phenotype_input": phenotype_name,
                        "pipeline": "Jev -> GPT-6 Luna",
                        "jev_model": jev_model,
                        "gpt_model": gpt_model,
                        "epoch": epoch,
                        "candidates_available": classification_run.candidates_available,
                        "candidates_classified": classification_run.candidates_classified,
                        "classification_batches": classification_run.batches_completed,
                        "stopped_early": classification_run.stopped_early,
                        "gpt_candidates_eligible": review_run.eligible_candidates,
                        "gpt_candidates_reviewed": review_run.candidates_reviewed,
                        "gpt_labels_changed": review_run.labels_changed,
                        **{
                            key: value
                            for key, value in score_dict.items()
                            if not isinstance(value, list)
                        },
                    }
                )
                print(
                    f"[{subset_name}] [{phenotype_key}] epoch {epoch}/{epochs}: "
                    f"{score.grade}; Jev classified "
                    f"{classification_run.candidates_classified}/{len(candidates)}; "
                    f"GPT reviewed {review_run.candidates_reviewed}"
                )

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "runs.csv", index=False)
    (run_dir / "runs.json").write_text(json.dumps(rows, indent=2))
    summary = _summary(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    _paper_comparison(summary).to_csv(run_dir / "paper_comparison.csv", index=False)
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
