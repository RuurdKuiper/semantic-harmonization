#!/usr/bin/env python3
"""Run our pipeline on the exact Matthewman et al. reference design."""

from __future__ import annotations

import argparse
import json
import sys
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
    file_sha256,
    load_cprd_browser,
    load_reference_codelists,
    scope_reference_to_corpus,
    score_as_dict,
    score_whole_codelist,
)
from src.data.loaders import EventDefinitionForm  # noqa: E402
from src.llm.classify import llm_classify  # noqa: E402
from src.llm.rank import skip_rank  # noqa: E402
from src.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from src.retrieval.hybrid import hybrid_retrieval  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent

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


def _parse_model(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Model must use provider:model syntax: {spec!r}")
    provider, model = spec.split(":", 1)
    if provider not in {"anthropic", "openai", "google", "jev"} or not model:
        raise ValueError(f"Unsupported model specification: {spec!r}")
    return provider, model


def _summary(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(["subset", "provider", "model"], as_index=False)
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
            irrelevant_predicted=("irrelevant_predicted_count", "sum"),
            hallucinated=("hallucinated_count", "sum"),
        )
    )
    summary["paper_reported_score"] = [
        PAPER_REPORTED_SCORES.get((row.subset, row.model)) for row in summary.itertuples()
    ]
    summary["score_delta_vs_paper"] = summary["paper_score"] - summary["paper_reported_score"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cprd-browser", required=True, type=Path)
    parser.add_argument("--paper-repo", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval pool size (default from config: 500).",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--phenotypes", nargs="+", choices=sorted(PHENOTYPES), default=None)
    parser.add_argument("--subsets", nargs="+", choices=("full", "read"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = yaml.safe_load(args.config.read_text())
    retrieval_config = config["retrieval"]
    top_k = args.top_k or int(retrieval_config["top_k"])
    if top_k < 1:
        parser.error("--top-k must be at least 1")
    design = config["design"]
    epochs = args.epochs or int(design["epochs"])
    if epochs < 1:
        parser.error("--epochs must be at least 1")
    model_specs = args.models or design["models"]
    models = [_parse_model(spec) for spec in model_specs]
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
        "models": model_specs,
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

    from sentence_transformers import SentenceTransformer

    embedding_model_name = retrieval_config["embedding_model"]
    query_model = SentenceTransformer(embedding_model_name)
    rows: list[dict] = []
    for subset_name in subset_names:
        corpus = corpora[subset_name]
        cache_path = EXPERIMENT_DIR / "cache" / (
            f"{subset_name}_{embedding_model_name.replace('/', '_')}.npz"
        )
        embedding_index = EmbeddingIndex.from_cache_or_build(
            corpus,
            cache_path=cache_path,
            model_name=embedding_model_name,
            model=query_model,
        )
        for phenotype_key in phenotype_keys:
            phenotype_name, _ = PHENOTYPES[phenotype_key]
            edf = EventDefinitionForm(
                name=phenotype_key,
                preferred_name=phenotype_name,
                definition=config["minimal_instruction"],
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
            for provider, model in models:
                for epoch in range(1, epochs + 1):
                    classified = llm_classify(
                        ranked,
                        edf,
                        provider=provider,
                        model=model,
                        batch_size=(
                            20 if provider == "jev" else int(retrieval_config["batch_size"])
                        ),
                        use_possible_category=False,
                    )
                    predicted = [candidate.code for candidate in classified if candidate.label == "Narrow"]
                    score = score_whole_codelist(
                        reference,
                        retrieved_codes=[candidate.code for candidate in candidates],
                        predicted_codes=predicted,
                    )
                    score_dict = score_as_dict(score)
                    detail_name = (
                        f"{subset_name}__{phenotype_key}__{provider}__"
                        f"{model.replace('/', '_')}__epoch-{epoch}.json"
                    )
                    detail = {
                        "subset": subset_name,
                        "phenotype": phenotype_key,
                        "phenotype_input": phenotype_name,
                        "provider": provider,
                        "model": model,
                        "epoch": epoch,
                        "score": score_dict,
                        "retrieved": [
                            {
                                "code": candidate.code,
                                "description": candidate.description,
                                "hybrid_score": candidate.score,
                                "lexical_score": candidate.lexical_score,
                                "embedding_score": candidate.embedding_score,
                            }
                            for candidate in candidates
                        ],
                        "classified": [
                            {
                                "code": candidate.code,
                                "description": candidate.description,
                                "label": candidate.label,
                                "confidence": candidate.confidence,
                                "explanation": candidate.explanation,
                            }
                            for candidate in classified
                        ],
                    }
                    (run_dir / "details" / detail_name).write_text(json.dumps(detail, indent=2))
                    rows.append(
                        {
                            "subset": subset_name,
                            "phenotype": phenotype_key,
                            "phenotype_input": phenotype_name,
                            "provider": provider,
                            "model": model,
                            "epoch": epoch,
                            **{key: value for key, value in score_dict.items() if not isinstance(value, list)},
                        }
                    )
                    print(
                        f"[{subset_name}] [{phenotype_key}] [{provider}:{model}] "
                        f"epoch {epoch}/{epochs}: {score.grade}"
                    )

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "runs.csv", index=False)
    (run_dir / "runs.json").write_text(json.dumps(rows, indent=2))
    _summary(rows).to_csv(run_dir / "summary.csv", index=False)
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
