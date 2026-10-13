"""End-to-end orchestration of the semantic harmonization pipeline.

Usage:
    python main.py --phenotype myocarditis --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path

from dotenv import load_dotenv

from src.data.code_systems import load_code_system_corpus
from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import preprocess_corpus
from src.evaluation.metrics import (
    evaluate,
    evaluate_gpt_review,
    evaluate_retrieval,
    filter_gold_by_available_codes,
    filter_gold_by_vocabulary,
    filter_records_by_vocabulary,
    narrow_loss_breakdown,
    to_predicted_codelist,
)
from src.llm.classify import (
    AdaptiveStoppingConfig,
    ClassificationRunInfo,
    GPTReviewRunInfo,
    gpt_review_low_confidence,
    llm_classify,
)
from src.llm.client import resolve_provider
from src.mapping.codemapper import expand_narrow_matches
from src.retrieval.embeddings import EmbeddingIndex, create_embedding_model
from src.retrieval.hybrid import hybrid_retrieval
from src.uncertainty.selection import review_rate, select_uncertain
from src.utils.config import PipelineConfig, load_config
from src.utils.logging import get_logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

logger = get_logger(__name__)



ProgressCallback = Callable[[str, int | None, int | None], None]


def run_pipeline(
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
    stage_callback: Callable[[str, list[dict]], None] | None = None,
    resource_cache: dict | None = None,
) -> dict:
    """Run the full semantic harmonization pipeline for a single phenotype.

    Stages: load EDF -> load full reference code system(s) -> hybrid
    retrieval -> Jev classification -> GPT review of low-confidence decisions
    -> optional direct CodeMapper expansion -> review selection -> evaluation
    against the expert-curated ground truth.

    Parameters
    ----------
    config : PipelineConfig
        Fully resolved pipeline configuration.

    Returns
    -------
    dict
        Summary containing the predicted codelist (DataFrame), review items,
        and evaluation metrics.
    """
    phenotype = config.phenotype

    def report(message: str, completed: int | None = None, total: int | None = None) -> None:
        if progress_callback is not None:
            progress_callback(message, completed, total)

    # Step 1: Load the structured phenotype definition (EDF) — the actual
    # input to the system. It encodes the clinical definition, inclusion/
    # exclusion criteria, decision rules, synonyms, etc.
    report("Step 1/9: Loading EDF")
    logger.info("Loading EDF for phenotype '%s'", phenotype)
    edf = load_edf(phenotype, edf_dir=config.paths.edf_dir)
    query = edf.to_prompt_context()
    resource_key = json.dumps({"retrieval": asdict(config.retrieval),
                               "sources": config.paths.code_systems,
                               "embeddings_dir": config.paths.embeddings_dir}, sort_keys=True)
    prepared = (resource_cache or {}).get(resource_key)

    report("Step 2/9: Loading query embedding model")
    query_model = prepared[0] if prepared else create_embedding_model(
        config.retrieval.embedding_provider,
        config.retrieval.embedding_model,
        dimensions=config.retrieval.embedding_dimensions,
        batch_size=config.retrieval.embedding_batch_size,
    )

    # Step 2: Load the full reference code system(s) to retrieve from (e.g.
    # the complete ICD-10-CM code list), not just phenotype-specific
    # candidates. Each vocabulary's processed corpus is cached to Parquet so
    # the (slow) source file is only parsed once across runs.
    report("Step 3/9: Loading terminology corpora")
    logger.info("Loading full code system corpus for vocabularies: %s", config.retrieval.vocabularies)
    raw_codes = prepared[1] if prepared else load_code_system_corpus(
        config.retrieval.vocabularies,
        source_paths=config.paths.code_systems,
        cache_dir=config.paths.codes_parquet_dir,
    )
    codes = raw_codes if prepared else preprocess_corpus(raw_codes)
    logger.info("Loaded %d codes across %s", len(codes), config.retrieval.vocabularies)

    # Step 3: Hybrid retrieval — rank every code in the full corpus against
    # the EDF text using a weighted combination of BM25 lexical matching and
    # sentence-embedding semantic similarity. By default all candidates are
    # ranked; top-k remains available as an optional hard compatibility cap.
    # The embedding index is cached to disk (per vocabulary set + model) so
    # the (potentially large) corpus is only embedded once, not every run.
    report("Step 4/9: Loading or building embedding indexes", 0, len(config.retrieval.vocabularies))
    embedding_progress = None
    if progress_callback is not None:
        embedding_progress = lambda completed, total: report(
            "Step 4/9: Loading or building embedding indexes",
            completed,
            total,
        )
    embedding_index = prepared[2] if prepared else EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=config.retrieval.vocabularies,
        cache_dir=config.paths.embeddings_dir,
        model_name=config.retrieval.embedding_model,
        model=query_model,
        provider=config.retrieval.embedding_provider,
        dimensions=config.retrieval.embedding_dimensions,
        progress_callback=embedding_progress,
    )
    if resource_cache is not None and not prepared:
        resource_cache[resource_key] = (query_model, codes, embedding_index)

    report("Step 5/9: Running hybrid retrieval")
    logger.info("Running hybrid retrieval over %d candidate codes", len(codes))
    retrieval_limit = config.retrieval.top_k if config.retrieval.use_top_k_limit else None
    candidates = hybrid_retrieval(
        query,
        codes,
        lexical_weight=config.retrieval.lexical_weight,
        embedding_weight=config.retrieval.embedding_weight,
        embedding_model=config.retrieval.embedding_model,
        top_k=retrieval_limit,
        per_vocabulary_top_k=config.retrieval.per_vocabulary_top_k,
        embedding_index=embedding_index,
        query_model=query_model,
        lexical_query=edf.to_lexical_query(),
        embedding_query=edf.to_embedding_query(),
    )
    logger.info(
        "Scored and ranked %d candidates%s",
        len(candidates),
        f" (top-k cap: {retrieval_limit})" if retrieval_limit is not None else "",
    )

    # Resolve which LLM provider/model to use (explicit config, or auto-detect
    # from whichever API key is set in the environment).
    provider = resolve_provider(config.llm.provider)
    model = getattr(config.llm, f"{provider}_model")

    # Step 4: LLM ranking is disabled; preserve the hybrid retrieval order and
    # pass candidates directly to classification.
    # Step 5: LLM classification — assign each candidate a Narrow/Exclude
    # label (optionally Possible), confidence score, and short explanation.
    # Adaptive stopping ends the scan once Narrow results stay sparse.
    report("Step 6/9: Stage-2 classification", 0, len(candidates))
    logger.info("Classifying candidates with LLM (provider=%s, model=%s)", provider, model)
    stopping_config = None
    if config.llm.adaptive_stopping_enabled:
        stopping_config = AdaptiveStoppingConfig(
            sparse_narrow_threshold=config.llm.sparse_narrow_threshold,
            consecutive_sparse_batches=config.llm.consecutive_sparse_batches,
            minimum_batches=config.llm.minimum_batches,
        )
    classification_run_info: ClassificationRunInfo | None = None

    def capture_run_info(info: ClassificationRunInfo) -> None:
        nonlocal classification_run_info
        classification_run_info = info

    classified = llm_classify(
        candidates,
        edf,
        provider=provider,
        model=model,
        max_retries=config.llm.max_retries,
        batch_size=20 if provider in {"jev", "openai_decisions", "laya", "jpt"} else 10,
        use_possible_category=config.llm.use_possible_category,
        adaptive_stopping=stopping_config,
        run_info_callback=capture_run_info,
        progress_callback=lambda completed, total: report(
            "Step 6/9: Stage-2 classification",
            completed,
            total,
        ),
        reasoning_effort=config.llm.classify_effort,
    )
    if classification_run_info is not None:
        logger.info(
            "Classification processed %d/%d candidates in %d batches. %s",
            classification_run_info.candidates_classified,
            classification_run_info.candidates_available,
            classification_run_info.batches_completed,
            classification_run_info.stop_reason,
        )
        if classification_run_info.stopped_early:
            report(
                "Step 6/9: Adaptive classification complete",
                classification_run_info.candidates_classified,
                classification_run_info.candidates_classified,
            )

    stage2_classified = list(classified)
    if stage_callback:
        stage_callback("jev", [asdict(c) for c in stage2_classified])

    # Step 7: Use GPT only for low-confidence stage-2 classifications.
    gpt_review_run_info: GPTReviewRunInfo | None = None
    if config.uncertainty.gpt_review_enabled:
        report("Step 7/9: GPT review of low-confidence decisions")
        classified, gpt_review_run_info = gpt_review_low_confidence(
            classified,
            edf,
            confidence_threshold=config.uncertainty.confidence_threshold,
            max_candidates=config.uncertainty.gpt_review_max_candidates,
            model=config.uncertainty.review_model or getattr(config.llm, f"{config.uncertainty.review_provider}_model"),
            provider=config.uncertainty.review_provider,
            max_retries=config.llm.max_retries,
            batch_size=config.uncertainty.gpt_review_batch_size,
            use_possible_category=config.llm.use_possible_category,
            progress_callback=lambda completed, total: report(
                "Step 7/9: GPT review of low-confidence decisions",
                completed,
                total,
            ),
            reasoning_effort=(
                "medium" if config.uncertainty.gpt_review_reasoning_enabled else "none"
            ),
        )
        logger.info(
            "GPT reviewed %d candidates and changed %d labels",
            gpt_review_run_info.candidates_reviewed,
            gpt_review_run_info.labels_changed,
        )
    else:
        report("Step 7/9: GPT review disabled")

    gpt_classified = list(classified)
    if stage_callback:
        stage_callback("gpt", [asdict(c) for c in gpt_classified])
    codemapper_run_info = {"enabled": False}
    if config.codemapper.enabled:
        report("After GPT review: expanding Narrow codes through CodeMapper (no descendants)")
        classified, codemapper_run_info = expand_narrow_matches(
            classified, codes, config.retrieval.vocabularies, config.codemapper,
        )
        logger.info("CodeMapper added %d unseen codes and promoted %d existing decisions",
                    len(codemapper_run_info["new_codes"]), len(codemapper_run_info["promoted_codes"]))
    if stage_callback:
        stage_callback("codemapper", [asdict(c) for c in classified])

    # Step 8: Add review annotations for any uncertainty that remains.
    report("Step 8/9: Adding manual-review flags")
    review_items = select_uncertain(
        classified,
        confidence_threshold=config.uncertainty.confidence_threshold,
        possible_requires_review=config.uncertainty.possible_requires_review,
    )
    logger.info(
        "%d/%d candidates flagged for human review (%.1f%%)",
        len(review_items),
        len(classified),
        100 * review_rate(classified, review_items),
    )

    # Step 9: Evaluation — render the predicted codelist in the same schema as
    # the ground-truth AESI export, then score it against that ground truth
    # (retrieval sensitivity/precision/F1 and classification accuracy/Cohen's
    # kappa), restricted to `evaluation.compare_vocabularies` and available codes.
    # Skipped entirely if no ground truth is registered for this phenotype.
    review_by_key = {(r.code, r.vocabulary): r.reason for r in review_items}
    classified_dicts = []
    for candidate in classified:
        review_reason = review_by_key.get((candidate.code, candidate.vocabulary), "")
        classified_dicts.append(
            {
                "code": candidate.code,
                "vocabulary": candidate.vocabulary,
                "description": candidate.description,
                "label": candidate.label,
                "confidence": candidate.confidence,
                "explanation": candidate.explanation,
                "manual_review": bool(review_reason),
                "review_reason": review_reason,
                "decision_origin": candidate.decision_origin,
                "mapping_provenance": candidate.mapping_provenance,
                "prior_label": candidate.prior_label,
                "prior_confidence": candidate.prior_confidence,
            }
        )
    stage2_dicts = [
        {
            "code": candidate.code,
            "vocabulary": candidate.vocabulary,
            "description": candidate.description,
            "label": candidate.label,
            "confidence": candidate.confidence,
            "explanation": candidate.explanation,
        }
        for candidate in stage2_classified
    ]
    predicted_codelist = to_predicted_codelist(classified_dicts)
    stage_decisions = {
        "jev": [asdict(c) for c in stage2_classified],
        "gpt": [asdict(c) for c in gpt_classified],
        "codemapper": [asdict(c) for c in classified],
    }
    stage_metrics = {}

    report("Step 9/9: Evaluating against available ground truth")
    metrics: dict | None = None
    aesi_path = config.paths.aesi_datasets.get(phenotype)
    if aesi_path:
        logger.info("Loading ground-truth codelist for '%s' from %s", phenotype, aesi_path)
        _, gold_labels = load_aesi_dataset(aesi_path)
        available_vocabularies = set(codes["vocabulary"].astype(str))
        compare_vocab = [
            vocabulary
            for vocabulary in config.evaluation.compare_vocabularies
            if vocabulary in available_vocabularies
        ]
        gold_vocabulary_scoped = filter_gold_by_vocabulary(gold_labels, compare_vocab)
        gold_scoped = filter_gold_by_available_codes(gold_vocabulary_scoped, codes)
        predicted_scoped = filter_records_by_vocabulary(classified_dicts, compare_vocab)
        metrics = evaluate(
            predicted_scoped,
            gold_scoped,
            narrow_only_as_positive=config.evaluation.narrow_only_as_positive,
            include_possible=config.llm.use_possible_category,
        )
        metrics["narrow_loss_breakdown"] = asdict(
            narrow_loss_breakdown(predicted_scoped, gold_scoped)
        )
        if gpt_review_run_info is not None:
            stage2_scoped = filter_records_by_vocabulary(stage2_dicts, compare_vocab)
            metrics["stage2_before_gpt"] = evaluate(
                stage2_scoped,
                gold_scoped,
                narrow_only_as_positive=config.evaluation.narrow_only_as_positive,
                include_possible=config.llm.use_possible_category,
            )["classification"]
            metrics["gpt_review_evaluation"] = asdict(
                evaluate_gpt_review(
                    stage2_scoped,
                    filter_records_by_vocabulary(stage_decisions["gpt"], compare_vocab),
                    gold_scoped,
                    confidence_threshold=gpt_review_run_info.confidence_threshold,
                    max_candidates=gpt_review_run_info.max_candidates,
                    include_possible=config.llm.use_possible_category,
                )
            )
        metrics["evaluation_scope"] = {
            "selected_vocabularies": compare_vocab,
            "gold_rows_in_selected_vocabularies": len(gold_vocabulary_scoped),
            "gold_rows_available_in_loaded_sources": len(gold_scoped),
            "gold_rows_excluded_as_unavailable": len(gold_vocabulary_scoped) - len(gold_scoped),
            "gold_narrow_in_selected_vocabularies": int((gold_vocabulary_scoped["label"] == "Narrow").sum()),
            "gold_narrow_available": int((gold_scoped["label"] == "Narrow").sum()),
            "gold_narrow_excluded_as_unavailable": int((gold_vocabulary_scoped["label"] == "Narrow").sum()
                                                      - (gold_scoped["label"] == "Narrow").sum()),
        }
        for stage, decisions in stage_decisions.items():
            scoped = filter_records_by_vocabulary(decisions, compare_vocab)
            stage_metrics[stage] = evaluate(
                scoped, gold_scoped,
                narrow_only_as_positive=config.evaluation.narrow_only_as_positive,
                include_possible=config.llm.use_possible_category,
            )
            stage_metrics[stage]["narrow_codelist"] = asdict(evaluate_retrieval(
                [c for c in scoped if c["label"] == "Narrow"], gold_scoped,
                narrow_only_as_positive=True,
            ))
            stage_metrics[stage]["narrow_loss_breakdown"] = asdict(narrow_loss_breakdown(scoped, gold_scoped))
        metrics["stages"] = stage_metrics
        logger.info("Evaluation metrics (vocabularies=%s): %s", compare_vocab, metrics)
    else:
        logger.warning("No ground-truth codelist registered for phenotype '%s'; skipping evaluation", phenotype)

    return {
        "phenotype": phenotype,
        "predicted_codelist": predicted_codelist,
        "review_items": [
            {
                "code": r.code,
                "vocabulary": r.vocabulary,
                "label": r.label,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r in review_items
        ],
        "review_rate": round(review_rate(classified, review_items), 4),
        "classification_run": (
            asdict(classification_run_info) if classification_run_info is not None else None
        ),
        "gpt_review_run": (
            asdict(gpt_review_run_info) if gpt_review_run_info is not None else None
        ),
        "metrics": metrics,
        "stage_decisions": stage_decisions,
        "codemapper_run": codemapper_run_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the semantic harmonization pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--phenotype", default=None, help="Override phenotype name from config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the predicted codelist CSV and metrics JSON (defaults to paths.results_dir).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.phenotype:
        config.phenotype = args.phenotype

    results = run_pipeline(config)

    # Write two outputs: a predicted codelist CSV in the same schema as the
    # ground-truth AESI export, and a separate metrics JSON.
    output_dir = Path(args.output_dir or config.paths.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codelist_path = output_dir / f"{config.phenotype}_predicted.csv"
    results["predicted_codelist"].to_csv(codelist_path, index=False)
    logger.info("Predicted codelist written to %s", codelist_path)

    metrics_path = output_dir / f"{config.phenotype}_metrics.json"
    metrics_payload = {
        "phenotype": results["phenotype"],
        "review_items": results["review_items"],
        "review_rate": results["review_rate"],
        "classification_run": results["classification_run"],
        "gpt_review_run": results["gpt_review_run"],
        "codemapper_run": results["codemapper_run"],
        "metrics": results["metrics"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Metrics written to %s", metrics_path)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
