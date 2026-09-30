"""End-to-end orchestration of the semantic harmonization pipeline.

Usage:
    python main.py --phenotype myocarditis --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path

from dotenv import load_dotenv

from src.data.code_systems import load_code_system_corpus
from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import preprocess_corpus
from src.evaluation.metrics import (
    evaluate,
    filter_gold_by_available_codes,
    filter_gold_by_vocabulary,
    filter_records_by_vocabulary,
    to_predicted_codelist,
)
from src.llm.classify import llm_classify
from src.llm.client import resolve_provider
from src.llm.rank import skip_rank
from src.retrieval.embeddings import EmbeddingIndex
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
) -> dict:
    """Run the full semantic harmonization pipeline for a single phenotype.

    Stages: load EDF -> load full reference code system(s) -> hybrid
    retrieval -> LLM ranking -> LLM classification -> uncertainty-based
    review selection -> evaluation against the expert-curated ground truth.

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
    report("Step 1/8: Loading EDF")
    logger.info("Loading EDF for phenotype '%s'", phenotype)
    edf = load_edf(phenotype, edf_dir=config.paths.edf_dir)
    query = edf.to_prompt_context()

    report("Step 2/8: Loading query embedding model")
    from sentence_transformers import SentenceTransformer

    query_model = SentenceTransformer(config.retrieval.embedding_model)

    # Step 2: Load the full reference code system(s) to retrieve from (e.g.
    # the complete ICD-10-CM code list), not just phenotype-specific
    # candidates. Each vocabulary's processed corpus is cached to Parquet so
    # the (slow) source file is only parsed once across runs.
    report("Step 3/8: Loading terminology corpora")
    logger.info("Loading full code system corpus for vocabularies: %s", config.retrieval.vocabularies)
    raw_codes = load_code_system_corpus(
        config.retrieval.vocabularies,
        source_paths=config.paths.code_systems,
        cache_dir=config.paths.codes_parquet_dir,
    )
    codes = preprocess_corpus(raw_codes)
    logger.info("Loaded %d codes across %s", len(codes), config.retrieval.vocabularies)

    # Step 3: Hybrid retrieval — rank every code in the full corpus against
    # the EDF text using a weighted combination of BM25 lexical matching and
    # sentence-embedding semantic similarity, keeping the top-k candidates.
    # The embedding index is cached to disk (per vocabulary set + model) so
    # the (potentially large) corpus is only embedded once, not every run.
    report("Step 4/8: Loading or building embedding indexes", 0, len(config.retrieval.vocabularies))
    embedding_progress = None
    if progress_callback is not None:
        embedding_progress = lambda completed, total: report(
            "Step 4/8: Loading or building embedding indexes",
            completed,
            total,
        )
    embedding_index = EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=config.retrieval.vocabularies,
        cache_dir=config.paths.embeddings_dir,
        model_name=config.retrieval.embedding_model,
        model=query_model,
        progress_callback=embedding_progress,
    )

    report("Step 5/8: Running hybrid retrieval")
    logger.info("Running hybrid retrieval over %d candidate codes", len(codes))
    candidates = hybrid_retrieval(
        query,
        codes,
        lexical_weight=config.retrieval.lexical_weight,
        embedding_weight=config.retrieval.embedding_weight,
        embedding_model=config.retrieval.embedding_model,
        top_k=config.retrieval.top_k,
        per_vocabulary_top_k=config.retrieval.per_vocabulary_top_k,
        embedding_index=embedding_index,
        query_model=query_model,
        lexical_query=edf.to_lexical_query(),
        embedding_query=edf.to_embedding_query(),
    )
    logger.info("Retrieved %d candidates", len(candidates))

    # Resolve which LLM provider/model to use (explicit config, or auto-detect
    # from whichever API key is set in the environment).
    provider = resolve_provider(config.llm.provider)
    model = getattr(config.llm, f"{provider}_model")

    # Step 4: Skip LLM ranking (disabled — see src/llm/rank.py::llm_rank, kept
    # for potential future use but not currently run) and pass retrieval
    # candidates straight through to classification.
    ranked = skip_rank(candidates)

    # Step 5: LLM classification — assign each candidate a Narrow/Exclude
    # label (optionally Possible), confidence score, and short explanation.
    report("Step 6/8: LLM classification", 0, len(ranked))
    logger.info("Classifying candidates with LLM (provider=%s, model=%s)", provider, model)
    classified = llm_classify(
        ranked,
        edf,
        provider=provider,
        model=model,
        max_retries=config.llm.max_retries,
        batch_size=20 if provider == "jev" else 10,
        use_possible_category=config.llm.use_possible_category,
        progress_callback=lambda completed, total: report(
            "Step 6/8: LLM classification",
            completed,
            total,
        ),
    )

    # Step 6: Add review annotations without filtering or relabeling results.
    report("Step 7/8: Adding manual-review flags")
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

    # Step 7: Evaluation — render the predicted codelist in the same schema as
    # the ground-truth AESI export, then score it against that ground truth
    # (retrieval sensitivity/precision/F1 and classification accuracy/Cohen's
    # kappa), restricted to `evaluation.compare_vocabularies` (ICD10CM only
    # for now, since that's the only full reference list currently loaded).
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
            }
        )
    predicted_codelist = to_predicted_codelist(classified_dicts)

    report("Step 8/8: Evaluating against available ground truth")
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
        metrics["evaluation_scope"] = {
            "selected_vocabularies": compare_vocab,
            "gold_rows_in_selected_vocabularies": len(gold_vocabulary_scoped),
            "gold_rows_available_in_loaded_sources": len(gold_scoped),
            "gold_rows_excluded_as_unavailable": len(gold_vocabulary_scoped) - len(gold_scoped),
        }
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
        "metrics": metrics,
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
        "metrics": results["metrics"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Metrics written to %s", metrics_path)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
