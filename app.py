"""Streamlit GUI for the semantic harmonization pipeline.

Run with:
    streamlit run app.py

The page walks through the pipeline step by step:
1. Select which clinical coding system(s) to retrieve from.
2. Provide the phenotype's Event Definition Form (EDF).
3. Run hybrid (lexical + embedding) retrieval over the full code system(s).
4. Run LLM classification (Narrow/Exclude, optionally Possible) on candidates.
5. Optionally score the result against a ground-truth AESI codelist.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.data.code_systems import load_code_system_corpus
from src.data.loaders import EventDefinitionForm, load_aesi_dataset
from src.data.preprocessing import preprocess_corpus, standardize_code
from src.evaluation import metrics as evaluation_metrics
from src.evaluation.metrics import (
    evaluate,
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
from src.llm.client import default_model_for, resolve_provider
from src.retrieval.embeddings import EmbeddingIndex
from src.retrieval.hybrid import hybrid_retrieval
from src.uncertainty.selection import review_rate, select_uncertain
from src.utils.config import load_config

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Semantic Harmonization", layout="wide")

CONFIG = load_config("configs/default.yaml")
EDF_DIR = Path(CONFIG.paths.edf_dir)

# Coding systems currently available in the local corpus.
AVAILABLE_VOCABULARIES = {
    "ICD10CM": "ICD-10-CM",
    "ICD9CM": "ICD-9-CM",
    "ICPC": "ICPC",
    "MDR": "MedDRA",
    "RCD2": "RCD2",
    "SNOMEDCT_US": "SNOMED CT (US)",
}


def _filter_gold_by_available_codes(
    gold_labels: pd.DataFrame,
    available_codes: pd.DataFrame,
) -> pd.DataFrame:
    """Use the current metrics helper, with support for rolling app reloads.

    Streamlit can briefly keep the previous revision of an imported module in
    memory while re-running a newly deployed ``app.py``. The fallback keeps
    startup and evaluation working when that older module predates the helper.
    """
    filter_fn = getattr(evaluation_metrics, "filter_gold_by_available_codes", None)
    if filter_fn is not None:
        return filter_fn(gold_labels, available_codes)

    required = {"code", "vocabulary"}
    missing = required - set(available_codes.columns)
    if missing:
        raise ValueError(f"Available code corpus missing required columns: {missing}")

    def standardized_key(code: str, vocabulary: str) -> tuple[str, str]:
        return standardize_code(code, vocabulary), (vocabulary or "").strip().upper()

    available_keys = {
        standardized_key(row["code"], row["vocabulary"])
        for _, row in available_codes.iterrows()
    }
    mask = gold_labels.apply(
        lambda row: standardized_key(row["code"], row["vocabulary"]) in available_keys,
        axis=1,
    )
    return gold_labels[mask].reset_index(drop=True)


def _init_state() -> None:
    defaults = {
        "edf_yaml_text": "",
        "phenotype_name": "",
        "retrieval_candidates": None,
        "classified": None,
        "stage2_classified": None,
        "classification_run": None,
        "gpt_review_run": None,
        "metrics_result": None,
        "predicted_codelist": None,
        "classification_use_possible": CONFIG.llm.use_possible_category,
        "retrieval_vocabularies": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()

st.title("Semantic Harmonization Pipeline")
st.caption(
    "AI-assisted phenotype codelist generation: EDF \u2192 hybrid retrieval \u2192 "
    "LLM classification \u2192 evaluation against ground truth."
)

# ---------------------------------------------------------------------------
# Block 1: Coding system selection
# ---------------------------------------------------------------------------
st.header("1. Select coding system(s)")
st.caption("Choose which full reference code systems to retrieve candidates from.")

vocab_cols = st.columns(len(AVAILABLE_VOCABULARIES))
selected_vocabularies: list[str] = []

for col, (vocab, label) in zip(vocab_cols, AVAILABLE_VOCABULARIES.items()):
    with col:
        if st.checkbox(label, value=vocab in CONFIG.retrieval.vocabularies, key=f"vocab_{vocab}"):
            selected_vocabularies.append(vocab)

if not selected_vocabularies:
    st.warning("Select at least one coding system to continue.")

st.divider()

# ---------------------------------------------------------------------------
# Block 2: EDF input
# ---------------------------------------------------------------------------
st.header("2. Provide the phenotype's Event Definition Form (EDF)")

example_files = sorted(EDF_DIR.glob("*.yaml"))
example_names = [f.stem for f in example_files]

edf_col1, edf_col2 = st.columns([1, 1])

with edf_col1:
    selected_example = st.selectbox(
        "Load an example EDF",
        options=["(none)"] + example_names,
        key="example_select",
    )
    if st.button("Load selected example"):
        if selected_example != "(none)":
            example_path = EDF_DIR / f"{selected_example}.yaml"
            st.session_state.edf_yaml_text = example_path.read_text()
            st.session_state.phenotype_name = selected_example
            st.rerun()

with edf_col2:
    uploaded_file = st.file_uploader("...or upload an EDF YAML file", type=["yaml", "yml"])
    if uploaded_file is not None and st.button("Load uploaded file"):
        st.session_state.edf_yaml_text = uploaded_file.getvalue().decode("utf-8")
        st.session_state.phenotype_name = Path(uploaded_file.name).stem
        st.rerun()

st.session_state.phenotype_name = st.text_input(
    "Phenotype name",
    value=st.session_state.phenotype_name,
    help="Used to label outputs and to auto-match a registered ground-truth AESI file in block 5.",
)

st.session_state.edf_yaml_text = st.text_area(
    "EDF YAML (paste, edit, or load from the options above)",
    value=st.session_state.edf_yaml_text,
    height=280,
)


def _parse_edf(name: str, yaml_text: str) -> EventDefinitionForm | None:
    if not yaml_text.strip():
        return None
    try:
        raw = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        st.error(f"Could not parse EDF YAML: {exc}")
        return None
    try:
        return EventDefinitionForm(name=name or "phenotype", **raw)
    except TypeError as exc:
        st.error(f"EDF YAML is missing required fields: {exc}")
        return None


edf = _parse_edf(st.session_state.phenotype_name, st.session_state.edf_yaml_text)
if edf is not None:
    with st.expander("Preview EDF prompt context"):
        st.text(edf.to_prompt_context())

st.divider()

# ---------------------------------------------------------------------------
# Block 3: Hybrid retrieval
# ---------------------------------------------------------------------------
st.header("3. Run hybrid retrieval")

top_k = st.number_input(
    "Candidates to preview (top-k)",
    min_value=1,
    max_value=500,
    value=CONFIG.retrieval.top_k,
    help="Only this many similarity-ranked candidates are shown on screen. Adaptive classification can still process candidates beyond this preview.",
)
use_top_k_limit = st.checkbox(
    "Use top-k as a hard classification limit",
    value=CONFIG.retrieval.use_top_k_limit,
    help="Optional compatibility mode. When disabled, every code receives a similarity score and adaptive classification decides when to stop.",
)
per_vocabulary_top_k = st.checkbox(
    "Retrieve top-K per code list",
    value=getattr(CONFIG.retrieval, "per_vocabulary_top_k", True),
    disabled=not use_top_k_limit,
    help="If enabled, retrieves the top-K candidates within each selected vocabulary. If disabled, retrieves the top-K overall after combining vocabularies.",
)


@st.cache_resource(show_spinner="Loading code system corpus...")
def _load_corpus(vocabularies: tuple[str, ...]) -> pd.DataFrame:
    raw_codes = load_code_system_corpus(
        list(vocabularies),
        source_paths=CONFIG.paths.code_systems,
        cache_dir=CONFIG.paths.codes_parquet_dir,
    )
    return preprocess_corpus(raw_codes)


@st.cache_resource(show_spinner="Building/loading embedding index (only happens once per corpus)...")
def _load_embedding_index(vocabularies: tuple[str, ...]) -> EmbeddingIndex:
    codes = _load_corpus(vocabularies)
    return EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=vocabularies,
        cache_dir=CONFIG.paths.embeddings_dir,
        model_name=CONFIG.retrieval.embedding_model,
    )


@st.cache_resource(show_spinner="Loading query embedding model...")
def _load_query_model() -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(CONFIG.retrieval.embedding_model)


run_retrieval = st.button("Run hybrid retrieval", disabled=(edf is None or not selected_vocabularies))

if run_retrieval and edf is not None and selected_vocabularies:
    vocab_tuple = tuple(sorted(selected_vocabularies))
    codes = _load_corpus(vocab_tuple)
    embedding_index = _load_embedding_index(vocab_tuple)
    query_model = _load_query_model()
    retrieval_limit = int(top_k) if use_top_k_limit else None
    spinner_text = (
        f"Retrieving top {top_k} candidates from {len(codes)} codes..."
        if retrieval_limit is not None
        else f"Scoring and ranking all {len(codes)} codes..."
    )
    with st.spinner(spinner_text):
        candidates = hybrid_retrieval(
            edf.to_prompt_context(),
            codes,
            lexical_weight=CONFIG.retrieval.lexical_weight,
            embedding_weight=CONFIG.retrieval.embedding_weight,
            embedding_model=CONFIG.retrieval.embedding_model,
            top_k=retrieval_limit,
            per_vocabulary_top_k=bool(per_vocabulary_top_k),
            embedding_index=embedding_index,
            query_model=query_model,
            lexical_query=edf.to_lexical_query(),
            embedding_query=edf.to_embedding_query(),
        )
    st.session_state.retrieval_candidates = candidates
    st.session_state.retrieval_vocabularies = vocab_tuple
    st.session_state.classified = None
    st.session_state.stage2_classified = None
    st.session_state.classification_run = None
    st.session_state.gpt_review_run = None
    st.session_state.metrics_result = None

if st.session_state.retrieval_candidates:
    preview_candidates = st.session_state.retrieval_candidates[: int(top_k)]
    st.success(f"Scored and ranked {len(st.session_state.retrieval_candidates)} candidates.")
    st.caption(
        f"Showing only the top {len(preview_candidates)} similarity-ranked candidates. "
        "Similarity is not a final Narrow/Exclude decision."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rank": c.rank,
                    "code": c.code,
                    "vocabulary": c.vocabulary,
                    "description": c.description,
                    "normalized_lexical_score": c.lexical_score,
                    "normalized_embedding_score": c.embedding_score,
                    "score": c.score,
                }
                for c in preview_candidates
            ]
        ),
        width='stretch',
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Block 4: LLM classification
# ---------------------------------------------------------------------------
st.header("4. Classify candidates with an LLM")
st.caption(
    "Note: the LLM re-ranking step is intentionally skipped here \u2014 candidates go "
    "directly from retrieval to classification."
)

use_possible_category = st.checkbox(
    "Enable the 'Possible' category",
    value=CONFIG.llm.use_possible_category,
    help="Off by default. When disabled, the model classifies codes as Narrow or Exclude.",
)

adaptive_stopping_enabled = st.checkbox(
    "Stop when Narrow results become sparse",
    value=CONFIG.llm.adaptive_stopping_enabled,
    help="Candidates are classified in similarity order. Classification stops after the configured sparse-batch streak.",
)
sparsity_col1, sparsity_col2, sparsity_col3 = st.columns(3)
with sparsity_col1:
    sparse_narrow_threshold = st.number_input(
        "Maximum Narrow results in a sparse batch",
        min_value=0,
        value=CONFIG.llm.sparse_narrow_threshold,
        disabled=not adaptive_stopping_enabled,
    )
with sparsity_col2:
    consecutive_sparse_batches = st.number_input(
        "Consecutive sparse batches",
        min_value=1,
        value=CONFIG.llm.consecutive_sparse_batches,
        disabled=not adaptive_stopping_enabled,
    )
with sparsity_col3:
    minimum_batches = st.number_input(
        "Minimum batches before stopping",
        min_value=1,
        value=CONFIG.llm.minimum_batches,
        disabled=not adaptive_stopping_enabled,
    )

llm_col1, llm_col2 = st.columns(2)
with llm_col1:
    provider_options = ["auto", "anthropic", "openai", "jev", "google"]
    configured_provider = CONFIG.llm.provider if CONFIG.llm.provider in provider_options else "auto"
    provider_choice = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(configured_provider),
    )
with llm_col2:
    try:
        resolved_provider_preview = resolve_provider(provider_choice)
        default_model = default_model_for(resolved_provider_preview)
    except Exception:
        resolved_provider_preview = None
        default_model = ""
    model_override = st.text_input(
        "Model",
        value=default_model,
        help="Defaults to this provider's configured model; edit to override.",
    )

run_classification = st.button(
    "Run LLM classification",
    disabled=not st.session_state.retrieval_candidates or edf is None,
)

if run_classification and st.session_state.retrieval_candidates and edf is not None:
    try:
        progress = st.progress(0.0, text="Classifying candidates...")

        def _update_progress(completed: int, total: int) -> None:
            progress.progress(completed / total if total else 1.0, text=f"Classifying candidates... {completed}/{total}")

        classification_run_holder: list[ClassificationRunInfo] = []

        def _capture_run_info(info: ClassificationRunInfo) -> None:
            classification_run_holder.append(info)

        stopping_config = None
        if adaptive_stopping_enabled:
            stopping_config = AdaptiveStoppingConfig(
                sparse_narrow_threshold=int(sparse_narrow_threshold),
                consecutive_sparse_batches=int(consecutive_sparse_batches),
                minimum_batches=int(minimum_batches),
            )

        with st.spinner("Classifying candidates..."):
            classified = llm_classify(
                st.session_state.retrieval_candidates,
                edf,
                provider=provider_choice,
                model=model_override or None,
                max_retries=CONFIG.llm.max_retries,
                batch_size=20 if resolved_provider_preview == "jev" else 10,
                progress_callback=_update_progress,
                use_possible_category=use_possible_category,
                adaptive_stopping=stopping_config,
                run_info_callback=_capture_run_info,
            )
        progress.progress(1.0, text="Classification complete")
        st.session_state.classified = classified
        st.session_state.stage2_classified = classified
        st.session_state.classification_run = (
            classification_run_holder[0] if classification_run_holder else None
        )
        st.session_state.gpt_review_run = None
        st.session_state.classification_use_possible = use_possible_category
        st.session_state.metrics_result = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"LLM classification failed: {exc}")

if st.session_state.classified:
    classification_use_possible = st.session_state.classification_use_possible
    review_items = select_uncertain(
        st.session_state.classified,
        confidence_threshold=CONFIG.uncertainty.confidence_threshold,
        possible_requires_review=classification_use_possible and CONFIG.uncertainty.possible_requires_review,
    )
    review_by_key = {(r.code, r.vocabulary): r.reason for r in review_items}
    classified_df = pd.DataFrame(
        [
            {
                "code": c.code,
                "vocabulary": c.vocabulary,
                "description": c.description,
                "label": c.label,
                "confidence": c.confidence,
                "explanation": c.explanation,
                "manual_review": (c.code, c.vocabulary) in review_by_key,
                "review_reason": review_by_key.get((c.code, c.vocabulary), ""),
            }
            for c in st.session_state.classified
        ]
    )
    st.success(f"Classified {len(classified_df)} candidates.")
    if st.session_state.classification_run is not None:
        run_info = st.session_state.classification_run
        if run_info.stopped_early:
            st.info(
                f"Adaptive stopping classified {run_info.candidates_classified:,} of "
                f"{run_info.candidates_available:,} ranked candidates in "
                f"{run_info.batches_completed} batches. {run_info.stop_reason}"
            )
        else:
            st.caption(run_info.stop_reason)
    st.dataframe(classified_df, width='stretch', hide_index=True)

    if st.session_state.gpt_review_run is None:
        st.info(
            f"{len(review_items)}/{len(st.session_state.classified)} candidates are below the "
            f"confidence threshold ({100 * review_rate(st.session_state.classified, review_items):.1f}%) "
            "and are eligible for final GPT review."
        )
    else:
        st.info(
            f"{len(review_items)}/{len(st.session_state.classified)} candidates remain flagged "
            f"after GPT review ({100 * review_rate(st.session_state.classified, review_items):.1f}%)."
        )

    predicted_codelist = to_predicted_codelist(classified_df.to_dict("records"))
    st.session_state.predicted_codelist = predicted_codelist
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", st.session_state.phenotype_name.strip()) or "phenotype"
    results_dir = Path(CONFIG.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predicted_codelist.to_csv(results_dir / f"{safe_name}_predicted.csv", index=False)
    st.download_button(
        "Export predicted codelist CSV",
        data=predicted_codelist.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_name}_predicted.csv",
        mime="text/csv",
    )

st.divider()

# ---------------------------------------------------------------------------
# Block 5: GPT review of low-confidence stage-2 decisions
# ---------------------------------------------------------------------------
st.header("5. Review low-confidence decisions with GPT")
st.caption(
    "Only stage-2 results below the confidence threshold are sent to GPT. "
    "All other Jev classifications remain unchanged."
)

gpt_col1, gpt_col2 = st.columns(2)
with gpt_col1:
    gpt_confidence_threshold = st.number_input(
        "Stage-2 confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(CONFIG.uncertainty.confidence_threshold),
        step=0.05,
    )
with gpt_col2:
    gpt_model = st.text_input("GPT review model", value=CONFIG.llm.openai_model)

stage2_classified = st.session_state.stage2_classified or st.session_state.classified or []
gpt_review_candidates = [
    item for item in stage2_classified if item.confidence < gpt_confidence_threshold
]
st.caption(
    f"{len(gpt_review_candidates):,} of {len(stage2_classified):,} stage-2 classifications "
    "would be reviewed."
)

run_gpt_review = st.button(
    "Run final GPT review",
    disabled=not stage2_classified or edf is None or not gpt_review_candidates,
)
if run_gpt_review and edf is not None:
    try:
        progress = st.progress(0.0, text="Reviewing low-confidence decisions with GPT...")

        def _update_gpt_progress(completed: int, total: int) -> None:
            progress.progress(
                completed / total if total else 1.0,
                text=f"Reviewing low-confidence decisions with GPT... {completed}/{total}",
            )

        with st.spinner("Running final GPT review..."):
            reviewed, review_run = gpt_review_low_confidence(
                stage2_classified,
                edf,
                confidence_threshold=float(gpt_confidence_threshold),
                model=gpt_model or None,
                max_retries=CONFIG.llm.max_retries,
                batch_size=CONFIG.uncertainty.gpt_review_batch_size,
                progress_callback=_update_gpt_progress,
                use_possible_category=st.session_state.classification_use_possible,
            )
        progress.progress(1.0, text="GPT review complete")
        st.session_state.classified = reviewed
        st.session_state.gpt_review_run = review_run
        st.session_state.metrics_result = None
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"GPT review failed: {exc}")

if st.session_state.gpt_review_run is not None:
    review_run: GPTReviewRunInfo = st.session_state.gpt_review_run
    st.success(
        f"GPT reviewed {review_run.candidates_reviewed:,} low-confidence decisions and "
        f"changed {review_run.labels_changed:,} labels: "
        f"{review_run.exclude_to_narrow:,} Exclude→Narrow and "
        f"{review_run.narrow_to_exclude:,} Narrow→Exclude."
    )

st.divider()

# ---------------------------------------------------------------------------
# Block 6: Ground truth & metrics
# ---------------------------------------------------------------------------
st.header("6. Evaluate against ground truth")

registered_aesi_path = CONFIG.paths.aesi_datasets.get(st.session_state.phenotype_name)
gt_col1, gt_col2 = st.columns(2)
with gt_col1:
    if registered_aesi_path:
        st.caption(f"Registered ground truth found for '{st.session_state.phenotype_name}': {registered_aesi_path}")
    ground_truth_file = st.file_uploader("Upload a ground-truth AESI CSV (optional)", type=["csv"])
with gt_col2:
    compare_vocab_options = selected_vocabularies or list(AVAILABLE_VOCABULARIES)
    compare_vocabularies = st.multiselect(
        "Vocabularies to compare",
        options=compare_vocab_options,
        default=compare_vocab_options,
    )

run_metrics = st.button(
    "Compute metrics",
    disabled=not st.session_state.classified or (not registered_aesi_path and ground_truth_file is None),
)

if run_metrics:
    try:
        if ground_truth_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(ground_truth_file.getvalue())
                gt_path = tmp.name
        else:
            gt_path = registered_aesi_path

        _, gold_labels = load_aesi_dataset(gt_path)
        retrieval_vocabularies = st.session_state.retrieval_vocabularies or tuple(compare_vocabularies)
        effective_compare_vocabularies = [
            vocabulary for vocabulary in compare_vocabularies if vocabulary in retrieval_vocabularies
        ]
        classified_dicts = [
            {
                "code": c.code,
                "vocabulary": c.vocabulary,
                "description": c.description,
                "label": c.label,
                "confidence": c.confidence,
                "explanation": c.explanation,
            }
            for c in st.session_state.classified
        ]
        gold_vocabulary_scoped = filter_gold_by_vocabulary(gold_labels, effective_compare_vocabularies)
        available_codes = _load_corpus(tuple(retrieval_vocabularies))
        gold_scoped = _filter_gold_by_available_codes(gold_vocabulary_scoped, available_codes)
        predicted_scoped = filter_records_by_vocabulary(classified_dicts, effective_compare_vocabularies)
        st.session_state.metrics_result = evaluate(
            predicted_scoped,
            gold_scoped,
            narrow_only_as_positive=CONFIG.evaluation.narrow_only_as_positive,
            include_possible=st.session_state.classification_use_possible,
        )
        st.session_state.metrics_result["narrow_loss_breakdown"] = asdict(
            narrow_loss_breakdown(predicted_scoped, gold_scoped)
        )
        st.session_state.metrics_result["evaluation_scope"] = {
            "selected_vocabularies": effective_compare_vocabularies,
            "gold_rows_in_selected_vocabularies": len(gold_vocabulary_scoped),
            "gold_rows_available_in_loaded_sources": len(gold_scoped),
            "gold_rows_excluded_as_unavailable": len(gold_vocabulary_scoped) - len(gold_scoped),
        }
        if st.session_state.classification_run is not None:
            st.session_state.metrics_result["classification_run"] = asdict(
                st.session_state.classification_run
            )
        if st.session_state.gpt_review_run is not None:
            st.session_state.metrics_result["gpt_review_run"] = asdict(
                st.session_state.gpt_review_run
            )
        st.session_state.metrics_gold_scoped = gold_scoped
        st.session_state.metrics_predicted_scoped = predicted_scoped
        st.session_state.metrics_classified_dicts = classified_dicts
        st.session_state.metrics_compare_vocabularies = compare_vocabularies
    except Exception as exc:  # noqa: BLE001
        st.error(f"Evaluation failed: {exc}")

if st.session_state.metrics_result:
    metrics = st.session_state.metrics_result
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", st.session_state.phenotype_name.strip()) or "phenotype"
    results_dir = Path(CONFIG.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{safe_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    st.download_button(
        "Export metrics JSON",
        data=json.dumps(metrics, indent=2).encode("utf-8"),
        file_name=f"{safe_name}_metrics.json",
        mime="application/json",
    )
    retrieval_metrics = metrics["retrieval"]
    classification_metrics = metrics["classification"]

    scope = metrics.get("evaluation_scope", {})
    if scope:
        st.caption(
            f"Evaluation uses {scope['gold_rows_available_in_loaded_sources']} available gold rows; "
            f"{scope['gold_rows_excluded_as_unavailable']} rows absent from the loaded terminology "
            "versions were excluded."
        )

    st.subheader("Metrics after initial retrieval")
    retrieval_summary = pd.DataFrame(
        [
            {
                "Correctly Retrieved (TP)": retrieval_metrics["true_positives"],
                "Incorrectly Retrieved (FP)": retrieval_metrics["false_positives"],
                "Missed Retrievals (FN)": retrieval_metrics["false_negatives"],
            }
        ]
    )
    st.dataframe(retrieval_summary, use_container_width=True, hide_index=True)
    with st.expander("Correct retrievals"):
        gold_scoped = st.session_state.get("metrics_gold_scoped")
        predicted_scoped = st.session_state.get("metrics_predicted_scoped", [])
        if gold_scoped is not None:
            predicted_keys = {
                (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper())
                for row in predicted_scoped
            }
            positive_labels = {"Narrow", "Possible"} if st.session_state.classification_use_possible else {"Narrow"}
            correct_retrievals = gold_scoped[
                gold_scoped.apply(
                    lambda r: (
                        r["label"] in positive_labels
                        and (str(r["code"]).strip(), str(r["vocabulary"]).strip().upper()) in predicted_keys
                    ),
                    axis=1,
                )
            ]
            correct_cols = [col for col in ["vocabulary", "code", "code_name", "label"] if col in correct_retrievals.columns]
            st.dataframe(
                correct_retrievals[correct_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Run metrics to see correct retrievals.")

    with st.expander("Missed retrievals"):
        gold_scoped = st.session_state.get("metrics_gold_scoped")
        predicted_scoped = st.session_state.get("metrics_predicted_scoped", [])
        if gold_scoped is not None:
            predicted_keys = {
                (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper())
                for row in predicted_scoped
            }
            positive_labels = {"Narrow", "Possible"} if st.session_state.classification_use_possible else {"Narrow"}
            missed_retrievals = gold_scoped[
                gold_scoped.apply(
                    lambda r: (
                        r["label"] in positive_labels
                        and (str(r["code"]).strip(), str(r["vocabulary"]).strip().upper()) not in predicted_keys
                    ),
                    axis=1,
                )
            ]
            missed_cols = [col for col in ["vocabulary", "code", "code_name", "label"] if col in missed_retrievals.columns]
            st.dataframe(
                missed_retrievals[missed_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Run metrics to see missed retrievals.")

    st.subheader("Metrics after LLM classification")
    gold_scoped = st.session_state.get("metrics_gold_scoped")
    classified_dicts = st.session_state.get("metrics_classified_dicts", [])
    if gold_scoped is not None:
        losses = narrow_loss_breakdown(classified_dicts, gold_scoped)
        run_info = st.session_state.get("classification_run")
        stopped_early = bool(run_info and run_info.stopped_early)
        pre_classifier_label = (
            "Missed: early stopping" if stopped_early else "Missed before classifier"
        )
        st.markdown("#### Gold Narrow recovery")
        recovery_columns = st.columns(5)
        recovery_columns[0].metric("Gold Narrow available", losses.gold_narrow_available)
        recovery_columns[1].metric("Correctly recovered", losses.correctly_classified)
        recovery_columns[2].metric(pre_classifier_label, losses.not_classified)
        recovery_columns[3].metric("Missed: misclassified", losses.misclassified)
        recovery_columns[4].metric("End-to-end recall", f"{losses.end_to_end_recall:.1%}")
        st.caption(
            f"{losses.total_missed} gold Narrow codes were missed in total: "
            f"{losses.not_classified} did not reach classification and "
            f"{losses.misclassified} reached the model but were assigned another label."
        )

    metric_columns = st.columns(6)
    metric_columns[0].metric("Accuracy", f"{classification_metrics['accuracy']:.1%}")
    metric_columns[1].metric("Narrow TN", classification_metrics["true_negatives"])
    metric_columns[2].metric("Narrow sensitivity", f"{classification_metrics['sensitivity']:.1%}")
    metric_columns[3].metric("Narrow precision", f"{classification_metrics['precision']:.1%}")
    metric_columns[4].metric("Narrow F1", f"{classification_metrics['f1']:.1%}")
    metric_columns[5].metric("Cohen's kappa", f"{classification_metrics['cohens_kappa']:.3f}")
    st.caption(
        "Sensitivity, precision, and F1 treat Narrow as the positive class. "
        "Macro averages for all enabled labels are available in Full metrics."
    )
    display_labels = ["Narrow", "Possible", "Exclude"] if st.session_state.classification_use_possible else ["Narrow", "Exclude"]
    classification_summary = pd.DataFrame(0, index=display_labels, columns=display_labels)
    if gold_scoped is not None:
        gold_by_key = {
            (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper()): str(row["label"]).strip().title()
            for _, row in gold_scoped.iterrows()
        }
        for item in classified_dicts:
            key = (str(item["code"]).strip(), str(item["vocabulary"]).strip().upper())
            gold_label = gold_by_key.get(key)
            if not st.session_state.classification_use_possible and gold_label == "Possible":
                gold_label = "Exclude"
            pred_label = str(item["label"]).strip().title()
            if gold_label is None:
                if pred_label in classification_summary.index:
                    classification_summary.loc[pred_label, "Exclude"] += 1
                continue
            if pred_label in classification_summary.index and gold_label in classification_summary.columns:
                classification_summary.loc[pred_label, gold_label] += 1
    classification_summary.index.name = "predicted \\ gold"
    st.dataframe(classification_summary, use_container_width=True, hide_index=False)
    with st.expander("Correct predictions"):
        if gold_scoped is not None:
            gold_by_key = {
                (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper()): str(row["label"]).strip().title()
                for _, row in gold_scoped.iterrows()
            }
            classified_by_key = {
                (str(item["code"]).strip(), str(item["vocabulary"]).strip().upper()): item
                for item in classified_dicts
            }
            correct_rows = []
            for key, item in classified_by_key.items():
                gold_label = gold_by_key.get(key)
                if not st.session_state.classification_use_possible and gold_label == "Possible":
                    gold_label = "Exclude"
                pred_label = str(item["label"]).strip().title()
                if gold_label == pred_label:
                    correct_rows.append(
                        {
                            "vocabulary": item["vocabulary"],
                            "code": item["code"],
                            "description": item.get("description", ""),
                            "predicted": pred_label,
                            "gold": gold_label,
                            "confidence": item.get("confidence", ""),
                            "explanation": item.get("explanation", ""),
                        }
                    )
                elif gold_label is None and pred_label == "Exclude":
                    correct_rows.append(
                        {
                            "vocabulary": item["vocabulary"],
                            "code": item["code"],
                            "description": item.get("description", ""),
                            "predicted": pred_label,
                            "gold": "Exclude",
                            "confidence": item.get("confidence", ""),
                            "explanation": item.get("explanation", ""),
                        }
                    )
            st.markdown("##### Correct predictions")
            st.dataframe(pd.DataFrame(correct_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Run metrics to see correct predictions.")

    with st.expander("Incorrect predictions"):
        if gold_scoped is not None:
            gold_by_key = {
                (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper()): row["label"]
                for _, row in gold_scoped.iterrows()
            }
            classified_by_key = {
                (str(item["code"]).strip(), str(item["vocabulary"]).strip().upper()): item
                for item in classified_dicts
            }
            incorrect_rows = []
            for key, item in classified_by_key.items():
                gold_label = gold_by_key.get(key)
                if not st.session_state.classification_use_possible and str(gold_label).strip().title() == "Possible":
                    gold_label = "Exclude"
                pred_label = str(item["label"]).strip().title()
                if gold_label is None:
                    if pred_label != "Exclude":
                        incorrect_rows.append(
                        {
                            "vocabulary": item["vocabulary"],
                            "code": item["code"],
                            "description": item.get("description", ""),
                            "predicted": pred_label,
                            "gold": "Exclude",
                            "confidence": item.get("confidence", ""),
                            "explanation": item.get("explanation", ""),
                        }
                    )
                    continue
                if gold_label == pred_label:
                    continue
                incorrect_rows.append(
                    {
                        "vocabulary": item["vocabulary"],
                        "code": item["code"],
                        "description": item.get("description", ""),
                        "predicted": pred_label,
                        "gold": gold_label,
                        "confidence": item.get("confidence", ""),
                        "explanation": item.get("explanation", ""),
                    }
                )
            st.markdown("##### Incorrect predictions")
            if incorrect_rows:
                st.dataframe(pd.DataFrame(incorrect_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No incorrect predictions.")
        else:
            st.caption("Run metrics to see incorrect predictions.")

    with st.expander("Full metrics"):
        st.json(metrics)
        st.markdown("#### Per-label classification metrics")
        per_label_df = pd.DataFrame(classification_metrics["per_label"]).T
        per_label_df.index.name = "label"
        st.dataframe(per_label_df, use_container_width=True)
