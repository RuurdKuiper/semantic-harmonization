"""Streamlit GUI for the semantic harmonization pipeline.

Run with:
    streamlit run app.py

The page walks through the pipeline step by step:
1. Select which clinical coding system(s) to retrieve from.
2. Provide the phenotype's Event Definition Form (EDF).
3. Run hybrid (lexical + embedding) retrieval over the full code system(s).
4. Run LLM classification (Narrow/Possible/Exclude) on the retrieved candidates.
5. Optionally score the result against a ground-truth AESI codelist.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.data.code_systems import load_code_system_corpus
from src.data.loaders import EventDefinitionForm, load_aesi_dataset
from src.data.preprocessing import preprocess_corpus
from src.evaluation.metrics import (
    evaluate,
    filter_gold_by_vocabulary,
    filter_records_by_vocabulary,
    to_predicted_codelist,
)
from src.llm.classify import llm_classify
from src.llm.client import default_model_for, resolve_provider
from src.llm.rank import skip_rank
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
    "ICPC": "ICPC",
    "RCD2": "RCD2",
    "SNOMEDCT_US": "SNOMED CT (US)",
}


def _init_state() -> None:
    defaults = {
        "edf_yaml_text": "",
        "phenotype_name": "",
        "retrieval_candidates": None,
        "classified": None,
        "metrics_result": None,
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
        if vocab == "SNOMEDCT_US":
            st.checkbox(f"{label} (offline only)", value=False, disabled=True, key=f"vocab_{vocab}")
        elif st.checkbox(label, value=True, key=f"vocab_{vocab}"):
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

top_k = st.number_input("Top-K candidates to retrieve", min_value=1, max_value=500, value=CONFIG.retrieval.top_k)
per_vocabulary_top_k = st.checkbox(
    "Retrieve top-K per code list",
    value=getattr(CONFIG.retrieval, "per_vocabulary_top_k", True),
    help="If enabled, retrieves the top-K candidates within each selected vocabulary. If disabled, retrieves the top-K overall after combining vocabularies.",
)


@st.cache_resource(show_spinner="Loading code system corpus...")
def _load_corpus(vocabularies: tuple[str, ...]) -> pd.DataFrame:
    raw_codes = load_code_system_corpus(
        list(vocabularies),
        source_paths=CONFIG.paths.code_systems,
        cache_dir=CONFIG.paths.processed_dir,
    )
    return preprocess_corpus(raw_codes)


@st.cache_resource(show_spinner="Building/loading embedding index (only happens once per corpus)...")
def _load_embedding_index(vocabularies: tuple[str, ...]) -> EmbeddingIndex:
    codes = _load_corpus(vocabularies)
    return EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=vocabularies,
        cache_dir=CONFIG.paths.processed_dir,
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
    with st.spinner(f"Retrieving top {top_k} candidates from {len(codes)} codes..."):
        candidates = hybrid_retrieval(
            edf.to_prompt_context(),
            codes,
            lexical_weight=CONFIG.retrieval.lexical_weight,
            embedding_weight=CONFIG.retrieval.embedding_weight,
            embedding_model=CONFIG.retrieval.embedding_model,
            top_k=int(top_k),
            per_vocabulary_top_k=bool(per_vocabulary_top_k),
            embedding_index=embedding_index,
            query_model=query_model,
            lexical_query=edf.to_lexical_query(),
            embedding_query=edf.to_embedding_query(),
        )
    st.session_state.retrieval_candidates = candidates
    st.session_state.classified = None
    st.session_state.metrics_result = None

if st.session_state.retrieval_candidates:
    st.success(f"Retrieved {len(st.session_state.retrieval_candidates)} candidates.")
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
                for c in st.session_state.retrieval_candidates
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

llm_col1, llm_col2 = st.columns(2)
with llm_col1:
    provider_choice = st.selectbox("Provider", options=["auto", "anthropic", "openai"], index=0)
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

        with st.spinner("Classifying candidates..."):
            ranked = skip_rank(st.session_state.retrieval_candidates)
            classified = llm_classify(
                ranked,
                edf,
                provider=provider_choice,
                model=model_override or None,
                max_retries=CONFIG.llm.max_retries,
                batch_size=10,
                progress_callback=_update_progress,
            )
        progress.progress(1.0, text="Classification complete")
        st.session_state.classified = classified
        st.session_state.metrics_result = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"LLM classification failed: {exc}")

if st.session_state.classified:
    classified_df = pd.DataFrame(
        [
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
    )
    st.success(f"Classified {len(classified_df)} candidates.")
    st.dataframe(classified_df, width='stretch', hide_index=True)

    review_items = select_uncertain(
        st.session_state.classified,
        confidence_threshold=CONFIG.uncertainty.confidence_threshold,
        possible_requires_review=CONFIG.uncertainty.possible_requires_review,
    )
    st.info(
        f"{len(review_items)}/{len(st.session_state.classified)} candidates flagged for human review "
        f"({100 * review_rate(st.session_state.classified, review_items):.1f}%)."
    )

st.divider()

# ---------------------------------------------------------------------------
# Block 5: Ground truth & metrics
# ---------------------------------------------------------------------------
st.header("5. Evaluate against ground truth")

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
        gold_scoped = filter_gold_by_vocabulary(gold_labels, compare_vocabularies)
        predicted_scoped = filter_records_by_vocabulary(classified_dicts, compare_vocabularies)
        st.session_state.metrics_result = evaluate(predicted_scoped, gold_scoped)
        st.session_state.predicted_codelist = to_predicted_codelist(classified_dicts)
        st.session_state.metrics_gold_scoped = gold_scoped
        st.session_state.metrics_predicted_scoped = predicted_scoped
        st.session_state.metrics_classified_dicts = classified_dicts
        st.session_state.metrics_compare_vocabularies = compare_vocabularies
    except Exception as exc:  # noqa: BLE001
        st.error(f"Evaluation failed: {exc}")

if st.session_state.metrics_result:
    metrics = st.session_state.metrics_result
    retrieval_metrics = metrics["retrieval"]
    classification_metrics = metrics["classification"]

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
            correct_retrievals = gold_scoped[
                gold_scoped.apply(
                    lambda r: (
                        r["label"] in {"Narrow", "Possible"}
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
            missed_retrievals = gold_scoped[
                gold_scoped.apply(
                    lambda r: (
                        r["label"] in {"Narrow", "Possible"}
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
    classification_summary = pd.DataFrame(0, index=["Narrow", "Possible", "Exclude"], columns=["Narrow", "Possible", "Exclude"])
    if gold_scoped is not None:
        gold_by_key = {
            (str(row["code"]).strip(), str(row["vocabulary"]).strip().upper()): str(row["label"]).strip().title()
            for _, row in gold_scoped.iterrows()
        }
        for item in classified_dicts:
            key = (str(item["code"]).strip(), str(item["vocabulary"]).strip().upper())
            gold_label = gold_by_key.get(key)
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
