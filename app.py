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

# Coding systems we can currently retrieve from vs. planned-but-not-yet-available.
AVAILABLE_VOCABULARIES = {"ICD10CM": "ICD-10-CM"}
PLANNED_VOCABULARIES = {"SNOMEDCT_US": "SNOMED CT (US)", "MDR": "MedDRA"}


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

vocab_cols = st.columns(len(AVAILABLE_VOCABULARIES) + len(PLANNED_VOCABULARIES))
selected_vocabularies: list[str] = []

for col, (vocab, label) in zip(vocab_cols, AVAILABLE_VOCABULARIES.items()):
    with col:
        if st.checkbox(label, value=True, key=f"vocab_{vocab}"):
            selected_vocabularies.append(vocab)

for col, (vocab, label) in zip(vocab_cols[len(AVAILABLE_VOCABULARIES):], PLANNED_VOCABULARIES.items()):
    with col:
        st.checkbox(f"{label} (no reference list loaded yet)", value=False, disabled=True, key=f"vocab_{vocab}")

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
    vocab_key = "-".join(sorted(vocabularies))
    model_slug = CONFIG.retrieval.embedding_model.replace("/", "_")
    cache_path = Path(CONFIG.paths.processed_dir) / f"embeddings_{vocab_key}_{model_slug}.npz"
    return EmbeddingIndex.from_cache_or_build(codes, cache_path=cache_path, model_name=CONFIG.retrieval.embedding_model)


run_retrieval = st.button("Run hybrid retrieval", disabled=(edf is None or not selected_vocabularies))

if run_retrieval and edf is not None and selected_vocabularies:
    vocab_tuple = tuple(sorted(selected_vocabularies))
    codes = _load_corpus(vocab_tuple)
    embedding_index = _load_embedding_index(vocab_tuple)
    with st.spinner(f"Retrieving top {top_k} candidates from {len(codes)} codes..."):
        candidates = hybrid_retrieval(
            edf.to_prompt_context(),
            codes,
            lexical_weight=CONFIG.retrieval.lexical_weight,
            embedding_weight=CONFIG.retrieval.embedding_weight,
            embedding_model=CONFIG.retrieval.embedding_model,
            top_k=int(top_k),
            embedding_index=embedding_index,
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
                    "lexical_score": c.lexical_score,
                    "embedding_score": c.embedding_score,
                    "score": c.score,
                }
                for c in st.session_state.retrieval_candidates
            ]
        ),
        use_container_width=True,
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
        with st.spinner("Classifying candidates..."):
            ranked = skip_rank(st.session_state.retrieval_candidates)
            classified = llm_classify(
                ranked,
                edf,
                provider=provider_choice,
                model=model_override or None,
                max_retries=CONFIG.llm.max_retries,
            )
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
    st.dataframe(classified_df, use_container_width=True, hide_index=True)

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
            {"code": c.code, "vocabulary": c.vocabulary, "description": c.description, "label": c.label}
            for c in st.session_state.classified
        ]
        gold_scoped = filter_gold_by_vocabulary(gold_labels, compare_vocabularies)
        predicted_scoped = filter_records_by_vocabulary(classified_dicts, compare_vocabularies)
        st.session_state.metrics_result = evaluate(predicted_scoped, gold_scoped)
        st.session_state.predicted_codelist = to_predicted_codelist(classified_dicts)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Evaluation failed: {exc}")

if st.session_state.metrics_result:
    metrics = st.session_state.metrics_result
    retrieval_metrics = metrics["retrieval"]
    classification_metrics = metrics["classification"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sensitivity", f"{retrieval_metrics['sensitivity']:.2f}")
    m2.metric("Precision", f"{retrieval_metrics['precision']:.2f}")
    m3.metric("F1", f"{retrieval_metrics['f1']:.2f}")
    m4.metric("Accuracy", f"{classification_metrics['accuracy']:.2f}")
    m5.metric("Cohen's kappa", f"{classification_metrics['cohens_kappa']:.2f}")

    with st.expander("Full metrics"):
        st.json(metrics)
