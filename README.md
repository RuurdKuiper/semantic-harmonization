# Semantic Harmonization Pipeline

An AI-assisted pipeline for phenotype codelist generation and reduce the time needed
for expert code review.

## Pipeline

The EDF (Event Definition Form) (in .yaml format, currenlty manually extracted 
from the phenotype's `.docx` description) is the input for this system.
The system's job is to search the **full** clinical coding systems (currently
only ICD-10-CM; SNOMED CT (US), MedDRas and others still need to be implemented) 
for codes matching that EDF. The AESI `.csv` files are used as the expert
generated ground truth used only to score the pipeline's output.

Each run (`main.py`) executes the following steps in order for a single
phenotype:

1. **Load the EDF** (`src/data/loaders.py::load_edf`) — reads the structured
   Event Definition Form YAML for the phenotype (`data/raw/edf/<phenotype>.yaml`):
   clinical definition, inclusion/exclusion criteria, narrow/exclude decision
   rules, synonyms, etc. This is rendered into a single text block
   (`EventDefinitionForm.to_prompt_context()`) that is reused as both the
   retrieval query and the LLM context in later steps.

   > **Note on the raw `.docx` source files:** the EDF YAMLs were authored by
   > hand from the original `.docx` phenotype descriptions in
   > `data/raw/<Phenotype>/*.docx`. There is no code that parses `.docx` at
   > runtime — this needs to be done manually when adding a new phenotype.
   > We will let an LLM do this in future versions.

2. **Load the full reference code system(s)** (`src/data/code_systems.py::load_code_system_corpus`) —
   loads the full code list for each vocabulary in `retrieval.vocabularies`
   (e.g. all ~74k ICD-10-CM codes from `data/codes/icd10.xlsx`). Each vocabulary
   is parsed once and cached to `data/processed/codes_<VOCAB>.parquet`; subsequent
   runs read the cache instead of re-parsing the source file. The corpus is then
   normalized by `src/data/preprocessing.py::preprocess_corpus` (code standardization,
   deduplication, text normalization).

4. **Hybrid retrieval** (`src/retrieval/hybrid.py::hybrid_retrieval`) —
   ranks every code in the full corpus against the EDF text using a weighted
   combination of BM25 lexical matching (`src/retrieval/lexical.py`) and
   sentence-embedding cosine similarity (`src/retrieval/embeddings.py`),
   returning the top `retrieval.top_k` candidates (default 50). The embedding
   index itself is cached per vocabulary set + model to
   `data/processed/embeddings_<VOCAB>_<model>.npz`, so a large corpus is only
   ever embedded once, not on every run (`EmbeddingIndex.from_cache_or_build`).

5. **LLM ranking** (`src/llm/rank.py::llm_rank`) — sends the EDF context and
   the retrieved candidates to the configured LLM, which re-orders them by
   clinical relevance (`relevance_score`). Any candidate the LLM omits from
   its response is appended at the end so nothing is silently dropped.

6. **LLM classification** (`src/llm/classify.py::llm_classify`) — sends the
   ranked candidates back to the LLM, which assigns each one a label
   (`Narrow`/`Possible`/`Exclude`), a confidence score, and a short
   explanation, using the same EDF context for grounding.

7. **Uncertainty selection** (`src/uncertainty/selection.py::select_uncertain`) —
   flags candidates for human review when their confidence is below
   `uncertainty.confidence_threshold` and/or they were classified as
   `Possible` (configurable via `uncertainty.possible_requires_review`).

8. **Evaluation** (`src/evaluation/metrics.py`) — the classified candidates are
   rendered into the ground-truth CSV schema via `to_predicted_codelist()`
   (`coding_system`, `code`, `code_name`, `concept`, `concept_name`, `tags`).
   If there is a ground-truth AESI export for the phenotype
   (`paths.aesi_datasets`), the predicted codelist is compared against it
   (both filtered to `evaluation.compare_vocabularies`, currently `[ICD10CM]`
   since that's the only full reference list loaded), computing retrieval
   sensitivity/precision/F1 ('Narrow codes' as positives) and classification
   accuracy/Cohen's kappa.

The LLM calls in steps 4–5 go through `src/llm/client.py`, which resolves
whether to use Anthropic or OpenAI (`resolve_provider()`, based on
`llm.provider` in config or auto-detection from whichever `*_API_KEY` is set
in `.env`) before dispatching the request.

> **Note:** step 4 (LLM ranking) is currently **disabled** — candidates go
> straight from retrieval to classification via `src/llm/rank.py::skip_rank`.
> The `llm_rank` function is still implemented and tested, just not called by
> `main.py` or `app.py`, so it can be re-enabled later without rewriting it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Usage

### Command line

```bash
python main.py --phenotype myocarditis --config configs/default.yaml
```

Each run writes two files to `--output-dir` (defaults to `paths.results_dir`,
i.e. `data/processed/results/`):

- `<phenotype>_predicted.csv` — the predicted codelist, in the same schema as
  the ground-truth AESI export (`coding_system`, `code`, `code_name`,
  `concept`, `concept_name`, `tags`).
- `<phenotype>_metrics.json` — review items, review rate, and evaluation
  metrics (also printed to stdout).

### Web UI

```bash
streamlit run app.py
```

A single-page Streamlit UI mirrors the same pipeline stages interactively:
coding-system selection, EDF input (paste, upload, or load a bundled
example), hybrid retrieval with a live results table, LLM classification
(provider/model picker), and a final block to upload a ground-truth AESI CSV
and view the computed metrics.

Note: the first run for a given vocabulary will be slow (~30s for the full
ICD-10-CM list) while it builds and caches the embedding index; subsequent
runs reuse that cache and are fast.

## Data

- `data/raw/edf/*.yaml` — structured phenotype definitions (EDFs), hand-authored
  from the source `.docx` files described in step 1 above. This is the actual
  input to the system.
- `data/codes/icd10.xlsx` — the full ICD-10-CM reference code list, registered
  in `configs/default.yaml` under `paths.code_systems`. This is the retrieval
  universe for step 2/3 (see `src/data/code_systems.py`). Its processed form
  and embeddings are cached under `data/processed/` (gitignored, regenerated
  automatically).
- `data/raw/<Phenotype>/*_AESI_*.csv` — expert-review exports used as **ground
  truth for evaluation only** (not for retrieval), registered per phenotype in
  `configs/default.yaml` under `paths.aesi_datasets`.
- `data/raw/codes/codes.csv` / `data/gold/<phenotype>_gold.csv` — legacy
  standalone corpus/gold format, still supported by `src/data/loaders.py` but
  no longer used by `main.py`.

Real curated data is provided for `myocarditis`, `erythema_multiforme`, and
`guillain_barre`. Add a new phenotype by creating an EDF YAML and registering
its ground-truth AESI CSV in `configs/default.yaml`. Adding a new vocabulary
(e.g. SNOMED CT (US), MedDRA) requires adding a loader to
`src/data/code_systems.py::CODE_SYSTEM_LOADERS` and a source file.

## Testing

```bash
pytest                      # full suite
pytest -m "not embeddings"  # skip tests that download a sentence-transformers model
```

