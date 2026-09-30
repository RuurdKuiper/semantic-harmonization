# Semantic Harmonization Pipeline

An AI-assisted pipeline for finding clinical terminology codes that match a
medical condition and reducing the amount of expert code review required.

## Pipeline

An Event Definition Form (EDF) is the semantic input. Original EDF documents
are retained as DOCX, while reviewed YAML representations are used at runtime.
AESI codelist CSVs are expert-generated ground truth used only for evaluation.

For a single phenotype, `main.py`:

1. Loads `data/edfs/yamls/<phenotype>.yaml`, including its definition,
   inclusion/exclusion criteria, decision rules, synonyms, anchor codes,
   ambiguities, and references.
2. Loads the selected complete terminology sources from `data/codes/csv/`.
   Parsed corpora are cached in `data/codes/parquet/`; a cache is rebuilt when
   its source CSV is newer. Non-codable dictionary range rows such as
   `N17-N19` or `N17–N19` are removed before retrieval and classification.
3. Gives every code a combined BM25 lexical and sentence-embedding similarity
   score, then ranks the complete corpus. Vector indexes are cached per
   vocabulary in `data/codes/embeddings/` and validated against the current
   corpus before reuse. An optional hard top-k cap is retained for fixed-size
   experiments.
4. Passes ranked candidates directly to LLM classification in batches. LLM
   re-ranking is implemented but currently disabled. By default, classification
   stops adaptively once Narrow results remain sparse for several consecutive
   batches.
5. Classifies each candidate as `Narrow` or `Exclude`. The `Possible` category
   is optional via `llm.use_possible_category` and is off by default.
6. Adds a `manual_review` flag and reason to low-confidence results. Review
   annotation never removes, relabels, or otherwise changes a result.
7. Evaluates the output against the phenotype's registered AESI codelist,
   restricted to selected vocabularies and gold codes actually present in the
   loaded terminology versions. The metrics report how many unavailable gold
   rows were excluded. Predictions and metrics are written to `results/`.

Available full code systems are ICD-10-CM, ICD-9-CM, ICPC, MedDRA (`MDR`),
RCD2, and SNOMED CT (US).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set ANTHROPIC_API_KEY, OPENAI_API_KEY, JEV_API_KEY, and/or GOOGLE_API_KEY
```

## Usage

Run one sample:

```bash
python main.py --phenotype myocarditis --config configs/default.yaml
```

This writes:

- `results/<phenotype>_predicted.csv` — AESI-compatible fields plus confidence,
  explanation, `manual_review`, and `review_reason`.
- `results/<phenotype>_metrics.json` — review summary and evaluation metrics.

Run the Streamlit app:

```bash
streamlit run app.py
```

The app supports every configured code system and bundled EDF. It writes the
same outputs to `results/` and provides download buttons for the classified CSV
and metrics JSON. The retrieval table only previews the configured top-k rows;
the full ranked list remains available to adaptive classification without being
rendered in the browser.

### Adaptive classification stopping

The default settings score the full selected terminology corpus and classify it
in similarity order. Classification stops after three consecutive batches each
containing at most one `Narrow` result, once at least three batches have been
processed. These controls are available in the Streamlit app and under `llm`
in `configs/default.yaml`:

```yaml
retrieval:
  top_k: 5                  # preview size
  use_top_k_limit: false    # true restores a hard retrieval cap

llm:
  adaptive_stopping_enabled: true
  sparse_narrow_threshold: 1
  consecutive_sparse_batches: 3
  minimum_batches: 3
```

All processed candidates still receive a `Narrow` or `Exclude` label. Adaptive
stopping only prevents lower-ranked, not-yet-processed candidates from being
sent to the classifier.

### Run all samples and compare settings

Run every registered EDF/AESI pair with the default settings:

```bash
python scripts/run_all_samples.py
```

Run a retrieval-settings grid:

```bash
python scripts/run_all_samples.py --top-k 5 10 25 --lexical-weight 0.25 0.5 0.75
```

The experiment runner treats `--top-k` as a hard cap and disables adaptive
stopping so fixed-size runs remain directly comparable.

The embedding weight is set to `1 - lexical weight`. Add `--possible` to test
three-label classification. Outputs are grouped beneath
`results/experiments/<run-id>/`; `summary.csv` and `summary.json` compare all
phenotype/setting combinations. By default, each phenotype is run only against
vocabularies present in its AESI ground truth; use `--all-vocabularies` to
override this. The command prints each pipeline stage and live progress bars
for embedding-index preparation and LLM classification.

Useful options:

```bash
python scripts/run_all_samples.py --phenotypes myocarditis kidney_disease
python scripts/run_all_samples.py --run-id baseline --top-k 10 25
```

Generate a publication-style Markdown report from an experiment:

```bash
python scripts/summarize_experiment.py results/experiments/top-k-50
```

This writes `experiment_report.md` beside the experiment summary. It includes
an abstract, settings-level averages, pooled Narrow TP/TN/FP/FN and derived
metrics, and a complete per-phenotype results table. Use `--output <path>` to
choose another destination.

## Data layout

```text
data/
├── codes/
│   ├── csv/          # source terminology exports
│   ├── parquet/      # regenerated normalized caches
│   └── embeddings/   # regenerated vector indexes
├── edfs/
│   ├── docx/         # original Event Definition Forms
│   └── yamls/        # structured runtime EDFs
└── codelists/
    └── AESI/         # expert reference codelists
results/              # predictions, metrics, and experiment summaries
```

Eight samples are registered: ADEM, AMI, eczema vaccinatum, erythema
multiforme, Guillain-Barré syndrome, kidney disease, myocarditis, and type 1
diabetes.

To add a phenotype, place its reviewed YAML in `data/edfs/yamls/`, retain the
source document in `data/edfs/docx/`, place its reference export in
`data/codelists/AESI/`, and register the AESI path in `configs/default.yaml`.

## Embedding release assets

To stage all configured precomputed embedding files for a GitHub Release:

```bash
python scripts/package_embeddings.py
gh release upload <tag> dist/release-assets/*
```

The app can download missing assets when `GITHUB_REPOSITORY_OWNER`,
`GITHUB_REPOSITORY_NAME`, and `GITHUB_RELEASE_TAG` are set. Otherwise it builds
missing indexes locally.

## Testing

```bash
pytest
pytest -m "not embeddings"
```
