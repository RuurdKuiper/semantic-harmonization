# Matthewman et al. 2026 matched-corpus replication

This experiment applies this repository's constrained hybrid-retrieval and
candidate-classification pipeline to the seven codelists evaluated in:

> Matthewman et al. *Large language models and retrieval augmented generation
> for complex clinical codelists: evaluating performance and assessing failure
> modes*. medRxiv 2026.04.23.26351098v1.

It deliberately does not substitute the seven reference codelists for the
retrieval corpus. Doing that would put every answer in the search index and
leak the gold standard into retrieval.

## Required external inputs

1. Clone the authors' public repository, which supplies the seven reference
   codelists:

   ```bash
   git clone https://github.com/julianmatthewman/llmcodelists_public.git
   git -C llmcodelists_public checkout bcef4a7d2274f08cdd6601ded8152dfb1be8847b
   ```

   That is the repository's initial and currently only commit, dated before the
   preprint. The public wrist-fracture CSV contains 214 required and 127
   optional codes; the manuscript reports 214 and 125. There are no blank or
   duplicate codes in the public file, so the runner preserves the authors'
   published data and records its hash rather than silently removing two rows.

2. Obtain the **CPRD Aurum September 2025 Medical Browser** used by the paper.
   It is not included in the public repository. CPRD distributes its
   dictionaries and browser subject to its access terms; request the matching
   release from CPRD and retain it outside this repository.

The runner accepts the authors' Stata `.dta` file directly. It expects the
fields `originalreadcode`, `term`, and `cleansedreadcode`. Column matching is
case-insensitive.

## Faithful controls

- Seven identical phenotype names and public required/optional reference lists.
- Full CPRD browser and Read-code-only corpus subsets.
- Three epochs by default.
- The same four model identifiers used in the paper.
- Minimal semantic input: only the phenotype name and the instruction
  "Include codes indicating that the person currently has the condition."
- Paper-compatible whole-codelist grades:
  - Correct: every required code, with no code outside required/optional gold.
  - Partially correct: every required code, plus at least one non-gold code.
  - Incorrect: one or more required codes omitted.

## Intentional treatment differences

These are the components being tested, rather than copied from the paper:

- deterministic 50/50 BM25 + `all-MiniLM-L6-v2` retrieval;
- one fixed phenotype-name query instead of LLM-controlled repeated tool calls;
- one fixed top-500 candidate pool per corpus subset;
- constrained binary Narrow/Exclude classification in batches of 10;
- exact validation against retrieved `(code, vocabulary)` pairs.

Consequently, generated code hallucinations are structurally impossible, but
retrieval failures and false Exclude classifications remain possible.

The paper exposes 100 results **per retrieval tool call** and allows the LLM to
make multiple calls. A single top-100 retrieval is not equivalent: it cannot
possibly contain all 214 required full-browser wrist-fracture codes, or even
all 111 required Read-only codes. This experiment therefore defaults to 500.
Use `--top-k 100` only as a deliberately capacity-limited ablation. A useful
retrieval-size sensitivity analysis is `100`, `250`, `500`, and `1000`, run as
separate named runs.

## Run

First install the updated requirements. Then run a small smoke experiment:

```bash
python experiments/matthewman_2026/run.py \
  --cprd-browser /secure/path/CPRDAurumMedical_2025_09.dta \
  --paper-repo /path/to/llmcodelists_public \
  --models openai:gpt-5.2 \
  --phenotypes eosinophilic_esophagitis \
  --subsets full \
  --epochs 1
```

Run the complete matched design:

```bash
python experiments/matthewman_2026/run.py \
  --cprd-browser /secure/path/CPRDAurumMedical_2025_09.dta \
  --paper-repo /path/to/llmcodelists_public
```

API keys are read from `.env`. The full design requires `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and either `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
Jev can be added as an extra treatment with `--models jev:jev-1.13`; it is not
part of the paper's original four-model design.

Outputs are written beneath `outputs/<run-id>/`:

- `run_manifest.json`: exact inputs, file hashes, settings, and corpus sizes;
- `runs.csv` / `runs.json`: one row per phenotype/subset/model/epoch;
- `summary.csv`: paper-style score by model and subset, the paper's reported
  score, the difference, and aggregate failure-mode counts;
- `details/*.json`: retrieved, predicted, missed, and irrelevant code lists.

Use `--dry-run` to validate schemas and report corpus/reference counts without
loading embedding models or calling an LLM.
