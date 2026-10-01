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

   For the experiment's default path, clone it to
   `experiments/matthewman_2026/data/llmcodelists_public`. The entire local
   data directory is ignored by Git.

   That is the repository's initial and currently only commit, dated before the
   preprint. The public wrist-fracture CSV contains 214 required and 127
   optional codes; the manuscript reports 214 and 125. There are no blank or
   duplicate codes in the public file, so the runner preserves the authors'
   published data and records its hash rather than silently removing two rows.

2. Obtain the **CPRD Aurum September 2025 Medical Browser** used by the paper.
   It is not included in the public repository. CPRD distributes its
   dictionaries and browser subject to its access terms; request the matching
   release from CPRD. Place it at
   `experiments/matthewman_2026/data/CPRDAurumMedical.txt`. This directory is
   ignored by Git so licensed source data and derived private artifacts are not
   accidentally committed.

The runner accepts Stata `.dta`, comma-delimited `.csv`, and tab-delimited
CPRD `.txt` files directly. It expects the fields `originalreadcode`, `term`,
and `cleansedreadcode`. Column matching is case-insensitive.

## Faithful controls

- Seven identical phenotype names and public required/optional reference lists.
- Full CPRD browser and Read-code-only corpus subsets.
- One repetition per phenotype and corpus subset.
- Minimal retrieval input: only the phenotype name. The Jev/GPT classifier also
  receives an experiment-only, phenotype-agnostic operationalization of the
  paper's reference-list scope. It includes explicit synonyms/subtypes,
  attributed manifestations and complications, combined anatomical concepts,
  sequelae, and procedures whose term explicitly names the target condition.
  It does not contain phenotype-specific codes or reference terms and is not
  used to form the retrieval query.
- Paper-compatible whole-codelist grades:
  - Correct: every required code, with no code outside required/optional gold.
  - Partially correct: every required code, plus at least one non-gold code.
  - Incorrect: one or more required codes omitted.

## Intentional treatment differences

These are the components being tested, rather than copied from the paper:

- deterministic 10% BM25 + 90% OpenAI `text-embedding-3-large` retrieval at
  3,072 dimensions;
- one fixed phenotype-name query instead of LLM-controlled repeated tool calls;
- the complete vocabulary is similarity-scored and ordered (no default top-k cap);
- Jev performs constrained binary Narrow/Exclude classification in batches of
  20 and stops after 10 consecutive batches containing zero Narrow results;
- GPT-6 Luna reviews decisions below 0.7 Jev confidence, capped at the 200
  lowest-confidence eligible decisions;
- exact validation against retrieved `(code, vocabulary)` pairs.

The expanded classification policy is a deliberate reference-alignment layer,
not an instruction present in the paper's original minimal prompt. It addresses
cases where the published gold codelists operationalize "currently has the
condition" more broadly than a strict direct-diagnosis reading. Because it is
classification-only, it cannot improve the retrieval rank of a required code.

Consequently, generated code hallucinations are structurally impossible, but
required codes can still be missed because adaptive stopping reaches them too
late or because Jev/GPT classifies them as Exclude. The reports separate these
two failure modes.

The paper exposes 100 results **per retrieval tool call** and allows the LLM to
make multiple calls. Our adaptive scan instead gives every vocabulary entry a
similarity score, classifies in that order, and stops only when Narrow results
have become sparse. `--top-k` remains available solely for a deliberately
capacity-limited ablation.

## Evaluation limitations and interpretation

### LLM grading versus deterministic scoring

The paper's primary Correct/Partially-correct/Incorrect grades were assigned by
an LLM judge (Gemini 3 Pro) given the generated and reference codelists, rather
than by deterministic set comparison. Claude Sonnet 4.6 was also used to parse
free-form result tables for the subsequent failure-mode analysis. This permits
the grader to tolerate harmless formatting differences, but introduces another
model-dependent step whose repeatability, error rate, and possible same-family
grading preference were not quantified with inter-rater agreement, independent
manual adjudication, or a second-grader sensitivity analysis. The paper clearly
reports the method and publishes its evaluation logs, but does not discuss the
single-LLM judge as a distinct limitation.

This experiment instead emits structured `(code, vocabulary)` decisions and
computes grades, required-code recall, acceptable-code precision, stopping
misses, and classification misses deterministically. This is stricter about
identifier equality but is appropriate here because predictions copy codes
directly from the source vocabulary. Differences from the paper's reported
grades can therefore reflect the evaluation method as well as pipeline quality.

### Iteratively constructed reference codelists

The authors initially constructed the seven reference codelists and then
iteratively revised them after inspecting RAG evaluation logs, adding or
correcting codes until remaining disagreements were considered unambiguous
model errors. This improves reference completeness, but it is not an
independently developed, blinded test set. It creates a form of incorporation
bias: codes and interpretations surfaced by the development-time RAG system
are more likely to enter the reference, while relevant codes missed by both the
authors and that system can remain absent. The resulting benchmark may also
favour models or retrieval behaviour similar to those used during its
construction.

The paper acknowledges this directly: it describes its scoring as deliberately
lenient, notes that the references were revised after viewing evaluation
results, and states that the reported performance may consequently be an upper
bound. It also acknowledges that simple phenotype specifications cannot produce
fully unambiguous reference standards.

An additional genuinely valid code that is absent from the reference has the
following effects under the paper-compatible scoring used here:

- If every required code is present, the extra code changes a whole-list grade
  from Correct to Partially correct (half credit), rather than Incorrect.
- If any required code is also missing, the grade is Incorrect regardless of
  whether the extra code was clinically valid.
- In code-level metrics, the absent-from-reference code is still counted as an
  irrelevant prediction/false positive, so acceptable precision is a lower
  bound if the reference is incomplete.
- Codes marked optional by the authors are accepted and do not incur a penalty.

Accordingly, this experiment foregrounds code-level required recall and
acceptable precision alongside the strict whole-codelist grade. Apparent false
positives should be reviewed clinically before being interpreted as genuine
errors. The experiment-only expanded classification policy is itself informed
by general categories observed in the published references; it contains no
phenotype-specific codes or terms, but its use must still be reported as a
reference-alignment choice rather than a blinded replication.

## Run

First install the updated requirements. Then run a small smoke experiment:

```bash
python experiments/matthewman_2026/run.py \
  --phenotypes eosinophilic_esophagitis \
  --subsets full
```

Run the complete matched design:

```bash
python experiments/matthewman_2026/run.py
```

Use `--cprd-browser /secure/other/path/file.dta` to override the default local
file without copying it into the experiment directory. Likewise,
`--paper-repo /path/to/llmcodelists_public` overrides the default reference
repository.

API keys are read from `.env`. The design requires `JEV_API_KEY` and
`OPENAI_API_KEY`. Use `--jev-model` or `--gpt-model` to override either model.
The OpenAI embedding index is stored under a provider/model/dimension-specific
name, alongside (and without overwriting) any existing MiniLM experiment cache.

Outputs are written beneath `outputs/<run-id>/`:

- `run_manifest.json`: exact inputs, file hashes, settings, and corpus sizes;
- `runs.csv` / `runs.json`: one row per phenotype/subset/repetition;
- `summary.csv`: our pipeline's paper-style score and aggregate failure modes;
- `paper_comparison.csv`: our score beside every model score reported by the paper;
- `details/*.json`: adaptive-stopping metadata, GPT-review metadata, final
  decisions, missed codes, and irrelevant codes.

Use `--dry-run` to validate schemas and report corpus/reference counts without
loading embedding models or calling an LLM.
