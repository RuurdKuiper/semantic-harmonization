# Pipeline design and evaluation study

The study separates three experiments. Settings are selected using seven EDFs
only. Type 1 diabetes is excluded from every selection step. Because it appeared
in earlier development runs, it is a **provisional retrospective test**, rather
than an independent, previously unseen hold-out set.

1. [Validation and design choices](experiment_1_validation/README.md).
2. [Frozen pipelines on test data](experiment_2_test/README.md).
3. [Published phenotype benchmark](experiment_3_benchmark/README.md).

The earlier fixed-candidate GPT-5-mini/Jev comparison is dropped from this study.
The previous eight-EDF CodeMapper report remains a pilot and is not relabeled as
the new seven-EDF validation experiment.

## Setup

Use an isolated environment for the local decision models, with the repository's
normal dependencies and `requirements-decision-models.txt`. The current run uses
`.venv-study/bin/python` on Apple silicon. Existing local MiniLM vocabulary caches
are required; the runner never creates paid vocabulary embedding indexes.

```sh
.venv-study/bin/python scripts/prepare_study_models.py
.venv-study/bin/python experiments/pipeline_design/run.py --phase all
```

Credentials are read from the ignored `.env`. API calls are authorized subject to
the account's automatic spending limit. The configured request ceiling is an
operational limit, rather than a monetary cap. Raw credentials are not saved.

## Progress and resumption

The runner prints the phase, model, EDF, and candidate range before each batch,
then the number completed. Local Qwen additionally prints generated-token
progress. Every successful batch is atomically checkpointed. Repeating the same
command resumes at the first unfinished batch, including the stopping history.
A failed request can be retried because its response was not saved; completed
responses are reused. Input changes are rejected within an existing result file.

Run decision models independently when a remote service is unavailable:

```sh
.venv-study/bin/python experiments/pipeline_design/run.py --phase retrieval
.venv-study/bin/python experiments/pipeline_design/run.py --phase decisions --models jev openai_decisions
.venv-study/bin/python experiments/pipeline_design/run.py --phase decisions --models jpt4b
.venv-study/bin/python experiments/pipeline_design/run.py --phase decisions
.venv-study/bin/python experiments/pipeline_design/run.py --phase review
.venv-study/bin/python experiments/pipeline_design/run.py --phase mapping
.venv-study/bin/python experiments/pipeline_design/run.py --phase freeze
.venv-study/bin/python experiments/pipeline_design/run.py --phase test
.venv-study/bin/python experiments/pipeline_design/run.py --phase benchmark
.venv-study/bin/python experiments/pipeline_design/report.py
```

To prioritize CodeMapper before unfinished local reviews, run:

```sh
.venv-study/bin/python experiments/pipeline_design/map_available.py --report
```

It maps the best proprietary reviewer after its compatible comparisons are
complete, plus completed EDFs from the fixed local reviewer. Results are saved
as partial mapping metrics; final selection still requires all validation EDFs.
The later full mapping phase reuses these per-EDF outputs and API caches. The
current API uses series-level `vocId` / `vocIds` fields; the adapter selects the
configured releases from its multi-version responses.

The report fills figures and tables with completed outputs and leaves unfinished
combinations pending. To rebuild the interim manuscript body, abstract, three
PDFs, and source archive without model calls, run:

```sh
.venv-study/bin/python experiments/pipeline_design/draft_manuscript.py
```

The main draft separates the revised experiments from archived pilot results.
The documented report-preparation suspension is retained in `report_pause.json`;
reported timing subtotals omit the entire affected batch, while retaining its
predictions. Raw batch wall times remain unchanged for auditing.

To keep the worker independent of the chat's command session, use:

```sh
.venv-study/bin/python experiments/pipeline_design/background.py
tail -f results/experiments/pipeline_design/20261008_cap10000/run.log
```

The launcher prints its PID, rejects a second active worker, and refreshes the
report after each phase. Errors stop the worker with an incomplete status and
saved batches. Run the launcher again after the blocking issue is resolved.

Resume evaluation after completed validation without rerunning its phases:

```sh
.venv-study/bin/python experiments/pipeline_design/background.py --from-phase test
```

This continues the provisional test, then the published benchmark, using the
saved frozen configurations. `--from-phase benchmark` resumes only that phase.
The worker verifies frozen source hashes before evaluation. All completed
decision/review checkpoints are reused.

The 22 October restart retains generation settings and adds local response
auditing under the run's `local_responses/` directory. Each JSON audit contains
raw generated text, token count, prompt hash, parser source hash, and parse status.
The parser reads the first complete visible JSON envelope when a model continues
after its answer; incomplete JSON is rejected. The restart is documented in
`resume_20261022.json`. Existing validation predictions remain unchanged.
When malformed output persists, the batch is recursively subdivided while
retaining its EDF instructions, model, and generation settings. Nominal review
batches remain at most ten candidates; recovered cases have smaller batch
contexts. Recovery is recorded in raw-response audits and in the manuscript
methods. Every input identifier must have exactly one returned classification.
Audited failed envelopes are reused to avoid regenerating an identical failed
batch before subdivision; they are never treated as completed classifications.

Separate model processes may be used for remote work. Run the local large models
sequentially to avoid GPU contention. Selection files are created only when all
configured competitors have completed all validation EDFs. Frozen configurations
retain source hashes, package versions, model revisions, and a checksum; test and
benchmark results cannot select pipeline settings. An independent hold-out set
can be added to a new study configuration before freezing, once available.

## Interpretation

Both branches share on-device MiniLM retrieval. The proprietary branch compares
Jev and OpenAI Decisions, then the two GPT reviewers. The local branch compares
JPT-4B and uses Qwen3.5-9B at 8 bits for review. Laya was stopped at the user's
request after excessive Narrow decisions in early validation runs; its incomplete
outputs remain available as a diagnostic run and are excluded from selection. All three reviewers are
also compared on the same best overall preliminary decisions. JPT-4B's
noncommercial license means **open-weight branch** is the accurate label, rather
than claiming that every component is open source. Terminology licenses remain
separate. CodeMapper is a hosted optional stage and makes a branch dependent on
that service if selected.

Confidence is the provider's reported signal. JPT's entropy-derived signal and
the other providers' signals are not assumed to be clinically calibrated or
equivalent. Counts of eligible and reviewed decisions must accompany comparisons.
Unclassified available reference Narrow codes count as false negatives; absent
terminology entries are reported separately. TN counts only assessed negatives.
Results use individual EDF tables and equal-EDF means rather than pooled counts.

Model documentation: [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions),
[Laya](https://huggingface.co/convaiinnovations/laya),
[JPT-4B](https://huggingface.co/kirp/jpt-4b), and
[Qwen 8-bit conversion](https://huggingface.co/mlx-community/Qwen3.5-9B-8bit).

## Rescoring published baselines

The primary Matthewman RAG comparison reuses eight published evaluation bundles
from repository commit `bcef4a7d2274f08cdd6601ded8152dfb1be8847b`: four models,
two corpus scopes, seven phenotypes, and three epochs (168 final submissions).
Alternative workflows are excluded. Published models are not called again.

`published_baselines.py` treats submitted codelists as identifier sets and applies
the benchmark's deterministic required/optional scoring to the same fixed
references used for our pipelines. It records source hashes, original LLM grades,
exact extracted codes, ambiguous-cell interpretations, unavailable identifiers,
and reference changes. It does not reconstruct retrieval coverage from final
outputs. Formatting wrappers and source annotations are distinguished from code
identifiers; no identifier spelling or case is corrected and no range is expanded.
One GPT psoriasis submission pairs retrieval-record IDs with clinical codes; the
leading source-record numbers are excluded after checking its retrieval records.
All such interpretations are retained in the extraction audit.

Figure 8 uses grouped bars for our single runs and the published three-run means.
Columns show required-code recall, acceptable-code precision, and their harmonic
mean (F1), calculated per run before averaging. An Average group gives equally
weighted means over completed phenotypes; panels label incomplete coverage.
Partial averages can cover different phenotype sets and remain provisional.
One empty bar-width separates our two pipelines from the four published systems
within each phenotype; fixed slots are retained for pending results.
The axes identify required-code recall and acceptable-code precision. These are
descriptive system comparisons, rather than original-paper grade reproductions.
Appendix Tables A10 and A11 separate original versus rescored grades and record
reference changes; Table A12 retains all three run scores per phenotype and corpus.
The primary comparison accepts missing ESCT carets and Read trailing-dot padding
when a variant resolves to one browser entry available in the tested corpus.
Rules use the full browser independently of reference labels, preserving case;
ambiguity is evaluated before restricting corpus scope. Exact browser identifiers
are retained and normalized duplicates are collapsed. Archived model-grader
explanations explicitly accept padded Read codes and missing ESCT carets in
inspected cases. This policy applies equally to all systems; our outputs already
use valid browser identifiers. Native SNOMED IDs remain unresolved because
one-to-many concept-to-browser mappings would change the evaluation unit.
Unknown codes, empty submissions, and actual code-selection errors retain their
penalties. `metrics.strict.csv` preserves strict matching for the same 168 runs;
Appendix Table A13 gives the sensitivity comparison. The extraction audit records
original codes, normalized codes, and each applied rule. No models are rerun.
