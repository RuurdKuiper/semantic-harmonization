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
