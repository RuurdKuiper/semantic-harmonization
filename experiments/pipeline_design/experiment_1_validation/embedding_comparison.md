# Controlled embedding comparison

This is a validation-only extension to Experiment 1. The older Matthewman
developmental runs changed both embedding model and classification policy;
they cannot isolate the embedding effect. The current frozen study uses MiniLM.

Compare MiniLM with OpenAI `text-embedding-3-large` (3,072 dimensions), using the
same seven validation EDFs, six-vocabulary corpus, description text, EDF query
strings, BM25 vectors, references, normalization, and ranking tie handling.
Native tokenization and truncation behavior can differ between encoders.

For each embedder, evaluate all 21 lexical weights from 0 to 1. Compare the fixed
50:50 weighting and the weight maximizing equally weighted EDF average precision.
Report per-EDF AP and recall at 200, 1,000, 5,000, and 10,000 candidates.

Run each distinct fixed/optimized ranking through the frozen proprietary Jev and
GPT-6 Luna choices. Retain batches of 20, ten consecutive zero-Narrow batches,
minimum three batches, and the 10,000-candidate ceiling. Review confidence below
0.3, at most 300 candidates, with medium reasoning. Record initial and post-review
confusion counts, recall, precision, F1, candidate and review counts, and ceiling
hits. CodeMapper is excluded. Identical conditions share checkpoints; existing
MiniLM results are reused only after exact decision/review signature checks.
References enter scoring and validation weight selection, rather than prompts.

The original frozen pipelines, tests, and benchmark remain unchanged. Neither
type 1 diabetes nor published-benchmark outcomes select settings for this extension.
Future evaluation of a revised selected pipeline is a subsequent study step.

Run or resume:

```sh
.venv-study/bin/python experiments/pipeline_design/embedding_comparison.py --detach
```

Configuration: [embedding_comparison.yaml](../embedding_comparison.yaml).
Output directory: `results/experiments/embedding_comparison/20261022/`.

- `run.log`, `status.json`, `worker.json`: progress and worker state.
- `protocol.json`, `runner.json`: fixed inputs and implementation provenance.
- `retrieval.csv`, `retrieval_selection.json`: all weights and selected settings.
- `pipeline_metrics.csv`: per-EDF initial and post-review metrics.
- `pipelines/`: decision/review checkpoints for each distinct ranking.
- `report.md`, `figures/`: completed values and explicit pending conditions.
- `embedding_usage.json`, `queries/*.json`: recorded request/token usage.

Corpus vectors are disk-backed arrays under
`data/codes/embeddings/openai_text-embedding-3-large_3072_resumable/`.
Each vocabulary has row metadata, an input signature, and a committed row offset.
Vectors are flushed before the offset advances. Successfully saved batches are
reused; responses lost before a checkpoint may need repeating. Incomplete vectors
never enter ranking. Worker and cache locks prevent duplicate jobs/writes.

Description batches contain at most 512 inputs. Conservative UTF-8 byte bounds
keep inputs below documented token limits without truncation. Malformed responses
are rejected. Usage limits stop the worker with an explicit incomplete status.
Credentials are excluded from reports and manifests.

API details: [official embedding reference](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create).
