# Experiment 3: published phenotype benchmark

Use the existing Matthewman et al. public reference codelists and September 2025
CPRD Aurum Medical Browser for seven phenotypes, on full and Read-only corpora.
Run both frozen branches. Retrieval uses only the published phenotype name;
classification uses a generic current-condition policy frozen before evaluation.
The reference-aligned developmental policy from the older benchmark is excluded
from new pipeline selection. MiniLM corpus embeddings are built locally if absent.

Report required-code recall, precision accepting required or optional codes, and
the deterministic whole-list grade by phenotype and branch. Required reference
codes outside a tested corpus are outside its denominator and should be described
as scope exclusions. Results are saved in `benchmark_metrics.csv`, with separate
resumable decision and review files for each corpus subset.

The local and proprietary branches can run concurrently using the same frozen
settings and checkpoints:

```sh
.venv-study/bin/python experiments/pipeline_design/parallel_benchmark.py --branch proprietary --detach --report
.venv-study/bin/python experiments/pipeline_design/parallel_benchmark.py --branch open_weight --detach --report
```

Stop an existing sequential study worker gracefully before switching to this
runner. Each branch holds an exclusive worker lock to prevent duplicate calls.
Scores are cached under locks; shared metrics are merged atomically by branch,
corpus, and phenotype. Model decisions and reviews retain their existing paths
and signatures. The frozen `study.py` implementation is reused unchanged.
Report generation is serialized when workers finish or stop, including on usage
limits. Relaunch the affected branch with the same command to resume.

Progress is in `run.benchmark.proprietary.log` and
`run.benchmark.open_weight.log` under the dated output directory.
`status.benchmark.<branch>.json` and `worker.benchmark.<branch>.json` record each
worker; `status.json` combines their states. `parallel_benchmark.<branch>.json`
records the scheduling wrapper and frozen-pipeline hashes.

The benchmark identifies codes by `OriginalReadCode` from the CPRD browser.
If validation selects CodeMapper, the runner adapts these identifiers through
explicit native Read and SNOMED columns in the same browser. Read identifiers
must agree between the padded five-character original code and the base of
`CleansedReadCode`; SNOMED identifiers come from `SnomedCTConceptId`. Aliases must
exist in the loaded native vocabulary releases. Ambiguous and unavailable aliases
are excluded and counted. Read case is preserved; native identifiers that collide
under the mapper's uppercase key normalization are excluded. Direct returned cross-vocabulary CUI links are projected
back to existing identifiers in the tested CPRD subset. Browser aliases alone do
not add codes. Mapping provenance retains both browser and native identifiers,
with coverage and unmapped seeds saved alongside the results. The adapter hash is
frozen before benchmark scoring. Corpus and reference files retain their access terms.
