# Definitive SNOMED source activation (30 October 2026)

The user designated `data/codes/csv/SNOMEDCT_US@2025_09_01-codes definitive.csv`
as the authoritative source, accepting its missing codes. This file is
byte-identical to the previously examined `new.csv` export (SHA-256
`2925788262144dc8bad27da140a84bac7e9e4aca85314aff8903c17a8fd077b2`).

`configs/default.yaml`, the configuration class defaults, and the terminology
loader defaults now select the definitive export. Active caches are isolated in
`data/codes/parquet/snomed_definitive_20261030/` and
`data/codes/embeddings/snomed_definitive_20261030/`.

The parsed SNOMED corpus contains 382,170 distinct code–vocabulary pairs, with
no blank descriptions. The readable import is `SNOMEDCT_US_imported.csv` in the
active Parquet cache directory. The five other terminology Parquets were copied
unchanged, and their existing MiniLM caches were cloned as independent files.
No SNOMED embeddings or external model calls were generated during activation.

The earlier comparison applies unchanged: 2,450 fewer populated identifiers than
the original corpus, no recovered missing reference codes, and 199,267 changed
first-selected descriptions among retained identifiers. Across the eight
phenotype reference codelists, 447 SNOMED Narrow entries are unavailable versus
446 in the original corpus. The additional unavailable code is
`295121000119101` (kidney disease). These are phenotype-specific reference entry
counts, rather than a count of pooled unique identifiers. See the detailed
comparison in `../../20261029/snomed_new_export/`.

Embedding cache reuse now requires matching descriptions as well as code and
vocabulary, including when selecting a subset of a larger cached corpus. Changed
wording rebuilds the embeddings; unchanged wording permits reuse. Streamlit's
resource cache keys now include the configured source paths and cache folders.

The historical study continues to use `configs/codemapper_experiment.yaml` and
the original corpus/cache directories. The current manuscript describes those
original results. Twenty-nine original corpus, configuration, study and manuscript
files were hash-checked and unchanged. See `protected_before.json` and
`activation_summary.json` for the verification records.

Validation: 20 targeted tests passed for imports, configuration, embedding cache
identity/progress, and the OpenAI embedding adapter. Python compilation and
`git diff --check` also passed. All tests used local data or dummy encoders;
no paid model calls were made.

## Embedding builds requested after activation

Both final SNOMED indexes have now been built in the active embedding cache
folder. Each contains all 382,170 codes: 182,903 vectors reused for exact
code/vocabulary/description matches and 199,267 vectors generated for changed
wording. No original vectors were overwritten.

| Model | Dimensions | Final cache filename |
| --- | ---: | --- |
| MiniLM | 384 | `embeddings_SNOMEDCT_US_sentence-transformers_all-MiniLM-L6-v2.npz` |
| OpenAI text-embedding-3-large | 3,072 | `embeddings_SNOMEDCT_US_openai_text-embedding-3-large_3072.npz` |

The OpenAI build used 390 successful requests and 1,987,574 input tokens, plus
two small independent validation requests. The sandbox initially blocked the
connection; the build resumed successfully with approved network access.
The MiniLM snapshot revision is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Five changed descriptions were independently re-embedded for each model. MiniLM
matched within the numerical tolerance. Fresh OpenAI calls showed small numerical
variation: the minimum cosine agreement was 0.99999916. An initial stricter
componentwise tolerance check failed; the measured differences and cosine
comparison are recorded in `openai_independent_validation.json`. The fresh
vectors were much closer to the new vectors than to vectors for the old wording.

`scripts/build_snomed_embeddings.py` provides a resumable build with exact text
matching, separate checkpoint files, a writer lock and atomic publication of the
final cache. Twenty-one targeted tests passed. All 29 protected original study
and manuscript files and four original SNOMED embedding files were checked by
SHA-256 and remained unchanged.

Both final archives reloaded successfully without rebuilding. Their metadata
matches the active corpus, every vector is finite and normalized, and all saved
vectors exactly match the corresponding build checkpoint. See
`embedding_build_summary.json` and the per-model validation reports.
