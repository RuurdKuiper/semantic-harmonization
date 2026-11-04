# Browser pipeline

Live app: https://ruurdkuiper.github.io/semantic-harmonization/

API: https://ruurd-semantic-harmonization-api.hf.space

Visitors can use the app without signing in, configuring connections or entering
access codes. They choose MiniLM or OpenAI for ranking. Lexical/semantic weights
are applied automatically from the study's validation selection: 0.50/0.50 for
MiniLM and 0.35/0.65 for OpenAI.

BM25, cosine scoring, normalization, ranking and reference scoring run in a Web
Worker on the visitor's computer. MiniLM also embeds the EDF in the browser.
OpenAI sends the EDF to the API to obtain a `text-embedding-3-large` query vector shortened to 256 dimensions;
Jev classification and GPT review send the EDF and the current candidate batch.
Provider keys and the Hugging Face dataset credential stay on the server.

The dataset `Ruurd/semantic-harmonization-minilm` remains private storage. With
`PIPELINE_PUBLIC_ACCESS=true`, the Space exposes its terminology assets to anyone
through the app's public download endpoints. It supplies temporary direct CDN links for vectors and streams metadata,
without keeping a vector index or generating terminology embeddings. Both models use precomputed
vectors for 617,077 codes across six terminologies. MiniLM packs total about
236 MiB using 8-bit storage (previously 912 MiB); the app's 256-dimensional, 8-bit OpenAI pack totals about 161 MiB. The 3,072-dimensional
full-precision study vectors and legacy packs remain unchanged. Only selected
terminologies download. Hash-verified chunks are saved in IndexedDB for reuse;
browser storage can be evicted. A bounded window prefetches up to four fragments in parallel; scoring preserves
corpus order. Cached fragments are reused without another download.

## Run locally

```sh
cd web
npm ci
npm run dev
```

Open the printed address, normally `http://127.0.0.1:5173`. The checked-in public
configuration connects to the hosted API automatically. Start with one small
terminology to reduce the initial download. Browser storage can be cleared in
the collapsed footer control. EDFs and decisions are not persisted; save exports
before closing the tab.

Classification runs sequential batches of 20 with a minimum of three batches,
ten consecutive zero-Narrow batches and a configurable 10,000-code ceiling.
Successful batches survive later failures within the tab; retry continues from
them. GPT review sends at most ten low-confidence decisions per batch. Step 4
keeps initial decisions; step 5 shows reviewed decisions and previous labels.
Changing the EDF, terminology selection or encoder invalidates results.

## Export precomputed terminology packs

From the repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/export_browser_embeddings.py
PYTHONPATH=. .venv/bin/python scripts/export_browser_embeddings.py --provider openai --dimensions 256 --dtype int8
```

The exporter checks the active corpus against saved vectors and refuses missing
or mismatched embeddings. It creates content-addressed packs, with 4,096-row
compact MiniLM/OpenAI chunks or 2,048-row full-precision OpenAI chunks by default. It never calls a provider to
generate corpus embeddings. MiniLM defaults to int8 packs in
`dist/browser-minilm-int8-assets`; use `--dtype fp32` for the original format.
Upload int8 MiniLM assets under `minilm-int8/`, compact OpenAI assets from
`dist/browser-openai-256-int8-assets` under `openai-256-int8/`, and legacy full-size
OpenAI assets under `openai/`, preserving the manifest's paths and raw compressed bytes.
Do not set `Content-Encoding: gzip` on `.json.gz` assets; the worker verifies their
hashes before decompressing. Older full-precision MiniLM manifests stay at the
dataset root for compatibility. To export full OpenAI packs explicitly, use
`--provider openai --dimensions 3072 --dtype fp32`. The int8 manifest lists obsolete fp32 vector
hashes so the browser can remove those cached vectors on upgrade, preserving
shared metadata and the other encoder's vectors. For an OpenAI upgrade, supply
`--previous-manifest` pointing to the legacy OpenAI manifest.

## Hosting

The frontend is published on the dedicated `browser-app-pages` branch. It
contains the interface, examples and MiniLM runtime, without vectors or secrets.
The GitHub Pages publishing source is GitHub Actions. Public build variables
`PIPELINE_API_URL` and `PIPELINE_MANIFEST_URL` supply the API base URL and MiniLM
manifest URL; the app's OpenAI manifest defaults to `/api/assets/openai-256-int8/manifest.json`
on that API. `web/public/config.json` provides defaults for local development.

The API runs as a Docker Hugging Face Space. To prepare its source and the
verified MiniLM dataset assets:

```sh
PYTHONPATH=. .venv/bin/python scripts/prepare_huggingface_deployment.py --account YOUR_ACCOUNT
```

The generated Dockerfile uses port 7860. The helper's `--upload` option checks
the intended account, uploads the listed files and sets configuration through
the Hugging Face API. Compact OpenAI packs are uploaded separately under `openai-256-int8/`.
Set `JEV_API_KEY`, `OPENAI_API_KEY` and a persistent read-only `HF_TOKEN` as Space secrets.
Initial classification uses Jev (`jev-latest`); GPT review uses OpenAI.
Configure these separately using `PIPELINE_CLASSIFIER_PROVIDER`,
`PIPELINE_CLASSIFIER_MODEL` and `PIPELINE_REVIEW_MODEL`.
Interactive OAuth login credentials are not copied into runtime secrets.
Set `PIPELINE_HF_DATASET`, `PIPELINE_ALLOWED_ORIGINS` and
`PIPELINE_PUBLIC_ACCESS=true` as Space variables. Existing tokens are unnecessary
for visitors in public mode. Legacy authenticated deployments can instead set
`PIPELINE_PUBLIC_ACCESS=false` and configure `PIPELINE_API_TOKEN`.

Anonymous paid endpoints have an in-process per-client limit of 60 requests per
minute and a shared daily limit of 10,000 requests, configurable using
`PIPELINE_MODEL_REQUESTS_PER_MINUTE` and `PIPELINE_DAILY_MODEL_REQUESTS`.
Counters reset when the Space restarts; they are not a billing quota. The browser
retries short rate-limit waits automatically. Provider/model choice for
classification and review stays on the server.

For a separate local API, install `backend/requirements.txt`, configure the
variables in `backend/.env.example`, then run:

```sh
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Set development URLs in `web/public/config.json` when using that local service.

## Retrieval compatibility and checks

MiniLM uses the pinned BERT-only Transformers.js runtime, full-precision
`Xenova/all-MiniLM-L6-v2`, mean pooling, L2 normalization and 256-token truncation.
Only stored corpus vectors are quantized: each row uses signed int8 components
scaled to its largest absolute component, followed by a little-endian float32
inverse norm. Cosine scores use the float32 query and normalized quantized row.
This uses 388 bytes per 384-dimensional vector instead of 1,536 bytes.
The online app uses the first 256 dimensions of existing `text-embedding-3-large`
vectors, renormalized and stored with the same per-row int8 format (260 bytes
per vector). Study settings and source vectors remain at 3,072 dimensions/fp32.
The embedding endpoint accepts an explicit dimension count; its legacy default
remains 3,072 so older clients can keep using the full-size pack.
BM25 follows the Python tokenizer, per-vocabulary statistics and repeated query
terms. Score normalization spans all selected vocabularies; combined scores are
rounded to six decimals, with corpus-order ties. EDF field selection matches the Python pipeline. Browser semantic queries contain
field values without generated headings or bullet prefixes; lexical queries
remain the preferred name and synonyms. Classification retains the full structured
EDF. The Python renderer keeps its historical labeled default for frozen-study
compatibility; browser OpenAI requests explicitly use `include_labels=False`. ONNX floating-point differences can alter close MiniLM ties;
this interface is not represented as a rerun of the frozen study.

```sh
cd web
npm test
npm run build
npx playwright test
```

Browser tests use controlled model and API responses without provider charges.
The inference check script can exercise either real encoder with the hosted
service. Python exporter and API checks are in `tests/test_browser_export.py`
and `tests/test_browser_api.py`.

The seven-EDF check on the current 617,077-code browser corpus (fixed 50/50
weighting, available Narrow references, validation EDFs only) found mean AP
0.6293607 for fp32 and 0.6293766 for int8. Top-100 sets matched for every EDF;
mean recall at 1,000 was 0.8768559 versus 0.8769131. This is a deployment check,
separate from the frozen manuscript results. Reproduce it after exporting both
formats with `PYTHONPATH=. .venv/bin/python scripts/validate_browser_quantization.py`.
The report records per-EDF metrics, manifest hashes and acceptance thresholds.

A separate seven-EDF query-label comparison used the same current corpus and
fixed weights. MiniLM int8 MAP was 0.6293766 labeled versus 0.6296649 values-only;
OpenAI fp32 MAP was 0.6516335 versus 0.6509634. Recall at 100 rose from 0.6501541
to 0.6603606 for MiniLM and from 0.6616180 to 0.6709790 for OpenAI. Recall at
1,000 fell from 0.8769131 to 0.8749818 and from 0.8776211 to 0.8746845. These are
mixed, small aggregate effects; values-only queries follow the app's requested
format. `scripts/compare_browser_query_labels.py` reproduces the comparison using
existing corpus checkpoints, with 14 small query embeddings per encoder. Its
report is separate from frozen study results. No corpus re-embedding or new
terminology download is needed for this query-format change.

For the compact OpenAI app variant, the seven validation EDFs at fixed 35/65
weighting gave MAP 0.6485793, recall at 1,000 of 0.8797354 and recall at 10,000
of 0.9603661. Full-size values were 0.6509634, 0.8746845 and 0.9698973. Kidney
disease recall at 10,000 fell from 0.7968 to 0.7376. The compact representation
was selected for the online app's practical download size after reviewing this
trade-off; the study retains full-size vectors. The comparison is reproducible
with `scripts/evaluate_openai_compaction.py`; it does not modify study outputs.
