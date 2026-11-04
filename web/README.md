# Browser MiniLM pipeline

Live frontend: https://ruurdkuiper.github.io/semantic-harmonization/

Classification API: https://ruurd-semantic-harmonization-api.hf.space

The terminology dataset is private under `Ruurd/semantic-harmonization-minilm`.
Downloads and model requests require the application access token. To choose or
rotate the colleague access code, update the Space secret `PIPELINE_API_TOKEN`.
The published build is on the dedicated `browser-app-pages` branch.

The GitHub Pages frontend runs BM25, MiniLM EDF embedding, cosine scoring,
normalization, ranking and reference scoring on the visitor's computer. A Web
Worker keeps retrieval off the interface thread. It never generates terminology
embeddings. The existing Streamlit application remains available separately.

Classification and GPT review use the Python API in `backend/`. Only the EDF and
the current candidate batch go to that service. Provider credentials stay on the
server; the frontend accepts an application access token, kept only in memory.
The API has no vector index, model download or terminology storage requirement. It can also stream protected packs from a private Hugging Face dataset without storing them.

## Run locally

From the repository root, prepare the existing MiniLM vectors for the browser:

```sh
PYTHONPATH=. .venv/bin/python scripts/export_browser_embeddings.py
```

This reads the active corpus and its precomputed MiniLM caches. It refuses
missing or mismatched vectors instead of creating new embeddings. It produces
content-addressed, integrity-checked packs in `dist/browser-assets`, with vector
chunks of at most 4,096 rows (6 MiB). The current six-terminology export contains
617,077 entries and approximately 912 MiB. Nothing in that output is committed.

Serve the packs in one terminal:

```sh
python3 scripts/serve_browser_assets.py
```

Run the frontend in another:

```sh
cd web
npm ci
npm run dev
```

Open the printed address, normally `http://127.0.0.1:5173`. In **Connections**, set
the terminology manifest URL to `http://127.0.0.1:8766/manifest.json`. Retrieval
works without an API service or API key. Start with one terminology to reduce
the first download; later searches reuse browser-cached, hash-verified chunks.
Browser storage can be evicted, especially in private mode or when disk space is
low. Only one vector chunk is needed in memory at a time. Metadata and ranking
scores for the selected corpus remain in memory.

To enable classification/review, install `backend/requirements.txt`, set its
documented environment variables, and run from the repository root:

```sh
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Set the classification service URL to `http://127.0.0.1:8000` and enter your
`PIPELINE_API_TOKEN` in Connections. **Do not enter a provider API key there.**
The API requires an application token and rejects requests if none is configured.
Provider/model selection is fixed on the server. OpenAI is the default for both classification and GPT review; Jev remains selectable on the server. Both model calls use the existing EDF prompts.

Classification runs sequential batches of 20 with the existing minimum of three
batches, ten consecutive zero-Narrow batches and a configurable 10,000-code cap.
Successful batches survive later request failures in the current tab; retry
continues from them. GPT review selects the lowest-confidence initial decisions
and sends at most ten per batch. Step 4 keeps initial decisions and step 5 shows
reviewed decisions with previous labels/confidence in separate columns.
Changing the EDF, terminology selection or retrieval weight invalidates results.
Save downloads before closing the tab: EDFs and decisions are not persisted.

## Host the services on Hugging Face

The supported Hugging Face setup keeps the terminology dataset **private** and
runs the small API as a Docker Space. Classification and review use OpenAI. The
Space's `OPENAI_API_KEY` and `HF_TOKEN` are secrets; only an application access
token is given to intended users. Browser downloads go through the protected
Space endpoint, which streams each file from the dataset without storing it.

Prepare both repositories without creating or uploading anything:

```sh
PYTHONPATH=. .venv/bin/python scripts/prepare_huggingface_deployment.py --account YOUR_ACCOUNT
```

This stages only the manifest's active, verified assets and the API's Python
source, dependencies and Dockerfile. It does not copy environment files or write
secrets. The generated Space uses port 7860.

After confirming the intended Hugging Face account, log in using that account,
set `OPENAI_API_KEY` and a long random `PIPELINE_API_TOKEN` in the environment,
and rerun with `--upload`. The script checks that the login matches the account,
creates or reuses a private dataset, uploads only the listed assets, and creates
a public Docker Space whose API and asset routes require the application token.
It sets secrets through Hugging Face's secrets API. A dedicated read-only dataset
token can be supplied through `PIPELINE_HF_READ_TOKEN`; otherwise a persistent personal access token can be used server-side. Interactive OAuth session credentials are not copied into Space secrets; when signed in through OAuth, add a persistent read-only `HF_TOKEN` directly in the Space settings. No token is printed or included in uploaded code.

The script returns the API URL and protected manifest URL. Use those two public
URLs for the GitHub Pages build variables below. The API URL will resemble
`https://ACCOUNT-semantic-harmonization-api.hf.space`; its manifest is at
`/api/assets/manifest.json`. Give authorized colleagues the application access
token separately; they do not need your OpenAI or Hugging Face credentials.
The Space must have an active runtime and outbound access to OpenAI and the Hub.
Do not assume a particular hardware tier or billing arrangement is available
until checked for the intended account.

## Publish on GitHub Pages

1. Publish the **contents** of `dist/browser-assets` on an HTTP object/static
   host. Preserve all paths and serve `.f32` and `.json.gz` as raw bytes. Do not
   set `Content-Encoding: gzip` for `.json.gz`: the worker decompresses it after
   checking its compressed-byte hash. Configure `Access-Control-Allow-Origin`
   for the Pages origin, and permit GET. Keep asset versions immutable;
   serve `manifest.json` with revalidation. Use HTTPS. Publish only terminology
   data you are authorized to distribute to the intended audience.
2. Deploy the classification API independently. For a container, build from the
   repository root with `docker build -f backend/Dockerfile -t phenotype-api .`.
   Configure `JEV_API_KEY`, `OPENAI_API_KEY`, `PIPELINE_API_TOKEN`, and
   `PIPELINE_ALLOWED_ORIGINS` on that host. The origin for this repository's
   default Pages URL is `https://ruurdkuiper.github.io` (no repository path).
   Provide the service access token to intended users separately. The service
   is stateless, accepts at most 20 candidates per request (10 for review), and
   never receives the full vector index. Add your hosting service's access and
   request limits before opening a shared paid service to a larger audience.
3. Set GitHub repository **Actions variables**, not secrets, for the public
   URLs: `PIPELINE_MANIFEST_URL` (including `/manifest.json`) and
   `PIPELINE_API_URL` (API base URL). Neither variable contains a credential.
4. Select **GitHub Actions** as the Pages publishing source and run the
   **Publish browser pipeline** workflow. It tests and builds `web/`, then
   deploys only the frontend, examples and browser runtime. Vectors and backend
   credentials are never included. Relative bundle paths support project Pages
   URLs such as `/semantic-harmonization/`.

The frontend can be built before either host exists, but production retrieval
needs a working manifest URL, and classification needs the API URL and access
token. Missing configuration is reported in Connections; it never falls back
to building the corpus. There is no anonymous, unmetered proxy to provider keys.

## Retrieval compatibility

The browser bundles only the BERT model and tokenizer components needed by MiniLM from the pinned Transformers.js version; unrelated generative model registries are excluded. The browser uses full-precision `Xenova/all-MiniLM-L6-v2` ONNX inference, mean
pooling, L2 normalization and a 256-token limit. Corpus packs are converted from
the saved Python MiniLM vectors into normalized little-endian float32 values.
BM25 uses the Python tokenizer conventions, per-vocabulary IDF/length statistics
and repeated query terms. Lexical and semantic scores are normalized across all
selected vocabularies, combined and rounded to six decimals, with corpus-order
ties. Query field selection matches the Python implementation. Backend/runtime
floating-point differences in ONNX query vectors can alter very close ties;
this is a new interface and is not represented as a rerun of the frozen study.

## Checks

```sh
cd web
npm test
npm run build
npx playwright test
```

Browser tests use controlled model/API responses so they do not download a model
or incur provider charges. Real MiniLM inference can be checked separately with
the locally served exported packs. Python exporter and API tests are in
`tests/test_browser_export.py` and `tests/test_browser_api.py`.
