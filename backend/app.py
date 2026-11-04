"""Small classification API for the static browser app. No embedding dependencies.

Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from dataclasses import asdict, fields
from collections import deque
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.loaders import EventDefinitionForm
from src.llm.classify import llm_classify
from src.llm.client import LLMCallError, default_model_for, _get_openai_client

load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI(title="Semantic Harmonization classification API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "PIPELINE_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,https://ruurdkuiper.github.io"
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Retry-After"],
)


def authorize(authorization: Annotated[str | None, Header()] = None):
    if os.getenv("PIPELINE_PUBLIC_ACCESS", "").lower() == "true":
        return
    token = os.getenv("PIPELINE_API_TOKEN", "")
    if not token:
        raise HTTPException(503, "The classification service needs an access token configured by its administrator.")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "Enter a valid classification service access token.")


_model_windows = {}
_model_usage_lock = threading.Lock()
_daily_model_usage = {"day": None, "count": 0}


def model_rate_limit(request: Request):
    """Bound anonymous paid requests; settings are server-side only."""
    if os.getenv("PIPELINE_PUBLIC_ACCESS", "").lower() != "true":
        return
    now = time.time()
    client = request.client.host if request.client else "unknown"
    with _model_usage_lock:
        if _daily_model_usage["day"] != int(now // 86400):
            _daily_model_usage.update(day=int(now // 86400), count=0)
        if _daily_model_usage["count"] >= int(os.getenv("PIPELINE_DAILY_MODEL_REQUESTS", "10000")):
            raise HTTPException(429, "The shared daily model allowance has been reached. Please try again tomorrow.", headers={"Retry-After": "3600"})
        window = _model_windows.setdefault(client, deque())
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= int(os.getenv("PIPELINE_MODEL_REQUESTS_PER_MINUTE", "60")):
            raise HTTPException(429, "The model service is busy. Please retry shortly.", headers={"Retry-After": "60"})
        window.append(now)
        _daily_model_usage["count"] += 1
        if len(_model_windows) > 2048:
            for address in list(_model_windows):
                if not _model_windows[address] or _model_windows[address][-1] <= now - 60:
                    del _model_windows[address]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    code: str = Field(min_length=1, max_length=128)
    vocabulary: Literal["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    description: str = Field(min_length=1, max_length=4096)


class EventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edf: dict

    @model_validator(mode="after")
    def validate_edf(self):
        allowed = {field.name: field for field in fields(EventDefinitionForm) if field.name != "name"}
        if not set(self.edf) <= set(allowed):
            raise ValueError("Unknown EDF field")
        if len(json.dumps(self.edf)) > 65_536:
            raise ValueError("EDF is too large")
        for key, value in self.edf.items():
            # Accept cleared YAML fields, matching the browser's normalization.
            is_list = key in {"inclusion_criteria", "exclusion_criteria", "narrow_decision_rules", "possible_decision_rules",
                              "exclude_decision_rules", "synonyms", "candidate_anchor_codes", "ambiguities", "references"}
            if value is None:
                self.edf[key] = [] if is_list else ""
            elif is_list and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError(f"{key} must be a list of text values")
            elif not is_list and not isinstance(value, str):
                raise ValueError(f"{key} must be text")
        if not self.edf.get("preferred_name", "").strip():
            raise ValueError("preferred_name is required")
        return self


class Batch(EventRequest):
    candidates: list[Candidate] = Field(min_length=1, max_length=20)
    use_possible_category: bool = False

    @model_validator(mode="after")
    def validate_candidates(self):
        if len({(row.code, row.vocabulary) for row in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidates in batch")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "retrieval": ["browser-minilm", "browser-openai"]}


_ASSET_PATH = re.compile(
    r"(?:(?:openai|minilm-int8)/)?(?:manifest\.json|(?:ICD10CM|ICD9CM|ICPC|MDR|RCD2|SNOMEDCT_US)-[0-9a-f]{20}/"
    r"(?:lexical\.json\.gz|[0-9]{7}\.(?:json\.gz|f32|i8)))\Z"
)


@app.get("/api/assets/{asset_path:path}", dependencies=[Depends(authorize)])
def terminology_asset(asset_path: str):
    """Stream HF dataset assets without exposing the server's dataset credential."""
    if not _ASSET_PATH.fullmatch(asset_path):
        raise HTTPException(404, "Unknown terminology asset.")
    repo = os.getenv("PIPELINE_HF_DATASET", "")
    revision = os.getenv("PIPELINE_HF_REVISION", "main")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(r"[A-Za-z0-9_-]+", revision):
        raise HTTPException(503, "The Hugging Face terminology dataset is not configured.")
    if not os.getenv("HF_TOKEN"):
        raise HTTPException(503, "Private terminology downloads need a read-only HF_TOKEN configured in the Space secrets.")
    headers = {"Accept-Encoding": "identity"}
    if os.getenv("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    # HTTPX strips Authorization on cross-origin CDN redirects. Neither the HF
    # token nor provider keys are forwarded to the browser.
    client = httpx.Client(timeout=60, follow_redirects=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{asset_path}"
    public = os.getenv("PIPELINE_PUBLIC_ACCESS", "").lower() == "true"
    if public and asset_path.endswith((".f32", ".i8")):
        # Let the CDN serve large vectors directly. Only the temporary link to
        # this file reaches the browser; the HF access token stays server-side.
        try:
            resolved = client.head(url, headers=headers, follow_redirects=False)
            location = resolved.headers.get("location", "")
            target = httpx.URL(location)
            trusted = target.scheme == "https" and any(
                target.host == domain or target.host.endswith("." + domain)
                for domain in ("hf.co", "huggingface.co"))
            if resolved.is_redirect and trusted:
                client.close()
                # Signed links expire: cache the verified bytes in the browser,
                # never the redirect itself.
                return RedirectResponse(location, status_code=307, headers={"Cache-Control": "no-store"})
        except (httpx.HTTPError, httpx.InvalidURL):
            pass  # Retain streaming as a fallback when resolution is unavailable.
    try:
        response = client.send(client.build_request("GET",
            url, headers=headers), stream=True)
    except httpx.HTTPError:
        client.close()
        raise HTTPException(502, "The terminology download could not be reached. Please retry.") from None
    if response.status_code != 200:
        status = response.status_code
        response.close()
        client.close()
        raise HTTPException(404 if status == 404 else 502,
                            "The terminology asset is unavailable. The administrator should check the dataset access settings.")

    def close():
        response.close()
        client.close()

    def chunks():
        try:
            yield from response.iter_raw(chunk_size=65536)
        finally:
            close()

    # Public mode permits shared caching; authenticated deployments stay private.
    cache_control = ("public, max-age=60" if asset_path.endswith("manifest.json") else "public, max-age=31536000, immutable") if public else "private, no-store"
    return StreamingResponse(chunks(), media_type="application/json" if asset_path.endswith("manifest.json") else "application/octet-stream",
                             headers={"Cache-Control": cache_control, "X-Content-Type-Options": "nosniff"},
                             background=BackgroundTask(close))


def classify_batch(batch: Batch, review: bool):
    provider = "openai" if review else os.getenv("PIPELINE_CLASSIFIER_PROVIDER", "jev")
    if provider not in {"jev", "openai", "anthropic", "google"}:
        raise HTTPException(503, "The classification provider is not configured correctly.")
    model = os.getenv("PIPELINE_REVIEW_MODEL" if review else "PIPELINE_CLASSIFIER_MODEL") or default_model_for(provider)
    try:
        records = llm_classify(
            batch.candidates, EventDefinitionForm(name="browser-edf", **batch.edf),
            provider=provider, model=model, max_retries=3, batch_size=10 if review else 20,
            use_possible_category=batch.use_possible_category, reasoning_effort="medium", adaptive_stopping=None,
        )
    except (LLMCallError, ValueError, RuntimeError):
        logger.exception("Classification request failed")
        # Provider responses may contain sensitive diagnostic text. Keep it server-side.
        raise HTTPException(502, "The model request failed. Retry the batch or ask the service administrator to check its configuration.") from None
    expected = {(row.code, row.vocabulary) for row in batch.candidates}
    labels = {"Narrow", "Exclude", "Possible"} if batch.use_possible_category else {"Narrow", "Exclude"}
    keys = [(row.code, row.vocabulary) for row in records]
    if len(keys) != len(expected) or set(keys) != expected or any(
        row.label not in labels or not math.isfinite(row.confidence) or not 0 <= row.confidence <= 1
        or row.explanation == "No classification returned by LLM; routed for manual review."
        for row in records
    ):
        raise HTTPException(502, "The model returned an incomplete or invalid batch. Retry to continue.")
    return {"decisions": [asdict(row) for row in records], "provider": provider, "model": model}


@app.post("/api/classify", dependencies=[Depends(authorize), Depends(model_rate_limit)])
def classify(batch: Batch):
    return classify_batch(batch, review=False)


@app.post("/api/review", dependencies=[Depends(authorize), Depends(model_rate_limit)])
def review(batch: Batch):
    if len(batch.candidates) > 10:
        raise HTTPException(422, "Review accepts at most ten candidates per batch.")
    return classify_batch(batch, review=True)


@app.post("/api/embed", dependencies=[Depends(authorize), Depends(model_rate_limit)])
def embed(event: EventRequest):
    """Encode one EDF query; terminology vectors are always precomputed."""
    query = EventDefinitionForm(name="browser-edf", **event.edf).to_embedding_query(include_labels=False)
    try:
        response = _get_openai_client().embeddings.create(
            model="text-embedding-3-large", input=query, dimensions=3072, encoding_format="float")
    except Exception as exc:
        logger.warning("Embedding request failed (%s)", type(exc).__name__)
        raise HTTPException(502, "The event definition could not be embedded. Please retry; very long definitions may need shortening.") from None
    vector = response.data[0].embedding
    if len(vector) != 3072 or any(not math.isfinite(value) for value in vector):
        raise HTTPException(502, "The embedding service returned an invalid vector.")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-12:
        raise HTTPException(502, "The embedding service returned an empty vector.")
    return {"vector": [value / norm for value in vector], "model": "text-embedding-3-large", "dimensions": 3072}
