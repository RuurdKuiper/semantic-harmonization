"""Small classification API for the static browser app. No embedding dependencies.

Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hmac
import json
import logging
import math
import os
import re
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.loaders import EventDefinitionForm
from src.llm.classify import llm_classify
from src.llm.client import LLMCallError, default_model_for

load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI(title="Semantic Harmonization classification API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "PIPELINE_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,https://ruurdkuiper.github.io"
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
)


def authorize(authorization: Annotated[str | None, Header()] = None):
    token = os.getenv("PIPELINE_API_TOKEN", "")
    if not token:
        raise HTTPException(503, "The classification service needs an access token configured by its administrator.")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "Enter a valid classification service access token.")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    code: str = Field(min_length=1, max_length=128)
    vocabulary: Literal["ICD10CM", "ICD9CM", "ICPC", "MDR", "RCD2", "SNOMEDCT_US"]
    description: str = Field(min_length=1, max_length=4096)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edf: dict
    candidates: list[Candidate] = Field(min_length=1, max_length=20)
    use_possible_category: bool = False

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
        if len({(row.code, row.vocabulary) for row in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidates in batch")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "retrieval": "browser-minilm"}


_ASSET_PATH = re.compile(
    r"(?:manifest\.json|(?:ICD10CM|ICD9CM|ICPC|MDR|RCD2|SNOMEDCT_US)-[0-9a-f]{20}/"
    r"(?:lexical\.json\.gz|[0-9]{7}\.(?:json\.gz|f32)))\Z"
)


@app.get("/api/assets/{asset_path:path}", dependencies=[Depends(authorize)])
def terminology_asset(asset_path: str):
    """Stream private HF dataset assets to authorized browsers without storing them."""
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
    try:
        response = client.send(client.build_request("GET",
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{asset_path}", headers=headers), stream=True)
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

    # Private terminology data must not be cached by shared HTTP proxies.
    # The authorized browser maintains its own content-addressed IndexedDB cache.
    return StreamingResponse(chunks(), media_type="application/json" if asset_path == "manifest.json" else "application/octet-stream",
                             headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
                             background=BackgroundTask(close))


def classify_batch(batch: Batch, review: bool):
    provider = "openai" if review else os.getenv("PIPELINE_CLASSIFIER_PROVIDER", "openai")
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


@app.post("/api/classify", dependencies=[Depends(authorize)])
def classify(batch: Batch):
    return classify_batch(batch, review=False)


@app.post("/api/review", dependencies=[Depends(authorize)])
def review(batch: Batch):
    if len(batch.candidates) > 10:
        raise HTTPException(422, "Review accepts at most ten candidates per batch.")
    return classify_batch(batch, review=True)
