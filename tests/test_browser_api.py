from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.app import app
from src.llm.classify import ClassifiedCandidate


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PIPELINE_API_TOKEN", "test-service-token")
    return TestClient(app)


def body():
    return {"edf": {"preferred_name": "Migraine"}, "candidates": [{"code": "A", "vocabulary": "ICD10CM", "description": "Migraine"}]}


def test_api_requires_service_auth_before_model_calls(client):
    with patch("backend.app.llm_classify") as classify:
        assert client.post("/api/classify", json=body()).status_code == 401
        classify.assert_not_called()


def test_api_does_not_run_publicly_when_token_is_missing(client, monkeypatch):
    monkeypatch.delenv("PIPELINE_API_TOKEN")
    assert client.post("/api/classify", json=body()).status_code == 503


@pytest.mark.parametrize("stage,provider", [("classify", "openai"), ("review", "openai")])
def test_name_only_edf_batches_use_server_provider_without_embeddings(client, stage, provider):
    result = ClassifiedCandidate("A", "Migraine", "ICD10CM", "Narrow", .9, "Matches the event")
    with patch("backend.app.llm_classify", return_value=[result]) as classify:
        response = client.post(f"/api/{stage}", json=body(), headers={"Authorization": "Bearer test-service-token"})
        assert response.status_code == 200, response.text
        assert response.json()["decisions"][0]["explanation"] == "Matches the event"
        assert classify.call_args.kwargs["provider"] == provider
        assert classify.call_args.args[1].definition == ""


def test_invalid_or_incomplete_model_batch_is_rejected(client):
    with patch("backend.app.llm_classify", return_value=[]):
        response = client.post("/api/classify", json=body(), headers={"Authorization": "Bearer test-service-token"})
        assert response.status_code == 502


def test_batch_bounds_and_edf_types_are_validated(client):
    request = body()
    request["candidates"] *= 21
    assert client.post("/api/classify", json=request, headers={"Authorization": "Bearer test-service-token"}).status_code == 422
    request = body()
    request["edf"]["synonyms"] = "not a list"
    assert client.post("/api/classify", json=request, headers={"Authorization": "Bearer test-service-token"}).status_code == 422


def test_cors_allows_pages_origin(client):
    response = client.options("/api/classify", headers={"Origin": "https://ruurdkuiper.github.io", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ruurdkuiper.github.io"


def test_private_hf_assets_are_streamed_with_server_credentials_only(client, monkeypatch):
    import httpx

    monkeypatch.setenv("PIPELINE_HF_DATASET", "owner/private-terminologies")
    monkeypatch.setenv("HF_TOKEN", "private-hf-test-token")
    real_client = httpx.Client
    seen = []

    def upstream(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer private-hf-test-token"
        assert request.headers["accept-encoding"] == "identity"
        assert str(request.url) == "https://huggingface.co/datasets/owner/private-terminologies/resolve/main/manifest.json"
        return httpx.Response(200, stream=httpx.ByteStream(b'{"format":"test"}'))

    with patch("backend.app.httpx.Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs)):
        response = client.get("/api/assets/manifest.json", headers={"Authorization": "Bearer test-service-token"})
    assert response.status_code == 200
    assert response.json() == {"format": "test"}
    assert "authorization" not in response.headers
    assert response.headers["cache-control"] == "private, no-store"
    assert len(seen) == 1


def test_asset_download_requires_auth_and_does_not_allow_arbitrary_paths(client):
    with patch("backend.app.httpx.Client") as upstream:
        assert client.get("/api/assets/manifest.json").status_code == 401
        assert client.get("/api/assets/secret.env", headers={"Authorization": "Bearer test-service-token"}).status_code == 404
        upstream.assert_not_called()
