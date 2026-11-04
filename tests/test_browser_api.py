from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.app import app
from src.llm.classify import ClassifiedCandidate


@pytest.fixture
def client(monkeypatch):
    from backend.app import _model_windows, _daily_model_usage
    _model_windows.clear()
    _daily_model_usage.update(day=None, count=0)
    monkeypatch.delenv("PIPELINE_PUBLIC_ACCESS", raising=False)
    monkeypatch.delenv("PIPELINE_CLASSIFIER_PROVIDER", raising=False)
    monkeypatch.delenv("PIPELINE_CLASSIFIER_MODEL", raising=False)
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


@pytest.mark.parametrize("stage,provider", [("classify", "jev"), ("review", "openai")])
def test_name_only_edf_batches_use_server_provider_without_embeddings(client, stage, provider):
    result = ClassifiedCandidate("A", "Migraine", "ICD10CM", "Narrow", .9, "Matches the event")
    with patch("backend.app.llm_classify", return_value=[result]) as classify:
        response = client.post(f"/api/{stage}", json=body(), headers={"Authorization": "Bearer test-service-token"})
        assert response.status_code == 200, response.text
        assert response.json()["decisions"][0]["explanation"] == "Matches the event"
        assert classify.call_args.kwargs["provider"] == provider
        assert classify.call_args.kwargs["model"] == ("jev-latest" if stage == "classify" else "gpt-6-luna")
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


@pytest.mark.parametrize("prefix,extension", [("", "f32"), ("minilm-int8/", "i8")])
def test_public_vectors_download_directly_without_exposing_hf_credentials(client, monkeypatch, prefix, extension):
    import httpx
    monkeypatch.setenv("PIPELINE_PUBLIC_ACCESS", "true")
    monkeypatch.setenv("PIPELINE_HF_DATASET", "owner/private-terminologies")
    monkeypatch.setenv("HF_TOKEN", "server-only-test-token")
    real_client = httpx.Client
    target = "https://cas-bridge.xethub.hf.co/vector?temporary-link=test"
    def upstream(request):
        assert request.method == "HEAD"
        assert request.headers["authorization"] == "Bearer server-only-test-token"
        return httpx.Response(302, headers={"Location": target})
    with patch("backend.app.httpx.Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs)):
        response = client.get("/api/assets/" + prefix + "ICPC-" + "a" * 20 + "/0000000." + extension, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == target
    assert response.headers["cache-control"] == "no-store"
    assert "server-only-test-token" not in str(response.headers)


def test_public_vector_resolution_failure_falls_back_to_streaming(client, monkeypatch):
    import httpx
    monkeypatch.setenv("PIPELINE_PUBLIC_ACCESS", "true")
    monkeypatch.setenv("PIPELINE_HF_DATASET", "owner/private-terminologies")
    monkeypatch.setenv("HF_TOKEN", "server-only-test-token")
    real_client = httpx.Client
    def upstream(request):
        if request.method == "HEAD":
            return httpx.Response(503)
        return httpx.Response(200, stream=httpx.ByteStream(b'vectors'))
    with patch("backend.app.httpx.Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs)):
        response = client.get("/api/assets/ICPC-" + "a" * 20 + "/0000000.f32")
    assert response.status_code == 200
    assert response.content == b'vectors'


def test_public_mode_allows_classification_without_user_credentials(client, monkeypatch):
    monkeypatch.setenv("PIPELINE_PUBLIC_ACCESS", "true")
    monkeypatch.delenv("PIPELINE_API_TOKEN", raising=False)
    record = ClassifiedCandidate("A", "Migraine", "ICD10CM", "Narrow", .9, "Matches")
    with patch("backend.app.llm_classify", return_value=[record]):
        assert client.post("/api/classify", json=body()).status_code == 200


def test_public_openai_embedding_encodes_only_semantic_edf_fields(client, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    monkeypatch.setenv("PIPELINE_PUBLIC_ACCESS", "true")
    create = Mock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[2.] + [0.] * 3071)]))
    sdk = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with patch("backend.app._get_openai_client", return_value=sdk):
        result = client.post("/api/embed", json={"edf": {"preferred_name": "Migraine", "exclusion_criteria": ["excluded concept"]}})
    assert result.status_code == 200
    assert result.json()["vector"] == [1.] + [0.] * 3071
    assert create.call_args.kwargs["input"] == "Migraine"
    assert create.call_args.kwargs["dimensions"] == 3072


def test_public_paid_model_requests_are_bounded_before_provider_call(client, monkeypatch):
    monkeypatch.setenv("PIPELINE_PUBLIC_ACCESS", "true")
    monkeypatch.setenv("PIPELINE_MODEL_REQUESTS_PER_MINUTE", "1")
    record = ClassifiedCandidate("A", "Migraine", "ICD10CM", "Narrow", .9, "Matches")
    with patch("backend.app.llm_classify", return_value=[record]) as classify:
        assert client.post("/api/classify", json=body()).status_code == 200
        response = client.post("/api/classify", json=body(), headers={"Origin": "https://ruurdkuiper.github.io"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert "Retry-After" in response.headers["access-control-expose-headers"]
    assert classify.call_count == 1
