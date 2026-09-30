"""Tests for shared Anthropic, OpenAI, and Google LLM client behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import (
    LLMCallError,
    call_jev_decisions,
    call_llm_json,
    default_model_for,
    extract_json,
    resolve_provider,
)


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_surrounding_text():
    text = 'Here is the result:\n{"a": 1, "b": [1, 2]}\nThanks.'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_resolve_provider_explicit():
    assert resolve_provider("anthropic") == "anthropic"
    assert resolve_provider("openai") == "openai"
    assert resolve_provider("google") == "google"
    assert resolve_provider("jev") == "jev"


def test_resolve_provider_invalid_raises():
    with pytest.raises(LLMCallError):
        resolve_provider("bogus")


def test_resolve_provider_auto_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert resolve_provider("auto") == "anthropic"


def test_resolve_provider_auto_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert resolve_provider("auto") == "openai"


def test_resolve_provider_auto_skips_placeholder_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your-api-key-here")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert resolve_provider("auto") == "openai"


def test_resolve_provider_auto_no_keys_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(LLMCallError):
        resolve_provider("auto")


def test_default_model_for():
    assert default_model_for("anthropic") == "claude-haiku-4-8"
    assert default_model_for("openai") == "gpt-5-mini"
    assert default_model_for("google") == "gemini-3-flash-preview"
    assert default_model_for("jev") == "jev-1.13"


def test_resolve_provider_auto_falls_back_to_jev(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JEV_API_KEY", "fake")
    assert resolve_provider("auto") == "jev"


def test_call_llm_json_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMCallError):
        call_llm_json("system", "user")


def test_call_llm_json_success_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = '{"result": "ok"}'
    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("src.llm.client._get_anthropic_client", return_value=mock_client):
        result = call_llm_json("system", "user", max_retries=1)

    assert result == {"result": "ok"}


def test_call_llm_json_success_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    mock_message = MagicMock()
    mock_message.content = '{"result": "ok"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("src.llm.client._get_openai_client", return_value=mock_client):
        result = call_llm_json("system", "user", provider="openai", max_retries=1)

    assert result == {"result": "ok"}


def test_call_llm_json_success_google():
    mock_response = MagicMock()
    mock_response.text = '{"result": "ok"}'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.llm.client._get_google_client", return_value=mock_client):
        result = call_llm_json(
            "system",
            "user",
            provider="google",
            model="gemini-test",
            max_retries=1,
        )

    assert result == {"result": "ok"}
    call = mock_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-test"
    assert call.kwargs["contents"] == "user"
    assert call.kwargs["config"].system_instruction == "system"


def test_call_jev_decisions(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"answers":{"candidate_0":{"choice":"Narrow"}}}'
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("src.llm.client.urlopen", return_value=mock_context) as mocked_urlopen:
        result = call_jev_decisions(
            state="phenotype",
            questions={
                "candidate_0": {
                    "type": "choice",
                    "instructions": "Classify it",
                    "criteria": {"Narrow": "include", "Exclude": "exclude"},
                }
            },
            max_retries=1,
        )

    assert result["answers"]["candidate_0"]["choice"] == "Narrow"
    request = mocked_urlopen.call_args.args[0]
    assert request.full_url.endswith("/api/v1/systemone/")
    assert request.get_header("Authorization") == "Bearer fake-key"


def test_call_llm_json_rejects_jev_interface(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fake-key")
    with pytest.raises(LLMCallError, match="typed decisions"):
        call_llm_json("system", "user", provider="jev")


def test_call_llm_json_explicit_provider_overrides_auto(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    mock_message = MagicMock()
    mock_message.content = '{"result": "openai"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("src.llm.client._get_openai_client", return_value=mock_client):
        result = call_llm_json("system", "user", provider="openai", max_retries=1)

    assert result == {"result": "openai"}


def test_call_llm_json_retries_then_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("boom")

    with patch("src.llm.client._get_anthropic_client", return_value=mock_client):
        with pytest.raises(LLMCallError):
            call_llm_json("system", "user", max_retries=2)

    assert mock_client.messages.create.call_count == 2
