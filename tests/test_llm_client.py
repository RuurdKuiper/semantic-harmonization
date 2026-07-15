"""Tests for the shared multi-provider LLM client: provider resolution, JSON
extraction, and error handling for both Anthropic and OpenAI backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import (
    LLMCallError,
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
    with pytest.raises(LLMCallError):
        resolve_provider("auto")


def test_default_model_for():
    assert default_model_for("anthropic") == "claude-haiku-4-8"
    assert default_model_for("openai") == "gpt-5-mini"


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
