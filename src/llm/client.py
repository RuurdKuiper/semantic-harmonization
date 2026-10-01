"""Shared model clients for generative JSON calls and Jev typed decisions.

Automatic provider selection preserves the original Anthropic-then-OpenAI
precedence; Google is selected explicitly by experiments that require it.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.utils.logging import get_logger

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-8"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_JEV_API_URL = "https://api.typesafe.ai/v1/systemone"

_PLACEHOLDER_KEY_VALUES = {"", "your-api-key-here"}


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails or returns unparsable output after retries."""


def _has_real_key(env_var: str) -> bool:
    """Return True if *env_var* is set to a non-empty, non-placeholder value.

    Guards against `.env` files copied from `.env.example` where a key is
    present but still holds the literal placeholder text.
    """
    value = os.environ.get(env_var, "").strip()
    return value.lower() not in _PLACEHOLDER_KEY_VALUES


def resolve_provider(provider: str = "auto") -> str:
    """Resolve the LLM provider to use, based on explicit choice or available API keys.

    Parameters
    ----------
    provider : str
        One of ``"auto"``, ``"anthropic"``, ``"openai"``, ``"google"``, or ``"jev"``.

    Returns
    -------
    str
        The resolved provider name.
    """
    provider = (provider or "auto").lower()
    if provider in {"anthropic", "openai", "google", "jev"}:
        return provider
    if provider != "auto":
        raise LLMCallError(
            f"Unknown LLM provider: {provider!r}. "
            "Use 'auto', 'anthropic', 'openai', 'google', or 'jev'."
        )

    if _has_real_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _has_real_key("OPENAI_API_KEY"):
        return "openai"
    if _has_real_key("JEV_API_KEY"):
        return "jev"
    raise LLMCallError(
        "No LLM API key found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in your .env file."
    )


def default_model_for(provider: str) -> str:
    """Return the default model identifier for the given provider."""
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "google":
        return DEFAULT_GOOGLE_MODEL
    if provider == "jev":
        return DEFAULT_JEV_MODEL
    return DEFAULT_OPENAI_MODEL


def _get_anthropic_client():
    """Lazily construct the Anthropic client so importing this module does not
    require an API key to be set (e.g. for unit tests that mock calls)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMCallError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.Anthropic(api_key=api_key)


def _get_openai_client():
    """Lazily construct the OpenAI client so importing this module does not
    require an API key to be set (e.g. for unit tests that mock calls)."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMCallError("OPENAI_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key)


def _get_google_client():
    """Lazily construct the Google Gen AI client."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMCallError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in *text*.

    Handles cases where the model wraps JSON in markdown code fences or adds
    surrounding commentary despite instructions not to.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(text)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]!r}")


def _call_anthropic(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Whether *model* is an OpenAI reasoning model (GPT-5/6 and o-series).

    Reasoning models spend part of their completion token budget on hidden
    reasoning tokens before producing visible output, and support a
    ``reasoning_effort`` parameter to control that trade-off. Non-reasoning
    models (e.g. gpt-4o) don't accept that parameter, so it must only be sent
    for models in this family.
    """
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    client = _get_openai_client()
    kwargs: dict[str, Any] = {}
    if reasoning_effort and _is_reasoning_model(model):
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content.strip():
        # Most commonly caused by a reasoning model spending its entire
        # max_completion_tokens budget on hidden reasoning, leaving nothing
        # for the visible answer (finish_reason == "length" in that case).
        raise ValueError(
            f"Empty response content from model '{model}' "
            f"(finish_reason={choice.finish_reason!r}, usage={response.usage!r}). "
            "Consider increasing max_tokens or lowering reasoning_effort."
        )
    return content


def _call_google(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    from google.genai import types

    client = _get_google_client()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text or ""
    if not text.strip():
        raise ValueError(f"Empty response content from Google model '{model}'.")
    return text


def call_jev_decisions(
    state: str,
    questions: dict[str, dict[str, Any]],
    model: str = DEFAULT_JEV_MODEL,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Submit typed questions to Jev and return its structured response.

    Jev is not a generative chat model, so it does not use
    :func:`call_llm_json`. A request shares one phenotype state across up to 20
    candidate-classification questions and receives declared labels plus
    probability distributions.
    """
    api_key = (
        os.environ.get("TYPESAFE_API_KEY", "").strip()
        or os.environ.get("JEV_API_KEY", "").strip()
    )
    if not api_key:
        raise LLMCallError(
            "TYPESAFE_API_KEY or JEV_API_KEY environment variable is not set."
        )
    if not questions or len(questions) > 20:
        raise ValueError("Jev requests must contain between 1 and 20 questions.")

    url = os.environ.get("JEV_API_URL", DEFAULT_JEV_API_URL).strip() or DEFAULT_JEV_API_URL
    payload = json.dumps(
        {"model": model, "state": state, "questions": questions},
        separators=(",", ":"),
    ).encode("utf-8")
    idempotency_key = str(uuid.uuid5(uuid.NAMESPACE_URL, payload.decode("utf-8")))
    retryable_statuses = {429, 502, 503, 504, 529}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        request = Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result.get("answers"), dict):
                raise ValueError("Jev response is missing an 'answers' object.")
            return result
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001 - preserve the original HTTP status
                error_body = ""
            last_error = RuntimeError(
                f"Jev HTTP {exc.code} from {url}"
                + (f": {error_body}" if error_body else "")
            )
            if exc.code not in retryable_statuses or attempt >= max_retries:
                break
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = min(float(retry_after), 30.0) if retry_after else min(2 ** (attempt - 1), 8)
            time.sleep(delay)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise LLMCallError(
        f"Jev decision call failed after {max_retries} attempts: {last_error}"
    ) from last_error


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    provider: str = "auto",
    model: str | None = None,
    max_retries: int = 3,
    max_tokens: int = 8192,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Call the configured LLM provider and parse a JSON object from the response,
    retrying on transient failures or malformed JSON output.

    Parameters
    ----------
    system_prompt : str
        System prompt establishing the LLM's role and output format.
    user_prompt : str
        The task-specific user prompt.
    provider : str
        ``"auto"``, ``"anthropic"``, ``"openai"``, or ``"google"``. ``"auto"`` selects
        Anthropic if ``ANTHROPIC_API_KEY`` is set, otherwise OpenAI.
    model : str, optional
        Model identifier. Defaults to a sensible default for the resolved provider.
    max_retries : int
        Number of attempts before raising ``LLMCallError``.
    max_tokens : int
        Maximum completion tokens to generate. For OpenAI reasoning models
        (gpt-5/o1/o3/o4), this budget is shared with hidden reasoning tokens,
        so it should be generous enough to leave room for the visible answer.
    reasoning_effort : str, optional
        Passed as ``reasoning_effort`` to OpenAI reasoning models only (e.g.
        ``"low"``/``"medium"``/``"high"``); ignored for Anthropic and for
        non-reasoning OpenAI models.

    Returns
    -------
    dict
        Parsed JSON response body.
    """
    resolved_provider = resolve_provider(provider)
    if resolved_provider == "jev":
        raise LLMCallError(
            "Jev uses typed decisions rather than generative JSON. "
            "Call call_jev_decisions(), or use llm_classify(provider='jev')."
        )
    resolved_model = model or default_model_for(resolved_provider)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if resolved_provider == "anthropic":
                text = _call_anthropic(system_prompt, user_prompt, resolved_model, max_tokens)
            elif resolved_provider == "openai":
                text = _call_openai(
                    system_prompt, user_prompt, resolved_model, max_tokens, reasoning_effort
                )
            else:
                text = _call_google(system_prompt, user_prompt, resolved_model, max_tokens)
            return extract_json(text)
        except Exception as exc:  # noqa: BLE001 - broad by design, retried below
            last_error = exc
            logger.warning(
                "LLM call attempt %d/%d (provider=%s) failed: %s",
                attempt, max_retries, resolved_provider, exc,
            )

    raise LLMCallError(
        f"LLM call failed after {max_retries} attempts (provider={resolved_provider}): {last_error}"
    ) from last_error
