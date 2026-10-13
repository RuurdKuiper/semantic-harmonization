"""Adapters for typed decision providers using the shared Jev question schema.

Keep provider confidence separate from option probabilities. A fallback to the
maximum option probability is recorded explicitly; it cannot be below 0.5 for
a normalized binary choice. No domain calibration is inferred from that value.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from urllib.request import Request, urlopen
from typing import Any

from src.llm.client import LLMCallError, _get_openai_client, call_jev_decisions

DECISION_PROVIDERS = frozenset({'jev', 'openai_decisions', 'laya', 'jpt'})
DEFAULT_DECISION_MODELS = {
    'jev': 'jev-latest', 'openai_decisions': 'gpt-6-luna',
    'laya': 'multilingual', 'jpt': 'kirp/jpt-4b',
}


def normalize_answers(response, names):
    """Validate and normalize typed answers without inventing missing decisions."""
    raw = response.get('answers')
    if isinstance(raw, list):
        keys = [answer.get('name') for answer in raw]
        if len(set(keys)) != len(keys):
            raise LLMCallError('Decision provider returned duplicate answer names.')
        raw = {answer['name']: answer for answer in raw}
    if not isinstance(raw, dict) or any(name not in raw for name in names):
        raise LLMCallError('Decision provider omitted one or more requested answers.')
    answers = {}
    for name in names:
        answer = dict(raw[name])
        if answer.get('type') == 'refusal' or not isinstance(answer.get('choice'), str):
            raise LLMCallError(f'Decision provider did not answer question {name}.')
        probabilities = answer.get('probabilities', {})
        if isinstance(probabilities, list):
            probabilities = {item['value']: item['probability'] for item in probabilities}
        probabilities = {key: float(value) for key, value in probabilities.items()}
        if not probabilities or any(not 0 <= p <= 1 for p in probabilities.values()):
            raise LLMCallError(f'Invalid choice probabilities for question {name}.')
        if abs(sum(probabilities.values())-1) > .02 or answer['choice'] not in probabilities:
            raise LLMCallError(f'Invalid choice distribution for question {name}.')
        if isinstance(names, dict) and set(probabilities) != set(names[name]['criteria']):
            raise LLMCallError(f'Unexpected choices for question {name}.')
        confidence = answer.get('confidence')
        answer['confidence_source'] = 'provider' if confidence is not None else 'max_probability'
        confidence = float(confidence) if confidence is not None else max(probabilities.values())
        if not 0 <= confidence <= 1:
            raise LLMCallError(f'Invalid confidence for question {name}.')
        answers[name] = {**answer, 'confidence': confidence, 'probabilities': probabilities}
    return {**response, 'answers': answers}


def call_openai_decisions(state, questions, model='gpt-6-luna', max_retries=3):
    """POST /v1/decisions, including SDKs predating the typed resource."""
    client = _get_openai_client().with_options(max_retries=max_retries-1)
    payload = {'model': model, 'input': state, 'questions': [
        {'name': name, 'type': 'choice', 'instructions': question['instructions'],
         'choices': [{'value': label, 'description': description}
                     for label, description in question['criteria'].items()]}
        for name, question in questions.items()
    ]}
    response = client.post('/decisions', cast_to=dict[str, Any], body=payload)
    return normalize_answers(response, questions)


@lru_cache(maxsize=1)
def _laya_router():
    try:
        from laya import Router
    except ImportError as exc:
        raise LLMCallError('Install requirements-decision-models.txt to use Laya.') from exc
    from src.llm.model_paths import prepared_model_path
    path=prepared_model_path('convaiinnovations/laya')
    return Router(models={'multilingual':(path,'multilingual')})


def call_decisions(state, questions, provider='jev', model=None, max_retries=3):
    """Dispatch an existing typed choice request to the selected backend."""
    if provider not in DECISION_PROVIDERS:
        raise ValueError(f'Unsupported decision provider: {provider}')
    model = model or DEFAULT_DECISION_MODELS[provider]
    if provider == 'jev':
        return call_jev_decisions(state, questions, model=model, max_retries=max_retries)
    if provider == 'openai_decisions':
        return call_openai_decisions(state, questions, model=model, max_retries=max_retries)
    if provider == 'laya':
        # Explicit multilingual checkpoint avoids the English router's shorter
        # input limit. The model/card documents 8192-token inference support.
        result = _laya_router().predict(state, questions, model=model, max_len=8192)
        return normalize_answers(result, questions)
    if not os.environ.get('JPT_API_URL'):
        from src.llm.jpt_runtime import predict_jpt
        return normalize_answers(predict_jpt(state, questions, model), questions)
    url = os.environ.get('JPT_API_URL', 'http://127.0.0.1:8080/v1/systemone')
    payload = json.dumps({'state': state, 'questions': questions}).encode()
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('JPT_API_KEY')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = Request(url, data=payload, headers=headers, method='POST')
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except Exception as exc:
        raise LLMCallError('JPT decision endpoint failed. Start llm2jev before running.') from exc
    return normalize_answers(result, questions)
