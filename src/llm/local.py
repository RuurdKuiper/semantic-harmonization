"""Local MLX reviewer for Apple silicon; no API calls or model substitution."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4


def visible_response(text):
    """Exclude reasoning, including a thinking block preopened by the template."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '</think>' in text:
        text = text.split('</think>',1)[1]
    if '<think>' in text:
        text = text.split('<think>',1)[0]
    return text.strip()


def first_response(text):
    """Read a complete JSON envelope, allowing subsequent prose or repeated output.

    Never repair incomplete JSON or extract a nested classification from a broken
    envelope. Generation and classification settings remain unchanged.
    """
    text=visible_response(text)
    start=text.find('{')
    if start<0:raise ValueError('No complete JSON envelope in local response')
    result,_=json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(result,dict):raise ValueError('Local response must be a JSON object')
    if not isinstance(result.get('classifications'),list):
        raise ValueError('Local response must contain a classifications array')
    return result


def candidate_parts(user_prompt):
    marker='Classify each of the following candidate codes:\n'
    if marker not in user_prompt:return None
    prefix,body=user_prompt.split(marker,1)
    lines,suffix=body.split('\n\nReturn a JSON object',1)
    candidates=lines.splitlines()
    if not candidates or any(not line.startswith('- code: ') for line in candidates):return None
    return prefix+marker,candidates,'\n\nReturn a JSON object'+suffix


def validate_candidates(response,user_prompt):
    parts=candidate_parts(user_prompt)
    if parts is None:return
    expected=[]
    for line in parts[1]:
        code,vocabulary,_=line.removeprefix('- code: ').split(' | ',2)
        expected.append((code,vocabulary.removeprefix('vocabulary: ')))
    actual=[(r.get('code'),r.get('vocabulary')) for r in response['classifications']]
    if len(actual)!=len(expected) or set(actual)!=set(expected):
        raise ValueError('Local answer does not cover every requested code exactly once')


@lru_cache(maxsize=1)
def load_mlx_reviewer(model_name):
    from mlx_vlm import load
    from src.llm.model_paths import prepared_model_path
    return load(prepared_model_path(model_name))


def call_mlx_json(system_prompt, user_prompt, model_name, max_tokens=8192, reasoning_effort='medium',
                  thinking_budget=1024):
    from mlx_vlm import stream_generate
    model, processor = load_mlx_reviewer(model_name)
    messages = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=reasoning_effort not in (None,'none'))
    folder=Path(os.environ.get('LOCAL_RESPONSE_DIR',Path(__file__).resolve().parents[2]/'results/local_responses'))
    folder.mkdir(parents=True,exist_ok=True)
    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    failure=folder/'failed'/f'{prompt_hash}.json'
    replay=json.loads(failure.read_text()) if failure.exists() else None
    if replay and (replay.get('model'),replay.get('max_tokens'),replay.get('thinking_budget'))!=(model_name,max_tokens,thinking_budget):
        replay=None
    if replay:
        print('[Qwen reviewer] Reusing audited failed output; subdividing its candidate batch.',flush=True)
    chunks = []; started = perf_counter(); reported = 0; generated_tokens=0
    finish_reason='model_stop_or_limit'
    outputs=() if replay else stream_generate(model, processor, prompt=prompt, max_tokens=max_tokens,
                                   temperature=.6, top_p=.95, top_k=20, seed=42,
                                   enable_thinking=reasoning_effort not in (None,'none'),
                                   thinking_budget=thinking_budget)
    for output in outputs:
        chunks.append(output.text)
        generated_tokens=output.generation_tokens
        if output.generation_tokens >= reported+256:
            reported = output.generation_tokens
            print(f'[Qwen reviewer] {reported} tokens generated, {perf_counter()-started:.1f}s',flush=True)
            current=''.join(chunks)
            if '</think>' in current or reasoning_effort in (None,'none'):
                try:
                    parsed=first_response(current)
                    validate_candidates(parsed,user_prompt)
                except json.JSONDecodeError as exc:
                    # An unescaped control character inside a JSON string cannot
                    # become valid by generating more tokens. Other incomplete
                    # prefixes remain eligible to finish normally.
                    if exc.msg.startswith('Invalid control character'):
                        finish_reason='invalid_json_syntax'
                        if hasattr(outputs,'close'):outputs.close()
                        break
                except (ValueError,TypeError):pass
                else:
                    finish_reason='complete_json'
                    if hasattr(outputs,'close'):outputs.close()
                    break
    text = replay['raw_response'] if replay else ''.join(chunks)
    path=folder/(uuid4().hex+'.json')
    audit={'model':model_name,'raw_response':text,'generation_tokens':generated_tokens,
           'max_tokens':max_tokens,'thinking_budget':thinking_budget,'seconds':perf_counter()-started,
           'prompt_sha256':prompt_hash,'replayed_failed_response':bool(replay),'finish_reason':finish_reason,
           'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        result=first_response(text)
        validate_candidates(result,user_prompt)
        audit['status']='parsed'
        return result
    except (ValueError,TypeError) as exc:
        audit.update(status='parse_error',error=str(exc))
        failure.parent.mkdir(exist_ok=True)
        failure.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        print(f'[Qwen reviewer] Raw failed response saved: {path}',flush=True)
        parts=candidate_parts(user_prompt)
        if parts and len(parts[1])>1:
            prefix,lines,suffix=parts;middle=len(lines)//2
            audit['recovery']='subdivide_batch'
            # Persist the failure before recovery, including if recovery is interrupted.
            path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
            print(f'[Qwen reviewer] Retrying failed batch as {middle} + {len(lines)-middle} candidates.',flush=True)
            recovered=[]
            for subset in (lines[:middle],lines[middle:]):
                answer=call_mlx_json(system_prompt,prefix+'\n'.join(subset)+suffix,model_name,
                                     max_tokens,reasoning_effort,thinking_budget)
                recovered.extend(answer['classifications'])
            result={'classifications':recovered}
            validate_candidates(result,user_prompt)
            audit['status']='recovered_by_subdivision'
            return result
        raise
    finally:
        path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
