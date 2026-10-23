"""Complete local answers remain usable when a model continues generating."""
import pytest
import json
import sys
from types import SimpleNamespace
from src.llm import local
from src.llm.local import first_response


def test_repeated_envelope_keeps_first_complete_answer():
    answer='{"classifications": [{"code": "x", "label": "Narrow"}]}'
    assert first_response('<think>{"draft": true}</think>```json\n'+answer+'\n```\n'+answer)=={
        'classifications':[{'code':'x','label':'Narrow'}]}


@pytest.mark.parametrize('text',[
    '<think>{"classifications": []}',
    '{"classifications": [{"code": "x"}',
    '{"classifications": [{"code": "x"} BROKEN]}'
])
def test_incomplete_or_reasoning_only_output_is_rejected(text):
    with pytest.raises(ValueError):first_response(text)


def test_template_preopened_thinking_is_removed():
    assert first_response('reasoning with {braces}</think>{"classifications": []}')=={'classifications':[]}


@pytest.mark.parametrize('raw,valid', [('{"classifications": []}',True),('{"classifications": [',False)])
def test_generation_audits_success_and_failure_without_changing_settings(monkeypatch,tmp_path,raw,valid):
    arguments={}
    def stream(*args,**kwargs):
        arguments.update(kwargs)
        yield SimpleNamespace(text=raw,generation_tokens=12)
    processor=SimpleNamespace(apply_chat_template=lambda *args,**kwargs:'rendered prompt')
    monkeypatch.setattr(local,'load_mlx_reviewer',lambda _:('model',processor))
    monkeypatch.setitem(sys.modules,'mlx_vlm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setenv('LOCAL_RESPONSE_DIR',str(tmp_path))
    if valid:assert local.call_mlx_json('system','user','local-model')=={'classifications':[]}
    else:
        with pytest.raises(ValueError):local.call_mlx_json('system','user','local-model')
    audit=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert audit['raw_response']==raw and audit['generation_tokens']==12
    assert audit['status']==('parsed' if valid else 'parse_error')
    assert {key:arguments[key] for key in ('temperature','top_p','top_k','seed','max_tokens','thinking_budget')}=={
        'temperature':.6,'top_p':.95,'top_k':20,'seed':42,'max_tokens':8192,'thinking_budget':1024}


def test_failed_batch_subdivision_preserves_all_requested_identifiers(monkeypatch,tmp_path):
    prompt=('EDF unchanged\nClassify each of the following candidate codes:\n'+
            '\n'.join(f'- code: {i} | vocabulary: V | description: item' for i in range(4))+
            '\n\nReturn a JSON object with classifications.')
    seen=[]
    def stream(*args,**kwargs):
        rendered=kwargs['prompt'];lines=local.candidate_parts(rendered)[1];seen.append(len(lines))
        if len(lines)>2:raw='{"classifications": ['
        else:raw=json.dumps({'classifications':[{'code':line.split(' | ')[0][8:],
                                                'vocabulary':'V','label':'Narrow'} for line in lines]})
        yield SimpleNamespace(text=raw,generation_tokens=5)
    processor=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:messages[1]['content'])
    monkeypatch.setattr(local,'load_mlx_reviewer',lambda _:('model',processor))
    monkeypatch.setitem(sys.modules,'mlx_vlm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setenv('LOCAL_RESPONSE_DIR',str(tmp_path))
    result=local.call_mlx_json('system',prompt,'local-model')
    assert seen==[4,2,2]
    assert [r['code'] for r in result['classifications']]==['0','1','2','3']
    audits=[json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert any(a['status']=='recovered_by_subdivision' for a in audits)
    seen.clear();local.call_mlx_json('system',prompt,'local-model')
    assert seen==[2,2]  # Audited malformed envelope is not generated again.


def test_missing_candidates_are_not_silently_converted_to_negative_decisions():
    prompt='Classify each of the following candidate codes:\n- code: x | vocabulary: V | description: item\n\nReturn a JSON object'
    with pytest.raises(ValueError):local.validate_candidates({'classifications':[]},prompt)


@pytest.mark.parametrize('response,valid', [('{"classifications": []}',True),
    ('{"classifications": [{"explanation": "bad\nstring',False)])
def test_stream_stops_at_complete_answer_or_irreparable_syntax(monkeypatch,tmp_path,response,valid):
    closed=[]
    def stream(*args,**kwargs):
        try:
            yield SimpleNamespace(text='reasoning</think>'+response,generation_tokens=256)
            raise AssertionError('Generation should have stopped')
        finally:closed.append(True)
    processor=SimpleNamespace(apply_chat_template=lambda *args,**kwargs:'prompt')
    monkeypatch.setattr(local,'load_mlx_reviewer',lambda _:('model',processor))
    monkeypatch.setitem(sys.modules,'mlx_vlm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setenv('LOCAL_RESPONSE_DIR',str(tmp_path))
    if valid:local.call_mlx_json('system','user','local-model')
    else:
        with pytest.raises(ValueError):local.call_mlx_json('system','user','local-model')
    assert closed==[True]
    audit=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert audit['finish_reason']==('complete_json' if valid else 'invalid_json_syntax')
