"""Typed provider schemas, confidence signals, and shared adaptive stopping."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm.client import LLMCallError
from src.llm.decisions import call_openai_decisions, normalize_answers
from src.llm.classify import AdaptiveStoppingConfig, llm_classify
from src.data.loaders import EventDefinitionForm


def test_openai_decisions_converts_questions_and_matches_names(monkeypatch):
    client=MagicMock();client.with_options.return_value=client
    client.post.return_value={'answers':[
        {'name':'second','type':'choice','choice':'Exclude','confidence':.2,
         'probabilities':[{'value':'Narrow','probability':.4},{'value':'Exclude','probability':.6}]},
        {'name':'first','type':'choice','choice':'Narrow','confidence':.9,
         'probabilities':[{'value':'Narrow','probability':.9},{'value':'Exclude','probability':.1}]}]}
    monkeypatch.setattr('src.llm.decisions._get_openai_client',lambda:client)
    questions={name:{'type':'choice','instructions':name,'criteria':{'Narrow':'specific','Exclude':'other'}}
               for name in ('first','second')}
    result=call_openai_decisions('EDF',questions)
    assert list(result['answers'])==['first','second']
    assert result['answers']['second']['confidence']==.2  # Separate provider confidence is preserved.
    assert client.post.call_args.args[0]=='/decisions'
    assert client.post.call_args.kwargs['body']['questions'][0]['choices'][0]=={'value':'Narrow','description':'specific'}


@pytest.mark.parametrize('answers', [[],[{'name':'x','type':'refusal'}],
    [{'name':'x','choice':'Narrow','probabilities':[{'value':'Narrow','probability':2}]}]])
def test_invalid_typed_answers_fail_instead_of_becoming_negative_codes(answers):
    with pytest.raises(LLMCallError):normalize_answers({'answers':answers},['x'])


def test_binary_max_probability_fallback_is_recorded():
    response={'answers':{'x':{'choice':'Narrow','probabilities':{'Narrow':.6,'Exclude':.4}}}}
    answer=normalize_answers(response,['x'])['answers']['x']
    assert answer['confidence']==.6
    assert answer['confidence_source']=='max_probability'


def test_openai_decisions_uses_the_same_twenty_code_stopping_batches(monkeypatch):
    calls=[]
    def decide(**kwargs):
        calls.append(kwargs)
        return {'answers':{key:{'choice':'Exclude','confidence':.9,'probabilities':{'Narrow':.1,'Exclude':.9}}
                           for key in kwargs['questions']}}
    monkeypatch.setattr('src.llm.decisions.call_decisions',decide)
    candidates=[SimpleNamespace(code=str(i),description='term',vocabulary='V') for i in range(100)]
    edf=EventDefinitionForm(name='x',preferred_name='X',definition='Current X')
    infos=[]
    results=llm_classify(candidates,edf,provider='openai_decisions',batch_size=20,
        adaptive_stopping=AdaptiveStoppingConfig(0,2,1),run_info_callback=infos.append)
    assert len(results)==40 and len(calls)==2
    assert infos[0].narrow_counts_by_batch==(0,0)
    assert infos[0].stopped_early
