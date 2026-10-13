"""CodeMapper must receive GPT-reviewed decisions and preserve stage snapshots."""
from dataclasses import asdict, replace
import json

import pandas as pd

import main
from src.llm.classify import ClassifiedCandidate, GPTReviewRunInfo
from src.utils.config import PipelineConfig


def test_codemapper_runs_after_review(monkeypatch):
    config = PipelineConfig()
    config.codemapper.enabled = True
    config.retrieval.vocabularies = ['ICD10CM']
    corpus = pd.DataFrame([{'code':'I40.0','description':'myocarditis','vocabulary':'ICD10CM'}])
    resource_key = json.dumps({'retrieval':asdict(config.retrieval),'sources':config.paths.code_systems,
                               'embeddings_dir':config.paths.embeddings_dir},sort_keys=True)
    resources = {resource_key:(object(),corpus,object())}
    original = ClassifiedCandidate('I40.0','myocarditis','ICD10CM','Narrow',.1,'initial')
    monkeypatch.setattr(main,'hybrid_retrieval',lambda *args,**kwargs: [original])
    monkeypatch.setattr(main,'llm_classify',lambda *args,**kwargs: [original])
    def review(items,*args,**kwargs):
        assert items[0].label == 'Narrow'
        return [replace(items[0],label='Exclude')], GPTReviewRunInfo(.3,300,1,1,1,1,0,'medium')
    monkeypatch.setattr(main,'gpt_review_low_confidence',review)
    def expand(items,*args):
        assert items[0].label == 'Exclude'
        return list(items), {'enabled':True,'new_codes':[],'promoted_codes':[]}
    monkeypatch.setattr(main,'expand_narrow_matches',expand)
    stages = []
    result = main.run_pipeline(config,stage_callback=lambda stage,records:stages.append(stage),
                               resource_cache=resources)
    assert stages == ['jev','gpt','codemapper']
    assert result['stage_decisions']['jev'][0]['label'] == 'Narrow'
    assert result['stage_decisions']['gpt'][0]['label'] == 'Exclude'
    assert original.label == 'Narrow'
