"""Ranking selection, end-to-end scoring, and resumability of the study."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline_design.study import Study, choose_winner, confusion_metrics, ranking_metrics
from src.llm.classify import ClassifiedCandidate


def test_ranking_metrics_use_positive_rank_and_fixed_budgets():
    result=ranking_metrics(np.array([0,1,2,3]),np.array([False,True,False,True]),[2,4])
    assert result['average_precision']==.5
    assert result['recall_at_2']==.5 and result['recall_at_4']==1
    assert result['last_positive_rank']==4


def test_confusion_counts_include_unclassified_positives_and_exclude_unscored_vocabularies():
    corpus=pd.DataFrame([{'code':c,'vocabulary':'V','description':c} for c in 'abcd'])
    gold=pd.DataFrame([{'code':'a','vocabulary':'V','label':'Narrow'},
                       {'code':'c','vocabulary':'V','label':'Narrow'},
                       {'code':'missing','vocabulary':'V','label':'Narrow'}])
    result=confusion_metrics([{'code':'a','vocabulary':'V','label':'Narrow'},
                              {'code':'b','vocabulary':'V','label':'Exclude'},
                              {'code':'d','vocabulary':'V','label':'Narrow'},
                              {'code':'x','vocabulary':'Unscored','label':'Narrow'}],gold,corpus)
    assert (result['tp'],result['tn'],result['fp'],result['fn'])==(1,1,1,1)
    assert result['fn_unclassified']==1 and result['unavailable_narrow']==1


def test_selection_gives_cases_equal_weight_and_rejects_missing_results():
    rows=[{'name':name,'phenotype':case,'f1':score} for name,case,score in
          [('a','small',1.),('a','large',.2),('b','small',.7),('b','large',.6)]]
    assert choose_winner(rows,['small','large'])['name']=='b'
    with pytest.raises(ValueError):choose_winner(rows[:-1],['small','large'])


def test_interrupted_decision_run_resumes_without_repeating_completed_batches(tmp_path,monkeypatch):
    worker=Study.__new__(Study);worker.output=tmp_path;worker.requests=0
    worker.config={'execution':{'max_requests':50},'classification':{
        'batch_size':2,'minimum_batches':1,'consecutive_sparse_batches':2,'sparse_narrow_threshold':0}}
    worker.corpus=pd.DataFrame([{'code':str(i),'description':'term','vocabulary':'V'} for i in range(8)])
    worker.edf=lambda _:SimpleNamespace()
    worker.source_signature=lambda _:'unchanged'
    worker.ordered_candidates=lambda *_:(np.arange(8),np.ones(8),np.ones(8),np.ones(8))
    calls=[]
    def classify(batch,*args,**kwargs):
        calls.append([c.code for c in batch])
        if len(calls)==2:raise RuntimeError('Spending limit reached')
        return [ClassifiedCandidate(c.code,c.description,c.vocabulary,
                                    'Narrow' if c.code=='0' else 'Exclude',.9,'typed decision') for c in batch]
    monkeypatch.setattr('experiments.pipeline_design.study.llm_classify',classify)
    competitor={'name':'fake','provider':'jev','model':'fake','branch':'proprietary'}
    with pytest.raises(RuntimeError):worker.classify('edf',competitor,.5)
    result=worker.classify('edf',competitor,.5)
    assert calls==[['0','1'],['2','3'],['2','3'],['4','5']]
    assert result['complete'] and len(result['records'])==6
    assert result['narrow_counts']==[1,0,0]


def test_shared_ceiling_stops_a_model_that_keeps_predicting_narrow(tmp_path,monkeypatch):
    worker=Study.__new__(Study);worker.output=tmp_path;worker.requests=0
    worker.config={'execution':{'max_requests':50},'classification':{
        'batch_size':2,'minimum_batches':1,'consecutive_sparse_batches':2,
        'sparse_narrow_threshold':0,'max_candidates':4}}
    worker.corpus=pd.DataFrame([{'code':str(i),'description':'term','vocabulary':'V'} for i in range(8)])
    worker.edf=lambda _:SimpleNamespace();worker.source_signature=lambda _:'unchanged'
    worker.ordered_candidates=lambda *_:(np.arange(8),np.ones(8),np.ones(8),np.ones(8))
    monkeypatch.setattr('experiments.pipeline_design.study.llm_classify',lambda batch,*a,**k:[
        ClassifiedCandidate(c.code,c.description,c.vocabulary,'Narrow',.6,'choice') for c in batch])
    result=worker.classify('edf',{'name':'fake','provider':'laya','model':'fake'},.5)
    assert len(result['records'])==4 and result['complete']
    assert result['candidate_ceiling_hit'] and result['stop_reason']=='candidate_ceiling'
    assert not result['stopped_early']
