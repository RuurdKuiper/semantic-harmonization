"""Concurrent publication must retain both branches and preserve frozen settings."""
from copy import deepcopy
import json
import multiprocessing

import pandas as pd
import pytest

from experiments.pipeline_design import study as study_module
from experiments.pipeline_design.parallel_benchmark import (
    atomic_json,branch_execution,merge_metrics,update_status,
)


def publish_cases(path,branch):
    for i in range(6):
        merge_metrics(path,pd.DataFrame([{'branch':branch,'subset':'full','phenotype':str(i),'score':i}]))


def test_processes_merge_both_branches_without_lost_or_duplicate_rows(tmp_path):
    path=tmp_path/'benchmark_metrics.csv'
    context=multiprocessing.get_context('spawn')
    workers=[context.Process(target=publish_cases,args=(path,branch)) for branch in ('proprietary','open_weight')]
    for worker in workers:worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode==0
    frame=pd.read_csv(path)
    assert len(frame)==12 and not frame.duplicated(['branch','subset','phenotype']).any()
    merge_metrics(path,pd.DataFrame([{'branch':'proprietary','subset':'full','phenotype':'0','score':99}]))
    frame=pd.read_csv(path)
    assert len(frame)==12
    assert frame[(frame.branch=='proprietary')&(frame.phenotype==0)].score.iloc[0]==99


def test_branch_adapter_preserves_frozen_parameters_and_restores_globals(tmp_path,monkeypatch):
    frozen={'branches':{'proprietary':{'decision':{'model':'jev'}},'open_weight':{'decision':{'model':'jpt'}}},
            'classification':{'max_candidates':10000},'review':{'confidence_threshold':.3}}
    expected=deepcopy(frozen)
    original=study_module.Study;original_csv=pd.DataFrame.to_csv
    monkeypatch.setattr(original,'frozen',lambda self:frozen)
    with branch_execution(tmp_path,'proprietary') as cls:
        result=cls.__new__(cls).frozen()
        assert list(result['branches'])==['proprietary']
        assert result['classification']==expected['classification'] and result['review']==expected['review']
        pd.DataFrame([{'branch':'proprietary','subset':'full','phenotype':'a','score':1}]).to_csv(tmp_path/'benchmark_metrics.csv',index=False)
        with pytest.raises(ValueError):
            pd.DataFrame([{'branch':'open_weight','subset':'read','phenotype':'a'}]).to_csv(tmp_path/'benchmark_metrics.csv',index=False)
    assert frozen==expected and study_module.Study is original and pd.DataFrame.to_csv is original_csv
    assert len(pd.read_csv(tmp_path/'benchmark_metrics.csv'))==1


def test_worker_status_requires_both_branches_for_completion(tmp_path):
    update_status(tmp_path,'proprietary','running')
    update_status(tmp_path,'open_weight','running')
    update_status(tmp_path,'proprietary','complete')
    assert json.loads((tmp_path/'status.json').read_text())['status']=='running'
    update_status(tmp_path,'open_weight','incomplete',error='Usage cap')
    assert json.loads((tmp_path/'status.json').read_text())['status']=='incomplete'
    update_status(tmp_path,'open_weight','complete')
    assert json.loads((tmp_path/'status.json').read_text())['status']=='complete'
    atomic_json(tmp_path/'deep/result.json',{'checkpoint':'saved'})
    assert json.loads((tmp_path/'deep/result.json').read_text())=={'checkpoint':'saved'}
