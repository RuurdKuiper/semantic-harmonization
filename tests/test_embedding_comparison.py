"""Paid work must resume, preserve input identity, and score rows in corpus order."""
import base64
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline_design.embedding_cache import build_cache,cosine_scores,input_batches,request_vectors
from experiments.pipeline_design.embedding_comparison import Comparison
from experiments.pipeline_design.study import fingerprint


SETTINGS={'model':'text-embedding-3-large','dimensions':2,'batch_size':2,'max_batch_bytes':20,'max_input_bytes':10}


def client_with_failure(calls,fail_at=None):
    def create(**kwargs):
        calls.append(kwargs['input'])
        if len(calls)==fail_at:raise RuntimeError('Usage cap reached')
        vectors=[np.array([len(t),1],dtype=np.float32) for t in kwargs['input']]
        return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=base64.b64encode(v.tobytes()).decode())
                                     for i,v in reversed(list(enumerate(vectors)))],usage=SimpleNamespace(total_tokens=len(vectors)))
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_embedding_build_resumes_saved_batches_and_rejects_changed_input(tmp_path):
    frame=pd.DataFrame({'code':['a','b','c','d'],'description':['one','two','three','four'],'vocabulary':['V']*4})
    calls=[]
    with pytest.raises(RuntimeError):build_cache(frame,tmp_path,SETTINGS,client_with_failure(calls,2))
    assert json.loads((tmp_path/'state.json').read_text())['completed_rows']==2
    state=build_cache(frame,tmp_path,SETTINGS,client_with_failure(calls))
    assert calls==[['one','two'],['three','four'],['three','four']]
    assert state['complete'] and state['requests']==2 and state['total_tokens']==4
    assert np.allclose(np.linalg.norm(np.load(tmp_path/'vectors.npy'),axis=1),1)
    build_cache(frame,tmp_path,SETTINGS,client_with_failure(calls))
    assert len(calls)==3
    changed=frame.copy();changed.loc[0,'description']='different'
    with pytest.raises(ValueError,match='inputs changed'):build_cache(changed,tmp_path,SETTINGS,client_with_failure(calls))


def test_token_byte_bounds_split_batches_and_never_truncate():
    settings={**SETTINGS,'max_batch_bytes':5}
    assert list(input_batches(['abc','def','g'],0,settings))==[(0,1),(1,3)]
    with pytest.raises(ValueError):list(input_batches([''],0,settings))
    with pytest.raises(ValueError):list(input_batches(['a'*11],0,settings))


def test_scoring_keeps_vocabulary_row_order_and_normalized_cosines(tmp_path):
    folders=[tmp_path/'first',tmp_path/'second']
    for folder,vectors in zip(folders,([ [1,0],[0,1] ],[[.6,.8]])):
        folder.mkdir();np.save(folder/'vectors.npy',np.array(vectors,dtype=np.float32))
    scores=cosine_scores(folders,np.array([[1,0],[0,1]],dtype=np.float32),block_size=1)
    assert scores.shape==(3,2)
    assert np.allclose(scores,[[1,0],[0,1],[.6,.8]])


def test_malformed_embedding_response_is_rejected_before_checkpoint():
    bad=SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(
        data=[SimpleNamespace(index=9,embedding=[1,0])],usage=SimpleNamespace(total_tokens=1))))
    with pytest.raises(ValueError,match='exactly once'):request_vectors(bad,['term'],SETTINGS)


def test_baseline_reuse_requires_identical_models_weights_and_review_settings(tmp_path):
    comparison=Comparison.__new__(Comparison);comparison.source_output=tmp_path/'original'
    decision={'name':'jev','provider':'jev'};reviewer={'name':'gpt6_luna','provider':'openai'}
    settings={'confidence_threshold':.3,'max_candidates':300,'qwen_thinking_budget':1024}
    worker=SimpleNamespace(output=tmp_path/'new',source_signature=lambda _: 'same-inputs',
                           config={'classification':{'max_candidates':10000},'review':settings})
    sig=fingerprint({'source':'same-inputs','competitor':decision,'weight':.5,'stopping':worker.config['classification']})
    rsig=fingerprint({'decision':sig,'competitor':reviewer,'review':{k:v for k,v in settings.items() if k!='qwen_thinking_budget'}})
    for relative,signature in [('decisions/jev/edf.json',sig),('reviews/gpt6_luna/jev/edf.json',rsig)]:
        path=comparison.source_output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'signature':signature,'complete':True,'records':[]}))
    comparison.reuse_baseline(worker,'edf',.5,decision,reviewer)
    assert (worker.output/'reviews/gpt6_luna/jev/edf.json').exists()
    worker.config['review']={**settings,'confidence_threshold':.4}
    with pytest.raises(ValueError,match='cannot be reused exactly'):comparison.reuse_baseline(worker,'edf',.5,decision,reviewer)
