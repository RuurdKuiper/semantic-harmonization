"""Mapping changes must be traceable, bounded to corpus, and cross-vocabulary."""
import json
import io
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

import pandas as pd

from src.llm.classify import ClassifiedCandidate
from src.mapping.codemapper import (CodeMapperClient, CodeMapperError, expand_narrow_matches,
                                  normalize_code_versions, normalize_concept_versions)
from src.utils.config import CodeMapperConfig
from src.evaluation.metrics import to_predicted_codelist
from dataclasses import asdict


def candidate(code, voc, label="Narrow"):
    return ClassifiedCandidate(code, code, voc, label, .9, "original")


class FakeClient:
    calls = []
    cache_hits = 0
    def __init__(self):
        self.seeds = []
        self.closed = False
    def cuis_for_codes(self, vocabulary, codes):
        self.seeds.extend(codes)
        return {c: {"cuis": ["C1"]} for c in codes if c != "unmapped"}
    def concepts(self, cuis, targets):
        return [{"cui": "C1", "sourceConcepts": [
            {"codingSystem": "ICD10CM@2026", "id": "I40.0"},
            {"codingSystem": "ICD10CM@2026", "id": "I40.9"},
            {"codingSystem": "SNOMEDCT_US@2025_09_01", "id": "1"},
            {"codingSystem": "SNOMEDCT_US@2025_09_01", "id": "2"},
            {"codingSystem": "SNOMEDCT_US@2025_09_01", "id": "3"},
            {"codingSystem": "SNOMEDCT_US@2025_09_01", "id": "not-in-corpus"},
        ]}]
    def close(self):
        self.closed = True


def test_expansion_adds_and_promotes_without_mutating_baseline_or_recursing():
    baseline = [candidate("I40.0", "ICD10CM"), candidate("1", "SNOMEDCT_US", "Exclude")]
    corpus = pd.DataFrame([{"code":c,"vocabulary":v,"description":c}
                          for c,v in [("I40.0","ICD10CM"),("I40.9","ICD10CM"),
                                      ("1","SNOMEDCT_US"),("2","SNOMEDCT_US"),("3","SNOMEDCT_US")]])
    client = FakeClient()
    merged, info = expand_narrow_matches(baseline, corpus, ["ICD10CM","SNOMEDCT_US"],
                                        CodeMapperConfig(), client=client)
    assert baseline[1].label == "Exclude"
    assert {(c.code,c.vocabulary) for c in merged if c.label == "Narrow"} == {
        ("I40.0","ICD10CM"),("1","SNOMEDCT_US"),("2","SNOMEDCT_US"),("3","SNOMEDCT_US")}
    assert merged[1].prior_label == "Exclude"
    assert merged[1].mapping_provenance[0]["seed_code"] == "I40.0"
    assert merged[1].confidence == 0
    assert len(info["new_codes"]) == 2
    assert len(info["promoted_codes"]) == 1
    assert len(info["unavailable_codes"]) == 1
    assert client.seeds == ["I40.0"]
    assert client.closed and not info["descendants_used"]


def test_no_narrow_seeds_make_no_mapping_calls():
    client = FakeClient()
    client.concepts = lambda *args: (_ for _ in ()).throw(AssertionError("should not call"))
    corpus = pd.DataFrame(columns=["code","vocabulary","description"])
    baseline = [candidate("I40.0","ICD10CM","Exclude")]
    merged, info = expand_narrow_matches(baseline, corpus, ["ICD10CM"], CodeMapperConfig(),client=client)
    assert merged == baseline and info['seeds'] == 0


def test_cache_reuses_successful_and_empty_code_mappings(tmp_path):
    config = CodeMapperConfig(cache_dir=str(tmp_path), batch_size=2)
    client = CodeMapperClient(config)
    with patch.object(client,'_login'), patch.object(client,'_request',return_value={"A": {"cuis":["C1"]}}) as request:
        assert client.cuis_for_codes("V@1",["A","B"])["A"]["cuis"] == ["C1"]
        assert client.cuis_for_codes("V@1",["B"]) == {}
        assert request.call_count == 1
        assert all('session' not in p.read_text() for p in tmp_path.glob('*.json'))
        # Versions are part of the cache identity.
        client.cuis_for_codes("V@2",["A"])
        assert request.call_count == 2


def test_current_api_code_versions_use_only_the_pinned_release(tmp_path):
    response={'A':[{'version':'2027','term':'changed','cuis':['WRONG']},
                   {'version':'2026','term':'original','cuis':['C1']}],
              'B':[{'version':'2027','term':'new code','cuis':['C2']}]}
    client=CodeMapperClient(CodeMapperConfig(cache_dir=str(tmp_path)))
    with patch.object(client,'_login'),patch.object(client,'_request',return_value=response) as request:
        result=client.cuis_for_codes('V@2026',['A','B'])
    assert result['A']['cuis']==['C1'] and 'B' not in result
    assert request.call_args.kwargs['form']=={'vocId':'V','codes':['A','B']}
    assert client.cuis_for_codes('V@2026',['B'])=={}  # Version gaps are cached as empty.


def test_current_api_concepts_exclude_other_versions_and_keep_requested_terms():
    response=[{'cui':'C1','sourceConcepts':[
        {'vocId':'V','id':'A','preferredTerm':'latest','versions':[
            {'version':'2027','term':'latest'},{'version':'2026','term':'pinned'}]},
        {'vocId':'V','id':'B','preferredTerm':'new','versions':[{'version':'2027','term':'new'}]},
        {'vocId':'Unrequested','id':'C','versions':[{'version':'2026','term':'other'}]}]}]
    result=normalize_concept_versions(response,['V@2026'])
    assert [(s['id'],s['codingSystem'],s['preferredTerm']) for s in result[0]['sourceConcepts']]==[('A','V@2026','pinned')]


def test_legacy_release_specific_cached_response_remains_compatible():
    value={'A':{'term':'original','cuis':['C1']}}
    assert normalize_code_versions(value,'V@2026')==value


def test_form_encoding_repeats_array_keys_and_records_no_session_response(tmp_path):
    client = CodeMapperClient(CodeMapperConfig(cache_dir=str(tmp_path)))
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return b'{"success":true,"user":{"username":"private"}}'
    with patch.object(client,'_pace'), patch.object(client.opener,'open',return_value=Response()) as request:
        client._request('/session',form={'codes':['A','B'],'codingSystem':'V@1'})
        assert request.call_args.args[0].data == b'codes=A&codes=B&codingSystem=V%401'
        assert 'private' not in json.dumps(client.calls)


def test_pacing_is_shared_between_clients(tmp_path):
    config = CodeMapperConfig(cache_dir=str(tmp_path), requests_per_minute=6)
    first, second = CodeMapperClient(config), CodeMapperClient(config)
    with patch('src.mapping.codemapper.time.time', side_effect=[100,100,102,110]), \
         patch('src.mapping.codemapper.time.sleep') as sleep:
        first._pace()
        second._pace()
        sleep.assert_called_once_with(8)


def test_mapping_snapshot_export_flags_manual_review_without_extra_annotations():
    item = candidate('1','SNOMEDCT_US')
    item.decision_origin = 'codemapper'
    item.confidence = 0
    item.mapping_provenance = [{'cui':'C1'}]
    row = to_predicted_codelist([asdict(item)]).iloc[0]
    assert row['manual_review']
    assert 'EDF scope' in row['review_reason']
    assert json.loads(row['mapping_provenance'])[0]['cui'] == 'C1'


def test_authentication_http_error_does_not_retry_or_expose_server_body(tmp_path):
    client = CodeMapperClient(CodeMapperConfig(cache_dir=str(tmp_path)))
    error = HTTPError('https://example.invalid/session',401,'Unauthorized',{},
                      io.BytesIO(b'private username and password'))
    with patch.object(client,'_pace'), patch.object(client.opener,'open',side_effect=error) as request:
        with pytest.raises(CodeMapperError,match='HTTP 401') as captured:
            client._request('/session',form={'username':'private','password':'secret'})
        assert request.call_count == 1
        assert 'private' not in str(captured.value)
        assert 'secret' not in json.dumps(client.calls)
