"""Native identifier adapters must not invent CPRD codes or alias-only links."""
import pandas as pd

from experiments.pipeline_design.cprd_mapping import browser_aliases, expand_cprd_matches
from src.llm.classify import ClassifiedCandidate
from src.utils.config import CodeMapperConfig


def corpus(codes,vocabulary='CPRD'):
    return pd.DataFrame([{'code':c,'description':c,'vocabulary':vocabulary} for c in codes])


def test_aliases_validate_padded_read_ids_and_exclude_ambiguous_snomed():
    browser=pd.DataFrame([
        ['S226','S226.00','101'],['S226','S226.00','102'],
        ['bad','OTHER00','999'],['outside','A000000','101']],
        columns=['OriginalReadCode','CleansedReadCode','SnomedCTConceptId'])
    native=pd.concat([corpus(['S226.'],'RCD2'),corpus(['101','102'],'SNOMEDCT_US')])
    aliases,_,info=browser_aliases(browser,corpus(['S226','bad']),native)
    assert aliases=={'S226':[('S226.','RCD2')]}
    assert info['ambiguous_aliases_excluded']==1
    assert info['unavailable_aliases_excluded']==1


def test_only_returned_cross_vocabulary_links_promote_browser_codes():
    browser=pd.DataFrame([['A0000','A000000',''],['B0000','B000000','202'],['C0000','C000000','']],
                         columns=['OriginalReadCode','CleansedReadCode','SnomedCTConceptId'])
    target=corpus(['A0000','B0000','C0000'])
    native=pd.concat([corpus(['A0000','B0000','C0000'],'RCD2'),corpus(['202'],'SNOMEDCT_US')])
    aliases,available,_=browser_aliases(browser,target,native)
    original=[ClassifiedCandidate('A0000','A','CPRD','Narrow',.9,'original'),
              ClassifiedCandidate('B0000','B','CPRD','Exclude',.8,'original')]
    class Client:
        calls=[];cache_hits=0
        def cuis_for_codes(self,vocabulary,codes):
            assert codes==['A0000']
            return {'A0000':{'cuis':['C1']}}
        def concepts(self,*args):
            return [{'cui':'C1','sourceConcepts':[
                {'codingSystem':'SNOMEDCT_US@2025_09_01','id':'202'},
                {'codingSystem':'SNOMEDCT_US@2025_09_01','id':'unavailable'},
                {'codingSystem':'RCD2@20200401','id':'C0000'}]}]
        def close(self):pass
    records,info=expand_cprd_matches(original,target,aliases,available,CodeMapperConfig(),client=Client())
    assert original[1].label=='Exclude'
    assert {r.code for r in records if r.label=='Narrow'}=={'A0000','B0000'}
    assert records[1].confidence==0 and records[1].prior_label=='Exclude'
    assert records[1].mapping_provenance[0]['seed_browser_code']=='A0000'
    assert records[1].mapping_provenance[0]['target_native_code']=='202'
    assert info['promoted_browser_codes']==['B0000']
    assert not info['descendants_used']


def test_read_case_is_preserved_and_case_collisions_are_excluded():
    browser=pd.DataFrame([['G73y0','G73y000',''],['Ab...','Ab...00','']],
                         columns=['OriginalReadCode','CleansedReadCode','SnomedCTConceptId'])
    native=corpus(['G73y0','Ab...','AB...'],'RCD2')
    aliases,_,info=browser_aliases(browser,corpus(['G73y0','Ab...']),native)
    assert aliases=={'G73y0':[('G73y0','RCD2')]}
    assert info['native_case_collisions_excluded']==2


def test_existing_native_seed_does_not_hide_a_cross_vocabulary_browser_link():
    browser=pd.DataFrame([['A0000','A000000',''],['B0000','B000000','202'],['D0000','','202']],
                         columns=['OriginalReadCode','CleansedReadCode','SnomedCTConceptId'])
    target=corpus(['A0000','B0000','D0000'])
    native=pd.concat([corpus(['A0000','B0000'],'RCD2'),corpus(['202'],'SNOMEDCT_US')])
    aliases,available,_=browser_aliases(browser,target,native)
    class Client:
        calls=[];cache_hits=0
        def cuis_for_codes(self,vocabulary,codes):return {c:{'cuis':['C1']} for c in codes}
        def concepts(self,*args):return [{'cui':'C1','sourceConcepts':[
            {'codingSystem':'SNOMEDCT_US@2025_09_01','id':'202'}]}]
        def close(self):pass
    seeds=[ClassifiedCandidate(c,c,'CPRD','Narrow',.9,'seed') for c in ['A0000','D0000']]
    records,_=expand_cprd_matches(seeds,target,aliases,available,CodeMapperConfig(),client=Client())
    assert {r.code for r in records}=={'A0000','B0000','D0000'}
    # A shared SNOMED alias alone is insufficient without a cross-vocabulary link.
    records,_=expand_cprd_matches(seeds[1:],target,aliases,available,CodeMapperConfig(),client=Client())
    assert {r.code for r in records}=={'D0000'}
