"""Project direct CodeMapper links through validated CPRD browser identifiers."""
from dataclasses import replace

import pandas as pd

from src.data.preprocessing import standardize_code
from src.llm.classify import ClassifiedCandidate
from src.mapping.codemapper import expand_narrow_matches

VOCABULARIES=('RCD2','SNOMEDCT_US')


def browser_aliases(browser, corpus, native_corpus):
    """Use explicit browser identifiers, validating membership and ambiguity.

    Read identifiers must agree between the padded original five-character code
    and the five-character base of CleansedReadCode. SNOMED identifiers come from
    SnomedCTConceptId. Only identifiers present in the loaded releases qualify.
    Multiple native identifiers for one browser code/vocabulary are excluded.
    """
    required={'OriginalReadCode','CleansedReadCode','SnomedCTConceptId'}
    if not required.issubset(browser.columns):raise ValueError('CPRD browser lacks native identifier columns.')
    eligible=set(corpus.code)
    available={(str(r.code).strip(),r.vocabulary):r
               for r in native_corpus.itertuples() if r.vocabulary in VOCABULARIES}
    # The legacy mapper normalizes keys to uppercase. Exclude identifiers that
    # collide under that normalization, while sending original Read case.
    canonical={}
    for key in available:canonical.setdefault((standardize_code(*key),key[1]),[]).append(key)
    collisions={key for keys in canonical.values() if len(keys)>1 for key in keys}
    available={key:row for key,row in available.items() if key not in collisions}
    candidates={}
    for original,read,snomed in browser[['OriginalReadCode','CleansedReadCode','SnomedCTConceptId']].fillna('').astype(str).itertuples(index=False,name=None):
        original,read=original.strip(),read.strip()
        if original not in eligible:continue
        pairs=[]
        if read and len(original)<=5 and len(read)>=5:
            padded=original.ljust(5,'.')
            if padded==read[:5]:pairs.append((padded,'RCD2'))
        if snomed.strip():pairs.append((standardize_code(snomed,'SNOMEDCT_US'),'SNOMEDCT_US'))
        for key in pairs:candidates.setdefault((original,key[1]),set()).add(key)
    aliases={};ambiguous=0;unavailable=0
    for (original,vocabulary),keys in candidates.items():
        if len(keys)!=1:ambiguous+=1;continue
        key=next(iter(keys))
        if key not in available:unavailable+=1;continue
        aliases.setdefault(original,[]).append(key)
    return aliases,available,{'browser_codes':len(eligible),'codes_with_valid_alias':len(aliases),
                              'ambiguous_aliases_excluded':ambiguous,'unavailable_aliases_excluded':unavailable,
                              'native_case_collisions_excluded':len(collisions)}


def expand_cprd_matches(classified, corpus, aliases, native_available, config, *, client=None):
    """Only a returned cross-vocabulary CUI link can add a CPRD identifier."""
    descriptions=dict(zip(corpus.code,corpus.description));vocabulary=corpus.vocabulary.iloc[0]
    source_ids={};target_ids={}
    for original,keys in aliases.items():
        for key in keys:target_ids.setdefault(key,set()).add(original)
    for record in classified:
        if record.label=='Narrow':
            for key in aliases.get(record.code,[]):source_ids.setdefault(key,set()).add(record.code)
    seeds=[ClassifiedCandidate(code,native_available[(code,voc)].description,voc,'Narrow',0.,
                              'Native browser alias of a post-review CPRD Narrow seed.')
           for code,voc in sorted(source_ids)]
    native=pd.DataFrame([{'code':code,'vocabulary':voc,'description':row.description}
                         for (code,voc),row in native_available.items()])
    if native.empty:native=pd.DataFrame(columns=['code','vocabulary','description'])
    _,info=expand_narrow_matches(seeds,native,list(VOCABULARIES),config,client=client)
    links={}
    for record in info['direct_links']:
        key=(record['code'],record['vocabulary'])
        for provenance in record['provenance']:
            source=(provenance['seed_code'],provenance['seed_vocabulary'])
            for seed_original in source_ids.get(source,[]):
                for target_original in target_ids.get(key,[]):
                    if target_original==seed_original:continue
                    links.setdefault(target_original,[]).append({**provenance,
                        'seed_browser_code':seed_original,'target_browser_code':target_original,
                        'target_native_code':record['code'],'target_native_vocabulary':record['vocabulary']})
    merged=list(classified);positions={r.code:i for i,r in enumerate(merged)};added=[];promoted=[]
    for code,provenance in sorted(links.items()):
        position=positions.get(code);previous=merged[position] if position is not None else None
        if previous is not None and previous.label=='Narrow':continue
        item=ClassifiedCandidate(code,descriptions[code],vocabulary,'Narrow',0.,
            'Included by direct cross-vocabulary CUI mapping through validated CPRD browser identifiers.',
            decision_origin='codemapper',mapping_provenance=provenance,
            prior_label=previous.label if previous else None,prior_confidence=previous.confidence if previous else None)
        if previous is None:merged.append(item);added.append(code)
        else:merged[position]=replace(item,retrieval_score=previous.retrieval_score);promoted.append(code)
    info={**info,'identifier_adapter':'explicit_browser_aliases_v1',
          'browser_seeds':sum(r.label=='Narrow' for r in classified),
          'browser_seeds_without_valid_alias':[r.code for r in classified if r.label=='Narrow' and not aliases.get(r.code)],
          'new_browser_codes':added,'promoted_browser_codes':promoted}
    return merged,info
