"""Import the new SNOMED export into isolated artifacts and compare reference coverage.

Run from the repository root with PYTHONPATH=. .venv-study/bin/python.
No model calls, existing corpus/cache/config changes or manuscript updates.
"""
from pathlib import Path
import csv
import hashlib
import json

import pandas as pd
import yaml

from src.data.code_systems import load_code_system_corpus, load_snomedct_full
from src.data.loaders import load_aesi_dataset
from src.data.preprocessing import standardize_code

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
OLD=ROOT/'data/codes/csv/SNOMEDCT_US@2025_09_01-codes.csv'
NEW=OLD.with_name('SNOMEDCT_US@2025_09_01-codes new.csv')
CACHE=ROOT/'data/codes/parquet/codes_SNOMEDCT_US.parquet'
CONFIG=ROOT/'configs/codemapper_experiment.yaml'
config=yaml.safe_load(CONFIG.read_text())
protected=[OLD,NEW,CACHE,CONFIG,ROOT/'configs/default.yaml',
           ROOT/'manuscript/jamia_open_colleague_merged/main.pdf',
           *[ROOT/p for p in config['paths']['aesi_datasets'].values()]]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

old=pd.read_parquet(CACHE)
pd.testing.assert_frame_equal(load_snomedct_full(OLD),old)
frozen=json.loads((ROOT/'results/experiments/pipeline_design/20261008_cap10000/frozen_pipelines.json').read_text())
assert hashes[str(CACHE.relative_to(ROOT))]==frozen['validation_source_hashes'][str(CACHE.relative_to(ROOT))]
new=load_code_system_corpus(['SNOMEDCT_US'],source_paths={'SNOMEDCT_US':str(NEW)},cache_dir=OUT/'parquet')
pd.testing.assert_frame_equal(new,load_snomedct_full(NEW))
assert new.code.str.isdecimal().all()
assert new.description.str.strip().ne('').all()
assert not new.duplicated(['code','vocabulary']).any()
new.to_csv(OUT/'SNOMEDCT_US_new_imported.csv',index=False)
old_ids=set(old.code)
new_ids=set(new.code)
removed=old_ids-new_ids
added=new_ids-old_ids

old_blanks=set()
with OLD.open(newline='') as handle:
    for row in csv.DictReader(handle):
        if not row['term'].strip():old_blanks.add(row['code'])
old_blanks-=old_ids
raw_rows=outer_split_terms=0
raw_codes=set()
with NEW.open(encoding='utf-8-sig',newline='') as handle:
    reader=csv.reader(handle,delimiter=';')
    assert next(reader)==['code,term','','','']
    for outer in reader:
        raw_rows+=1
        outer_split_terms+=any(value.strip() for value in outer[1:])
        inner=next(csv.reader([';'.join(outer).rstrip(';')],strict=True))
        assert len(inner)==2 and inner[0].strip() and inner[1].strip()
        raw_codes.add(inner[0].strip())
assert raw_codes==new_ids

joined=old.set_index('code').join(new.set_index('code'),how='inner',lsuffix='_old',rsuffix='_new')
different=joined[joined.description_old!=joined.description_new]
different[['description_old','description_new']].to_csv(OUT/'changed_first_descriptions.csv')
old[old.code.isin(removed)][['code','description']].sort_values('code').to_csv(OUT/'removed_codes.csv',index=False)
new[new.code.isin(added)][['code','description']].sort_values('code').to_csv(OUT/'added_codes.csv',index=False)

coverage=[]
loss=[]
previously_missing=[]
for phenotype,path in config['paths']['aesi_datasets'].items():
    _,gold=load_aesi_dataset(ROOT/path)
    gold=gold[(gold.vocabulary=='SNOMEDCT_US')&(gold.label=='Narrow')]
    positives={standardize_code(code,'SNOMEDCT_US') for code in gold.code}
    before=positives&old_ids
    after=positives&new_ids
    coverage.append({'phenotype':phenotype,'snomed_reference_narrow':len(positives),
                     'available_old':len(before),'available_new':len(after),
                     'unavailable_old':len(positives-old_ids),'unavailable_new':len(positives-new_ids),
                     'recovered':len(after-before),'newly_unavailable':len(before-after)})
    for row in gold.itertuples():
        code=standardize_code(row.code,'SNOMEDCT_US')
        if code in removed:loss.append({'phenotype':phenotype,'code':code,'reference_description':row.code_name})
        if code not in old_ids:
            previously_missing.append({'phenotype':phenotype,'code':code,'reference_description':row.code_name,
                                       'old_cause':'empty description' if code in old_blanks else 'absent identifier',
                                       'present_in_new_export':code in new_ids})
pd.DataFrame(coverage).to_csv(OUT/'reference_coverage.csv',index=False)
pd.DataFrame(loss,columns=['phenotype','code','reference_description']).to_csv(OUT/'newly_unavailable_reference_entries.csv',index=False)
pd.DataFrame(previously_missing).to_csv(OUT/'previously_unavailable_snomed.csv',index=False)
for path in protected:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path.relative_to(ROOT))],path

summary={'date':'2026-10-29','focus':'SNOMEDCT_US only; MedDRA not changed or analysed',
         'source_hashes':hashes,'old_populated_identifiers':len(old_ids),
         'old_blank_description_identifiers':len(old_blanks),'new_source_rows':raw_rows,
         'new_populated_identifiers':len(new_ids),'new_blank_description_rows':0,
         'added_identifiers':len(added),'removed_populated_identifiers':len(removed),
         'old_blank_identifiers_now_populated':len(old_blanks&new_ids),
         'description_rows_with_semicolons_split_across_outer_columns':outer_split_terms,
         'changed_first_descriptions':len(different),
         'reference_coverage':coverage,
         'previously_unavailable_reference_entries':len(previously_missing),
         'previously_unavailable_reference_entries_recovered':sum(row['present_in_new_export'] for row in previously_missing),
         'newly_unavailable_reference_entries':loss,
         'imported_parquet':str((OUT/'parquet/codes_SNOMEDCT_US.parquet').relative_to(ROOT)),
         'original_csv_cache_configs_references_and_manuscript_unchanged':True,
         'old_loader_output_unchanged':True,
         'outcome':'Import succeeds; new export does not repair missing reference coverage and was not activated.',
         'embedding_note':'The first available description changed for many identifiers. Any future evaluation using this import needs an isolated/rebuilt embedding index matching these descriptions.'}
assert summary['previously_unavailable_reference_entries']==446
assert summary['previously_unavailable_reference_entries_recovered']==0
(OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['old_populated_identifiers','new_populated_identifiers','added_identifiers',
                                         'removed_populated_identifiers','previously_unavailable_reference_entries_recovered',
                                         'newly_unavailable_reference_entries','old_loader_output_unchanged','outcome']},indent=2))
