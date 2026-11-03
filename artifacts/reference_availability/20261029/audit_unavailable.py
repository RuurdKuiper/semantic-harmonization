"""Read-only audit of the reference entries unavailable to the frozen EDF study.

Run from the repository root with .venv-study/bin/python. No source/corpus,
reference, embedding, prediction, configuration or manuscript file is changed.
"""
from pathlib import Path
import collections
import hashlib
import json
import re
import zipfile

import pandas as pd
import yaml

from src.data.loaders import load_aesi_dataset
from src.data.preprocessing import is_code_range, standardize_code

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
config = yaml.safe_load((ROOT/'configs/codemapper_experiment.yaml').read_text())
freeze = json.loads((ROOT/'results/experiments/pipeline_design/20261008_cap10000/frozen_pipelines.json').read_text())
source_checks = []
available = set()
for vocab in config['retrieval']['vocabularies']:
    path = ROOT/'data/codes/parquet'/f'codes_{vocab}.parquet'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert freeze['validation_source_hashes'][str(path.relative_to(ROOT))] == digest
    frame = pd.read_parquet(path)
    keys = {(standardize_code(r.code, vocab), vocab) for r in frame.itertuples() if not is_code_range(r.code)}
    available.update(keys)
    source_checks.append({'terminology': vocab, 'cache_sha256': digest,
                          'matches_frozen_study': True, 'eligible_pairs': len(keys)})

missing = []
phenotypes = []
reference_hashes = {}
for phenotype, rel in config['paths']['aesi_datasets'].items():
    path = ROOT/rel
    reference_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    if rel in freeze['validation_source_hashes']:
        assert reference_hashes[rel] == freeze['validation_source_hashes'][rel]
    _, gold = load_aesi_dataset(path)
    narrow = gold[gold.label == 'Narrow']
    present = 0
    for row in narrow.itertuples():
        code = standardize_code(row.code, row.vocabulary)
        if (code, row.vocabulary) in available:
            present += 1
        else:
            missing.append({'phenotype': phenotype, 'vocabulary': row.vocabulary,
                            'reference_code': row.code, 'normalised_code': code,
                            'description': row.code_name})
    phenotypes.append({'phenotype': phenotype, 'narrow': len(narrow),
                       'available': present, 'unavailable': len(narrow)-present})
assert len(missing) == 894
assert sum(p['available'] for p in phenotypes) == 3051

raw_hits = {}
for vocab in sorted({r['vocabulary'] for r in missing}):
    wanted = {r['normalised_code'] for r in missing if r['vocabulary'] == vocab}
    hits = collections.defaultdict(list)
    source = ROOT/config['paths']['code_systems'][vocab]
    for number, line in enumerate(source.read_text().splitlines()[1:], 2):
        if ',' not in line:
            continue
        raw_code, term = line.split(',', 1)
        code = standardize_code(raw_code.strip().strip('"'), vocab)
        if code in wanted:
            hits[code].append((number, term.strip().strip('"').strip(';').strip()))
    raw_hits[vocab] = hits

# The independently downloaded official CMS diagnosis list verifies the invalid reference.
cms = OUT/'external/icd9_v32.zip'
assert cms.exists(), 'Download the official CMS v32 code-titles archive before running.'
with zipfile.ZipFile(cms) as archive:
    cms_text = archive.read('CMS32_DESC_LONG_DX.txt').decode('utf8', 'replace')
cms_codes = {line.split()[0] for line in cms_text.splitlines() if line.strip()}
assert '26061' not in cms_codes
assert '25061' in cms_codes

for row in missing:
    vocab, code = row['vocabulary'], row['normalised_code']
    found = raw_hits[vocab].get(code, [])
    row.update({'source_file': config['paths']['code_systems'][vocab],
                'source_lines': ','.join(str(n) for n, _ in found),
                'raw_source_rows': len(found),
                'nonempty_raw_descriptions': sum(bool(term) for _, term in found),
                'raw_descriptions': ' | '.join(term for _, term in found)})
    if found:
        assert not any(term for _, term in found), 'Unexpected parsing/normalisation loss'
        row['diagnosis'] = 'code_present_but_description_empty'
        row['repair'] = 'Obtain independent descriptions plus status/hierarchy metadata; assess eligibility before rebuilding the corpus.'
    elif vocab == 'SNOMEDCT_US':
        assert len(code) > 10 and code[-10:-3] == '1000000' and code[-3:-1] == '10'
        row['namespace'] = '1000000 (NHS Digital UK)'
        row['diagnosis'] = 'UK_namespace_code_absent_from_US_export'
        row['repair'] = 'Align the reference and corpus editions; use a documented UK extension or adjudicated edition-scoping policy.'
    elif vocab == 'ICPC':
        row['diagnosis'] = 'ICPC2_code_absent_from_1993_export' if code == 'T89' else 'Dutch_ICPC_subcode_absent_from_base_export'
        row['repair'] = 'Use the matching ICPC edition/national extension; do not collapse subcodes to broader parents automatically.'
    elif vocab == 'ICD9CM':
        assert code == '260.61'
        row['diagnosis'] = 'reference_code_absent_from_official_CMS_diagnosis_list'
        row['repair'] = 'Reference-author adjudication; do not assume the intended replacement code.'
    else:
        raise AssertionError(row)

entries = pd.DataFrame(missing)
entries.to_csv(OUT/'unavailable_reference_entries_final.csv', index=False)
pd.DataFrame(phenotypes).to_csv(OUT/'counts_by_phenotype.csv', index=False)
pd.DataFrame(source_checks).to_csv(OUT/'frozen_corpus_verification.csv', index=False)
causes = entries.groupby(['vocabulary', 'diagnosis']).size().reset_index(name='reference_entries')
causes.to_csv(OUT/'counts_by_cause.csv', index=False)

local = pd.read_parquet(ROOT/'data/codes/parquet/codes_ICD10CM.parquet')
local_codes = set(local.code.str.replace('.', '', regex=False))
versions = []
for year in [2025, 2026]:
    with zipfile.ZipFile(OUT/f'external/icd10_{year}.zip') as archive:
        text = archive.read(f'icd10cm-order-{year}.txt').decode('utf8', 'replace')
    codes = {line.split()[1] for line in text.splitlines() if line.strip()}
    versions.append({'official_year': year, 'official_codes': len(codes),
                     'official_missing_from_local': len(codes-local_codes),
                     'local_codes_not_in_official': len(local_codes-codes)})
assert versions[1]['official_missing_from_local'] == 0
header = local[(local.code == 'ICD-10-CM')].to_dict('records')
assert len(header) == 1
summary = {'date': '2026-10-29', 'available_reference_entries': 3051,
           'unavailable_reference_entries': len(entries),
           'unique_unavailable_code_terminology_pairs': len(entries.drop_duplicates(['vocabulary','normalised_code'])),
           'empty_description_entries': sum(entries.diagnosis == 'code_present_but_description_empty'),
           'all_six_corpus_hashes_match_frozen_study': True,
           'reference_sha256': reference_hashes, 'causes': causes.to_dict('records'),
           'official_icd10_comparisons': versions, 'additional_noncode_header_in_corpus': header,
           'snomed_exact_release_independently_verified': False,
           'limitations': ['Raw sources have no active-status or hierarchy metadata; blank descriptions do not establish inactivity or MedDRA term level.',
                           'SNOMED public-browser requests were unavailable (HTTP 405/423); no per-code status claims were inferred.',
                           'SNOMED export/configuration label is 2025_09_01; original package metadata is needed to distinguish a correctly labelled release from a mislabelled export.'],
           'production_inputs_predictions_and_manuscript_unchanged': True}
(OUT/'audit_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
