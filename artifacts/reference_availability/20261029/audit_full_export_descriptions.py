"""Count identifiers with no wording in any row; never change corpus inputs."""
from pathlib import Path
import collections
import csv
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
records=[]
for path in sorted((ROOT/'data/codes/csv').glob('*.csv')):
    states=collections.defaultdict(lambda:[0,0])
    rows=blank=0
    with path.open() as handle:
        values=(line.rstrip('\n').split(',',1) for line in handle) if path.name.startswith('RCD2') else csv.reader(handle)
        next(values)
        for row in values:
            if not row:continue
            code=row[0].strip().strip('"')
            term=row[1].strip().strip('"').rstrip(';').strip() if len(row)>1 else ''
            if not code:continue
            rows+=1
            blank+=not bool(term)
            states[code][int(bool(term))]+=1
    no_words={code for code,(empty,named) in states.items() if not named}
    records.append({'source':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'raw_code_rows':rows,'unique_code_identifiers':len(states),
                    'blank_description_rows':blank,'unique_codes_without_any_description':len(no_words),
                    'unique_codes_with_description':len(states)-len(no_words),
                    'codes_with_both_blank_and_populated_rows':sum(bool(empty and named) for empty,named in states.values()),
                    'percent_unique_codes_without_description':round(100*len(no_words)/len(states),2),
                    'examples_without_description':sorted(no_words)[:8]})
(OUT/'full_export_description_completeness.json').write_text(json.dumps(records,indent=2)+'\n')

# These counts are an inference about possible export-level filtering, not term-level validation.
official={'LLT_including_PT':89774,'PT':26920,'HLT':1739,'HLGT':337,'SOC':27,'SMQ':230}
observed=next(row for row in records if row['source'].startswith('MDR'))
fingerprint={'official_source':'https://alt.meddra.org/files_acrobat/whatsnew_28_0_English.pdf',
             'official_version_28_0_counts':official,
             'predicted_labels_if_PT_HLT_and_SMQ_only':official['PT']+official['HLT']+official['SMQ'],
             'predicted_blanks_if_LLT_only_HLGT_and_SOC':official['LLT_including_PT']-official['PT']+official['HLGT']+official['SOC'],
             'predicted_total_identifiers':official['LLT_including_PT']+official['HLT']+official['HLGT']+official['SOC']+official['SMQ'],
             'interpretation':'Exact numerical agreement is a strong lead for hierarchy-level-dependent label export. No per-code term-type validation or export-query inspection has been performed.'}
assert fingerprint['predicted_labels_if_PT_HLT_and_SMQ_only']==observed['unique_codes_with_description']
assert fingerprint['predicted_blanks_if_LLT_only_HLGT_and_SOC']==observed['unique_codes_without_any_description']
assert fingerprint['predicted_total_identifiers']==observed['unique_code_identifiers']

probe=ROOT/'results/codemapper/eosinophilic_esophagitis.json'
cached=json.loads(probe.read_text())
matches=[s for c in cached['concepts'] for s in c.get('sourceConcepts',[])
         if s.get('codingSystem')=='MDR@28.0' and str(s.get('id'))=='10064220']
assert matches and matches[0]['preferredTerm']=='Eosinophilic esophagitis'
source=ROOT/'data/codes/csv/MDR@28.0-codes.csv'
raw=[row for row in csv.DictReader(source.open()) if row['code']=='10064220']
assert raw and all(not row['term'].strip() for row in raw)
evidence={'csv_code':'10064220','csv_term':'','cached_api_source':str(probe.relative_to(ROOT)),
          'cached_api_source_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),
          'cached_api_object':matches[0],
          'interpretation':'At least one blank CSV label is available independently from CodeMapper for the same release identifier; blanks cannot universally mean no wording exists upstream.'}
followup={'date':'2026-10-29','counts':records,'meddra_level_filter_hypothesis':fingerprint,
          'independent_cached_label_example':evidence,
          'limitations':'Counts describe exported identifiers, not a count of all clinically valid codes. ICPC blank identifiers 1–7 are structural components. SNOMED status/description selection remains unverified.',
          'production_sources_corpora_predictions_and_manuscript_unchanged':True}
(OUT/'full_export_followup.json').write_text(json.dumps(followup,indent=2)+'\n')
print(json.dumps({'counts':[(r['source'],r['unique_codes_without_any_description'],r['unique_code_identifiers']) for r in records],
                  'meddra_level_counts_match':True,'independent_api_label_found':True},indent=2))
