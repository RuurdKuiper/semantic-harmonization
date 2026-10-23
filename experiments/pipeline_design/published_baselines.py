"""Rescore archived primary RAG submissions, without model calls or gold-based extraction."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from experiments.matthewman_2026.experiment import (
    PHENOTYPES,load_cprd_browser,load_reference_codelists,scope_reference_to_corpus,
    score_whole_codelist,score_as_dict,
)

# These eight bundles match the primary four-model, two-corpus experiment.
# The separate 200-result wrist experiment and MedCPT bundle are excluded.
SOURCES=[
    ('gpt-5.2','full','2026-01-21T12-26-16+00-00_RAG_f5178a12554e983adb7209.json'),
    ('gpt-5.2','read','2026-01-22T15-38-02+00-00_RAG_f5178a12554e983adb7209.json'),
    ('gemini-3-pro-preview','read','2026-01-26T10-41-03+00-00_RAG_f5178a12554e983adb7209.json'),
    ('gemini-3-flash-preview','full','2026-02-04T12-03-25+00-00_RAG_f5178a12554e983adb7209.json'),
    ('claude-sonnet-4-6','full','2026-02-19T12-55-50+00-00_RAG_5ab6567c11a9be5c54356c.json'),
    ('claude-sonnet-4-6','read','2026-02-19T14-19-44+00-00_RAG_5ab6567c11a9be5c54356c.json'),
    ('gemini-3-pro-preview','full','2026-03-04T11-47-58+00-00_RAG_5ab6567c11a9be5c54356c.json'),
    ('gemini-3-flash-preview','read','2026-03-04T14-46-00+00-00_RAG_5ab6567c11a9be5c54356c.json'),
]
LABELS={'gemini-3-pro-preview':'Gemini 3 Pro','gemini-3-flash-preview':'Gemini 3 Flash',
        'gpt-5.2':'GPT-5.2','claude-sonnet-4-6':'Claude Sonnet 4.6'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_cell(cell):
    cell=cell.strip().replace(r'\|','|')
    for wrapper in ('**','`'):
        if cell.startswith(wrapper) and cell.endswith(wrapper):cell=cell[len(wrapper):-len(wrapper)].strip()
    return cell


def extract_codes(text,source_id_pairs=False):
    """Extract only explicitly submitted code-column rows, including unknown codes."""
    if not text.strip():return [],{'empty_submission':True,'tables':0,'ignored_groups':[],'excluded_tables':0}
    codes=[];column=None;tables=0;groups=[];context='';excluded_tables=0;excluded=False;interpreted=[]
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            column=None
            if line.lstrip().startswith(('#','**')):context=line.strip()
            continue
        cells=[clean_cell(c) for c in re.split(r'(?<!\\)\|',line.strip().strip('|'))]
        headings=[c.lower() for c in cells]
        if 'code' in headings:
            column=headings.index('code');tables+=1
            excluded=bool(re.search(r'\b(excluded|not included|omitted)\b',context,re.I))
            excluded_tables+=int(excluded)
            continue
        if column is None or column>=len(cells) or excluded:continue
        if all(not c or re.fullmatch(r':?-{2,}:?',c) for c in cells):continue
        code=cells[column]
        if not code:raise ValueError('Empty code cell in a submitted table')
        if re.search(r'\s',code) and all(not c for i,c in enumerate(cells) if i!=column):
            groups.append(code);continue
        raw=code
        if re.search(r'\bexcluded\b',code,re.I):
            interpreted.append({'cell':raw,'codes':[],'reason':'Explicitly excluded by the submitted answer'})
            continue
        # Parenthetical via/source notes are annotations, not additional list entries.
        code=re.sub(r'\([^)]*\)','',code).replace('**','').replace('`','').strip()
        identifiers=[c.strip() for c in re.split(r'\s+/\s+',code)]
        if source_id_pairs and len(identifiers)==2 and identifiers[0].isdigit():
            # Verified against the retrieval records in this archived submission.
            identifiers=identifiers[1:]
        if any(not c or re.search(r'\s',c) for c in identifiers):
            raise ValueError(f'Ambiguous submitted code cell requires review: {raw}')
        if raw!=code or len(identifiers)>1 or (source_id_pairs and ' / ' in raw):
            interpreted.append({'cell':raw,'codes':identifiers,'reason':'Separate declared identifiers from presentation and source annotations'})
        codes.extend(identifiers)
    if not tables:raise ValueError('Nonempty submission has no recognized codelist table')
    return sorted(set(codes)),{'empty_submission':False,'tables':tables,'ignored_groups':groups,
                              'excluded_tables':excluded_tables,'duplicate_rows':len(codes)-len(set(codes)),
                              'interpreted_cells':interpreted}


def final_text(sample):
    choices=sample['output']['choices']
    if len(choices)!=1:raise ValueError('Expected one submitted answer per sample')
    content=choices[0]['message']['content']
    return '\n'.join(block['text'] for block in content if block.get('type')=='text')


def format_aliases(full,read):
    """Resolve only unique presentation variants, independently of reference labels.

    Index the full browser before restricting corpus scope, so a smaller test
    corpus cannot make an ambiguous alias appear unique.
    """
    aliases={}
    def add(alias,code,rule):
        aliases.setdefault(alias,{}).setdefault(code,set()).add(rule)
    for code in full.code:
        if code.startswith('^ESCT'):add(code[1:],code,'restore_esct_caret')
    for code in read.code:
        base=code.rstrip('.')
        if re.fullmatch(r'[A-Za-z0-9]{1,5}',base):
            for padding in range(6-len(base)):
                add(base+'.'*padding,code,'read_trailing_dot_padding')
    full_codes=set(full.code)
    for alias in list(aliases):
        if alias in full_codes:add(alias,alias,'exact_browser_identifier')
    return aliases


def normalize_codes(codes,available,aliases):
    """Keep exact identifiers; accept unique in-scope format aliases; retain unknowns."""
    normalized=[];changes=[]
    for code in codes:
        candidates=aliases.get(code,{})
        if code not in available and len(candidates)==1:
            target=next(iter(candidates))
            if target in available:
                changes.append({'submitted':code,'canonical':target,'rules':sorted(candidates[target])})
                normalized.append(target)
                continue
        normalized.append(code)
    return sorted(set(normalized)),changes


def rescore(root,output,config):
    repo=root/config['benchmark']['references'];logs=repo/'docs/logs'
    paths=[logs/name for _,_,name in SOURCES]
    browser=root/config['benchmark']['browser']
    refs_paths=[repo/'codelists'/name for _,name in PHENOTYPES.values()]
    signature={str(p.relative_to(root)):sha(p) for p in [*paths,*refs_paths,browser,Path(__file__)]}
    folder=output/'published_baselines';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'manifest.json';metrics=folder/'metrics.csv'
    strict_metrics=folder/'metrics.strict.csv'
    if receipt.exists() and metrics.exists() and strict_metrics.exists() and json.loads(receipt.read_text())['sources']==signature:
        return pd.read_csv(metrics)
    full,read=load_cprd_browser(browser);corpora={'full':full,'read':read};refs=load_reference_codelists(repo)
    available_sets={s:set(c.code) for s,c in corpora.items()}
    aliases=format_aliases(full,read)
    scoped={(s,p):scope_reference_to_corpus(refs[p],c) for s,c in corpora.items() for p in PHENOTYPES}
    names={title:key for key,(title,_) in PHENOTYPES.items()}
    rows=[];strict_rows=[];audits=[];seen=set();bundles=[]
    for model,subset,filename in SOURCES:
        path=logs/filename;bundle=json.loads(path.read_text())
        if bundle['status']!='success' or len(bundle['samples'])!=21:raise ValueError('Incomplete primary bundle')
        if model not in bundle['eval']['model']:raise ValueError('Unexpected model metadata')
        if bundle['eval']['task']!='RAG':raise ValueError('Baseline must use primary RAG workflow')
        reported=sum({'C':1,'P':.5,'I':0}[next(iter(s['scores'].values()))['value']] for s in bundle['samples'])/21
        bundles.append({'model':model,'subset':subset,'file':filename,'published_grade_mean':reported})
        print(f'[published rescoring] {model}, {subset}: 21 archived submissions',flush=True)
        available=available_sets[subset]
        for sample in bundle['samples']:
            phenotype=names[sample['input']];epoch=sample['epoch'];key=(model,subset,phenotype,epoch)
            if key in seen:raise ValueError('Duplicate published run')
            seen.add(key);text=final_text(sample)
            codes,extraction=extract_codes(text,source_id_pairs=key==('gpt-5.2','full','psoriasis',1))
            normalized,changes=normalize_codes(codes,available,aliases)
            reference=scoped[subset,phenotype]
            current=dict(zip(reference.code,reference.required))
            original={line.split()[0]:int(line.split()[-1]) for line in sample['target'].splitlines()}
            # Subset membership comes from the supplied gold target, independently of predictions.
            if subset=='read' and (set(original)&available_sets['full'])-available:
                raise ValueError('Logged target identifies a different corpus subset')
            # Final-set metrics do not require reconstructing retrieval traces.
            def final_score(identifiers):
                measured=score_as_dict(score_whole_codelist(reference,(),identifiers))
                return {k:measured[k] for k in ('grade','grade_value','required_total','optional_total',
                    'predicted_total','required_predicted','irrelevant_predicted_count','missing_required',
                    'irrelevant_predicted','final_required_recall','final_acceptable_precision')}
            grade=next(iter(sample['scores'].values()))['value']
            identity={'model':model,'subset':subset,'phenotype':phenotype,'epoch':epoch,'published_grade':grade}
            rows.append({**identity,**final_score(normalized)})
            strict_rows.append({**identity,**final_score(codes)})
            audits.append({'model':model,'subset':subset,'phenotype':phenotype,'epoch':epoch,
                'source_file':filename,'source_sample_id':sample['id'],'output_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'codes':codes,'normalized_codes':normalized,'normalizations':changes,
                'unmatched_codes':sorted(set(normalized)-available),'strict_unmatched_codes':sorted(set(codes)-available),
                'extraction':extraction,
                'logged_target_unavailable':sorted(set(original)-available),
                'target_added':sorted(current.keys()-original.keys()),'target_removed':sorted(original.keys()-current.keys()),
                'target_relabelled':{c:[original[c],int(current[c])] for c in current.keys()&original.keys() if original[c]!=current[c]}})
    expected={(m,s,p,e) for m in LABELS for s in ('full','read') for p in PHENOTYPES for e in (1,2,3)}
    if seen!=expected:raise ValueError('Incomplete 168-run comparison grid')
    frame=pd.DataFrame(rows);frame.to_csv(metrics,index=False)
    pd.DataFrame(strict_rows).to_csv(strict_metrics,index=False)
    (folder/'extraction_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    manifest={'sources':signature,'repository':'https://github.com/julianmatthewman/llmcodelists_public',
              'commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
              'runs':len(frame),'bundles':bundles,
              'normalization':'Presentation wrappers and source annotations removed. Missing ESCT carets and Read trailing-dot padding accepted only for unique browser aliases within the tested corpus. Case preserved; no spelling correction, range expansion, or native SNOMED translation. Full-browser alias ambiguity is retained before corpus restriction.',
              'strict_sensitivity':'metrics.strict.csv retains exact-identifier rescoring before format normalization.',
              'manual_interpretation':'Compound cells were reviewed from submitted text without reference labels. In GPT-5.2/full/psoriasis/epoch 1, paired leading numeric retrieval-record IDs are source annotations; the following original code is the submitted identifier.',
              'precision_for_empty_submission':0,'reference_policy':'All submissions rescored against the same current, corpus-scoped required/optional lists.'}
    receipt.write_text(json.dumps(manifest,indent=2))
    return frame


def write_appendix(output,destination):
    from experiments.pipeline_design.manuscript_results import tex
    folder=output/'published_baselines'
    if not (folder/'metrics.csv').exists():
        (destination/'study_published_comparison.tex').write_text(
            r'\section*{Table A10. Published-output rescoring}'+'\nPending.\n'+
            r'\section*{Table A11. Reference harmonization}'+'\nPending.\n'+
            r'\section*{Table A12. Individual published run scores}'+'\nPending.\n'+
            r'\section*{Table A13. Sensitivity to identifier formatting}'+'\nPending.\n')
        return
    frame=pd.read_csv(folder/'metrics.csv');audits=json.loads((folder/'extraction_audit.json').read_text())
    parts=[r'\clearpage',r'\section*{Table A10. Published primary RAG outputs: original grades and harmonized rescoring}',
        'R and P are mean required-code recall and acceptable-code precision. These code-level percentages use the same fixed references as our pipelines. '
        'Original and rescored list scores assign 1 to correct, 0.5 to partially correct, and 0 to incorrect. '
        'Original grades were assigned by the published LLM judge; rescored list grades use format-normalized identifier sets and the fixed references. '
        'They are distinct evaluations. Each row contains seven phenotypes with three runs each; all 21 runs are retained, including empty submissions.',
        r'\begin{center}',r'\begin{tabular}{p{0.25\textwidth}lrrrr}',r'\toprule',
        r'Model & Corpus & R (\%) & P (\%) & Original list (\%) & Rescored list (\%) \\',r'\midrule']
    for model in LABELS:
        for subset in ('full','read'):
            group=frame[(frame.model==model)&(frame.subset==subset)]
            original=group.published_grade.map({'C':1,'P':.5,'I':0}).mean()
            parts.append(tex(LABELS[model])+' & '+subset+
                f' & {100*group.final_required_recall.mean():.1f} & {100*group.final_acceptable_precision.mean():.1f}'+
                f' & {100*original:.1f} & {100*group.grade_value.mean():.1f}'+r' \\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{center}',r'\clearpage',
        r'\section*{Table A11. Reference harmonization for the published-output comparison}',
        'Differences compare archived evaluation targets with the fixed, corpus-scoped reference files. Counts are distinct identifiers across the four models and three runs. '
        'Added and removed refer to the reference membership; relabeled refers to required versus optional status. '
        'Unavailable identifiers occur in an archived target but outside the tested browser subset. '
        'All systems are rescored against the fixed reference; an added optional entry is accepted but creates no additional recall requirement.',
        r'\begin{longtable}{p{0.30\textwidth}lrrrr}',r'\toprule',
        r'Phenotype & Corpus & Added & Removed & Relabeled & Unavailable \\',r'\midrule',r'\endhead'])
    for phenotype in PHENOTYPES:
        for subset in ('full','read'):
            group=[a for a in audits if a['phenotype']==phenotype and a['subset']==subset]
            counts=[len(set().union(*(set(a[key]) for a in group))) for key in
                ('target_added','target_removed','target_relabelled','logged_target_unavailable')]
            parts.append(tex(phenotype.replace('_',' '))+' & '+subset+' & '+' & '.join(map(str,counts))+r' \\')
    parts.extend([r'\bottomrule',r'\end{longtable}'])
    parts.extend([r'\clearpage',r'\section*{Table A12. Individual published run scores}',
        'Each cell lists percentages for runs 1, 2, and 3 in that order. R is required-code recall; P is acceptable-code precision. '
        'Figure 8 shows the arithmetic mean of each triplet after accepting verified formatting variants. Unresolved submitted identifiers remain in the precision denominator. '
        'Empty submissions have R and P of zero. Variation includes differences in code selection, unresolved identifier representation, and empty responses.',
        r'\begingroup',r'\small',r'\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{p{0.18\textwidth}lp{0.22\textwidth}rr}',r'\toprule',
        r'Model & Corpus & Phenotype & R: runs 1 / 2 / 3 (\%) & P: runs 1 / 2 / 3 (\%) \\',
        r'\midrule',r'\endhead'])
    for subset in ('full','read'):
        for phenotype in PHENOTYPES:
            for model,label in LABELS.items():
                group=frame[(frame.model==model)&(frame.subset==subset)&(frame.phenotype==phenotype)].sort_values('epoch')
                scores=[' / '.join(f'{100*v:.1f}' for v in group[metric]) for metric in
                        ('final_required_recall','final_acceptable_precision')]
                parts.append(tex(label)+' & '+subset+' & '+tex(phenotype.replace('_',' '))+' & '+' & '.join(scores)+r' \\')
    parts.extend([r'\bottomrule',r'\end{longtable}',r'\endgroup'])
    strict=pd.read_csv(folder/'metrics.strict.csv')
    parts.extend([r'\clearpage',r'\section*{Table A13. Sensitivity to identifier formatting}',
        'Each row averages the same 21 published submissions. Strict scoring requires literal browser identifiers. '
        'Normalized scoring additionally accepts missing ESCT carets and Read trailing-dot padding when the alias identifies one browser entry in scope. '
        'Alias rules are based on the browser, independently of reference labels; case and code spelling are preserved. '
        'Native SNOMED identifiers are outside these formatting rules. Changed runs contain at least one normalized identifier. '
        'All original submissions and both scoring versions are retained.',
        r'\begin{center}',r'\begin{tabular}{p{0.22\textwidth}lrrrrr}',r'\toprule',
        r'Model & Corpus & Strict R & Norm. R & Strict P & Norm. P & Changed runs \\',r'\midrule'])
    for model,label in LABELS.items():
        for subset in ('full','read'):
            group=frame[(frame.model==model)&(frame.subset==subset)]
            literal=strict[(strict.model==model)&(strict.subset==subset)]
            changed=sum(bool(a['normalizations']) for a in audits if a['model']==model and a['subset']==subset)
            values=[literal.final_required_recall.mean(),group.final_required_recall.mean(),
                    literal.final_acceptable_precision.mean(),group.final_acceptable_precision.mean()]
            parts.append(tex(label)+' & '+subset+' & '+' & '.join(f'{100*v:.1f}' for v in values)+f' & {changed}'+r' \\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{center}',
        'R and P are percentages. Their definitions and empty-submission convention are identical across both scoring versions.'])
    (destination/'study_published_comparison.tex').write_text('\n'.join(parts)+'\n')
