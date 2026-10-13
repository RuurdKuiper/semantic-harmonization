#!/usr/bin/env python3
"""Reuse compatible preliminary outputs when introducing a shared candidate cap.

The original run directory is preserved. Only a verified ranked prefix is
adopted; changed review inputs are not imported.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from experiments.pipeline_design.study import Study,fingerprint,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir',required=True)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text());study=Study(config)
    source=ROOT/args.from_dir
    if source.resolve()==study.output.resolve():raise ValueError('Use a separate destination.')
    cap=config['classification']['max_candidates'];old_settings=deepcopy(config['classification'])
    old_settings.pop('max_candidates')
    selection=json.loads((source/'retrieval_selection.json').read_text());weight=selection['lexical_weight']
    selection['config_hash']=fingerprint(config);save_json(study.output/'retrieval_selection.json',selection)
    shutil.copyfile(source/'retrieval.csv',study.output/'retrieval.csv')
    for path in (source/'scores').glob('*.npz'):
        target=study.output/'scores'/path.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copyfile(path,target)
    adopted=[]
    for model in config['decision_models']:
        for name in config['validation_edfs']:
            path=source/'decisions'/model['name']/f'{name}.json'
            if not path.exists():continue
            old=json.loads(path.read_text())
            signature={'source':study.source_signature(name),'competitor':model,'weight':weight,'stopping':old_settings}
            if old['signature']!=fingerprint(signature):raise ValueError(f'Incompatible source: {path}')
            result=deepcopy(old);result['records']=old['records'][:cap]
            batches=len(result['records'])//config['classification']['batch_size']
            result['narrow_counts']=old['narrow_counts'][:batches]
            result['batch_seconds']=old.get('batch_seconds',[])[:batches]
            result['model_seconds']=sum(result['batch_seconds'])
            capped=len(result['records'])==cap and not (old.get('stopped_early') and len(old['records'])<=cap)
            result.update(complete=old['complete'] or capped,stopped_early=old.get('stopped_early',False) and not capped,
                          candidate_ceiling_hit=capped,stop_reason='candidate_ceiling' if capped else
                          'adaptive_zero_narrow' if old.get('stopped_early') else 'in_progress',
                          signature=fingerprint({**signature,'stopping':config['classification']}),
                          adopted_from=str(path.relative_to(ROOT)),source_record_count=len(old['records']))
            target=study.output/'decisions'/model['name']/f'{name}.json'
            if target.exists():raise ValueError(f'Existing destination checkpoint: {target}')
            save_json(target,result);adopted.append({'model':model['name'],'edf':name,'records':len(result['records']),
                                                   'ceiling_hit':capped,'complete':result['complete']})
            # An unchanged decision set gives exactly the same reviewer inputs.
            if old['records']!=result['records']:continue
            for reviewer in config['review_models']:
                path=source/'reviews'/reviewer['name']/model['name']/f'{name}.json'
                if not path.exists():continue
                review=json.loads(path.read_text())
                settings={k:v for k,v in config['review'].items() if k!='qwen_thinking_budget' or reviewer['provider']=='mlx'}
                expected=fingerprint({'decision':old['signature'],'competitor':reviewer,'review':settings})
                if review['signature']!=expected:raise ValueError(f'Incompatible review source: {path}')
                review['signature']=fingerprint({'decision':result['signature'],'competitor':reviewer,'review':settings})
                review['adopted_from']=str(path.relative_to(ROOT))
                save_json(study.output/'reviews'/reviewer['name']/model['name']/f'{name}.json',review)
    save_json(study.output/'checkpoint_adoption.json',{'source':str(source.relative_to(ROOT)),
              'reason':'User selected shared 10,000-candidate ceiling after observing non-stopping Laya output.',
              'candidate_ceiling':cap,'adopted':adopted})
    print(json.dumps(adopted,indent=2))


if __name__=='__main__':main()
