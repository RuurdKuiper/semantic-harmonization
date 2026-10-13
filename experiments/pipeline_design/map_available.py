#!/usr/bin/env python3
"""Map completed validation reviews before the remaining local-model runs."""
import argparse
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
import pandas as pd
import yaml
from experiments.pipeline_design.study import Study,choose_winner,confusion_metrics,save_json
from src.data.loaders import load_aesi_dataset


def completed_review_choices(config,frame):
    choices={}
    for branch in ('proprietary','open_weight'):
        models=[m['name'] for m in config['review_models'] if m['branch']==branch]
        compatible=frame[frame.branch==branch]
        if len(models)==1:
            choices[branch]={'name':models[0],'basis':'sole configured reviewer; mapping completed EDFs only'}
        elif all(compatible[compatible.name==name].phenotype.nunique()==len(config['validation_edfs']) for name in models):
            selected=choose_winner(compatible.to_dict('records'),config['validation_edfs'])
            choices[branch]={**selected,'basis':'all compatible reviewer comparisons complete'}
    return choices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/pipeline_design/config.yaml')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    config=yaml.safe_load((ROOT/args.config).read_text());study=Study(config);output=study.output
    path=output/'review_metrics.csv'
    if not path.exists():path=output/'review_metrics.partial.csv'
    frame=pd.read_csv(path);choices=completed_review_choices(config,frame)
    decisions=json.loads((output/'decision_selection.json').read_text());rows=[]
    status=output/'status.mapping_available.json'
    save_json(status,{'phase':'mapping_available','status':'running','updated_at':datetime.now(timezone.utc).isoformat()})
    try:
        for branch,choice in choices.items():
            for case in config['validation_edfs']:
                path=output/'reviews'/choice['name']/decisions[branch]['name']/(case+'.json')
                if not path.exists():continue
                baseline=json.loads(path.read_text())
                if not baseline.get('complete'):continue
                mapped=study.mapping_case(branch,case,baseline)
                _,gold=load_aesi_dataset(ROOT/study.base.paths.aesi_datasets[case])
                for stage,result in [('review',baseline),('codemapper',mapped)]:
                    rows.append({'branch':branch,'phenotype':case,'stage':stage,
                                 **confusion_metrics(result['records'],gold,study.load_corpus())})
                pd.DataFrame(rows).to_csv(output/'mapping_metrics.partial.csv',index=False)
                before,after=rows[-2:]
                print(f"[mapping/{branch}] {case}: F1 {before['f1']:.4f} -> {after['f1']:.4f}; TP {before['tp']} -> {after['tp']}; FP {before['fp']} -> {after['fp']}",flush=True)
        save_json(output/'mapping_available_choices.json',{'reviewers':choices,'completed_branch_edf_pairs':len(rows)//2,
                  'complete_validation_mapping':len(rows)==len(config['validation_edfs'])*4,
                  'selection_rule':'Final mapping selection waits for both complete validation branches.'})
        save_json(status,{'phase':'mapping_available','status':'complete','completed_branch_edf_pairs':len(rows)//2,
                          'updated_at':datetime.now(timezone.utc).isoformat()})
    except (Exception,KeyboardInterrupt) as exc:
        # Mapping exceptions never expose session credentials or server bodies.
        save_json(status,{'phase':'mapping_available','status':'incomplete','error_type':type(exc).__name__,
                          'updated_at':datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if args.report:
            subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),'--config',args.config],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
