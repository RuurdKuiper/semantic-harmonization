#!/usr/bin/env python3
"""Run one resumable study phase, or the entire ordered validation/test study."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from dotenv import load_dotenv
import yaml
from experiments.pipeline_design.study import Study, save_json

PHASES=('retrieval','decisions','review','mapping','freeze','test','benchmark')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--phase',choices=(*PHASES,'all'),default='retrieval')
    parser.add_argument('--output-dir',help='Override the dated results directory.')
    parser.add_argument('--models',nargs='+',help='Run selected decision-model names; selection waits for all models.')
    parser.add_argument('--report',action='store_true',help='Refresh the report after each phase and on interruption.')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    # Keep new model downloads inside the workspace. Existing terminology
    # embedding indexes are read from their configured directory.
    os.environ.setdefault('HF_HOME',str(ROOT/'.model-cache/huggingface'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
    config=yaml.safe_load(Path(args.config).read_text())
    if args.output_dir:config['output_dir']=args.output_dir
    study=Study(config)
    if args.models and args.phase!='decisions':parser.error('--models applies to --phase decisions only.')
    status=study.output/('status.'+'-'.join(args.models)+'.json' if args.models else 'status.json')
    def refresh_report():
        if not args.report:return
        command=[sys.executable,str(Path(__file__).with_name('report.py')),'--config',args.config]
        if args.output_dir:command+=['--output-dir',args.output_dir]
        subprocess.run(command,check=True)
    for phase in PHASES if args.phase=='all' else (args.phase,):
        print(f'=== Study phase: {phase} ===',flush=True)
        save_json(status,{'phase':phase,'status':'running','updated_at':datetime.now(timezone.utc).isoformat()})
        try:
            if phase=='freeze' and (study.output/'frozen_pipelines.json').exists():
                study.frozen()  # Verify the existing immutable freeze before continuing.
            elif phase=='decisions':study.decisions(args.models)
            else:getattr(study,phase)()
        except (Exception,KeyboardInterrupt) as exc:
            message=str(exc)
            for key,value in os.environ.items():
                if any(word in key for word in ('KEY','TOKEN','PASSWORD')) and len(value)>8:
                    message=message.replace(value,'[redacted]')
            save_json(status,{'phase':phase,'status':'incomplete','error_type':type(exc).__name__,
                              'error':message[:1000],'updated_at':datetime.now(timezone.utc).isoformat()})
            print(f'{phase} stopped: {type(exc).__name__}: {message[:500]}',file=sys.stderr,flush=True)
            print('Completed batches are saved. Rerun the same phase to resume.',file=sys.stderr,flush=True)
            refresh_report()
            return 1
        save_json(status,{'phase':phase,'status':'complete','updated_at':datetime.now(timezone.utc).isoformat()})
        refresh_report()
    return 0


if __name__=='__main__':raise SystemExit(main())
