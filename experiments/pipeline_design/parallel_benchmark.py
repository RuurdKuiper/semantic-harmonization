#!/usr/bin/env python3
"""Run a frozen benchmark branch with shared checkpoints and protected results.

The frozen Study implementation and all inference settings are reused unchanged.
Only branch scheduling, cache locking, and atomic result publication differ.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import pandas as pd
import yaml
from dotenv import load_dotenv
from experiments.pipeline_design import study as study_module

BRANCHES=('open_weight','proprietary')
KEYS=['branch','subset','phenotype']


@contextmanager
def locked(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:json.dump(value,handle,indent=2,default=str)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def merge_metrics(path,incoming,csv_writer=None):
    """Upsert completed cases under a process lock without dropping other branches."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if incoming.empty:return
    incoming=incoming.copy()
    incoming[KEYS]=incoming[KEYS].astype(str)
    if incoming.duplicated(KEYS).any():raise ValueError('Duplicate benchmark case in incoming results.')
    writer=csv_writer or pd.DataFrame.to_csv
    with locked(path.with_suffix('.lock')):
        previous=pd.read_csv(path,dtype={key:str for key in KEYS}) if path.exists() else pd.DataFrame()
        combined=pd.concat([previous,incoming],ignore_index=True).drop_duplicates(KEYS,keep='last')
        combined=combined.sort_values(KEYS).reset_index(drop=True)
        fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
        try:
            writer(combined,name,index=False)
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)


def update_status(output,branch,status,**details):
    now=datetime.now(timezone.utc).isoformat()
    item={'phase':'benchmark','branch':branch,'status':status,'pid':os.getpid(),'updated_at':now,**details}
    with locked(output/'status.benchmark.lock'):
        atomic_json(output/f'status.benchmark.{branch}.json',item)
        workers={b:json.loads(p.read_text()) for b in BRANCHES
                 if (p:=output/f'status.benchmark.{b}.json').exists()}
        states=[workers.get(b,{}).get('status','pending') for b in BRANCHES]
        overall='complete' if all(s=='complete' for s in states) else 'running' if 'running' in states else 'incomplete'
        atomic_json(output/'status.json',{'phase':'benchmark','status':overall,'updated_at':now,'workers':workers})


@contextmanager
def branch_execution(output,branch):
    """Adapt scheduling and publication without editing frozen scientific code."""
    original_class=study_module.Study
    original_save=study_module.save_json
    original_csv=pd.DataFrame.to_csv
    metrics=(output/'benchmark_metrics.csv').resolve()

    class BranchStudy(original_class):
        def frozen(self):
            frozen=super().frozen()
            if branch not in frozen['branches']:raise ValueError('Requested branch is absent from the frozen pipelines.')
            return {**frozen,'branches':{branch:frozen['branches'][branch]}}

        def scores(self,phenotype):
            with locked(self.output/'scores'/f'{phenotype}.lock'):
                return super().scores(phenotype)

    def publish(frame,path_or_buf=None,*args,**kwargs):
        if isinstance(path_or_buf,(str,os.PathLike)) and Path(path_or_buf).resolve()==metrics:
            if not frame.empty and set(frame.branch)!={branch}:
                raise ValueError('A branch worker attempted to publish another branch.')
            return merge_metrics(metrics,frame,csv_writer=original_csv)
        return original_csv(frame,path_or_buf,*args,**kwargs)

    study_module.Study=BranchStudy
    study_module.save_json=atomic_json
    pd.DataFrame.to_csv=publish
    try:yield BranchStudy
    finally:
        study_module.Study=original_class
        study_module.save_json=original_save
        pd.DataFrame.to_csv=original_csv


def redact(message):
    for key,value in os.environ.items():
        if any(word in key for word in ('KEY','TOKEN','PASSWORD')) and len(value)>8:
            message=message.replace(value,'[redacted]')
    return message


def run(config,config_path,branch,report):
    output=ROOT/config['output_dir']
    # Hold this lock for the entire process; the next worker cannot duplicate calls.
    with locked(output/f'worker.benchmark.{branch}.lock',blocking=False):
        update_status(output,branch,'running')
        success=False
        try:
            with branch_execution(output,branch) as Study:
                worker=Study(config)
                os.environ.setdefault('LOCAL_RESPONSE_DIR',str(output/'local_responses'))
                receipt={'branch':branch,'pid':os.getpid(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'frozen_sha256':(output/'frozen_pipelines.sha256').read_text().strip(),
                         'scheduling':'Independent branch processes, locked score caches, atomic merged metrics; unchanged frozen Study methods.'}
                atomic_json(output/f'parallel_benchmark.{branch}.json',receipt)
                print(f'[parallel benchmark] {branch}: resuming saved checkpoints; independent model execution',flush=True)
                worker.benchmark()
            update_status(output,branch,'complete')
            print(f'[parallel benchmark] {branch}: all 14 cases complete',flush=True)
            success=True
        except (Exception,KeyboardInterrupt) as exc:
            message=redact(str(exc))
            update_status(output,branch,'incomplete',error_type=type(exc).__name__,error=message[:1000])
            print(f'[parallel benchmark] {branch} stopped: {type(exc).__name__}: {message[:500]}',flush=True)
            print('Completed batches are saved. Rerun the same branch to resume.',flush=True)
        if report:
            # Both workers can finish at once; serialize manuscript generation too.
            with locked(output/'report.lock'):
                subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),'--config',str(config_path)],check=True)
        return 0 if success else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--branch',choices=BRANCHES,required=True)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();config_path=Path(args.config).resolve()
    config=yaml.safe_load(config_path.read_text());output=ROOT/config['output_dir']
    load_dotenv(ROOT/'.env')
    os.environ.setdefault('HF_HOME',str(ROOT/'.model-cache/huggingface'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
    if args.detach:
        # Reserve startup so two launcher calls cannot race before child takes its lock.
        with locked(output/f'launch.benchmark.{args.branch}.lock'):
            with locked(output/f'worker.benchmark.{args.branch}.lock',blocking=False):pass
            marker=output/f'worker.benchmark.{args.branch}.json'
            if marker.exists():
                try:os.kill(json.loads(marker.read_text())['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('This benchmark branch already has a live worker.')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(config_path),'--branch',args.branch]
            if args.report:command.append('--report')
            log=output/f'run.benchmark.{args.branch}.log'
            with log.open('a') as handle:
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(marker,{'pid':process.pid,'command':command,'log':str(log.relative_to(ROOT))})
            print(f'Worker PID {process.pid}. Progress: {log}',flush=True)
        return 0
    return run(config,config_path,args.branch,args.report)


if __name__=='__main__':raise SystemExit(main())
