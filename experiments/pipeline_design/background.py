#!/usr/bin/env python3
"""Launch one detached, resumable study worker and print its log location."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from experiments.pipeline_design.study import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    output=ROOT/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    marker=output/'worker.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f"Worker PID {previous['pid']} is still alive. Inspect it before starting another run.")
    command=[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--config',args.config,'--phase','all','--report']
    log=output/'run.log'
    with log.open('a') as handle:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    save_json(marker,{'pid':process.pid,'command':command,'log':str(log.relative_to(ROOT))})
    print(f'Worker PID {process.pid}. Progress: {log}')
    print('Successful batches are checkpointed; rerun this launcher after a stopped worker to resume.')


if __name__=='__main__':main()
