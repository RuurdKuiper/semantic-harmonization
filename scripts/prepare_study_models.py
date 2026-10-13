"""Download and pin the exact public checkpoints used in the local study."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.model-cache/huggingface'))
from huggingface_hub import HfApi, snapshot_download

models={
    'convaiinnovations/laya':['multilingual/**'],
    'kirp/jpt-4b':['*.json','*.safetensors','*.txt','*.jinja','*.model'],
    'mlx-community/Qwen3.5-9B-8bit':['*.json','*.safetensors','*.txt','*.jinja','*.model'],
}
receipt_path=ROOT/'.model-cache/model_revisions.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
for repo,patterns in models.items():
    revision=receipt.get(repo,{}).get('revision') or HfApi().model_info(repo).sha
    print(f'[download] {repo} at {revision}',flush=True)
    path=snapshot_download(repo,revision=revision,allow_patterns=patterns)
    receipt[repo]={'revision':revision,'path':path}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2))
    print(f'[download] {repo} complete',flush=True)
