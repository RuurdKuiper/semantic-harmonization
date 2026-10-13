"""Use prepared local snapshots when available, retaining exact model revisions."""
import json
import gc
import sys
from pathlib import Path


def prepared_model_path(repo):
    root=Path(__file__).resolve().parents[2]
    receipt=root/'.model-cache/model_revisions.json'
    if receipt.exists():
        info=json.loads(receipt.read_text()).get(repo)
        if info and Path(info['path']).is_dir():return info['path']
    return repo


def release_local_models():
    """Release cached weights before another local model or phase is loaded."""
    for module_name,function in (('src.llm.decisions','_laya_router'),
                                 ('src.llm.jpt_runtime','load_jpt'),
                                 ('src.llm.local','load_mlx_reviewer')):
        module=sys.modules.get(module_name)
        if module is not None:getattr(module,function).cache_clear()
    gc.collect()
    torch=sys.modules.get('torch')
    if torch is not None and torch.backends.mps.is_available():torch.mps.empty_cache()
    mx=sys.modules.get('mlx.core')
    if mx is not None:mx.clear_cache()
