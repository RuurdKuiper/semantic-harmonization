"""Disk-backed OpenAI embeddings with a durable checkpoint after every batch."""
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from experiments.pipeline_design.parallel_benchmark import atomic_json,locked
from experiments.pipeline_design.study import fingerprint


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:np.savez_compressed(handle,**arrays)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def input_batches(texts,start,settings):
    """Never truncate input; byte bounds conservatively satisfy API token limits."""
    while start<len(texts):
        end=start;size=0
        while end<len(texts) and end-start<settings['batch_size']:
            length=len(texts[end].encode('utf-8'))
            if not length or length>settings['max_input_bytes']:
                raise ValueError(f'Input row {end} is empty or exceeds the conservative byte limit; no input is truncated.')
            if length>settings['max_batch_bytes']:raise ValueError('One input exceeds the batch byte budget.')
            if size+length>settings['max_batch_bytes']:break
            size+=length;end+=1
        yield start,end
        start=end


def request_vectors(client,texts,settings):
    response=client.embeddings.create(model=settings['model'],input=texts,
        dimensions=settings['dimensions'],encoding_format='base64')
    data=sorted(response.data,key=lambda x:x.index)
    if [x.index for x in data]!=list(range(len(texts))):raise ValueError('Embedding response does not cover inputs exactly once.')
    vectors=np.asarray([np.frombuffer(base64.b64decode(x.embedding),dtype='<f4')
        if isinstance(x.embedding,str) else x.embedding for x in data],dtype=np.float32)
    if vectors.shape!=(len(texts),settings['dimensions']) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding response shape or values.')
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if (norms<=1e-12).any():raise ValueError('Zero-length embedding response.')
    vectors/=norms
    return vectors,int(response.usage.total_tokens),getattr(response,'_request_id',None)


def build_cache(frame,folder,settings,client,progress=None):
    """Commit vectors before advancing offset; resume without repeating saved calls."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    texts=frame.description.fillna('').astype(str).tolist()
    signature=fingerprint({'rows':list(frame[['code','vocabulary','description']].itertuples(index=False,name=None)),
                           'model':settings['model'],'dimensions':settings['dimensions'],'normalized':True})
    manifest=folder/'state.json';matrix=folder/'vectors.npy'
    with locked(folder/'writer.lock'):
        if manifest.exists():
            state=json.loads(manifest.read_text())
            if state['signature']!=signature:raise ValueError(f'Embedding inputs changed: {folder}')
            if not matrix.exists():raise ValueError('Embedding checkpoint exists without its vector matrix.')
            vectors=np.load(matrix,mmap_mode='r+')
            if vectors.shape!=(len(frame),settings['dimensions']):raise ValueError('Embedding matrix shape changed.')
        else:
            # Validate every input before the first paid request.
            for _ in input_batches(texts,0,settings):pass
            vectors=np.lib.format.open_memmap(matrix,mode='w+',dtype=np.float32,
                                               shape=(len(frame),settings['dimensions']))
            state={'signature':signature,'rows':len(frame),'dimensions':settings['dimensions'],
                   'model':settings['model'],'completed_rows':0,'total_tokens':0,'requests':0,'seconds':0.,
                   'complete':False,'byte_count':sum(len(t.encode('utf-8')) for t in texts)}
            frame[['code','vocabulary','description']].to_parquet(folder/'rows.parquet',index=False)
            atomic_json(manifest,state)
        if state['complete']:
            if state['completed_rows']!=len(frame):raise ValueError('Inconsistent completed embedding checkpoint.')
            return state
        for start,end in input_batches(texts,state['completed_rows'],settings):
            print(f'[embeddings/{frame.vocabulary.iloc[0]}] {start+1:,}--{end:,} of {len(frame):,}',flush=True)
            began=perf_counter();batch,tokens,request_id=request_vectors(client,texts[start:end],settings)
            vectors[start:end]=batch;vectors.flush()
            state.update(completed_rows=end,complete=end==len(frame),
                         total_tokens=state['total_tokens']+tokens,requests=state['requests']+1,
                         seconds=state['seconds']+perf_counter()-began,
                         last_request_id=request_id,updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(manifest,state)
            if progress:progress(state)
        return state


def cached_queries(queries,folder,settings,client):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    result=[]
    for name,text in queries.items():
        signature=fingerprint({'text':text,'model':settings['model'],'dimensions':settings['dimensions']})
        path=folder/f'{name}_{signature}.npz'
        if path.exists():
            with np.load(path) as saved:vector=saved['embedding']
        else:
            for _ in input_batches([text],0,settings):pass
            vectors,tokens,request_id=request_vectors(client,[text],settings);vector=vectors[0]
            atomic_npz(path,embedding=vector)
            atomic_json(path.with_suffix('.json'),{'signature':signature,'query_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'tokens':tokens,'request_id':request_id})
        if vector.shape!=(settings['dimensions'],) or not np.isfinite(vector).all():raise ValueError('Invalid cached query.')
        result.append(vector)
    return np.stack(result)


def cosine_scores(cache_folders,queries,block_size=4096):
    """Read matrix blocks once; keep the 3072-dimensional corpus off the Python heap."""
    matrices=[np.load(Path(folder)/'vectors.npy',mmap_mode='r') for folder in cache_folders]
    scores=np.empty((sum(len(matrix) for matrix in matrices),len(queries)),dtype=np.float64)
    offset=0
    for matrix in matrices:
        for start in range(0,len(matrix),block_size):
            block=matrix[start:start+block_size]
            for j,query in enumerate(queries):
                # Match EmbeddingIndex.score_all's six-decimal rounding before normalization.
                scores[offset+start:offset+start+len(block),j]=[float(round(v,6)) for v in block@query]
        offset+=len(matrix)
    return scores
