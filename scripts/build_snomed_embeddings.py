"""Build resumable MiniLM/OpenAI indexes for the active SNOMED corpus only.

Run with PYTHONPATH=. .venv/bin/python scripts/build_snomed_embeddings.py
--provider local|openai. Exact code/vocabulary/description matches may reuse
the original read-only vectors; changed wording is always embedded again.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from src.data.code_systems import load_code_system_corpus
from src.data.preprocessing import preprocess_corpus
from src.retrieval.embeddings import EmbeddingIndex, embedding_cache_key
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def matching_positions(current, previous):
    columns = ['code', 'vocabulary', 'description']
    lookup = {row: i for i, row in enumerate(previous[columns].itertuples(index=False, name=None))}
    return np.array([lookup.get(row, -1) for row in current[columns].itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', required=True, choices=['local', 'openai'])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error('Batch size and worker count must be positive')
    os.chdir(ROOT)
    load_dotenv(ROOT / '.env')
    config = load_config(ROOT / 'configs/default.yaml')
    corpus = preprocess_corpus(load_code_system_corpus(
        ['SNOMEDCT_US'], source_paths=config.paths.code_systems,
        cache_dir=config.paths.codes_parquet_dir,
    ))[['code', 'vocabulary', 'description']]
    model_name = 'sentence-transformers/all-MiniLM-L6-v2' if args.provider == 'local' else 'text-embedding-3-large'
    dimensions = 384 if args.provider == 'local' else 3072
    identity = embedding_cache_key(args.provider, model_name, None if args.provider == 'local' else dimensions)
    destination = ROOT / config.paths.embeddings_dir
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / f'embeddings_SNOMEDCT_US_{identity}.npz'
    work = destination / f'{identity}_resumable' / 'SNOMEDCT_US'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'writer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signature = hashlib.sha256(corpus.to_csv(index=False).encode()).hexdigest()
    state_path = work / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (state['signature'], state['model'], state['dimensions']) != (signature, model_name, dimensions):
            raise ValueError('Checkpoint belongs to a different corpus/model; refusing reuse')
        vectors = np.load(work / 'vectors.npy', mmap_mode='r+')
        done = np.load(work / 'done.npy', mmap_mode='r+')
    else:
        vectors = np.lib.format.open_memmap(work / 'vectors.npy', mode='w+', dtype=np.float32, shape=(len(corpus), dimensions))
        done = np.lib.format.open_memmap(work / 'done.npy', mode='w+', dtype=bool, shape=(len(corpus),))
        done[:] = False
        state = dict(signature=signature, model=model_name, provider=args.provider,
                     dimensions=dimensions, rows=len(corpus), requests=0, total_tokens=0,
                     reused_rows=0, generated_rows=0, complete=False)
        if args.provider == 'local':
            old_cache = ROOT / 'data/codes/embeddings' / cache.name
            old = EmbeddingIndex.load(old_cache, load_model=False)
            previous = pd.DataFrame(dict(code=old.codes, vocabulary=old.vocabularies, description=old.descriptions))
            old_vectors = old.embeddings
        else:
            old_dir = ROOT / 'data/codes/embeddings/openai_text-embedding-3-large_3072_resumable/SNOMEDCT_US'
            old_state = json.loads((old_dir / 'state.json').read_text())
            if not old_state['complete'] or old_state['completed_rows'] != old_state['rows']:
                raise ValueError('Original OpenAI checkpoint is incomplete')
            if (old_state['model'], old_state['dimensions']) != (model_name, dimensions):
                raise ValueError('Original OpenAI model/dimensions do not match')
            previous = pd.read_parquet(old_dir / 'rows.parquet')
            old_vectors = np.load(old_dir / 'vectors.npy', mmap_mode='r')
        positions = matching_positions(corpus, previous)
        reuse = np.flatnonzero(positions >= 0)
        for start in range(0, len(reuse), 1024):
            ids = reuse[start:start + 1024]
            vectors[ids] = old_vectors[positions[ids]]
            done[ids] = True
        vectors.flush()
        done.flush()
        state['reused_rows'] = len(reuse)
        state['reuse_source'] = str(old_cache if args.provider == 'local' else old_dir)
        corpus.to_parquet(work / 'rows.parquet', index=False)
        del old_vectors, previous
        if args.provider == 'local':
            del old
        atomic_json(state_path, state)

    def checkpoint():
        vectors.flush()
        done.flush()
        state['completed_rows'] = int(done.sum())
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)

    missing = np.flatnonzero(~done)
    print(json.dumps(dict(provider=args.provider, rows=len(corpus), reused=state['reused_rows'], pending=len(missing))), flush=True)
    started = time.monotonic()
    if args.provider == 'local' and len(missing):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        # A cached snapshot avoids network refreshes and preserves the revision.
        hub = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
        revision = (hub / 'refs/main').read_text().strip()
        model = SentenceTransformer(str(hub / 'snapshots' / revision), device='cpu', local_files_only=True)
        state['model_revision'] = revision
        for start in range(0, len(missing), 4096):
            ids = missing[start:start + 4096]
            vectors[ids] = model.encode(corpus.iloc[ids].description.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
            done[ids] = True
            state['generated_rows'] += len(ids)
            checkpoint()
            print(f"local: {state['completed_rows']}/{len(corpus)} rows ready", flush=True)
    elif len(missing):
        from openai import OpenAI
        import base64
        client = OpenAI(timeout=90, max_retries=5)

        def request(ids):
            response = client.embeddings.create(
                model=model_name, dimensions=dimensions, encoding_format='base64',
                input=corpus.iloc[ids].description.tolist(),
            )
            returned = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in returned] != list(range(len(ids))):
                raise ValueError('Embedding response indices are incomplete')
            batch = np.stack([np.frombuffer(base64.b64decode(item.embedding), dtype=np.float32) for item in returned])
            if batch.shape != (len(ids), dimensions) or not np.isfinite(batch).all():
                raise ValueError('Invalid embedding response matrix')
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError('Embedding response contains zero vectors')
            return ids, batch / norms, response.usage.total_tokens

        batches = iter(missing[start:start + args.batch_size] for start in range(0, len(missing), args.batch_size))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(request, ids) for _, ids in zip(range(args.workers), batches)}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    ids, batch, tokens = future.result()
                    vectors[ids] = batch
                    done[ids] = True
                    state['generated_rows'] += len(ids)
                    state['requests'] += 1
                    state['total_tokens'] += tokens
                    checkpoint()
                    print(f"openai: {state['completed_rows']}/{len(corpus)} rows ready; {state['requests']} requests", flush=True)
                    ids = next(batches, None)
                    if ids is not None:
                        pending.add(pool.submit(request, ids))
        client.close()

    # Check every vector in bounded blocks before publishing an atomic NPZ.
    if not done.all():
        raise ValueError('Corpus build is incomplete')
    for start in range(0, len(corpus), 4096):
        block = vectors[start:start + 4096]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-5):
            raise ValueError('Non-finite or unnormalised embeddings')
    print(f'{args.provider}: saving final index', flush=True)
    temporary = cache.with_suffix('.partial.npz')
    EmbeddingIndex(
        model=None, descriptions=corpus.description.tolist(), codes=corpus.code.tolist(),
        vocabularies=corpus.vocabulary.tolist(), embeddings=vectors,
    ).save(temporary)
    temporary.replace(cache)
    state['complete'] = True
    state['cache_path'] = str(cache.relative_to(ROOT))
    state['elapsed_seconds_last_run'] = round(time.monotonic() - started, 1)
    checkpoint()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
