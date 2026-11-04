"""Export existing MiniLM vectors for browser retrieval; never call an encoder.

Run from the repository root with PYTHONPATH=. and the project Python.
Output is deliberately outside the static website and Git-tracked files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data.code_systems import load_code_system_corpus
from src.data.preprocessing import preprocess_corpus
from src.retrieval.embeddings import EmbeddingIndex, DEFAULT_LOCAL_EMBEDDING_MODEL, embedding_cache_key
from src.retrieval.lexical import _tokenize
from src.utils.config import load_config


def write_asset(path: Path, data: bytes) -> dict:
    path.write_bytes(data)
    return {"url": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def json_gzip(value) -> bytes:
    return gzip.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)


def export_vocabulary(corpus, index: EmbeddingIndex, output: Path, vocabulary: str, chunk_rows=4096) -> dict:
    """Require exact code/vocabulary/description matches with the saved vectors."""
    if index.embeddings.ndim != 2 or index.embeddings.shape[1] != 384:
        raise ValueError("Browser MiniLM packs require 384-dimensional vectors")
    lookup = {(code, vocab, desc): i for i, (code, vocab, desc) in
              enumerate(zip(index.codes, index.vocabularies, index.descriptions))}
    rows = list(corpus[["code", "vocabulary", "description"]].itertuples(index=False, name=None))
    try:
        positions = [lookup[row] for row in rows]
    except KeyError as exc:
        raise ValueError(f"Saved MiniLM vectors do not match the configured {vocabulary} corpus") from exc
    if not rows:
        raise ValueError(f"Empty corpus: {vocabulary}")

    documents = [_tokenize(desc) or [desc] for _, _, desc in rows]
    frequencies = Counter(token for document in documents for token in dict.fromkeys(document))
    count = len(rows)
    idfs = {token: math.log(count - df + .5) - math.log(df + .5) for token, df in frequencies.items()}
    average_idf = sum(idfs.values()) / len(idfs)
    lexical = {"count": count, "avgdl": sum(map(len, documents)) / count,
               "averageIdf": average_idf, "frequencies": frequencies, "k1": 1.5, "b": .75, "epsilon": .25}
    # Content-address the vocabulary so cached old and new releases cannot mix.
    digest = hashlib.sha256(json_gzip(lexical))
    chunks = []
    staging = output / (vocabulary + ".staging")
    staging.mkdir(parents=True, exist_ok=True)
    lexical_asset = write_asset(staging / "lexical.json.gz", json_gzip(lexical))
    for start in range(0, count, chunk_rows):
        selected = rows[start:start + chunk_rows]
        vectors = np.asarray(index.embeddings[positions[start:start + chunk_rows]], dtype="<f4")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite vectors in {vocabulary}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError(f"Zero vectors in {vocabulary}")
        vectors = np.asarray(vectors / norms, dtype="<f4")
        metadata = json_gzip([[code, desc] for code, _, desc in selected])
        binary = vectors.tobytes()
        digest.update(metadata)
        digest.update(binary)
        stem = f"{start:07d}"
        chunks.append({"start": start, "count": len(selected),
                       "rows": write_asset(staging / (stem + ".json.gz"), metadata),
                       "vectors": write_asset(staging / (stem + ".f32"), binary)})
    identity = vocabulary + "-" + digest.hexdigest()[:20]
    destination = output / identity
    if destination.exists():
        # Same content has the same identity; replace only these generated assets.
        for path in staging.iterdir():
            path.replace(destination / path.name)
        staging.rmdir()
    else:
        staging.rename(destination)
    for asset in [lexical_asset, *(asset for chunk in chunks for asset in (chunk["rows"], chunk["vectors"]))]:
        asset["url"] = identity + "/" + asset["url"]
    return {"id": vocabulary, "version": identity, "count": count, "lexical": lexical_asset,
            "chunks": chunks, "bytes": sum(a["bytes"] for a in
            [lexical_asset, *(a for chunk in chunks for a in (chunk["rows"], chunk["vectors"]))])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="dist/browser-assets")
    parser.add_argument("--vocabularies", nargs="+")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocabularies = args.vocabularies or config.retrieval.vocabularies
    manifest = {"format": "semantic-harmonization-minilm-v1", "dimensions": 384,
                "model": "Xenova/all-MiniLM-L6-v2", "dtype": "fp32", "maxLength": 256,
                "vocabularies": []}
    for vocabulary in vocabularies:
        print(f"Exporting saved MiniLM vectors: {vocabulary}", flush=True)
        corpus = preprocess_corpus(load_code_system_corpus(
            [vocabulary], source_paths=config.paths.code_systems, cache_dir=config.paths.codes_parquet_dir))
        cache = Path(config.paths.embeddings_dir) / f"embeddings_{vocabulary}_{embedding_cache_key('local', DEFAULT_LOCAL_EMBEDDING_MODEL)}.npz"
        if not cache.exists():
            raise FileNotFoundError(f"Precomputed MiniLM index is missing for {vocabulary}; export does not build embeddings")
        index = EmbeddingIndex.load(cache, load_model=False)
        manifest["vocabularies"].append(export_vocabulary(corpus, index, output, vocabulary, args.chunk_rows))
        del index, corpus
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {sum(v['count'] for v in manifest['vocabularies']):,} vectors; no embedding API calls.")


if __name__ == "__main__":
    main()
