"""Export saved MiniLM or OpenAI vectors for browser retrieval; never call an encoder.

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


def quantize_unit_vectors(vectors):
    """Symmetric per-row int8, followed by float32 inverse quantized norms.

    Cosine scoring needs only the direction of each vector: its original scale
    cancels when normalizing the quantized row. No calibration EDFs are used.
    """
    maximum = np.max(np.abs(vectors), axis=1, keepdims=True)
    quantized = np.rint(vectors / maximum * 127).clip(-127, 127).astype(np.int8)
    inverse_norms = np.asarray(1 / np.linalg.norm(quantized.astype(np.float64), axis=1), dtype="<f4")
    return quantized.tobytes() + inverse_norms.tobytes()


def export_vocabulary(corpus, index: EmbeddingIndex, output: Path, vocabulary: str, chunk_rows=4096, dimensions=384, dtype="fp32", source_dimensions=None) -> dict:
    """Require exact code/vocabulary/description matches with the saved vectors."""
    expected = source_dimensions or dimensions
    if index.embeddings.ndim != 2 or index.embeddings.shape[1] != expected or dimensions > expected:
        raise ValueError(f"Browser packs require {expected}-dimensional source vectors")
    if dtype not in {"fp32", "int8"}:
        raise ValueError("Unsupported vector storage format")
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
        vectors = np.asarray(index.embeddings[positions[start:start + chunk_rows], :dimensions], dtype="<f4")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite vectors in {vocabulary}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError(f"Zero vectors in {vocabulary}")
        vectors = np.asarray(vectors / norms, dtype="<f4")
        metadata = json_gzip([[code, desc] for code, _, desc in selected])
        binary = quantize_unit_vectors(vectors) if dtype == "int8" else vectors.tobytes()
        digest.update(metadata)
        digest.update(binary)
        stem = f"{start:07d}"
        chunks.append({"start": start, "count": len(selected),
                       "rows": write_asset(staging / (stem + ".json.gz"), metadata),
                       "vectors": write_asset(staging / (stem + (".i8" if dtype == "int8" else ".f32")), binary)})
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
    parser.add_argument("--output-dir")
    parser.add_argument("--vocabularies", nargs="+")
    parser.add_argument("--provider", choices=["minilm", "openai"], default="minilm")
    parser.add_argument("--dtype", choices=["fp32", "int8"])
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--previous-manifest", type=Path, default=Path("dist/browser-assets/manifest.json"))
    parser.add_argument("--chunk-rows", type=int)
    args = parser.parse_args()
    dtype = args.dtype or "int8"
    is_openai = args.provider == "openai"
    dimensions = args.dimensions if args.dimensions is not None else (256 if is_openai and dtype == "int8" else 3072 if is_openai else 384)
    if (is_openai and not 1 <= dimensions <= 3072) or (not is_openai and dimensions != 384):
        parser.error("MiniLM requires 384 dimensions; OpenAI supports 1 to 3072")
    chunk_rows = args.chunk_rows if args.chunk_rows is not None else (2048 if is_openai and dtype == "fp32" else 4096)
    if chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    config = load_config(args.config)
    default_output = (f"dist/browser-openai-{dimensions}-int8-assets" if dtype == "int8" else "dist/browser-openai-assets") if is_openai else ("dist/browser-minilm-int8-assets" if dtype == "int8" else "dist/browser-assets")
    output = Path(args.output_dir or default_output)
    output.mkdir(parents=True, exist_ok=True)
    vocabularies = args.vocabularies or config.retrieval.vocabularies
    manifest = {"format": f"semantic-harmonization-{args.provider}-v1", "dimensions": dimensions,
                "model": "text-embedding-3-large" if is_openai else "Xenova/all-MiniLM-L6-v2", "dtype": dtype, "maxLength": 8192 if is_openai else 256,
                "vocabularies": []}
    if dtype == "int8":
        manifest["quantization"] = "symmetric-int8-unit-vectors-v1"
    for vocabulary in vocabularies:
        print(f"Exporting saved {args.provider} vectors: {vocabulary}", flush=True)
        corpus = preprocess_corpus(load_code_system_corpus(
            [vocabulary], source_paths=config.paths.code_systems, cache_dir=config.paths.codes_parquet_dir))
        identity = embedding_cache_key('openai', 'text-embedding-3-large', 3072) if is_openai else embedding_cache_key('local', DEFAULT_LOCAL_EMBEDDING_MODEL)
        cache = Path(config.paths.embeddings_dir) / f"embeddings_{vocabulary}_{identity}.npz"
        checkpoint = next((path for path in [Path(config.paths.embeddings_dir) / 'openai_text-embedding-3-large_3072_resumable' / vocabulary,
                                            Path('data/codes/embeddings/openai_text-embedding-3-large_3072_resumable') / vocabulary]
                           if (path / 'state.json').is_file()), None) if is_openai else None
        if checkpoint is not None:
            state = json.loads((checkpoint / 'state.json').read_text())
            if not state.get('complete') or state.get('dimensions') != 3072 or state.get('model') != 'text-embedding-3-large':
                raise ValueError(f"The saved OpenAI checkpoint for {vocabulary} is incomplete or incompatible")
            import pandas as pd
            rows = pd.read_parquet(checkpoint / 'rows.parquet')
            index = EmbeddingIndex(None, rows.description.tolist(), rows.code.tolist(), rows.vocabulary.tolist(),
                                   np.load(checkpoint / 'vectors.npy', mmap_mode='r'))
        elif cache.exists():
            index = EmbeddingIndex.load(cache, load_model=False)
        else:
            raise FileNotFoundError(f"Precomputed index is missing for {vocabulary}; export does not build embeddings")
        manifest["vocabularies"].append(export_vocabulary(corpus, index, output, vocabulary, chunk_rows, dimensions, dtype,
                                                         source_dimensions=3072 if is_openai else 384))
        del index, corpus
    if dtype == "int8" and args.previous_manifest.is_file():
        previous = json.loads(args.previous_manifest.read_text())
        if previous.get("format") == manifest["format"] and previous.get("dtype") == "fp32":
            manifest["obsoleteVectorHashes"] = sorted({chunk["vectors"]["sha256"]
                for vocabulary in previous["vocabularies"] for chunk in vocabulary["chunks"]})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {sum(v['count'] for v in manifest['vocabularies']):,} vectors; no embedding API calls.")


if __name__ == "__main__":
    main()
