"""Compare labeled and values-only semantic queries on seven validation EDFs.

Uses deployed MiniLM int8 and OpenAI fp32 representations at their fixed ranking
weights. Reads corpus checkpoints in bounded chunks; embeds only 14 queries per
encoder and leaves frozen study outputs unchanged.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from scripts.export_browser_embeddings import quantize_unit_vectors
from scripts.validate_browser_quantization import metrics, normalize
from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import standardize_code
from src.retrieval.embeddings import create_embedding_model, DEFAULT_LOCAL_EMBEDDING_MODEL
from src.retrieval.lexical import build_lexical_index, score_lexical
from src.utils.config import load_config


def main():
    load_dotenv()
    config = load_config("configs/default.yaml")
    root = Path("dist/browser-assets")
    manifest = json.loads((root / "manifest.json").read_text())
    names = json.loads(Path("results/experiments/embedding_comparison/20261022/protocol.json").read_text())["validation_edfs"]
    edfs = [load_edf(name, config.paths.edf_dir) for name in names]
    inputs = [edf.to_embedding_query(include_labels=labels) for labels in (True, False) for edf in edfs]
    query_vectors = {}
    for provider, model, dimensions in (("local", DEFAULT_LOCAL_EMBEDDING_MODEL, 384), ("openai", "text-embedding-3-large", 3072)):
        encoder = create_embedding_model(provider, model, dimensions=dimensions)
        query_vectors[provider] = encoder.encode(inputs, normalize_embeddings=True, show_progress_bar=False).astype(np.float64).T
        print(f"Embedded {len(inputs)} {provider} queries", flush=True)
        del encoder
    count = sum(v["count"] for v in manifest["vocabularies"])
    lexical = np.zeros((count, len(names)))
    semantic = {provider: np.zeros((count, len(inputs))) for provider in query_vectors}
    keys = []
    cursor = 0
    for vocabulary in manifest["vocabularies"]:
        name = vocabulary["id"]
        directory = (Path(config.paths.embeddings_dir) if name == "SNOMEDCT_US" else Path("data/codes/embeddings")) / "openai_text-embedding-3-large_3072_resumable" / name
        state = json.loads((directory / "state.json").read_text())
        assert state["complete"] and state["dimensions"] == 3072 and state["model"] == "text-embedding-3-large"
        rows = pd.read_parquet(directory / "rows.parquet")
        lookup = {(str(r.code), str(r.vocabulary), str(r.description)): i for i, r in enumerate(rows.itertuples())}
        vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        descriptions = []
        start = cursor
        for chunk in vocabulary["chunks"]:
            metadata = json.loads(gzip.decompress((root / chunk["rows"]["url"]).read_bytes()))
            local = np.fromfile(root / chunk["vectors"]["url"], dtype="<f4").reshape(-1, 384)
            binary = quantize_unit_vectors(local)
            width = len(metadata) * 384
            quantized = np.frombuffer(binary, dtype=np.int8, count=width).reshape(-1, 384)
            inverse = np.frombuffer(binary, dtype="<f4", offset=width)
            positions = [lookup[(code, name, description)] for code, description in metadata]
            remote = np.asarray(vectors[positions], dtype=np.float32)
            remote /= np.linalg.norm(remote, axis=1, keepdims=True)
            end = cursor + len(metadata)
            semantic["local"][cursor:end] = (quantized.astype(np.float64) @ query_vectors["local"]) * inverse[:, None]
            semantic["openai"][cursor:end] = remote.astype(np.float64) @ query_vectors["openai"]
            descriptions.extend(description for code, description in metadata)
            keys.extend((standardize_code(code, name), name) for code, description in metadata)
            cursor = end
        index = build_lexical_index(pd.DataFrame({"description": descriptions}))
        for i, edf in enumerate(edfs):
            lexical[start:cursor, i] = score_lexical(edf.to_lexical_query(), index, cursor - start)
        del index, vectors, lookup, rows, remote
        print(f"Compared {name}: {cursor:,}/{count:,} codes", flush=True)
    lexical = normalize(lexical)
    report = {"scope": "Current browser corpus; seven validation EDFs; fixed encoder-specific weights; no test-set selection",
              "corpus_size": count, "query_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in inputs],
              "corpus_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(), "encoders": {}}
    for provider, weight in (("local", .5), ("openai", .35)):
        scores = normalize(semantic[provider])
        cases = []
        for i, name in enumerate(names):
            _, gold = load_aesi_dataset(config.paths.aesi_datasets[name])
            relevant = {(standardize_code(r.code, r.vocabulary), r.vocabulary) for r in gold.itertuples() if r.label == "Narrow"}
            mask = np.array([key in relevant for key in keys])
            result = {"phenotype": name}
            for label, offset in (("labeled", 0), ("unlabeled", len(names))):
                combined = np.floor((weight * lexical[:, i] + (1 - weight) * scores[:, offset + i]) * 1e6 + .5)
                result[label] = metrics(np.argsort(-combined, kind="stable"), mask)
            cases.append(result)
        means = {label: {metric: float(np.mean([case[label][metric] for case in cases]))
                         for metric in ("average_precision", "recall_at_100", "recall_at_1000")}
                 for label in ("labeled", "unlabeled")}
        report["encoders"][provider] = {"lexical_weight": weight, "means": means, "cases": cases}
        print(provider, json.dumps(means), flush=True)
    output = Path("results/browser_query_labels/comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
