"""Compare exported fp32 and int8 MiniLM retrieval on validation EDFs only.

Uses the current browser corpus, fixed 50/50 weighting and available Narrow
references. It does not change frozen study outputs or generate corpus vectors.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import standardize_code
from src.retrieval.embeddings import create_embedding_model, DEFAULT_LOCAL_EMBEDDING_MODEL
from src.retrieval.lexical import build_lexical_index, score_lexical
from src.utils.config import load_config


def metrics(order, positives):
    hits = positives[order]
    ranks = np.flatnonzero(hits) + 1
    count = len(ranks)
    return {"average_precision": float(np.mean(np.arange(1, count + 1) / ranks)) if count else 0.,
            "recall_at_100": float(hits[:100].sum() / count) if count else 0.,
            "recall_at_1000": float(hits[:1000].sum() / count) if count else 0.,
            "available_narrow": count}


def normalize(values):
    span = np.ptp(values, axis=0)
    return (values - values.min(axis=0)) / np.maximum(span, 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("dist/browser-assets"))
    parser.add_argument("--quantized", type=Path, default=Path("dist/browser-minilm-int8-assets"))
    parser.add_argument("--output", type=Path, default=Path("results/browser_minilm_quantization/validation.json"))
    args = parser.parse_args()
    base = load_config("configs/default.yaml")
    protocol = json.loads(Path("results/experiments/embedding_comparison/20261022/protocol.json").read_text())
    names = protocol["validation_edfs"]
    edfs = [load_edf(name, base.paths.edf_dir) for name in names]
    encoder = create_embedding_model("local", DEFAULT_LOCAL_EMBEDDING_MODEL)
    queries = encoder.encode([edf.to_embedding_query() for edf in edfs], normalize_embeddings=True, show_progress_bar=False).astype(np.float64).T
    original = json.loads((args.original / "manifest.json").read_text())
    quantized = json.loads((args.quantized / "manifest.json").read_text())
    assert original["dtype"] == "fp32" and quantized["quantization"] == "symmetric-int8-unit-vectors-v1"
    count = sum(v["count"] for v in original["vocabularies"])
    lexical = np.zeros((count, len(names)))
    fp32 = np.zeros_like(lexical)
    int8 = np.zeros_like(lexical)
    keys = []
    cursor = 0
    for vocabulary, packed in zip(original["vocabularies"], quantized["vocabularies"], strict=True):
        assert vocabulary["id"] == packed["id"] and vocabulary["count"] == packed["count"]
        descriptions = []
        start = cursor
        for chunk, reduced in zip(vocabulary["chunks"], packed["chunks"], strict=True):
            assert chunk["rows"]["sha256"] == reduced["rows"]["sha256"]
            metadata = json.loads(gzip.decompress((args.original / chunk["rows"]["url"]).read_bytes()))
            vectors = np.fromfile(args.original / chunk["vectors"]["url"], dtype="<f4").reshape(-1, 384)
            data = (args.quantized / reduced["vectors"]["url"]).read_bytes()
            width = len(metadata) * 384
            values = np.frombuffer(data, dtype=np.int8, count=width).reshape(-1, 384)
            norms = np.frombuffer(data, dtype="<f4", offset=width)
            end = cursor + len(metadata)
            fp32[cursor:end] = vectors.astype(np.float64) @ queries
            int8[cursor:end] = (values.astype(np.float64) @ queries) * norms[:, None]
            descriptions.extend(description for code, description in metadata)
            keys.extend((standardize_code(code, vocabulary["id"]), vocabulary["id"]) for code, description in metadata)
            cursor = end
        index = build_lexical_index(pd.DataFrame({"description": descriptions}))
        for i, edf in enumerate(edfs):
            lexical[start:cursor, i] = score_lexical(edf.to_lexical_query(), index, cursor - start)
        print(f'Compared {vocabulary["id"]}: {cursor:,}/{count:,} vectors', flush=True)
    lexical = normalize(lexical)
    scores = {"fp32": np.floor((.5 * lexical + .5 * normalize(fp32)) * 1e6 + .5),
              "int8": np.floor((.5 * lexical + .5 * normalize(int8)) * 1e6 + .5)}
    rows = []
    for i, name in enumerate(names):
        _, gold = load_aesi_dataset(base.paths.aesi_datasets[name])
        relevant = {(standardize_code(r.code, r.vocabulary), r.vocabulary) for r in gold.itertuples() if r.label == "Narrow"}
        mask = np.array([key in relevant for key in keys])
        orders = {dtype: np.argsort(-values[:, i], kind="stable") for dtype, values in scores.items()}
        result = {dtype: metrics(order, mask) for dtype, order in orders.items()}
        result.update(phenotype=name, top_100_overlap=len(set(orders["fp32"][:100]) & set(orders["int8"][:100])))
        rows.append(result)
        print(name, json.dumps(result), flush=True)
    means = {dtype: {metric: float(np.mean([row[dtype][metric] for row in rows]))
                    for metric in ("average_precision", "recall_at_100", "recall_at_1000")} for dtype in scores}
    passed = (means["int8"]["average_precision"] >= means["fp32"]["average_precision"] - .001
              and means["int8"]["recall_at_1000"] >= means["fp32"]["recall_at_1000"] - .002
              and all(row["int8"]["average_precision"] >= row["fp32"]["average_precision"] - .005 for row in rows))
    report = {"scope": "Current browser corpus; seven validation EDFs; fixed lexical weight 0.5; no test-set selection",
              "corpus_size": count, "means": means, "cases": rows, "passed": passed,
              "gates": {"maximum_mean_ap_loss": .001, "maximum_case_ap_loss": .005, "maximum_mean_recall_at_1000_loss": .002},
              "manifest_sha256": {dtype: hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
                                  for dtype, folder in (("fp32", args.original), ("int8", args.quantized))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"means": means, "passed": passed}), flush=True)
    if not passed:
        raise SystemExit("Quantization quality gates failed; do not deploy")


if __name__ == "__main__":
    main()
