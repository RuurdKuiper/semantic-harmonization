"""Test shortened/quantized OpenAI retrieval without changing the live app.

Seven validation EDFs, current values-only queries and fixed 35/65 BM25/semantic
weights. Reads saved corpus vectors in bounded chunks; never re-embeds the
terminology corpus or writes a full converted index to disk.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from scripts.export_browser_embeddings import quantize_unit_vectors
from scripts.validate_browser_quantization import normalize
from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import standardize_code
from src.retrieval.embeddings import create_embedding_model
from src.retrieval.lexical import build_lexical_index, score_lexical
from src.utils.config import load_config


def ranking_metrics(order, mask):
    hits = mask[order]
    ranks = np.flatnonzero(hits) + 1
    count = len(ranks)
    result = {"average_precision": float(np.mean(np.arange(1, count + 1) / ranks)) if count else 0.,
              "available_narrow": count}
    for budget in (100, 1000, 10000):
        result[f"recall_at_{budget}"] = float(hits[:budget].sum() / count) if count else 0.
    return result


def main():
    load_dotenv()
    config = load_config("configs/default.yaml")
    root = Path("dist/browser-assets")
    output = Path("results/browser_openai_compaction")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    names = json.loads(Path("results/experiments/embedding_comparison/20261022/protocol.json").read_text())["validation_edfs"]
    edfs = [load_edf(name, config.paths.edf_dir) for name in names]
    inputs = [edf.to_embedding_query(include_labels=False) for edf in edfs]
    query_hashes = [hashlib.sha256(text.encode()).hexdigest() for text in inputs]
    signature = hashlib.sha256(json.dumps({"model": "text-embedding-3-large", "dimensions": 3072,
                                          "queries": inputs}, sort_keys=True).encode()).hexdigest()
    cache = output / ("queries_" + signature[:20] + ".npz")
    if cache.exists():
        with np.load(cache) as saved:
            encoded = saved["embedding"]
    else:
        encoder = create_embedding_model("openai", "text-embedding-3-large", dimensions=3072)
        encoded = encoder.encode(inputs, normalize_embeddings=True).astype(np.float32)
        np.savez_compressed(cache, embedding=encoded)
    assert encoded.shape == (len(names), 3072) and np.isfinite(encoded).all()
    print(f"Prepared {len(names)} query embeddings; no corpus embedding calls", flush=True)
    variants = {"3072_fp32": (3072, False), "3072_int8": (3072, True),
                "512_fp32": (512, False), "512_int8": (512, True),
                "256_fp32": (256, False), "256_int8": (256, True)}
    queries = {}
    for label, (dimensions, quantized) in variants.items():
        shortened = encoded[:, :dimensions].copy()
        shortened /= np.linalg.norm(shortened, axis=1, keepdims=True)
        queries[label] = shortened.astype(np.float64).T
    count = sum(v["count"] for v in manifest["vocabularies"])
    lexical = np.zeros((count, len(names)))
    semantic = {label: np.zeros_like(lexical) for label in variants}
    keys = []
    cursor = 0
    metadata_bytes = sum(v["lexical"]["bytes"] + sum(c["rows"]["bytes"] for c in v["chunks"])
                         for v in manifest["vocabularies"])
    scoring_seconds = {label: 0. for label in variants}
    sources = {}
    for vocabulary in manifest["vocabularies"]:
        name = vocabulary["id"]
        directory = (Path(config.paths.embeddings_dir) if name == "SNOMEDCT_US" else Path("data/codes/embeddings")) / "openai_text-embedding-3-large_3072_resumable" / name
        state = json.loads((directory / "state.json").read_text())
        assert state["complete"] and state["dimensions"] == 3072 and state["model"] == "text-embedding-3-large"
        sources[name] = {"state_sha256": hashlib.sha256((directory / "state.json").read_bytes()).hexdigest(),
                         "rows_sha256": hashlib.sha256((directory / "rows.parquet").read_bytes()).hexdigest()}
        rows = pd.read_parquet(directory / "rows.parquet")
        lookup = {(str(r.code), str(r.vocabulary), str(r.description)): i for i, r in enumerate(rows.itertuples())}
        saved = np.load(directory / "vectors.npy", mmap_mode="r")
        descriptions = []
        start = cursor
        for chunk in vocabulary["chunks"]:
            metadata = json.loads(gzip.decompress((root / chunk["rows"]["url"]).read_bytes()))
            positions = [lookup[(code, name, description)] for code, description in metadata]
            full = np.asarray(saved[positions], dtype=np.float32)
            assert np.isfinite(full).all()
            end = cursor + len(metadata)
            for label, (dimensions, quantized) in variants.items():
                vectors = full[:, :dimensions].copy()
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                assert (norms > 1e-12).all()
                vectors /= norms
                inverse = None
                if quantized:
                    binary = quantize_unit_vectors(vectors)
                    width = len(metadata) * dimensions
                    vectors = np.frombuffer(binary, dtype=np.int8, count=width).reshape(-1, dimensions)
                    inverse = np.frombuffer(binary, dtype="<f4", offset=width)
                    assert len(binary) == len(metadata) * (dimensions + 4)
                numeric = vectors.astype(np.float64)
                began = perf_counter()
                scores = numeric @ queries[label]
                if inverse is not None:
                    scores *= inverse[:, None]
                semantic[label][cursor:end] = scores
                scoring_seconds[label] += perf_counter() - began
            descriptions.extend(description for code, description in metadata)
            keys.extend((standardize_code(code, name), name) for code, description in metadata)
            cursor = end
        index = build_lexical_index(pd.DataFrame({"description": descriptions}))
        for i, edf in enumerate(edfs):
            lexical[start:cursor, i] = score_lexical(edf.to_lexical_query(), index, cursor - start)
        del index, saved, lookup, rows, full, numeric
        print(f"Compared {name}: {cursor:,}/{count:,} codes", flush=True)
    lexical = normalize(lexical)
    combined = {label: np.floor((.35 * lexical + .65 * normalize(scores)) * 1e6 + .5)
                for label, scores in semantic.items()}
    cases = []
    for i, name in enumerate(names):
        _, gold = load_aesi_dataset(config.paths.aesi_datasets[name])
        relevant = {(standardize_code(r.code, r.vocabulary), r.vocabulary) for r in gold.itertuples() if r.label == "Narrow"}
        mask = np.array([key in relevant for key in keys])
        orders = {label: np.argsort(-scores[:, i], kind="stable") for label, scores in combined.items()}
        result = {"phenotype": name}
        for label, order in orders.items():
            result[label] = ranking_metrics(order, mask)
            result[label]["top_100_overlap_with_baseline"] = len(set(order[:100]) & set(orders["3072_fp32"][:100])) / 100
        cases.append(result)
        print(name, json.dumps({label: result[label]["average_precision"] for label in variants}), flush=True)
    metric_names = ("average_precision", "recall_at_100", "recall_at_1000", "recall_at_10000", "top_100_overlap_with_baseline")
    means = {label: {metric: float(np.mean([case[label][metric] for case in cases])) for metric in metric_names}
             for label in variants}
    prior = json.loads(Path("results/browser_query_labels/comparison.json").read_text())
    assert prior["query_sha256"][len(names):] == query_hashes
    prior_map = prior["encoders"]["openai"]["means"]["unlabeled"]["average_precision"]
    assert abs(means["3072_fp32"]["average_precision"] - prior_map) < 1e-5, "Baseline did not reproduce the current app comparison"
    sizes = {label: {"bytes": count * (dimensions + 4 if quantized else dimensions * 4) + metadata_bytes,
                     "dimensions": dimensions, "storage": "int8 with inverse norms" if quantized else "fp32"}
             for label, (dimensions, quantized) in variants.items()}
    report = {"scope": "Test only; current browser corpus, seven validation EDFs, values-only queries, fixed 35/65 weights; no live changes",
              "corpus_size": count, "means": means, "cases": cases, "sizes_excluding_model_and_manifest": sizes,
              "query_sha256": query_hashes, "query_cache_signature": signature,
              "source_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
              "source_checkpoints": sources, "baseline_reproduced": True,
              "numeric_scoring_seconds": scoring_seconds,
              "timing_scope": "Local NumPy scoring of seven queries; excludes I/O, conversion and ranking; not browser/download latency"}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame([{"phenotype": case["phenotype"], "variant": label, **case[label]}
                  for case in cases for label in variants]).to_csv(output / "comparison.csv", index=False)
    lines = ["# OpenAI compaction evaluation", "", report["scope"], "",
             "| Variant | Estimated pack MiB | MAP | Recall@100 | Recall@1,000 | Recall@10,000 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, result in means.items():
        size = sizes[label]["bytes"] / 1024**2
        lines.append(f"| {label} | {size:.2f} | {result['average_precision']:.6f} | "
                     f"{100*result['recall_at_100']:.2f}% | {100*result['recall_at_1000']:.2f}% | {100*result['recall_at_10000']:.2f}% |")
    kidney = next(case for case in cases if case["phenotype"] == "kidney_disease")
    lines += ["", "## Kidney-disease recall at the 10,000-candidate ceiling", ""]
    for label in variants:
        lines.append(f"- {label}: {100*kidney[label]['recall_at_10000']:.2f}%")
    lines += ["", "The full-precision baseline reproduced the earlier values-only OpenAI comparison. "
              "Sizes include compressed terminology metadata and per-row inverse norms. "
              "Numeric scoring timings in the JSON report are not browser or network benchmarks. "
              "The live app was not changed."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"means": means, "sizes": sizes, "baseline_reproduced": True}), flush=True)


if __name__ == "__main__":
    main()
