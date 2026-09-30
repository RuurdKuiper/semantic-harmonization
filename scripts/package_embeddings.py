#!/usr/bin/env python3
"""Package precomputed embedding assets for GitHub Release uploads.

This script copies the four vocabulary-specific embedding indexes from
``data/codes/embeddings`` into a release staging directory and writes a JSON manifest
with file metadata. The resulting directory is ready to upload with
``gh release upload``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EMBEDDING_FILES = [
    "embeddings_ICD10CM_sentence-transformers_all-MiniLM-L6-v2.npz",
    "embeddings_ICD9CM_sentence-transformers_all-MiniLM-L6-v2.npz",
    "embeddings_ICPC_sentence-transformers_all-MiniLM-L6-v2.npz",
    "embeddings_MDR_sentence-transformers_all-MiniLM-L6-v2.npz",
    "embeddings_RCD2_sentence-transformers_all-MiniLM-L6-v2.npz",
    "embeddings_SNOMEDCT_US_sentence-transformers_all-MiniLM-L6-v2.npz",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package embedding assets for a GitHub release.")
    parser.add_argument(
        "--source-dir",
        default="data/codes/embeddings",
        help="Directory containing the precomputed embedding .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="dist/release-assets",
        help="Directory to stage release assets into.",
    )
    parser.add_argument(
        "--manifest-name",
        default="embeddings-manifest.json",
        help="Name of the JSON manifest file to create in the output directory.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for filename in EMBEDDING_FILES:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing embedding asset: {source_path}")
        target_path = output_dir / filename
        shutil.copy2(source_path, target_path)
        manifest.append(
            {
                "filename": filename,
                "bytes": target_path.stat().st_size,
            }
        )

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(EMBEDDING_FILES)} assets to {output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
