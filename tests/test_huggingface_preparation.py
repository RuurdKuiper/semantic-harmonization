import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_browser_embeddings import export_vocabulary
from scripts.prepare_huggingface_deployment import prepare, manifest_assets
from src.retrieval.embeddings import EmbeddingIndex


def test_hf_preparation_stages_private_dataset_and_port_7860_space_without_secrets(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    corpus = pd.DataFrame({"code": ["A"], "description": ["alpha"], "vocabulary": ["ICPC"]})
    index = EmbeddingIndex(None, ["alpha"], ["A"], ["ICPC"], np.ones((1, 384), dtype=np.float32))
    pack = export_vocabulary(corpus, index, source, "ICPC")
    (source / "manifest.json").write_text(json.dumps({"format": "semantic-harmonization-minilm-v1", "vocabularies": [pack]}))
    result = prepare(source, tmp_path / "prepared")
    assert result["count"] == 1
    assert "Keep this repository private" in (result["dataset"] / "README.md").read_text()
    assert "ENV PORT=7860" in (result["space"] / "Dockerfile").read_text()
    assert "app_port: 7860" in (result["space"] / "README.md").read_text()
    assert all(not path.endswith(".env") and "__pycache__" not in path for path in result["space_paths"])
    assert all((result["dataset"] / path).exists() for path in result["dataset_paths"])


def test_hf_preparation_refuses_manifest_path_traversal(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "semantic-harmonization-minilm-v1", "vocabularies": [{"lexical": {"url": "../.env"}, "chunks": []}]}))
    with pytest.raises(ValueError, match="relative paths"):
        manifest_assets(tmp_path)
