import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_browser_embeddings import export_vocabulary
from src.retrieval.embeddings import EmbeddingIndex


def test_export_uses_only_exact_matching_saved_vectors_and_shards_rows(tmp_path):
    corpus = pd.DataFrame({"code": ["A", "B", "C"], "description": ["alpha condition", "beta condition", "gamma condition"], "vocabulary": ["TEST"] * 3})
    vectors = np.zeros((3, 384), dtype=np.float32)
    vectors[:, :3] = np.eye(3)
    index = EmbeddingIndex(None, corpus.description.tolist(), corpus.code.tolist(), corpus.vocabulary.tolist(), vectors)
    pack = export_vocabulary(corpus, index, tmp_path, "TEST", chunk_rows=2)
    assert pack["count"] == 3
    assert [chunk["count"] for chunk in pack["chunks"]] == [2, 1]
    data = (tmp_path / pack["chunks"][0]["vectors"]["url"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pack["chunks"][0]["vectors"]["sha256"]
    assert np.array_equal(np.frombuffer(data, dtype="<f4").reshape(2, 384), vectors[:2])
    metadata = json.loads(gzip.decompress((tmp_path / pack["chunks"][1]["rows"]["url"]).read_bytes()))
    assert metadata == [["C", "gamma condition"]]
    assert export_vocabulary(corpus, index, tmp_path, "TEST", chunk_rows=2)["version"] == pack["version"]
    changed = corpus.copy()
    changed.loc[0, "description"] = "changed condition"
    with pytest.raises(ValueError, match="do not match"):
        export_vocabulary(changed, index, tmp_path, "TEST")


def test_export_rejects_wrong_encoder_dimensions(tmp_path):
    corpus = pd.DataFrame({"code": ["A"], "description": ["alpha"], "vocabulary": ["TEST"]})
    with pytest.raises(ValueError, match="384-dimensional"):
        export_vocabulary(corpus, EmbeddingIndex(None, ["alpha"], ["A"], ["TEST"], np.ones((1, 3072))), tmp_path, "TEST")
