import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_browser_embeddings import export_vocabulary
from src.retrieval.embeddings import EmbeddingIndex


@pytest.mark.parametrize("dimensions", [384, 3072])
def test_export_uses_only_exact_matching_saved_vectors_and_shards_rows(tmp_path, dimensions):
    corpus = pd.DataFrame({"code": ["A", "B", "C"], "description": ["alpha condition", "beta condition", "gamma condition"], "vocabulary": ["TEST"] * 3})
    vectors = np.zeros((3, dimensions), dtype=np.float32)
    vectors[:, :3] = np.eye(3)
    index = EmbeddingIndex(None, corpus.description.tolist(), corpus.code.tolist(), corpus.vocabulary.tolist(), vectors)
    pack = export_vocabulary(corpus, index, tmp_path, "TEST", chunk_rows=2, dimensions=dimensions)
    assert pack["count"] == 3
    assert [chunk["count"] for chunk in pack["chunks"]] == [2, 1]
    data = (tmp_path / pack["chunks"][0]["vectors"]["url"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pack["chunks"][0]["vectors"]["sha256"]
    assert np.array_equal(np.frombuffer(data, dtype="<f4").reshape(2, dimensions), vectors[:2])
    metadata = json.loads(gzip.decompress((tmp_path / pack["chunks"][1]["rows"]["url"]).read_bytes()))
    assert metadata == [["C", "gamma condition"]]
    assert export_vocabulary(corpus, index, tmp_path, "TEST", chunk_rows=2, dimensions=dimensions)["version"] == pack["version"]
    changed = corpus.copy()
    changed.loc[0, "description"] = "changed condition"
    with pytest.raises(ValueError, match="do not match"):
        export_vocabulary(changed, index, tmp_path, "TEST", dimensions=dimensions)


def test_export_rejects_wrong_encoder_dimensions(tmp_path):
    corpus = pd.DataFrame({"code": ["A"], "description": ["alpha"], "vocabulary": ["TEST"]})
    with pytest.raises(ValueError, match="384-dimensional"):
        export_vocabulary(corpus, EmbeddingIndex(None, ["alpha"], ["A"], ["TEST"], np.ones((1, 3072))), tmp_path, "TEST")


def test_int8_export_preserves_signed_directions_with_unit_normalizers(tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(9, 384)).astype(np.float32)
    corpus = pd.DataFrame({"code": [str(i) for i in range(9)], "description": [f"condition {i}" for i in range(9)], "vocabulary": ["TEST"] * 9})
    index = EmbeddingIndex(None, corpus.description.tolist(), corpus.code.tolist(), corpus.vocabulary.tolist(), vectors)
    pack = export_vocabulary(corpus, index, tmp_path, "TEST", dtype="int8")
    asset = pack["chunks"][0]["vectors"]
    data = (tmp_path / asset["url"]).read_bytes()
    assert asset["url"].endswith(".i8")
    assert len(data) == 9 * (384 + 4)
    values = np.frombuffer(data, dtype=np.int8, count=9 * 384).reshape(9, 384)
    assert values.min() < 0 < values.max()
    norms = np.frombuffer(data, dtype="<f4", offset=9 * 384)
    decoded = values * norms[:, None]
    np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1, atol=1e-6)
    expected = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert np.min(np.sum(decoded * expected, axis=1)) > .9999
