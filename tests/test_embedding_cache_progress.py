"""Regression tests for cached embedding progress plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.retrieval.embeddings import EmbeddingIndex


def test_load_for_vocabularies_accepts_progress_callback_with_cached_index(tmp_path):
    codes = pd.DataFrame(
        [{"code": "A1", "description": "alpha", "vocabulary": "TEST"}]
    )
    cache_path = tmp_path / "embeddings_TEST_test_model.npz"
    EmbeddingIndex(
        model=None,
        descriptions=["alpha"],
        codes=["A1"],
        vocabularies=["TEST"],
        embeddings=np.array([[1.0]], dtype=np.float32),
    ).save(cache_path)

    updates: list[tuple[int, int]] = []
    statuses: list[str] = []
    loaded = EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=["TEST"],
        cache_dir=tmp_path,
        model_name="test/model",
        progress_callback=lambda completed, total: updates.append((completed, total)),
        status_callback=statuses.append,
    )

    assert loaded.codes == ["A1"]
    assert updates == [(0, 1), (1, 1)]
    assert any("Loading and checking saved embeddings for TEST" in status for status in statuses)
    assert statuses[-1] == "Combining terminology embeddings..."


def test_single_vocabulary_does_not_copy_embedding_matrix(tmp_path):
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    index = EmbeddingIndex(None, ["alpha"], ["A1"], ["TEST"], matrix)
    codes = pd.DataFrame({"code": index.codes, "description": index.descriptions,
                          "vocabulary": index.vocabularies})
    with patch.object(EmbeddingIndex, "from_cache_or_build", return_value=index):
        loaded = EmbeddingIndex.load_for_vocabularies(
            codes, ["TEST"], tmp_path, provider="openai",
        )
    assert loaded.embeddings is matrix
    assert EmbeddingIndex.combine([index]).embeddings is matrix
