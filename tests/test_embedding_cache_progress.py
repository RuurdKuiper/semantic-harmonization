"""Regression tests for cached embedding progress plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    loaded = EmbeddingIndex.load_for_vocabularies(
        codes,
        vocabularies=["TEST"],
        cache_dir=tmp_path,
        model_name="test/model",
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert loaded.codes == ["A1"]
    assert updates == [(0, 1), (1, 1)]
