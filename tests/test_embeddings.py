"""Tests for embedding-based retrieval. Uses a lightweight sentence-transformers
model; network access to download the model is required the first time."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from src.retrieval.embeddings import EmbeddingIndex, retrieve_embeddings

pytestmark = pytest.mark.embeddings


def test_retrieve_embeddings_returns_all(sample_codes):
    results = retrieve_embeddings("myocarditis", sample_codes)
    assert len(results) == len(sample_codes)
    assert all(hasattr(r, "score") for r in results)


def test_retrieve_embeddings_top_k(sample_codes):
    results = retrieve_embeddings("myocarditis", sample_codes, top_k=2)
    assert len(results) == 2


def test_retrieve_embeddings_ranks_relevant_higher(sample_codes):
    results = retrieve_embeddings("inflammation of the heart muscle myocarditis", sample_codes)
    top_codes = {results[0].code, results[1].code}
    assert top_codes & {"I40.0", "I40.9", "50920009"}


def test_embedding_index_reuse(sample_codes):
    index = EmbeddingIndex.from_codes(sample_codes)
    results = index.retrieve("myocarditis")
    assert len(results) == len(sample_codes)
    assert all(r.code for r in results)


def test_embedding_index_save_and_load_roundtrip(sample_codes, tmp_path):
    index = EmbeddingIndex.from_codes(sample_codes)
    cache_path = tmp_path / "index.npz"
    index.save(cache_path)
    assert cache_path.exists()

    loaded = EmbeddingIndex.load(cache_path)
    assert loaded.codes == index.codes
    assert loaded.descriptions == index.descriptions
    assert loaded.vocabularies == index.vocabularies
    assert loaded.embeddings.shape == index.embeddings.shape

    results = loaded.retrieve("myocarditis")
    assert len(results) == len(sample_codes)


def test_embedding_index_from_cache_or_build_builds_then_reuses(sample_codes, tmp_path):
    cache_path = tmp_path / "index.npz"
    assert not cache_path.exists()

    built = EmbeddingIndex.from_cache_or_build(sample_codes, cache_path=cache_path)
    assert cache_path.exists()

    reloaded = EmbeddingIndex.from_cache_or_build(sample_codes, cache_path=cache_path)
    assert reloaded.codes == built.codes


def test_score_all_matches_retrieve_scores_without_sorting(sample_codes):
    class QueryModel:
        def encode(self, text, **kwargs):
            return np.array([0.8, 0.6], dtype=np.float32)

    embeddings = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]],
        dtype=np.float32,
    )
    index = EmbeddingIndex(
        model=None,
        descriptions=sample_codes["description"].tolist(),
        codes=sample_codes["code"].tolist(),
        vocabularies=sample_codes["vocabulary"].tolist(),
        embeddings=embeddings,
    )

    scores = index.score_all("query", model=QueryModel())
    retrieved = index.retrieve("query", model=QueryModel())
    retrieved_by_code = {item.code: item.score for item in retrieved}

    assert scores == [retrieved_by_code[code] for code in index.codes]


def test_embedding_cache_reuses_vectors_when_preprocessing_removes_rows(sample_codes, tmp_path):
    cache_path = tmp_path / "index.npz"
    embeddings = np.arange(len(sample_codes) * 2, dtype=np.float32).reshape(-1, 2)
    cached = EmbeddingIndex(
        model=None,
        descriptions=sample_codes["description"].tolist(),
        codes=sample_codes["code"].tolist(),
        vocabularies=sample_codes["vocabulary"].tolist(),
        embeddings=embeddings,
    )
    cached.save(cache_path)
    filtered = sample_codes.iloc[[0, 2, 4]].reset_index(drop=True)

    with patch.object(EmbeddingIndex, "from_codes", side_effect=AssertionError("must not rebuild")):
        reused = EmbeddingIndex.from_cache_or_build(
            filtered,
            cache_path=cache_path,
            load_model=False,
        )

    assert reused.codes == filtered["code"].tolist()
    assert np.array_equal(reused.embeddings, embeddings[[0, 2, 4]])
