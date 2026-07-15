"""Tests for embedding-based retrieval. Uses a lightweight sentence-transformers
model; network access to download the model is required the first time."""

from __future__ import annotations

import pytest

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

