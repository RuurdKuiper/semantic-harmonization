"""Tests for BM25 lexical retrieval."""

from __future__ import annotations

from src.retrieval.lexical import build_lexical_index, retrieve_lexical, _tokenize


def test_tokenize_basic():
    assert _tokenize("Acute Myocarditis, unspecified!") == ["acute", "myocarditis", "unspecified"]


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize(None) == []  # type: ignore[arg-type]


def test_retrieve_lexical_ranks_relevant_first(sample_codes):
    results = retrieve_lexical("myocarditis inflammation of heart muscle", sample_codes, top_k=3)
    assert results[0].code in {"I40.0", "I40.9", "50920009"}
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_retrieve_lexical_top_k(sample_codes):
    results = retrieve_lexical("myocarditis", sample_codes, top_k=2)
    assert len(results) == 2


def test_build_lexical_index_reuse(sample_codes):
    index = build_lexical_index(sample_codes)
    results = retrieve_lexical("myocarditis", sample_codes, index=index)
    assert len(results) == len(sample_codes)


def test_retrieve_lexical_empty_query(sample_codes):
    results = retrieve_lexical("", sample_codes)
    assert len(results) == len(sample_codes)
    assert all(r.score == 0.0 for r in results)
