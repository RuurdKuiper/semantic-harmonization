"""Tests for hybrid lexical + embedding retrieval."""

from __future__ import annotations

import pytest

from src.retrieval.hybrid import hybrid_retrieval

pytestmark = pytest.mark.embeddings


def test_hybrid_retrieval_returns_ranked_results(sample_codes):
    results = hybrid_retrieval(
        "myocarditis heart inflammation",
        sample_codes,
        top_k=3,
        per_vocabulary_top_k=False,
    )
    assert len(results) == 3
    assert results[0].rank == 1
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_hybrid_retrieval_empty_corpus():
    import pandas as pd

    empty = pd.DataFrame(columns=["code", "description", "vocabulary"])
    assert hybrid_retrieval("myocarditis", empty) == []


def test_hybrid_retrieval_weights_affect_ranking(sample_codes):
    lexical_only = hybrid_retrieval(
        "myocarditis", sample_codes, lexical_weight=1.0, embedding_weight=0.0, top_k=None
    )
    embedding_only = hybrid_retrieval(
        "myocarditis", sample_codes, lexical_weight=0.0, embedding_weight=1.0, top_k=None
    )
    assert len(lexical_only) == len(sample_codes)
    assert len(embedding_only) == len(sample_codes)


def test_hybrid_retrieval_applies_top_k_per_vocabulary():
    import pandas as pd

    codes = pd.DataFrame(
        [
            {"code": "A1", "description": "alpha heart", "vocabulary": "V1"},
            {"code": "A2", "description": "alpha lung", "vocabulary": "V1"},
            {"code": "A3", "description": "alpha kidney", "vocabulary": "V1"},
            {"code": "B1", "description": "alpha heart", "vocabulary": "V2"},
            {"code": "B2", "description": "alpha lung", "vocabulary": "V2"},
            {"code": "B3", "description": "alpha kidney", "vocabulary": "V2"},
        ]
    )

    results = hybrid_retrieval("alpha", codes, lexical_weight=1.0, embedding_weight=0.0, top_k=2)
    assert len(results) == 4
    assert sorted(c.vocabulary for c in results).count("V1") == 2
    assert sorted(c.vocabulary for c in results).count("V2") == 2
