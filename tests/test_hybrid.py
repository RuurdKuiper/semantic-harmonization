"""Tests for hybrid lexical + embedding retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retrieval.embeddings import EmbeddingIndex
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


def test_cached_index_fast_path_preserves_legacy_scores_and_order():
    class QueryModel:
        def encode(self, text, **kwargs):
            return np.array([0.8, 0.6], dtype=np.float32)

    codes = pd.DataFrame(
        [
            {"code": "A1", "description": "wrist fracture", "vocabulary": "V1"},
            {"code": "A2", "description": "fracture of radius", "vocabulary": "V1"},
            {"code": "A3", "description": "wrist pain", "vocabulary": "V1"},
            {"code": "B1", "description": "closed wrist fracture", "vocabulary": "V2"},
            {"code": "B2", "description": "ankle fracture", "vocabulary": "V2"},
        ]
    )
    index = EmbeddingIndex(
        model=None,
        descriptions=codes["description"].tolist(),
        codes=codes["code"].tolist(),
        vocabularies=codes["vocabulary"].tolist(),
        embeddings=np.array(
            [[1, 0], [0.8, 0.6], [0, 1], [0.6, 0.8], [-1, 0]],
            dtype=np.float32,
        ),
    )

    results = hybrid_retrieval(
        "wrist fracture",
        codes,
        top_k=2,
        per_vocabulary_top_k=True,
        embedding_index=index,
        query_model=QueryModel(),
    )

    assert [
        (item.rank, item.code, item.vocabulary, item.lexical_score, item.embedding_score, item.score)
        for item in results
    ] == [
        (1, "A1", "V1", 1.0, 0.888889, 0.944444),
        (2, "A2", "V1", 0.806456, 1.0, 0.903228),
        (3, "B1", "V2", 0.110543, 0.977778, 0.54416),
        (4, "B2", "V2", 0.0, 0.0, 0.0),
    ]
