"""Hybrid retrieval combining lexical (BM25) and embedding (semantic) scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.retrieval.embeddings import EmbeddingIndex, retrieve_embeddings
from src.retrieval.lexical import build_lexical_index, retrieve_lexical


@dataclass(slots=True)
class HybridCandidate:
    rank: int
    code: str
    description: str
    vocabulary: str
    lexical_score: float
    embedding_score: float
    score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HybridCandidate):
            return NotImplemented
        return self.score > other.score


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_retrieval(
    query: str,
    codes: pd.DataFrame,
    lexical_weight: float = 0.5,
    embedding_weight: float = 0.5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = 25,
    embedding_index: EmbeddingIndex | None = None,
) -> list[HybridCandidate]:
    """Combine BM25 lexical retrieval with embedding-based semantic retrieval.

    Both signals are min-max normalized to [0, 1] and combined with the given
    weights before ranking. The combined ranking is truncated to *top_k*.

    Parameters
    ----------
    query : str
        Phenotype / EDF text used as the search query.
    codes : pd.DataFrame
        Candidate code corpus with ``code``, ``description``, ``vocabulary`` columns.
    lexical_weight, embedding_weight : float
        Relative weighting of each retrieval signal. Need not sum to 1.
    embedding_model : str
        sentence-transformers model identifier used when *embedding_index* is None.
    top_k : int, optional
        Number of ranked results to return. ``None`` returns the full corpus.
    embedding_index : EmbeddingIndex, optional
        Pre-built embedding index to reuse across multiple queries.

    Returns
    -------
    list[HybridCandidate]
    """
    if codes.empty:
        return []

    df = codes.reset_index(drop=True).copy()
    for col in ("code", "vocabulary"):
        if col not in df.columns:
            df[col] = ""

    lexical_index = build_lexical_index(df)
    lexical_results = retrieve_lexical(query, df, index=lexical_index, top_k=None)
    lexical_scores_by_code = {
        (c.code, c.vocabulary): c.score for c in lexical_results
    }

    if embedding_index is not None:
        embedding_results = embedding_index.retrieve(query, top_k=None)
    else:
        embedding_results = retrieve_embeddings(query, df, model_name=embedding_model, top_k=None)
    embedding_scores_by_code = {
        (c.code, c.vocabulary): c.score for c in embedding_results
    }

    keys = [(row["code"], row["vocabulary"]) for _, row in df.iterrows()]
    raw_lexical = [lexical_scores_by_code.get(k, 0.0) for k in keys]
    raw_embedding = [embedding_scores_by_code.get(k, 0.0) for k in keys]

    norm_lexical = _min_max_normalize(raw_lexical)
    norm_embedding = _min_max_normalize(raw_embedding)

    combined: list[HybridCandidate] = []
    for i, (_, row) in enumerate(df.iterrows()):
        score = lexical_weight * norm_lexical[i] + embedding_weight * norm_embedding[i]
        combined.append(
            HybridCandidate(
                rank=0,
                code=row["code"],
                description=row["description"],
                vocabulary=row["vocabulary"],
                lexical_score=round(raw_lexical[i], 6),
                embedding_score=round(raw_embedding[i], 6),
                score=round(score, 6),
            )
        )

    combined.sort(key=lambda c: -c.score)
    if top_k is not None:
        combined = combined[:top_k]
    for r, c in enumerate(combined):
        c.rank = r + 1
    return combined
