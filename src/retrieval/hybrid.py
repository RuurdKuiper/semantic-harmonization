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
    lexical_query: str | None = None,
    embedding_query: str | None = None,
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
        Number of ranked results to return per vocabulary. ``None`` returns the
        full corpus.
    embedding_index : EmbeddingIndex, optional
        Pre-built embedding index to reuse across multiple queries.
    lexical_query : str, optional
        Text used for the BM25 lexical score. If omitted, *query* is used.
    embedding_query : str, optional
        Text used for the embedding score. If omitted, *query* is used.

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

    lexical_query = lexical_query or query
    embedding_query = embedding_query or query
    per_vocab_candidates: list[tuple[str, list[tuple[pd.Series, float, float]]]] = []
    for vocabulary, vocab_df in df.groupby("vocabulary", sort=False):
        lexical_index = build_lexical_index(vocab_df)
        lexical_results = retrieve_lexical(lexical_query, vocab_df, index=lexical_index, top_k=None)
        lexical_scores_by_code = {(c.code, c.vocabulary): c.score for c in lexical_results}

        if embedding_index is not None:
            embedding_results = embedding_index.retrieve(embedding_query, top_k=None)
            embedding_results = [c for c in embedding_results if c.vocabulary == vocabulary]
        else:
            embedding_results = retrieve_embeddings(embedding_query, vocab_df, model_name=embedding_model, top_k=None)
        embedding_scores_by_code = {(c.code, c.vocabulary): c.score for c in embedding_results}

        vocab_candidates: list[tuple[pd.Series, float, float]] = []
        for _, row in vocab_df.iterrows():
            key = (row["code"], row["vocabulary"])
            vocab_candidates.append(
                (
                    row,
                    lexical_scores_by_code.get(key, 0.0),
                    embedding_scores_by_code.get(key, 0.0),
                )
            )
        per_vocab_candidates.append((vocabulary, vocab_candidates))

    all_lexical = [lex for _, vocab_candidates in per_vocab_candidates for _, lex, _ in vocab_candidates]
    all_embedding = [emb for _, vocab_candidates in per_vocab_candidates for _, _, emb in vocab_candidates]
    norm_all_lexical = _min_max_normalize(all_lexical)
    norm_all_embedding = _min_max_normalize(all_embedding)

    combined: list[HybridCandidate] = []
    offset = 0
    for _, vocab_candidates in per_vocab_candidates:
        scored: list[HybridCandidate] = []
        for idx, (row, raw_lexical, raw_embedding) in enumerate(vocab_candidates):
            norm_lexical = norm_all_lexical[offset + idx]
            norm_embedding = norm_all_embedding[offset + idx]
            score = lexical_weight * norm_lexical + embedding_weight * norm_embedding
            scored.append(
                HybridCandidate(
                    rank=0,
                    code=row["code"],
                    description=row["description"],
                    vocabulary=row["vocabulary"],
                    lexical_score=round(norm_lexical, 6),
                    embedding_score=round(norm_embedding, 6),
                    score=round(score, 6),
                )
            )
        scored.sort(key=lambda c: -c.score)
        if top_k is not None:
            scored = scored[:top_k]
        combined.extend(scored)
        offset += len(vocab_candidates)

    combined.sort(key=lambda c: -c.score)
    for r, c in enumerate(combined):
        c.rank = r + 1
    return combined
