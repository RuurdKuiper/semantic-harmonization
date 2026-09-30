"""Hybrid retrieval combining lexical (BM25) and embedding (semantic) scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.embeddings import EmbeddingIndex, retrieve_embeddings
from src.retrieval.lexical import build_lexical_index, score_lexical


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
    per_vocabulary_top_k: bool = True,
    embedding_index: EmbeddingIndex | None = None,
    query_model: SentenceTransformer | None = None,
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
    per_vocabulary_top_k : bool
        If ``True`` (default), apply ``top_k`` separately within each selected
        vocabulary. If ``False``, apply ``top_k`` after combining all selected
        vocabularies.
    embedding_index : EmbeddingIndex, optional
        Pre-built embedding index to reuse across multiple queries.
    query_model : SentenceTransformer, optional
        Cached transformer used to embed the EDF/query text.
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
    embedding_scores_by_code: dict[tuple[str, str], float] | None = None
    if embedding_index is not None:
        # Score the combined corpus exactly once. Previously this full matrix
        # multiplication, full sort, and candidate-object construction ran
        # once per vocabulary even though every pass produced identical raw
        # scores. The corpus is deduplicated on (code, vocabulary), so the map
        # is identical to the maps produced by the former repeated searches.
        all_embedding_scores = embedding_index.score_all(embedding_query, model=query_model)
        embedding_scores_by_code = {
            (code, vocabulary): score
            for code, vocabulary, score in zip(
                embedding_index.codes,
                embedding_index.vocabularies,
                all_embedding_scores,
            )
        }

    per_vocab_candidates: list[
        tuple[str, list[tuple[str, str, str, float, float]]]
    ] = []
    for vocabulary, vocab_df in df.groupby("vocabulary", sort=False):
        lexical_index = build_lexical_index(vocab_df)
        lexical_scores = score_lexical(lexical_query, lexical_index, len(vocab_df))

        if embedding_scores_by_code is None:
            embedding_results = retrieve_embeddings(embedding_query, vocab_df, model_name=embedding_model, top_k=None)
            vocabulary_embedding_scores = {
                (c.code, c.vocabulary): c.score for c in embedding_results
            }
        else:
            vocabulary_embedding_scores = embedding_scores_by_code

        codes_values = vocab_df["code"].tolist()
        description_values = vocab_df["description"].tolist()
        vocabulary_values = vocab_df["vocabulary"].tolist()
        vocab_candidates = [
            (
                code,
                description,
                candidate_vocabulary,
                lexical_score,
                vocabulary_embedding_scores.get((code, candidate_vocabulary), 0.0),
            )
            for code, description, candidate_vocabulary, lexical_score in zip(
                codes_values,
                description_values,
                vocabulary_values,
                lexical_scores,
            )
        ]
        per_vocab_candidates.append((vocabulary, vocab_candidates))

    all_lexical = [
        lexical
        for _, vocab_candidates in per_vocab_candidates
        for _, _, _, lexical, _ in vocab_candidates
    ]
    all_embedding = [
        embedding
        for _, vocab_candidates in per_vocab_candidates
        for _, _, _, _, embedding in vocab_candidates
    ]
    norm_all_lexical = _min_max_normalize(all_lexical)
    norm_all_embedding = _min_max_normalize(all_embedding)

    # Keep lightweight tuples until after top-k selection. Constructing a
    # dataclass (and formerly a Pandas Series) for all 620k rows dominated the
    # runtime even when only a handful of results were requested.
    selected: list[tuple[str, str, str, float, float, float]] = []
    flat_idx = 0
    for _, vocab_candidates in per_vocab_candidates:
        scored: list[tuple[str, str, str, float, float, float]] = []
        for idx, (code, description, vocabulary, _, _) in enumerate(vocab_candidates):
            norm_lexical = norm_all_lexical[flat_idx + idx]
            norm_embedding = norm_all_embedding[flat_idx + idx]
            score = lexical_weight * norm_lexical + embedding_weight * norm_embedding
            scored.append(
                (
                    code,
                    description,
                    vocabulary,
                    round(norm_lexical, 6),
                    round(norm_embedding, 6),
                    round(score, 6),
                )
            )
        if per_vocabulary_top_k and top_k is not None:
            scored.sort(key=lambda candidate: -candidate[5])
            scored = scored[:top_k]
        selected.extend(scored)
        flat_idx += len(vocab_candidates)

    selected.sort(key=lambda candidate: -candidate[5])
    if not per_vocabulary_top_k and top_k is not None:
        selected = selected[:top_k]
    return [
        HybridCandidate(
            rank=rank,
            code=code,
            description=description,
            vocabulary=vocabulary,
            lexical_score=lexical_score,
            embedding_score=embedding_score,
            score=score,
        )
        for rank, (code, description, vocabulary, lexical_score, embedding_score, score)
        in enumerate(selected, start=1)
    ]
