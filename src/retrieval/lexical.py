"""BM25 lexical retrieval over code descriptions."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25 indexing."""
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [tok for tok in text.split() if len(tok) > 1]


@dataclass(slots=True)
class LexicalCandidate:
    rank: int
    code: str
    description: str
    vocabulary: str
    score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LexicalCandidate):
            return NotImplemented
        return self.score > other.score  # higher = more relevant


def build_lexical_index(codes: pd.DataFrame) -> BM25Okapi:
    """Build a BM25 index from the code corpus descriptions.

    Parameters
    ----------
    codes : pd.DataFrame
        Must contain ``description`` column with full description strings.

    Returns
    -------
    BM25Okapi
        The built tokenised index.
    """
    doc_tokens: list[list[str]] = []
    for desc in codes["description"].astype(str):
        tokens = _tokenize(desc)
        doc_tokens.append(tokens if tokens else [desc])

    return BM25Okapi(doc_tokens)


def score_lexical(query: str, index: BM25Okapi, corpus_size: int) -> list[float]:
    """Return corpus-order BM25 scores without allocating result objects."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * corpus_size
    return [round(float(score), 6) for score in index.get_scores(query_tokens)]


def retrieve_lexical(
    query: str,
    codes: pd.DataFrame | list[dict],
    index: BM25Okapi | None = None,
    top_k: int | None = None,
) -> list[LexicalCandidate]:
    """Retrieve candidate codes ranked by BM25 lexical similarity.

    Parameters
    ----------
    query : str
        The phenotype description / EDF summary to search against.
    codes : pd.DataFrame or list[dict]
        Code corpus containing ``description`` (mandatory) and optionally
        ``code``, ``vocabulary`` columns (DataFrame) or dicts with those keys.
    index : BM25Okapi, optional
        Pre-built index. If omitted one is built on-the-fly from *codes*.
    top_k : int, optional
        Maximum number of results to return.  Defaults to the full corpus size.

    Returns
    -------
    list[LexicalCandidate]
        Ranked candidates sorted descending by BM25 score.
    """
    if isinstance(codes, pd.DataFrame):
        df = codes.copy()
        keys = ("description", "code", "vocabulary")
        for k in keys[1:]:
            if k not in df.columns:
                df[k] = ""  # type: ignore[operator]
    else:
        df = pd.DataFrame(codes)
        for k in ("code", "vocabulary"):
            if k not in df.columns:
                df[k] = ""

    if index is None:
        index = build_lexical_index(df)

    query_tokens = _tokenize(query)
    if not query_tokens:
        # Fallback: rank all codes uniformly (no lexical signal)
        return [
            LexicalCandidate(
                rank=0, code=r["code"], description=r["description"], vocabulary=r["vocabulary"], score=0.0
            )
            for i, r in df.iterrows()
        ]

    scores = index.get_scores(query_tokens).tolist()  # type: ignore[union-attr]

    if top_k is not None:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_scores = [scores[i] for i in sorted(top_indices, key=lambda x: -scores[x])]
        results: list[LexicalCandidate] = [
            LexicalCandidate(
                rank=r + 1,
                code=df.iloc[idx]["code"],
                description=df.iloc[idx]["description"],
                vocabulary=df.iloc[idx].get("vocabulary", ""),
                score=round(float(s), 6),
            )
            for r, (idx, s) in enumerate(sorted(enumerate(top_scores), key=lambda x: -x[1]))
        ]
    else:
        results = [
            LexicalCandidate(
                rank=r + 1,
                code=df.iloc[r]["code"],
                description=df.iloc[r]["description"],
                vocabulary=df.iloc[r].get("vocabulary", ""),
                score=round(float(scores[r]), 6),
            )
            for r in range(len(df))
        ]

    return results
