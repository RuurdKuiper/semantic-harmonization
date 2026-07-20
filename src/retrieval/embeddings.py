"""Embedding-based retrieval using sentence-transformers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class EmbeddingCandidate:
    rank: int
    code: str
    description: str
    vocabulary: str
    score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingCandidate):
            return NotImplemented
        return self.score > other.score


@dataclass
class EmbeddingIndex:
    """Wraps embedding model + corpus embeddings for fast cosine-sim search."""

    model: SentenceTransformer
    descriptions: list[str]
    codes: list[str]
    vocabularies: list[str]
    embeddings: np.ndarray  # shape (n, dim)

    @classmethod
    def from_codes(
        cls,
        codes: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "EmbeddingIndex":
        df = codes.copy()
        for col in ("code", "vocabulary"):
            if col not in df.columns:
                df[col] = ""
        model = SentenceTransformer(model_name)
        descs = df["description"].fillna("").astype(str).tolist()
        embs = model.encode(descs, show_progress_bar=False, normalize_embeddings=True)
        return cls(
            model=model,
            descriptions=descs,
            codes=df["code"].astype(str).tolist(),
            vocabularies=df["vocabulary"].astype(str).tolist(),
            embeddings=np.asarray(embs),
        )

    def save(self, path: str | Path) -> None:
        """Persist the corpus embeddings + metadata to disk (npz) so they can
        be reloaded without re-running the embedding model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            codes=np.array(self.codes, dtype=object),
            descriptions=np.array(self.descriptions, dtype=object),
            vocabularies=np.array(self.vocabularies, dtype=object),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "EmbeddingIndex":
        """Load a previously-saved embedding index from disk."""
        data = np.load(path, allow_pickle=True)
        model = SentenceTransformer(model_name)
        return cls(
            model=model,
            descriptions=list(data["descriptions"]),
            codes=list(data["codes"]),
            vocabularies=list(data["vocabularies"]),
            embeddings=data["embeddings"],
        )

    @classmethod
    def from_cache_or_build(
        cls,
        codes: pd.DataFrame,
        cache_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "EmbeddingIndex":
        """Load a cached embedding index if present, otherwise build it from
        *codes* and cache the result for subsequent runs.

        This is the key mechanism that avoids re-embedding large reference
        code systems (e.g. the ~74k-row full ICD-10-CM corpus) on every run.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            return cls.load(cache_path, model_name=model_name)
        index = cls.from_codes(codes, model_name=model_name)
        index.save(cache_path)
        return index

    @classmethod
    def combine(cls, indexes: Sequence["EmbeddingIndex"]) -> "EmbeddingIndex":
        """Concatenate multiple embedding indexes into a single searchable index."""
        if not indexes:
            raise ValueError("At least one embedding index must be provided.")

        model = indexes[0].model
        descriptions: list[str] = []
        codes: list[str] = []
        vocabularies: list[str] = []
        embeddings: list[np.ndarray] = []

        for index in indexes:
            descriptions.extend(index.descriptions)
            codes.extend(index.codes)
            vocabularies.extend(index.vocabularies)
            embeddings.append(np.asarray(index.embeddings))

        return cls(
            model=model,
            descriptions=descriptions,
            codes=codes,
            vocabularies=vocabularies,
            embeddings=np.vstack(embeddings) if embeddings else np.empty((0, 0)),
        )

    @classmethod
    def load_for_vocabularies(
        cls,
        codes: pd.DataFrame,
        vocabularies: Sequence[str],
        cache_dir: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "EmbeddingIndex":
        """Load per-vocabulary embedding caches and concatenate them.

        Each vocabulary is cached independently as ``embeddings_<vocab>_<model>.npz``.
        Selected vocabularies are then combined in memory for retrieval.
        """
        cache_dir = Path(cache_dir)
        indexes: list[EmbeddingIndex] = []
        for vocabulary in vocabularies:
            vocab_df = codes[codes["vocabulary"].astype(str).str.upper() == vocabulary.upper()].reset_index(drop=True)
            if vocab_df.empty:
                continue
            cache_path = cache_dir / f"embeddings_{vocabulary}_{model_name.replace('/', '_')}.npz"
            _ensure_embedding_asset(cache_path)
            indexes.append(cls.from_cache_or_build(vocab_df, cache_path=cache_path, model_name=model_name))

        return cls.combine(indexes)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[EmbeddingCandidate]:
        q_emb = self.model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        scores = self.embeddings @ q_emb  # cosine similarity (both normalised)
        if top_k is not None and len(scores) > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            sorted_local = sorted(top_indices, key=lambda i: -scores[i])[:top_k]
        else:
            sorted_local = np.argsort(-scores)

        return [
            EmbeddingCandidate(
                rank=r + 1,
                code=self.codes[idx],
                description=self.descriptions[idx],
                vocabulary=self.vocabularies[idx],
                score=float(round(scores[idx], 6)),
            )
            for r, idx in enumerate(sorted_local)
        ]


def _release_asset_url(asset_name: str) -> str | None:
    owner = os.getenv("GITHUB_REPOSITORY_OWNER")
    repo = os.getenv("GITHUB_REPOSITORY_NAME")
    tag = os.getenv("GITHUB_RELEASE_TAG")
    if not owner or not repo or not tag:
        return None
    return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset_name}"


def _ensure_embedding_asset(cache_path: Path) -> None:
    """Ensure an embedding asset exists locally, downloading it from a GitHub Release if needed."""
    if cache_path.exists():
        return

    asset_name = cache_path.name
    url = _release_asset_url(asset_name)
    if url is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response, cache_path.open("wb") as fh:
            fh.write(response.read())
    except URLError:
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
        raise


def retrieve_embeddings(
    query: str,
    codes: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = None,
) -> list[EmbeddingCandidate]:
    """Build an embedding index on *codes* and retrieve candidates for *query*.

    Parameters
    ----------
    query : str
        Phenotype / EDF description to search.
    codes : pd.DataFrame
        Corpus with a ``description`` column (and optionally ``code``/``vocabulary``).
    model_name : str
        sentence-transformers model identifier.
    top_k : int, optional
        Number of results to return.

    Returns
    -------
    list[EmbeddingCandidate]
    """
    df = codes.copy()
    for col in ("code", "vocabulary"):
        if col not in df.columns:
            df[col] = ""

    descs = df["description"].fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    embs = model.encode(descs, show_progress_bar=False, normalize_embeddings=True)
    q_emb = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
    scores = embs @ q_emb

    sorted_idx = np.argsort(-scores)
    if top_k is not None:
        sorted_idx = sorted_idx[:top_k]

    return [
        EmbeddingCandidate(
            rank=r + 1,
            code=df.iloc[idx]["code"],
            description=descs[idx],
            vocabulary=str(df.iloc[idx].get("vocabulary", "")),
            score=float(round(scores[idx], 6)),
        )
        for r, idx in enumerate(sorted_idx)
    ]
