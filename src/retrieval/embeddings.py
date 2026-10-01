"""Embedding-based retrieval using local or API-backed encoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Protocol, Sequence
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


def _create_local_embedding_model(model_name: str):
    """Import the optional local stack only when that backend is selected."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingEncoder(Protocol):
    """Common interface for local and API-backed embedding models."""

    def encode(self, sentences, **kwargs): ...


class OpenAIEmbeddingModel:
    """SentenceTransformer-compatible adapter for OpenAI embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        *,
        dimensions: int | None = 3072,
        batch_size: int = 512,
    ) -> None:
        if dimensions is not None and dimensions < 1:
            raise ValueError("Embedding dimensions must be at least 1")
        if batch_size < 1:
            raise ValueError("Embedding batch size must be at least 1")
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def encode(
        self,
        sentences,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **_kwargs,
    ) -> np.ndarray:
        del show_progress_bar
        single = isinstance(sentences, str)
        inputs = [sentences] if single else [str(item) for item in sentences]
        if not inputs:
            width = self.dimensions or 0
            return np.empty((0, width), dtype=np.float32)

        # Convert each response to float32 immediately. Keeping hundreds of
        # thousands of embeddings as Python float lists can require many times
        # more memory than the final NumPy matrix.
        chunks: list[np.ndarray] = []
        client = self._get_client()
        for start in range(0, len(inputs), self.batch_size):
            request = {
                "model": self.model_name,
                "input": inputs[start : start + self.batch_size],
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = client.embeddings.create(**request)
            chunks.append(
                np.asarray(
                    [
                        item.embedding
                        for item in sorted(response.data, key=lambda item: item.index)
                    ],
                    dtype=np.float32,
                )
            )

        result = np.vstack(chunks)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)
        return result[0] if single else result


def create_embedding_model(
    provider: str,
    model_name: str,
    *,
    dimensions: int | None = None,
    batch_size: int = 512,
) -> EmbeddingEncoder:
    """Construct a local SentenceTransformer or OpenAI embedding adapter."""
    provider = provider.strip().lower()
    if provider == "local":
        return _create_local_embedding_model(model_name)
    if provider == "openai":
        return OpenAIEmbeddingModel(
            model_name,
            dimensions=dimensions,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported embedding provider: {provider!r}")


def embedding_cache_key(
    provider: str,
    model_name: str,
    dimensions: int | None = None,
) -> str:
    """Return a filesystem-safe cache identity without mixing providers/models."""
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    if provider.strip().lower() == "local":
        return model_key
    dimension_key = f"_{dimensions}" if dimensions is not None else ""
    return f"{provider.strip().lower()}_{model_key}{dimension_key}"


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

    model: EmbeddingEncoder | None
    descriptions: list[str]
    codes: list[str]
    vocabularies: list[str]
    embeddings: np.ndarray  # shape (n, dim)

    @classmethod
    def from_codes(
        cls,
        codes: pd.DataFrame,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        model: EmbeddingEncoder | None = None,
        show_progress_bar: bool = False,
    ) -> "EmbeddingIndex":
        df = codes.copy()
        for col in ("code", "vocabulary"):
            if col not in df.columns:
                df[col] = ""
        embedding_model = model or _create_local_embedding_model(model_name)
        descs = df["description"].fillna("").astype(str).tolist()
        embs = embedding_model.encode(
            descs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        return cls(
            model=embedding_model,
            descriptions=descs,
            codes=df["code"].astype(str).tolist(),
            vocabularies=df["vocabulary"].astype(str).tolist(),
            embeddings=np.asarray(embs, dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Persist the corpus embeddings + metadata to disk (npz) so they can
        be reloaded without re-running the embedding model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=np.asarray(self.embeddings, dtype=np.float32),
            codes=np.array(self.codes, dtype=object),
            descriptions=np.array(self.descriptions, dtype=object),
            vocabularies=np.array(self.vocabularies, dtype=object),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        model: EmbeddingEncoder | None = None,
        load_model: bool = True,
    ) -> "EmbeddingIndex":
        """Load a previously-saved embedding index from disk."""
        data = np.load(path, allow_pickle=True)
        return cls(
            model=model or (_create_local_embedding_model(model_name) if load_model else None),
            descriptions=list(data["descriptions"]),
            codes=list(data["codes"]),
            vocabularies=list(data["vocabularies"]),
            embeddings=np.asarray(data["embeddings"], dtype=np.float32),
        )

    @classmethod
    def from_cache_or_build(
        cls,
        codes: pd.DataFrame,
        cache_path: str | Path,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        model: EmbeddingEncoder | None = None,
        load_model: bool = True,
        show_progress_bar: bool = False,
    ) -> "EmbeddingIndex":
        """Load a cached embedding index if present, otherwise build it from
        *codes* and cache the result for subsequent runs.

        This is the key mechanism that avoids re-embedding large reference
        code systems (e.g. the ~74k-row full ICD-10-CM corpus) on every run.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = cls.load(
                cache_path,
                model_name=model_name,
                model=model,
                load_model=load_model,
            )
            expected_codes = codes.get("code", pd.Series(dtype=str)).astype(str).tolist()
            expected_vocabularies = (
                codes.get("vocabulary", pd.Series(dtype=str)).astype(str).tolist()
            )
            if cached.codes == expected_codes and cached.vocabularies == expected_vocabularies:
                return cached
            subset_positions = _ordered_subset_positions(
                cached.codes,
                cached.vocabularies,
                expected_codes,
                expected_vocabularies,
            )
            if subset_positions is not None:
                # Corpus preprocessing can intentionally remove non-codable
                # dictionary rows (for example ICD range headers). Reuse the
                # matching cached vectors instead of re-embedding every code.
                subset = cls(
                    model=cached.model,
                    descriptions=codes.get("description", pd.Series(dtype=str))
                    .astype(str)
                    .tolist(),
                    codes=expected_codes,
                    vocabularies=expected_vocabularies,
                    embeddings=np.asarray(cached.embeddings[subset_positions], dtype=np.float32),
                )
                subset.save(cache_path)
                return subset
        index = cls.from_codes(
            codes,
            model_name=model_name,
            model=model,
            show_progress_bar=show_progress_bar,
        )
        index.save(cache_path)
        return index

    @classmethod
    def combine(cls, indexes: Sequence["EmbeddingIndex"]) -> "EmbeddingIndex":
        """Concatenate multiple embedding indexes into a single searchable index."""
        if not indexes:
            raise ValueError("At least one embedding index must be provided.")

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
            model=indexes[0].model,
            descriptions=descriptions,
            codes=codes,
            vocabularies=vocabularies,
            embeddings=np.vstack(embeddings).astype(np.float32)
            if embeddings
            else np.empty((0, 0), dtype=np.float32),
        )

    @classmethod
    def load_for_vocabularies(
        cls,
        codes: pd.DataFrame,
        vocabularies: Sequence[str],
        cache_dir: str | Path,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        model: EmbeddingEncoder | None = None,
        provider: str = "local",
        dimensions: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "EmbeddingIndex":
        """Load per-vocabulary embedding caches and concatenate them.

        Each vocabulary is cached independently by vocabulary, provider,
        model, and output dimensions.
        Selected vocabularies are then combined in memory for retrieval.
        """
        cache_dir = Path(cache_dir)
        indexes: list[EmbeddingIndex] = []
        total = len(vocabularies)
        if progress_callback is not None:
            progress_callback(0, total)
        for position, vocabulary in enumerate(vocabularies, start=1):
            vocab_df = codes[
                codes["vocabulary"].astype(str).str.upper() == vocabulary.upper()
            ].reset_index(drop=True)
            if vocab_df.empty:
                continue
            cache_identity = embedding_cache_key(provider, model_name, dimensions)
            cache_path = cache_dir / f"embeddings_{vocabulary}_{cache_identity}.npz"
            # Existing release assets contain the legacy local MiniLM indexes.
            # OpenAI indexes are private, paid-to-generate local artifacts and
            # must not be mistaken for downloadable release assets.
            if provider.strip().lower() == "local":
                _ensure_embedding_asset(cache_path)
            indexes.append(
                cls.from_cache_or_build(
                    vocab_df,
                    cache_path=cache_path,
                    model_name=model_name,
                    model=model,
                    load_model=False,
                    show_progress_bar=progress_callback is not None,
                )
            )
            if progress_callback is not None:
                progress_callback(position, total)
        return cls(
            model=None,
            descriptions=[d for index in indexes for d in index.descriptions],
            codes=[c for index in indexes for c in index.codes],
            vocabularies=[v for index in indexes for v in index.vocabularies],
            embeddings=np.vstack(
                [np.asarray(index.embeddings, dtype=np.float32) for index in indexes]
            ).astype(np.float32)
            if indexes
            else np.empty((0, 0), dtype=np.float32),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        model: EmbeddingEncoder | None = None,
    ) -> list[EmbeddingCandidate]:
        query_model = model or self.model
        if query_model is None:
            raise ValueError("An embedding model is required to embed the query.")
        q_emb = query_model.encode(query, show_progress_bar=False, normalize_embeddings=True)
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

    def score_all(
        self,
        query: str,
        model: EmbeddingEncoder | None = None,
    ) -> list[float]:
        """Return corpus-order cosine scores without ranking or object creation.

        Hybrid retrieval needs every embedding score for global min-max
        normalization, but it does not need the embedding-only ranking. This
        path preserves the same six-decimal scores as :meth:`retrieve` while
        avoiding a full sort and hundreds of thousands of temporary
        ``EmbeddingCandidate`` objects.
        """
        query_model = model or self.model
        if query_model is None:
            raise ValueError("An embedding model is required to embed the query.")
        q_emb = query_model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        scores = self.embeddings @ q_emb
        return [float(round(score, 6)) for score in scores]


def _ordered_subset_positions(
    cached_codes: Sequence[str],
    cached_vocabularies: Sequence[str],
    expected_codes: Sequence[str],
    expected_vocabularies: Sequence[str],
) -> list[int] | None:
    """Locate an order-preserving expected corpus inside a cached corpus."""
    if (
        len(expected_codes) != len(expected_vocabularies)
        or len(expected_codes) > len(cached_codes)
    ):
        return None
    positions: list[int] = []
    expected_index = 0
    for cached_index, (code, vocabulary) in enumerate(
        zip(cached_codes, cached_vocabularies)
    ):
        if expected_index >= len(expected_codes):
            break
        if (
            code == expected_codes[expected_index]
            and vocabulary == expected_vocabularies[expected_index]
        ):
            positions.append(cached_index)
            expected_index += 1
    return positions if expected_index == len(expected_codes) else None


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
    model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
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
    model = _create_local_embedding_model(model_name)
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
