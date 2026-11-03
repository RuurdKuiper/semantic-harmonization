"""Corpus wording is part of the identity of cached embedding vectors."""

import numpy as np
import pytest

from src.retrieval.embeddings import EmbeddingIndex


@pytest.mark.parametrize("subset", [False, True])
def test_changed_description_rebuilds_vectors(sample_codes, tmp_path, subset):
    class RecordingEncoder:
        def __init__(self):
            self.inputs = []

        def encode(self, descriptions, **kwargs):
            self.inputs.append(descriptions)
            return np.ones((len(descriptions), 2), dtype=np.float32)

    path = tmp_path / "index.npz"
    EmbeddingIndex(
        model=None,
        descriptions=sample_codes.description.tolist(),
        codes=sample_codes.code.tolist(),
        vocabularies=sample_codes.vocabulary.tolist(),
        embeddings=np.zeros((len(sample_codes), 2), dtype=np.float32),
    ).save(path)
    updated = sample_codes.iloc[[0, 2, 4]].copy() if subset else sample_codes.copy()
    updated.loc[0, "description"] = "Revised clinical wording"
    encoder = RecordingEncoder()

    rebuilt = EmbeddingIndex.from_cache_or_build(
        updated, path, model=encoder, load_model=False
    )

    assert encoder.inputs == [updated.description.tolist()]
    assert np.array_equal(rebuilt.embeddings, np.ones((len(updated), 2)))
    assert EmbeddingIndex.load(path, load_model=False).descriptions == updated.description.tolist()


def test_unchanged_description_reuses_vectors(sample_codes, tmp_path):
    class RejectEncoder:
        def encode(self, *args, **kwargs):
            raise AssertionError("Unchanged wording must not be re-embedded")

    path = tmp_path / "index.npz"
    original = np.arange(len(sample_codes) * 2, dtype=np.float32).reshape(-1, 2)
    EmbeddingIndex(
        model=None,
        descriptions=sample_codes.description.tolist(),
        codes=sample_codes.code.tolist(),
        vocabularies=sample_codes.vocabulary.tolist(),
        embeddings=original,
    ).save(path)

    loaded = EmbeddingIndex.from_cache_or_build(
        sample_codes, path, model=RejectEncoder(), load_model=False
    )

    assert np.array_equal(loaded.embeddings, original)
