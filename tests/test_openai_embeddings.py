"""Unit tests for selectable, separately cached OpenAI embeddings."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.retrieval.embeddings import OpenAIEmbeddingModel, embedding_cache_key


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        # Deliberately return each batch out of order to exercise index-order
        # restoration. Values vary between batches so batching is observable.
        base = len(self.requests) * 10
        data = [
            SimpleNamespace(index=index, embedding=[float(base + index), 1.0])
            for index in reversed(range(len(request["input"])))
        ]
        return SimpleNamespace(data=data)


def test_openai_embedding_adapter_batches_orders_and_normalizes():
    embeddings = _FakeEmbeddings()
    model = OpenAIEmbeddingModel("text-embedding-3-large", dimensions=2, batch_size=2)
    model._client = SimpleNamespace(embeddings=embeddings)

    result = model.encode(["a", "b", "c"])

    assert result.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(3))
    assert [request["input"] for request in embeddings.requests] == [["a", "b"], ["c"]]
    assert all(request["dimensions"] == 2 for request in embeddings.requests)
    assert result[0, 0] < result[1, 0] < result[2, 0]


def test_embedding_cache_keys_keep_openai_and_local_indexes_separate():
    local = embedding_cache_key("local", "sentence-transformers/all-MiniLM-L6-v2")
    openai = embedding_cache_key("openai", "text-embedding-3-large", 3072)

    assert local == "sentence-transformers_all-MiniLM-L6-v2"
    assert openai == "openai_text-embedding-3-large_3072"
    assert local != openai
