"""Integration test: load a real AESI dataset + EDF and run hybrid retrieval,
verifying the full data path (loader -> preprocessing -> retrieval) works end
to end on the real curated phenotype data."""

from __future__ import annotations

import pytest

from src.data.loaders import load_aesi_dataset, load_edf
from src.data.preprocessing import preprocess_corpus
from src.retrieval.hybrid import hybrid_retrieval

pytestmark = pytest.mark.embeddings


def test_hybrid_retrieval_on_real_gbs_dataset():
    edf = load_edf("guillain_barre")
    raw_codes, gold = load_aesi_dataset("data/codelists/AESI/N_GBS_AESI_filtered.csv")
    codes = preprocess_corpus(raw_codes)

    candidates = hybrid_retrieval(
        edf.to_prompt_context(),
        codes,
        top_k=10,
        per_vocabulary_top_k=False,
    )

    assert len(candidates) == 10
    retrieved_codes = {c.code for c in candidates}
    # The anchor Guillain-Barre code should be retrievable from its own reviewed corpus.
    assert retrieved_codes & set(gold.loc[gold["label"] == "Narrow", "code"])
