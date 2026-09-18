"""RAG: vector store ranking, de-duplication, persistence and prompt context.

Embeddings are supplied by hand so the tests assert retrieval behaviour rather
than the quality of a downloaded sentence-transformers model.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_studio.core.errors import StudioError
from ai_studio.rag.retrieval import build_context
from ai_studio.rag.vector_store import SearchHit, StoredChunk, VectorStore, faiss_available

BACKENDS = ["numpy"] + (["faiss"] if faiss_available() else [])


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype="float32")
    return vector / np.linalg.norm(vector)


def chunk(text: str, index: int = 0) -> StoredChunk:
    return StoredChunk(
        text=text, document_id="doc-1", source="handbook.pdf", title="Handbook", chunk_index=index
    )


@pytest.fixture(params=BACKENDS)
def store(request) -> VectorStore:
    return VectorStore(dimension=3, use_faiss=request.param == "faiss")


def test_empty_store_returns_no_hits(store):
    assert store.search(unit(1, 0, 0), top_k=4) == []


def test_ranks_by_cosine_similarity(store):
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0.9, 0.1, 0)])
    store.add(vectors, [chunk("exact", 0), chunk("unrelated", 1), chunk("close", 2)])

    hits = store.search(unit(1, 0, 0), top_k=3)
    assert [hit.chunk.text for hit in hits] == ["exact", "close", "unrelated"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_top_k_limits_results(store):
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    store.add(vectors, [chunk(f"chunk {i}", i) for i in range(3)])
    assert len(store.search(unit(1, 0, 0), top_k=2)) == 2


def test_min_score_filters_weak_matches(store):
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
    store.add(vectors, [chunk("relevant", 0), chunk("orthogonal", 1)])

    hits = store.search(unit(1, 0, 0), top_k=4, min_score=0.5)
    assert [hit.chunk.text for hit in hits] == ["relevant"]


def test_overlapping_chunks_are_deduplicated(store):
    # Overlapping chunkers emit the same passage twice; only the best copy is kept.
    duplicate = "The warranty period is twenty four months from delivery."
    vectors = np.stack([unit(1, 0, 0), unit(0.99, 0.01, 0), unit(0, 1, 0)])
    store.add(vectors, [chunk(duplicate, 0), chunk(duplicate, 1), chunk("something else", 2)])

    hits = store.search(unit(1, 0, 0), top_k=3)
    texts = [hit.chunk.text for hit in hits]
    assert texts.count(duplicate) == 1, "the same passage must not be returned twice"
    assert "something else" in texts


def test_add_rejects_count_mismatch(store):
    with pytest.raises(StudioError):
        store.add(np.stack([unit(1, 0, 0), unit(0, 1, 0)]), [chunk("only one", 0)])


def test_save_and_load_round_trip(store, tmp_path):
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
    store.add(vectors, [chunk("alpha", 0), chunk("beta", 1)])
    store.save(tmp_path / "index", meta={"embedding_model": "unit-test"})

    reloaded = VectorStore.load(tmp_path / "index")
    assert len(reloaded) == 2
    assert reloaded.dimension == 3
    assert reloaded.backend == store.backend
    hits = reloaded.search(unit(1, 0, 0), top_k=1)
    assert hits[0].chunk.text == "alpha"
    assert hits[0].chunk.source == "handbook.pdf", "chunk metadata survives persistence"


def test_load_without_an_index_is_an_error(tmp_path):
    with pytest.raises(StudioError):
        VectorStore.load(tmp_path / "not-an-index")


def test_build_context_cites_sources_and_admits_gaps():
    hits = [
        SearchHit(chunk=chunk("Warranty lasts 24 months.", 0), score=0.91, rank=1),
        SearchHit(chunk=chunk("Returns are free within 30 days.", 1), score=0.77, rank=2),
    ]
    context = build_context(hits)
    assert "Warranty lasts 24 months." in context
    assert "handbook.pdf" in context and "0.910" in context
    assert "say so explicitly" in context, "the model must be told not to invent an answer"


def test_build_context_respects_the_character_budget():
    hits = [
        SearchHit(chunk=chunk("x" * 500, index), score=0.9, rank=index + 1) for index in range(10)
    ]
    context = build_context(hits, max_chars=1200)
    assert len(context) < 2000
    assert context.count("similarity") < 10, "context is truncated, not silently concatenated"


def test_build_context_of_nothing_is_empty():
    assert build_context([]) == ""
