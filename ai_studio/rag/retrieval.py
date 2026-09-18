"""Building and querying knowledge indexes over the Data Library."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import NotFoundError, StudioError, ValidationError
from ai_studio.core.paths import dir_size, remove_path, slugify
from ai_studio.data.chunking import chunk_by_characters
from ai_studio.rag.embeddings import DEFAULT_MODEL, get_embedding_model
from ai_studio.rag.vector_store import SearchHit, StoredChunk, VectorStore


@dataclass
class RetrievalSettings:
    index_id: str | None = None
    top_k: int = 4
    min_score: float = 0.0
    enabled: bool = True
    max_context_chars: int = 4000


def build_index(
    name: str,
    document_ids: list[str],
    *,
    embedding_model: str = DEFAULT_MODEL,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    progress: Any = None,
) -> dict[str, Any]:
    """Chunk, embed and index the selected documents."""
    if not document_ids:
        raise ValidationError("Select at least one document to index.")

    db = get_db()
    if db.get("rag_indexes", name, key="name"):
        raise ValidationError(f"An index named {name!r} already exists.")

    from ai_studio.data.ingestion import document_text

    model = get_embedding_model(embedding_model)
    dimension = model.dimension  # raises early if the dependency is missing

    chunks: list[StoredChunk] = []
    used: list[str] = []
    skipped: list[str] = []

    for position, document_id in enumerate(document_ids):
        if progress:
            progress(
                (position + 1) / max(1, len(document_ids)) * 0.5,
                desc=f"Chunking {position + 1}/{len(document_ids)}",
            )
        record = db.get("documents", document_id)
        if record is None:
            skipped.append(document_id)
            continue
        try:
            text = document_text(document_id)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{record['filename']}: {exc}")
            continue
        pieces = chunk_by_characters(
            text, chunk_size=chunk_size, overlap=chunk_overlap, document_id=document_id
        )
        for piece in pieces:
            chunks.append(
                StoredChunk(
                    text=piece.text,
                    document_id=document_id,
                    source=record["filename"],
                    title=record.get("title") or record["filename"],
                    chunk_index=piece.index,
                    meta={"doc_type": record["doc_type"]},
                )
            )
        if pieces:
            used.append(document_id)

    if not chunks:
        raise ValidationError("No text chunks were produced from the selected documents.")

    if progress:
        progress(0.6, desc=f"Embedding {len(chunks):,} chunks")
    started = time.perf_counter()
    vectors = model.encode([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - started

    store = VectorStore(dimension)
    store.add(vectors, chunks)

    directory = get_config().indexes_dir / slugify(name)
    if directory.exists():
        remove_path(directory)
    store.save(directory, meta={"embedding_model": embedding_model, "name": name})

    record = {
        "id": new_id("idx"),
        "name": name,
        "embedding_model": embedding_model,
        "path": str(directory),
        "dimension": dimension,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_ids": used,
        "config": {"backend": store.backend, "embedding_model": embedding_model},
        "stats": {
            "documents": len(used),
            "skipped": skipped,
            "embedding_seconds": round(elapsed, 2),
            "size_bytes": dir_size(directory),
        },
        "status": "ready",
        "created_at": time.time(),
    }
    db.insert("rag_indexes", record)
    log.info(
        f"Built index '{name}': {len(chunks):,} chunks from {len(used)} documents "
        f"({store.backend}, {elapsed:.1f}s)",
        source="rag",
        context={"index_id": record["id"]},
    )
    return db.require("rag_indexes", record["id"])


_loaded: dict[str, VectorStore] = {}


def load_index(index_id: str) -> tuple[VectorStore, dict[str, Any]]:
    record = get_db().require("rag_indexes", index_id)
    store = _loaded.get(index_id)
    if store is None:
        path = Path(record["path"])
        if not path.exists():
            raise NotFoundError(
                f"Index files for '{record['name']}' are missing.", hint="Rebuild the index."
            )
        store = VectorStore.load(path)
        _loaded[index_id] = store
    return store, record


def search(index_id: str, query: str, *, top_k: int = 4, min_score: float = 0.0) -> list[SearchHit]:
    if not query.strip():
        return []
    store, record = load_index(index_id)
    model = get_embedding_model(record["embedding_model"])
    vector = model.encode_one(query)
    return store.search(vector, top_k=top_k, min_score=min_score)


def build_context(hits: list[SearchHit], *, max_chars: int = 4000) -> str:
    """Render retrieved chunks as a prompt block with visible sources."""
    if not hits:
        return ""
    blocks: list[str] = ["Relevant excerpts from your documents:"]
    used = 0
    for hit in hits:
        block = (
            f"\n[{hit.rank}] {hit.chunk.title} ({hit.chunk.source}, similarity {hit.score:.3f})\n"
            f"{hit.chunk.text.strip()}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    blocks.append(
        "\nAnswer using these excerpts. If they do not contain the answer, say so explicitly."
    )
    return "\n".join(blocks)


def retrieve_for_prompt(
    settings: RetrievalSettings, query: str
) -> tuple[str, list[SearchHit]]:
    """Fetch context for a chat turn. Failures degrade to no-context, not a crash."""
    if not settings.enabled or not settings.index_id:
        return "", []
    try:
        hits = search(settings.index_id, query, top_k=settings.top_k, min_score=settings.min_score)
    except StudioError as exc:
        log.warning(f"Retrieval failed: {exc.message}", source="rag")
        return "", []
    return build_context(hits, max_chars=settings.max_context_chars), hits


def list_indexes() -> list[dict[str, Any]]:
    return get_db().list("rag_indexes", order_by="created_at DESC")


def delete_index(index_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("rag_indexes", index_id)
    _loaded.pop(index_id, None)
    if remove_files and record.get("path"):
        remove_path(Path(record["path"]))
    db.delete("rag_indexes", index_id)
    log.warning(f"Deleted index {record['name']}", source="rag")
