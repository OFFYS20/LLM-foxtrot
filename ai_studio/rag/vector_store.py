"""Local vector store.

FAISS when available (fast, scales to large libraries); otherwise an exact
NumPy cosine search, which is slower but gives identical results on small
collections. Which backend ran is always reported.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.errors import DependencyMissingError, StudioError

INDEX_FILENAME = "index.faiss"
VECTORS_FILENAME = "vectors.npy"
CHUNKS_FILENAME = "chunks.jsonl"
META_FILENAME = "store.json"


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class StoredChunk:
    text: str
    document_id: str | None = None
    source: str = ""
    title: str = ""
    chunk_index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredChunk":
        return cls(
            text=data.get("text", ""),
            document_id=data.get("document_id"),
            source=data.get("source", ""),
            title=data.get("title", ""),
            chunk_index=int(data.get("chunk_index", 0)),
            meta=data.get("meta", {}),
        )


@dataclass
class SearchHit:
    chunk: StoredChunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.chunk.to_dict(), "score": round(self.score, 4), "rank": self.rank}


class VectorStore:
    """Cosine-similarity store over normalised embeddings."""

    def __init__(self, dimension: int, *, use_faiss: bool | None = None) -> None:
        self.dimension = dimension
        self.chunks: list[StoredChunk] = []
        self._index: Any = None
        self._vectors: Any = None
        self.backend = "faiss" if (faiss_available() if use_faiss is None else use_faiss) else "numpy"
        if self.backend == "faiss":
            import faiss

            self._index = faiss.IndexFlatIP(dimension)  # inner product on unit vectors == cosine

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, vectors: Any, chunks: list[StoredChunk]) -> None:
        import numpy as np

        if len(vectors) != len(chunks):
            raise StudioError("Vector/chunk count mismatch while indexing.")
        if not len(chunks):
            return
        vectors = np.asarray(vectors, dtype="float32")
        if self.backend == "faiss":
            self._index.add(vectors)
        else:
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: Any, top_k: int = 4, *, min_score: float = 0.0) -> list[SearchHit]:
        import numpy as np

        if not self.chunks:
            return []
        # Over-fetch so de-duplication can still return `top_k` distinct chunks.
        requested = max(1, min(top_k, len(self.chunks)))
        top_k = min(len(self.chunks), requested * 3)
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        if self.backend == "faiss":
            scores, indices = self._index.search(query, top_k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            similarities = (self._vectors @ query.T).ravel()
            order = np.argsort(-similarities)[:top_k]
            pairs = [(int(i), float(similarities[i])) for i in order]

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for index, score in pairs:
            if index < 0 or index >= len(self.chunks) or score < min_score:
                continue
            chunk = self.chunks[index]
            # Overlapping chunks repeat text; keep only the best-scoring copy.
            fingerprint = " ".join(chunk.text.split()).lower()[:400]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            hits.append(SearchHit(chunk=chunk, score=float(score), rank=len(hits) + 1))
            if len(hits) >= requested:
                break
        return hits

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path | str, *, meta: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss":
            import faiss

            faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        else:
            import numpy as np

            np.save(directory / VECTORS_FILENAME, self._vectors)

        with (directory / CHUNKS_FILENAME).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        (directory / META_FILENAME).write_text(
            json.dumps(
                {"dimension": self.dimension, "backend": self.backend, "chunks": len(self.chunks), **(meta or {})},
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "VectorStore":
        directory = Path(directory)
        meta_path = directory / META_FILENAME
        if not meta_path.exists():
            raise StudioError(f"No vector store found in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        backend = meta.get("backend", "numpy")
        if backend == "faiss" and not faiss_available():
            raise DependencyMissingError("faiss-cpu", "Loading this FAISS index", install="faiss-cpu")

        store = cls(int(meta["dimension"]), use_faiss=backend == "faiss")
        if backend == "faiss":
            import faiss

            store._index = faiss.read_index(str(directory / INDEX_FILENAME))
        else:
            import numpy as np

            path = directory / VECTORS_FILENAME
            store._vectors = np.load(path) if path.exists() else None

        chunks_path = directory / CHUNKS_FILENAME
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as handle:
                store.chunks = [StoredChunk.from_dict(json.loads(line)) for line in handle if line.strip()]
        return store
