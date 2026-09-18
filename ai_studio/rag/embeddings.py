"""Sentence embeddings for retrieval.

Uses sentence-transformers. If it is not installed the error says so — there is
no hashing fallback pretending to be semantic search.
"""

from __future__ import annotations

import threading
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.errors import DependencyMissingError, StudioError

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUGGESTED_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "Fast, 384-dim, good default (~80 MB)"),
    ("sentence-transformers/all-mpnet-base-v2", "Higher quality, 768-dim (~420 MB)"),
    ("BAAI/bge-small-en-v1.5", "Strong retrieval quality, 384-dim"),
    ("intfloat/e5-small-v2", "Retrieval-tuned, 384-dim"),
]


class EmbeddingModel:
    """Thin wrapper so the vector store does not depend on the library directly."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DependencyMissingError(
                    "sentence-transformers", "Embeddings and RAG", install="sentence-transformers"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                raise StudioError(
                    f"Could not load the embedding model {self.model_name!r}: {exc}",
                    hint="Check the model name and your network connection (first use downloads it).",
                ) from exc
            log.info(f"Loaded embedding model {self.model_name}", source="rag")
        return self._model

    @property
    def dimension(self) -> int:
        model = self.load()
        # Newer sentence-transformers renamed this accessor.
        for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(model, attribute, None)
            if callable(getter):
                return int(getter())
        return int(self.encode(["dimension probe"]).shape[1])

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
        progress: Any = None,
    ):
        """Return an (n, dim) float32 numpy array."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        model = self.load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_one(self, text: str, *, normalize: bool = True):
        return self.encode([text], normalize=normalize)[0]


_models: dict[str, EmbeddingModel] = {}
_models_lock = threading.Lock()


def get_embedding_model(name: str = DEFAULT_MODEL) -> EmbeddingModel:
    with _models_lock:
        if name not in _models:
            _models[name] = EmbeddingModel(name)
        return _models[name]
