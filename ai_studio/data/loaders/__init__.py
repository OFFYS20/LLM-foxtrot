"""Loader registry — extension → loader."""

from __future__ import annotations

from pathlib import Path

from ai_studio.core.errors import IngestionError
from ai_studio.data.loaders.base import DocumentLoader, LoadedDocument
from ai_studio.data.loaders.rich_loaders import DocxLoader, EPUBLoader, PDFLoader
from ai_studio.data.loaders.text_loaders import (
    MAX_PREVIEW_ROWS,
    CSVLoader,
    HTMLLoader,
    JSONLoader,
    MarkdownLoader,
    TextLoader,
    html_to_document,
    text_from_string,
)

LOADERS: list[DocumentLoader] = [
    TextLoader(),
    MarkdownLoader(),
    PDFLoader(),
    DocxLoader(),
    EPUBLoader(),
    HTMLLoader(),
    CSVLoader(),
    JSONLoader(),
]

SUPPORTED_EXTENSIONS: set[str] = {ext for loader in LOADERS for ext in loader.extensions}


def get_loader(path: Path) -> DocumentLoader:
    suffix = path.suffix.lower()
    for loader in LOADERS:
        if suffix in loader.extensions:
            return loader
    raise IngestionError(
        f"Unsupported file type {suffix or '(none)'}",
        hint=f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
    )


def load_document(path: Path, *, max_rows: int | None = MAX_PREVIEW_ROWS) -> LoadedDocument:
    """Read one file. ``max_rows=None`` reads every row of a CSV or JSONL.

    The default suits a preview. Training wants the whole file, and passes None.
    """
    return get_loader(path).load(path, max_rows=max_rows)


__all__ = [
    "LOADERS",
    "MAX_PREVIEW_ROWS",
    "SUPPORTED_EXTENSIONS",
    "DocumentLoader",
    "LoadedDocument",
    "get_loader",
    "load_document",
    "html_to_document",
    "text_from_string",
]
