"""Loader registry — extension → loader."""

from __future__ import annotations

from pathlib import Path

from ai_studio.core.errors import IngestionError
from ai_studio.data.loaders.base import DocumentLoader, LoadedDocument
from ai_studio.data.loaders.rich_loaders import DocxLoader, EPUBLoader, PDFLoader
from ai_studio.data.loaders.text_loaders import (
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


def load_document(path: Path) -> LoadedDocument:
    return get_loader(path).load(path)


__all__ = [
    "LOADERS",
    "SUPPORTED_EXTENSIONS",
    "DocumentLoader",
    "LoadedDocument",
    "get_loader",
    "load_document",
    "html_to_document",
    "text_from_string",
]
