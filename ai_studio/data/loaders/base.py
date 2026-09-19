"""Document loader contract.

One loader per file family. A loader that needs an optional package raises
``DependencyMissingError`` naming the install command rather than returning
empty text — a silent empty document would look like a successful import.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedDocument:
    text: str
    title: str | None = None
    author: str | None = None
    doc_type: str = "txt"
    meta: dict[str, Any] = field(default_factory=dict)
    #: Optional per-page/per-section text, used by the PDF cleaner.
    sections: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class DocumentLoader(abc.ABC):
    """Reads one file family into plain text plus metadata."""

    #: Lower-case suffixes this loader claims, e.g. {".pdf"}
    extensions: set[str] = set()
    #: Short identifier stored on the document record.
    doc_type: str = "txt"

    @abc.abstractmethod
    def load(self, path: Path, *, max_rows: int | None = None) -> LoadedDocument:
        """Read ``path``. Raise IngestionError/DependencyMissingError on failure.

        ``max_rows`` caps how many rows a row-oriented file contributes;
        ``None`` means every row, bounded only by memory. Loaders that are not
        row-oriented ignore it.
        """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions
