"""Reading the material you want the model to learn from.

Accepts files, folders and raw text, using AI Studio's loaders so books, PDFs,
EPUBs and Word documents all work — not just .txt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_studio.data.loaders import SUPPORTED_EXTENSIONS, load_document
from ai_studio.data.preprocessing import CleaningOptions, clean_text

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".idea", "storage"}


@dataclass
class Material:
    """Everything gathered for one lesson."""

    text: str = ""
    sources: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Files that were read only in part, and by how many rows. A row-oriented
    #: file has a ceiling on how much is read; losing the rest quietly would be
    #: the worst kind of bug, because the lesson still looks like it worked.
    partial: list[tuple[str, int]] = field(default_factory=list)

    @property
    def characters(self) -> int:
        return len(self.text)

    def summary(self) -> str:
        parts = [f"{len(self.sources)} source(s)", f"{self.characters:,} characters"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.partial:
            parts.append(f"{len(self.partial)} read only in part")
        return ", ".join(parts)


def _files_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return found


def gather(paths: list[str], *, raw_text: str = "", clean: bool = True) -> Material:
    """Read every readable file under ``paths`` and join it into one corpus."""
    material = Material()
    chunks: list[str] = []

    if raw_text.strip():
        chunks.append(raw_text.strip())
        material.sources.append("(text given on the command line)")

    for entry in paths:
        root = Path(entry).expanduser()
        if not root.exists():
            material.skipped.append((entry, "no such file or folder"))
            continue

        candidates = _files_under(root)
        if not candidates:
            material.skipped.append((entry, "nothing readable inside"))
            continue

        for path in candidates:
            try:
                # Training wants every row of a CSV or JSONL, not a preview of it.
                document = load_document(path, max_rows=None)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a lesson
                material.skipped.append((str(path), str(exc)))
                continue

            text = document.text
            if clean:
                text = clean_text(
                    text,
                    CleaningOptions.for_type(document.doc_type),
                    sections=document.sections,
                ).text
            if not text.strip():
                material.skipped.append((str(path), "empty after cleaning"))
                continue

            dropped = int(document.meta.get("truncated") or 0)
            if dropped:
                material.partial.append((str(path), dropped))

            chunks.append(text.strip())
            material.sources.append(str(path))

    material.text = "\n\n".join(chunks)
    return material
