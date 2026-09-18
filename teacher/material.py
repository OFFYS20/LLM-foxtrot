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

    @property
    def characters(self) -> int:
        return len(self.text)

    def summary(self) -> str:
        parts = [f"{len(self.sources)} source(s)", f"{self.characters:,} characters"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
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
                document = load_document(path)
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

            chunks.append(text.strip())
            material.sources.append(str(path))

    material.text = "\n\n".join(chunks)
    return material
