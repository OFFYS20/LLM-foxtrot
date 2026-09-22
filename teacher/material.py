"""Reading the material you want the model to learn from.

Accepts files, folders and raw text, using AI Studio's loaders so books, PDFs,
EPUBs and Word documents all work — not just .txt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import hashlib
import re

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
    #: Passages dropped for being repeats, and how many characters they were.
    repeats: int = 0
    repeated_characters: int = 0

    @property
    def characters(self) -> int:
        return len(self.text)

    def summary(self) -> str:
        parts = [f"{len(self.sources)} source(s)", f"{self.characters:,} characters"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.partial:
            parts.append(f"{len(self.partial)} read only in part")
        if self.repeats:
            parts.append(f"{self.repeats:,} repeat(s) dropped")
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


#: A passage shorter than this repeats for honest reasons — a heading, a name,
#: a refrain — and dropping it would quietly edit the text.
MIN_REPEAT = 200


def _fingerprint(passage: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", passage.lower()).strip().encode()).hexdigest()


def drop_repeats(text: str) -> tuple[str, int, int]:
    """Remove passages that have already appeared. Returns (text, dropped, characters).

    The cleaner already drops repeated paragraphs *within* a document. What
    this catches is repetition *across* sources — the same book in two folders,
    a chapter that is also on the web page you gathered, an export that
    overlaps the one beside it. Those teach a model to reproduce the material
    rather than learn from it, and the held-out number falls while the model
    gets worse, which is the kind of progress nobody wants.
    """
    kept: list[str] = []
    seen: set[str] = set()
    dropped = characters = 0

    # Split on any run of newlines, keeping the separators, because cleaning
    # collapses a blank line between paragraphs into a single one — looking for
    # blank lines alone would find no paragraphs at all in cleaned text.
    pieces = re.split(r"(\n+)", text)
    for index, piece in enumerate(pieces):
        if index % 2:                      # a separator, not a passage
            kept.append(piece)
            continue
        stripped = piece.strip()
        if len(stripped) < MIN_REPEAT:
            kept.append(piece)
            continue
        mark = _fingerprint(stripped)
        if mark in seen:
            dropped += 1
            characters += len(stripped)
            continue                        # its separator stays, so nothing runs together
        seen.add(mark)
        kept.append(piece)

    return "".join(kept).strip(), dropped, characters


def gather(paths: list[str], *, raw_text: str = "", clean: bool = True,
           dedupe: bool = True) -> Material:
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
    if dedupe:
        material.text, material.repeats, material.repeated_characters = drop_repeats(
            material.text)
    return material
