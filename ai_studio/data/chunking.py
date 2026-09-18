"""Chunking: turn long documents into training sequences.

Token chunking uses the real tokenizer when one is supplied, so a "2048-token
block" is genuinely 2048 tokens rather than a character estimate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int = 0
    document_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        normalised = re.sub(r"\W+", " ", self.text.lower()).strip()
        return hashlib.sha1(normalised.encode()).hexdigest()


def chunk_by_tokens(
    text: str,
    tokenizer: Any,
    *,
    block_size: int = 1024,
    overlap: int = 0,
    document_id: str | None = None,
    min_tokens: int = 16,
    drop_last_partial: bool = False,
) -> list[Chunk]:
    """Split text into token blocks with optional overlap (a real token count)."""
    if block_size < 8:
        raise ValueError("block_size must be at least 8 tokens")
    overlap = max(0, min(overlap, block_size - 1))
    stride = block_size - overlap

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, max(1, len(ids)), stride)):
        window = ids[start : start + block_size]
        if not window:
            break
        if len(window) < min_tokens:
            break
        if drop_last_partial and len(window) < block_size:
            break
        chunks.append(
            Chunk(
                text=tokenizer.decode(window, skip_special_tokens=True),
                index=index,
                token_count=len(window),
                document_id=document_id,
                meta={"start_token": start},
            )
        )
        if start + block_size >= len(ids):
            break
    return chunks


def chunk_by_characters(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
    document_id: str | None = None,
    min_chars: int = 64,
    respect_paragraphs: bool = True,
) -> list[Chunk]:
    """Character-window chunking used by RAG and when no tokenizer is selected."""
    text = text.strip()
    if not text:
        return []
    overlap = max(0, min(overlap, chunk_size - 1))

    if respect_paragraphs:
        units = [p.strip() for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
    else:
        units = [text]

    chunks: list[Chunk] = []
    buffer = ""

    def _flush() -> None:
        nonlocal buffer
        cleaned = buffer.strip()
        if len(cleaned) >= min_chars:
            chunks.append(
                Chunk(
                    text=cleaned,
                    index=len(chunks),
                    token_count=max(1, len(cleaned) // 4),
                    document_id=document_id,
                )
            )
        buffer = ""

    for unit in units:
        # A single oversized paragraph is split on sentence boundaries.
        if len(unit) > chunk_size:
            _flush()
            for piece in _split_long(unit, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        index=len(chunks),
                        token_count=max(1, len(piece) // 4),
                        document_id=document_id,
                    )
                )
            continue

        if len(buffer) + len(unit) + 2 > chunk_size:
            tail = buffer[-overlap:] if overlap and buffer else ""
            _flush()
            buffer = f"{tail}\n\n{unit}" if tail else unit
        else:
            buffer = f"{buffer}\n\n{unit}" if buffer else unit

    _flush()
    return chunks


def _split_long(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    sentences = SENTENCE_SPLIT.split(text)
    buffer = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:  # no sentence boundary — hard window
            if buffer.strip():
                yield buffer.strip()
                buffer = ""
            for start in range(0, len(sentence), max(1, chunk_size - overlap)):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    yield piece
            continue
        if len(buffer) + len(sentence) + 1 > chunk_size:
            if buffer.strip():
                yield buffer.strip()
            buffer = (buffer[-overlap:] + " " + sentence) if overlap else sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer.strip():
        yield buffer.strip()


def deduplicate(chunks: list[Chunk], *, similarity_shingle: int = 0) -> tuple[list[Chunk], int]:
    """Drop exact duplicates, and near-duplicates when a shingle size is given."""
    seen: set[str] = set()
    shingles: set[str] = set()
    kept: list[Chunk] = []
    removed = 0

    for chunk in chunks:
        fingerprint = chunk.fingerprint
        if fingerprint in seen:
            removed += 1
            continue
        if similarity_shingle:
            head = _shingle(chunk.text, similarity_shingle)
            if head and head in shingles:
                removed += 1
                continue
            if head:
                shingles.add(head)
        seen.add(fingerprint)
        kept.append(chunk)

    for position, chunk in enumerate(kept):
        chunk.index = position
    return kept, removed


def _shingle(text: str, words: int) -> str:
    tokens = re.sub(r"\W+", " ", text.lower()).split()
    if len(tokens) < words:
        return ""
    return hashlib.sha1(" ".join(tokens[:words]).encode()).hexdigest()
