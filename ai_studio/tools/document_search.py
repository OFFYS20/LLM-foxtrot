"""Keyword search across the Data Library (no embeddings required)."""

from __future__ import annotations

import re
from typing import Any

from ai_studio.core.errors import ValidationError


def search_documents(query: str, top_k: int = 4, *, max_chars: int = 600) -> list[dict[str, Any]]:
    """Return the best-matching excerpts by term overlap."""
    if not query or not query.strip():
        raise ValidationError("Enter a search query")
    top_k = max(1, min(int(top_k), 20))

    from ai_studio.data.ingestion import document_text, list_documents

    terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 2]
    if not terms:
        raise ValidationError("The query needs at least one word of three or more characters")

    results: list[dict[str, Any]] = []
    for record in list_documents(limit=200):
        try:
            text = document_text(record["id"])
        except Exception:  # noqa: BLE001 - skip unreadable documents
            continue
        lowered = text.lower()
        score = sum(lowered.count(term) for term in terms)
        if not score:
            continue
        position = min((lowered.find(term) for term in terms if lowered.find(term) >= 0), default=0)
        start = max(0, position - max_chars // 3)
        results.append(
            {
                "document_id": record["id"],
                "title": record.get("title") or record["filename"],
                "source": record["filename"],
                "score": score,
                "excerpt": text[start : start + max_chars].strip(),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
