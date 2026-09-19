"""Looking something up on the web, as a tool a model can be given.

Two separate capabilities, because they carry different weight: `web_search`
sends a query to a search engine and returns what it found, and `read_web_page`
fetches one address in full. Both are allow-listed http(s) only, size-capped and
refused for anything on this machine or its local network.

A word on which models this is for. Deciding to call a tool — noticing that a
question needs a fact it does not have, choosing a query, reading the answer
back — is an ability that appears in large instruction-tuned models. A model
you train here, at one million to a few hundred million parameters, will not do
it. This tool is useful with a capable model you have imported, and useful to
you and to any assistant driving Studio; it will not make a small model
resourceful.
"""

from __future__ import annotations

from typing import Any

from ai_studio.core.errors import ValidationError
from ai_studio.data import web

#: How much of a page one tool call hands back. Enough to answer from, not so
#: much that it swamps a context window.
DEFAULT_PAGE_CHARACTERS = 4_000
MAX_PAGE_CHARACTERS = 40_000


def web_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search the web and return titles, addresses and snippets."""
    if not query or not query.strip():
        raise ValidationError("Enter something to search for")
    results = web.search(query, limit=max(1, min(int(top_k), web.MAX_RESULTS)))
    return {
        "query": query.strip(),
        "count": len(results),
        "results": [result.to_dict() for result in results],
    }


def read_web_page(url: str, max_chars: int = DEFAULT_PAGE_CHARACTERS) -> dict[str, Any]:
    """Fetch one web page and return its readable text."""
    if not url or not url.strip():
        raise ValidationError("Enter an address to read")
    limit = max(200, min(int(max_chars), MAX_PAGE_CHARACTERS))

    page = web.fetch_page(url.strip())
    text = page.text[:limit]
    return {
        "url": page.url,
        "title": page.title,
        "characters": page.characters,
        "truncated": page.characters > limit,
        "text": text,
    }
