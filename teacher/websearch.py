"""Looking things up on the web, and keeping what comes back as material.

The fetching itself lives in ``ai_studio.data.web`` — the same code the chat
tool uses. This module is what Teacher adds on top: reading several pages in
one go, politely and one at a time, and writing each one to a text file that
says where it came from.

That last part is the point. A page you found is someone else's writing, under
someone else's terms. A corpus with no record of its sources cannot be checked
later, so every file saved here carries its address and the date it was read.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_studio.core.errors import StudioError
from ai_studio.data import web
from teacher.workspace import TeacherError

#: Seconds between fetches. One at a time, and not in a hurry.
PAUSE = 1.0


@dataclass
class Harvest:
    """What one search-and-read run brought back."""

    query: str
    results: list[web.Result] = field(default_factory=list)
    pages: list[web.Page] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def characters(self) -> int:
        return sum(page.characters for page in self.pages)

    def summary(self) -> str:
        parts = [f"{len(self.pages)} page(s)", f"{self.characters:,} characters"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


def search(query: str, *, limit: int = 5) -> list[web.Result]:
    """Titles, addresses and snippets for a query — nothing is fetched yet."""
    try:
        return web.search(query, limit=limit)
    except StudioError as exc:
        raise TeacherError(exc.display()) from exc


def read_page(url: str) -> web.Page:
    """One page, as text. Raises TeacherError with a reason you can act on."""
    try:
        return web.fetch_page(url)
    except StudioError as exc:
        raise TeacherError(exc.display()) from exc


def harvest(
    query: str = "",
    *,
    urls: list[str] | None = None,
    limit: int = 5,
    on_step=None,
) -> Harvest:
    """Search, then read what it found. Addresses given directly are read too.

    A page that cannot be read is recorded in ``skipped`` rather than stopping
    the run — one dead link should not cost you the other nine pages.
    """
    collected = Harvest(query=query)
    targets: list[web.Result] = []

    if query and query.strip():
        collected.results = search(query, limit=limit)
        targets.extend(collected.results)
    for address in urls or []:
        if address and address.strip():
            targets.append(web.Result(title="", url=address.strip()))

    if not targets:
        raise TeacherError("Give something to search for, or at least one address.")

    for index, target in enumerate(targets, start=1):
        if on_step:
            on_step(index, len(targets), target.url)
        try:
            page = read_page(target.url)
        except TeacherError as exc:
            collected.skipped.append((target.url, str(exc)))
        else:
            page.title = page.title or target.title or target.url
            collected.pages.append(page)
        if index < len(targets):
            time.sleep(PAUSE)

    if not collected.pages:
        raise TeacherError(
            "Nothing readable came back.\n  "
            + "\n  ".join(f"{url} — {why}" for url, why in collected.skipped[:5])
        )
    return collected


def save(collected: Harvest, destination: Path) -> list[Path]:
    """Write each page as its own text file, with the address it came from."""
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, page in enumerate(collected.pages, start=1):
        stem = re.sub(r"[^\w-]+", "-", (page.title or "page").lower()).strip("-")[:60]
        path = destination / f"{index:02d}-{stem or 'page'}.txt"
        path.write_text(
            f"{page.title}\n"
            f"Source: {page.url}\n"
            f"Retrieved: {time.strftime('%Y-%m-%d')}\n\n"
            f"{page.text}\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def folder_for(query: str, root: Path) -> Path:
    """Where a harvest for this query is kept, under ``root``."""
    label = re.sub(r"^https?://(www\.)?", "", (query or "pages").strip())
    stem = re.sub(r"[^\w-]+", "-", label.lower()).strip("-")[:40] or "pages"
    return Path(root) / "web-material" / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}"
