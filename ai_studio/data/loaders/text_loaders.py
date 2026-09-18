"""Plain text, Markdown, HTML and structured-data loaders."""

from __future__ import annotations

import csv as csv_module
import io
import json
import re
from pathlib import Path
from typing import Any

from ai_studio.core.errors import DependencyMissingError, IngestionError
from ai_studio.data.loaders.base import DocumentLoader, LoadedDocument

MAX_PREVIEW_ROWS = 2000


def read_text_file(path: Path) -> str:
    """Read a text file, tolerating unknown encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise IngestionError(f"Could not read {path.name}: {exc}") from exc
    raise IngestionError(f"Could not decode {path.name} as text")


class TextLoader(DocumentLoader):
    extensions = {".txt", ".text", ".log", ".rst"}
    doc_type = "txt"

    def load(self, path: Path) -> LoadedDocument:
        text = read_text_file(path)
        return LoadedDocument(text=text, title=path.stem, doc_type=self.doc_type)


class MarkdownLoader(DocumentLoader):
    extensions = {".md", ".markdown", ".mdx"}
    doc_type = "markdown"

    def load(self, path: Path) -> LoadedDocument:
        text = read_text_file(path)
        title = path.stem
        # A leading "# Heading" is a better title than the filename.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
            if stripped:
                break
        headings = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
        return LoadedDocument(
            text=text,
            title=title,
            doc_type=self.doc_type,
            meta={"headings": headings[:100], "heading_count": len(headings)},
        )


class HTMLLoader(DocumentLoader):
    extensions = {".html", ".htm", ".xhtml"}
    doc_type = "html"

    def load(self, path: Path) -> LoadedDocument:
        raw = read_text_file(path)
        return html_to_document(raw, fallback_title=path.stem, doc_type=self.doc_type)


def html_to_document(raw: str, *, fallback_title: str = "", doc_type: str = "html") -> LoadedDocument:
    """Extract readable text from HTML, dropping script/style/nav chrome."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise DependencyMissingError("beautifulsoup4", "HTML import", install="beautifulsoup4 lxml") from exc

    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:  # noqa: BLE001 - lxml missing or malformed markup
        soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "iframe"]):
        tag.decompose()

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else fallback_title

    author = None
    meta_author = soup.find("meta", attrs={"name": re.compile("author", re.I)})
    if meta_author and meta_author.get("content"):
        author = str(meta_author["content"]).strip()

    # Keep block structure so paragraphs survive as paragraphs.
    for block in soup.find_all(["p", "div", "section", "article", "li", "br", "tr"]):
        block.append("\n")
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        heading.insert_before("\n\n")
        heading.append("\n")

    text = soup.get_text()
    return LoadedDocument(
        text=text,
        title=title or fallback_title,
        author=author,
        doc_type=doc_type,
        meta={"links": len(soup.find_all("a"))},
    )


class CSVLoader(DocumentLoader):
    extensions = {".csv", ".tsv"}
    doc_type = "csv"

    def load(self, path: Path) -> LoadedDocument:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv_module.DictReader(handle, delimiter=delimiter)
                columns = reader.fieldnames or []
                for index, row in enumerate(reader):
                    if index >= MAX_PREVIEW_ROWS:
                        break
                    rows.append({k: v for k, v in row.items() if k is not None})
        except OSError as exc:
            raise IngestionError(f"Could not read {path.name}: {exc}") from exc

        total_rows = _count_lines(path) - 1
        # Render as readable text so the corpus is usable for language modelling.
        lines = []
        for row in rows:
            parts = [f"{key}: {value}" for key, value in row.items() if str(value or "").strip()]
            if parts:
                lines.append("\n".join(parts))
        return LoadedDocument(
            text="\n\n".join(lines),
            title=path.stem,
            doc_type=self.doc_type,
            meta={"columns": columns, "rows": max(total_rows, len(rows)), "previewed_rows": len(rows)},
            sections=lines,
        )


class JSONLoader(DocumentLoader):
    extensions = {".json", ".jsonl", ".ndjson"}
    doc_type = "json"

    def load(self, path: Path) -> LoadedDocument:
        raw = read_text_file(path)
        records: list[Any] = []
        malformed = 0

        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise IngestionError(
                    f"{path.name} is not valid JSON: {exc.msg} (line {exc.lineno})"
                ) from exc
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                for key in ("data", "rows", "examples", "train", "items"):
                    if isinstance(data.get(key), list):
                        records = data[key]
                        break
                else:
                    records = [data]

        sections = [_record_to_text(record) for record in records[:MAX_PREVIEW_ROWS]]
        sections = [section for section in sections if section.strip()]
        return LoadedDocument(
            text="\n\n".join(sections),
            title=path.stem,
            doc_type="jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "json",
            meta={
                "records": len(records),
                "malformed_lines": malformed,
                "keys": sorted({k for r in records[:200] if isinstance(r, dict) for k in r})[:40],
            },
            sections=sections,
        )


def _record_to_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        if "messages" in record and isinstance(record["messages"], list):
            return "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in record["messages"]
                if isinstance(m, dict)
            )
        if "text" in record:
            return str(record["text"])
        if "instruction" in record:
            parts = [str(record.get("instruction", ""))]
            if record.get("input"):
                parts.append(str(record["input"]))
            if record.get("output"):
                parts.append(str(record["output"]))
            return "\n".join(part for part in parts if part)
        return "\n".join(f"{key}: {value}" for key, value in record.items())
    return str(record)


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def text_from_string(raw: str, *, title: str | None = None) -> LoadedDocument:
    """Used by the 'paste text' path."""
    buffer = io.StringIO(raw)
    first_line = buffer.readline().strip()
    derived = title or (first_line[:80] if first_line else "Pasted text")
    return LoadedDocument(text=raw, title=derived, doc_type="paste")
