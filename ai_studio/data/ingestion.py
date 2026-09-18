"""Data Library: import documents, keep originals, store cleaned copies.

Import is always two artefacts: the original file, copied verbatim into
``storage/documents/original``, and a cleaned text rendering in
``storage/documents/cleaned``. The original is never modified or deleted by a
cleaning run, so re-cleaning with different options is always possible.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import IngestionError, NotFoundError, StudioError, ValidationError
from ai_studio.core.paths import (
    dir_size,
    file_sha256,
    remove_path,
    safe_filename,
    text_sha256,
    unique_path,
)
from ai_studio.data.loaders import SUPPORTED_EXTENSIONS, load_document, text_from_string
from ai_studio.data.preprocessing import (
    CleaningOptions,
    clean_text,
    estimate_tokens,
)

SKIP_DIRECTORIES = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".DS_Store"}


@dataclass
class ImportOutcome:
    imported: list[dict[str, Any]]
    failed: list[tuple[str, str]]

    @property
    def summary(self) -> str:
        parts = [f"{len(self.imported)} imported"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def _config():
    return get_config()


def import_file(
    path: str | Path,
    *,
    clean: bool = True,
    options: CleaningOptions | None = None,
    title: str | None = None,
    copy_original: bool = True,
) -> dict[str, Any]:
    """Import a single file into the Data Library."""
    source = Path(path).expanduser()
    if not source.exists():
        raise IngestionError(f"File not found: {source}")
    if not source.is_file():
        raise IngestionError(f"Not a file: {source}")

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type {suffix or '(none)'}",
            hint=f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    config = _config()
    max_bytes = config.data.max_upload_mb * 1024 * 1024
    size = source.stat().st_size
    if size > max_bytes:
        raise IngestionError(
            f"{source.name} is {size / 1024**2:.0f} MB, above the "
            f"{config.data.max_upload_mb} MB limit.",
            hint="Raise data.max_upload_mb in config.yaml or split the file.",
        )

    loaded = load_document(source)  # may raise IngestionError/DependencyMissingError

    stored_path = source
    if copy_original:
        stored_path = unique_path(config.originals_dir, source.name)
        shutil.copy2(source, stored_path)

    document_id = new_id("doc")
    record: dict[str, Any] = {
        "id": document_id,
        "filename": source.name,
        "title": (title or loaded.title or source.stem)[:300],
        "author": loaded.author,
        "doc_type": loaded.doc_type,
        "source": "file",
        "original_path": str(stored_path),
        "cleaned_path": None,
        "size_bytes": size,
        "char_count": loaded.char_count,
        "word_count": loaded.word_count,
        "token_estimate": estimate_tokens(loaded.text),
        "sha256": file_sha256(stored_path),
        "status": "imported",
        "meta": {**loaded.meta, "extension": suffix, "sections": len(loaded.sections)},
        "imported_at": time.time(),
    }
    get_db().insert("documents", record)
    log.info(
        f"Imported {source.name} ({loaded.char_count:,} chars)",
        source="ingestion",
        context={"document_id": document_id, "type": loaded.doc_type},
    )

    if clean:
        try:
            record = clean_document(
                document_id,
                options or CleaningOptions.for_type(loaded.doc_type),
                _loaded_text=loaded.text,
                _sections=loaded.sections,
            )
        except StudioError as exc:
            log.warning(f"Cleaning failed for {source.name}: {exc.message}", source="ingestion")
    return record


def import_upload(temp_path: str | Path, original_name: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Import a file handed over by the UI upload control."""
    temp = Path(temp_path)
    name = safe_filename(original_name or temp.name)
    if temp.suffix.lower() not in SUPPORTED_EXTENSIONS and Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
        staged = temp.parent / name
        shutil.copy2(temp, staged)
        temp = staged
    return import_file(temp, title=kwargs.pop("title", None), **kwargs)


def import_text(
    text: str,
    *,
    title: str | None = None,
    clean: bool = True,
    options: CleaningOptions | None = None,
    doc_type: str = "paste",
) -> dict[str, Any]:
    """Import pasted text (article, notes, a chapter, a conversation)."""
    if not text or not text.strip():
        raise ValidationError("Nothing to import — the text is empty.")

    config = _config()
    loaded = text_from_string(text, title=title)
    document_id = new_id("doc")
    stored = unique_path(config.originals_dir, f"{(title or 'pasted').strip()[:60] or 'pasted'}.txt")
    stored.write_text(text, encoding="utf-8")

    record: dict[str, Any] = {
        "id": document_id,
        "filename": stored.name,
        "title": (title or loaded.title or "Pasted text")[:300],
        "author": None,
        "doc_type": doc_type,
        "source": "paste",
        "original_path": str(stored),
        "cleaned_path": None,
        "size_bytes": len(text.encode("utf-8")),
        "char_count": len(text),
        "word_count": len(text.split()),
        "token_estimate": estimate_tokens(text),
        "sha256": text_sha256(text),
        "status": "imported",
        "meta": {"extension": ".txt", "pasted": True},
        "imported_at": time.time(),
    }
    get_db().insert("documents", record)
    log.info(f"Imported pasted text ({len(text):,} chars)", source="ingestion",
             context={"document_id": document_id})

    if clean:
        record = clean_document(
            document_id,
            options or CleaningOptions.for_type("txt"),
            _loaded_text=text,
            _sections=[],
        )
    return record


def import_folder(
    folder: str | Path,
    *,
    recursive: bool = True,
    clean: bool = True,
    options: CleaningOptions | None = None,
    extensions: Iterable[str] | None = None,
    limit: int | None = None,
) -> ImportOutcome:
    """Import every supported document in a directory.

    One unreadable file never aborts the batch — failures are collected and
    reported alongside the successes.
    """
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise IngestionError(f"Not a directory: {root}")

    wanted = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in wanted
        and not any(part in SKIP_DIRECTORIES or part.startswith(".") for part in path.parts)
    )
    if limit:
        candidates = candidates[:limit]

    imported: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    for path in candidates:
        try:
            imported.append(import_file(path, clean=clean, options=options))
        except StudioError as exc:
            failed.append((path.name, exc.message))
            log.warning(f"Skipped {path.name}: {exc.message}", source="ingestion")
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
            log.exception(f"Unexpected failure importing {path.name}", exc, source="ingestion")

    log.info(
        f"Folder import from {root}: {len(imported)} imported, {len(failed)} failed",
        source="ingestion",
    )
    return ImportOutcome(imported=imported, failed=failed)


def clean_document(
    document_id: str,
    options: CleaningOptions | None = None,
    *,
    _loaded_text: str | None = None,
    _sections: list[str] | None = None,
) -> dict[str, Any]:
    """(Re-)clean a document. The original file is never touched."""
    db = get_db()
    record = db.require("documents", document_id)

    text = _loaded_text
    sections = _sections
    if text is None:
        original = Path(record["original_path"] or "")
        if not original.exists():
            raise IngestionError(
                f"Original file for {record['filename']} is missing.",
                hint="Re-import the document.",
            )
        loaded = load_document(original)
        text, sections = loaded.text, loaded.sections

    options = options or CleaningOptions.for_type(record["doc_type"])
    result = clean_text(text, options, sections=sections or [])

    cleaned_path = _config().cleaned_dir / f"{document_id}.txt"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text(result.text, encoding="utf-8")

    meta = dict(record.get("meta") or {})
    meta["cleaning"] = {"options": options.to_dict(), "report": result.report.to_dict()}

    updates = {
        "cleaned_path": str(cleaned_path),
        "char_count": len(result.text),
        "word_count": len(result.text.split()),
        "token_estimate": estimate_tokens(result.text),
        "status": "cleaned",
        "meta": meta,
        "cleaned_at": time.time(),
        "error": None,
    }
    db.update("documents", document_id, updates)
    log.info(
        f"Cleaned {record['filename']}: {result.report.summary()}",
        source="ingestion",
        context={"document_id": document_id},
    )
    return db.require("documents", document_id)


def preview_cleaning(
    document_id: str, options: CleaningOptions | None = None, *, limit: int = 2400
) -> dict[str, Any]:
    """Before/after preview without writing anything."""
    record = get_db().require("documents", document_id)
    original = Path(record["original_path"] or "")
    if not original.exists():
        raise IngestionError(f"Original file for {record['filename']} is missing.")

    loaded = load_document(original)
    options = options or CleaningOptions.for_type(record["doc_type"])
    result = clean_text(loaded.text, options, sections=loaded.sections)
    return {
        "before": loaded.text[:limit],
        "after": result.text[:limit],
        "report": result.report,
        "options": options,
    }


def document_text(document_id: str, *, prefer_cleaned: bool = True) -> str:
    """Text for downstream use (dataset building, RAG)."""
    record = get_db().require("documents", document_id)
    cleaned = record.get("cleaned_path")
    if prefer_cleaned and cleaned and Path(cleaned).exists():
        return Path(cleaned).read_text(encoding="utf-8", errors="replace")
    original = record.get("original_path")
    if original and Path(original).exists():
        return load_document(Path(original)).text
    raise NotFoundError(f"No readable text for document {document_id}")


def list_documents(
    *, search: str | None = None, doc_type: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if search:
        clauses.append("(title LIKE ? OR filename LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])
    if doc_type and doc_type != "all":
        clauses.append("doc_type = ?")
        params.append(doc_type)
    return get_db().list(
        "documents",
        where=" AND ".join(clauses) or None,
        params=params,
        order_by="imported_at DESC",
        limit=limit,
    )


def delete_document(document_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("documents", document_id)
    if remove_files:
        for key in ("original_path", "cleaned_path"):
            path = record.get(key)
            if path:
                remove_path(Path(path))
    db.delete("documents", document_id)
    log.warning(f"Deleted document {record['filename']}", source="ingestion")


def library_stats() -> dict[str, Any]:
    db = get_db()
    rows = db.query(
        "SELECT doc_type, COUNT(*) AS count, SUM(char_count) AS chars, "
        "SUM(token_estimate) AS tokens FROM documents GROUP BY doc_type"
    )
    return {
        "documents": db.count("documents"),
        "cleaned": db.count("documents", "cleaned_path IS NOT NULL"),
        "characters": int(db.scalar("SELECT COALESCE(SUM(char_count), 0) FROM documents") or 0),
        "tokens_estimated": int(db.scalar("SELECT COALESCE(SUM(token_estimate), 0) FROM documents") or 0),
        "by_type": {row["doc_type"]: row["count"] for row in rows},
        "disk_bytes": dir_size(_config().documents_dir),
    }
