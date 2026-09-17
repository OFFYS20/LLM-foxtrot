"""Readers for every supported dataset container format."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.errors import UnsupportedError, ValidationError
from app.db.models.enums import DatasetFormat

SUFFIX_TO_FORMAT = {
    ".json": DatasetFormat.JSON,
    ".jsonl": DatasetFormat.JSONL,
    ".ndjson": DatasetFormat.JSONL,
    ".csv": DatasetFormat.CSV,
    ".tsv": DatasetFormat.CSV,
    ".txt": DatasetFormat.TXT,
    ".parquet": DatasetFormat.PARQUET,
}

MAX_TEXT_ROW_CHARS = 200_000


def detect_format(path: Path) -> DatasetFormat:
    fmt = SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise ValidationError(
            f"Cannot infer dataset format from {path.suffix!r}",
            details={"supported": sorted(SUFFIX_TO_FORMAT)},
        )
    return fmt


def iter_rows(path: Path, fmt: DatasetFormat, *, limit: int | None = None) -> Iterator[Any]:
    """Stream rows without loading the whole file into memory where possible."""
    if fmt == DatasetFormat.JSONL:
        yield from _iter_jsonl(path, limit)
    elif fmt == DatasetFormat.JSON:
        yield from _iter_json(path, limit)
    elif fmt == DatasetFormat.CSV:
        yield from _iter_csv(path, limit)
    elif fmt == DatasetFormat.TXT:
        yield from _iter_txt(path, limit)
    elif fmt == DatasetFormat.PARQUET:
        yield from _iter_parquet(path, limit)
    else:
        raise UnsupportedError(f"No reader for format {fmt}")


def _iter_jsonl(path: Path, limit: int | None) -> Iterator[Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"__parse_error__": f"line {count + 1}: {exc.msg}"}


def _iter_json(path: Path, limit: int | None) -> Iterator[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc.msg} (line {exc.lineno})") from exc

    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "train"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValidationError("JSON dataset must be an array or an object containing one")
    yield from (data[:limit] if limit is not None else data)


def _iter_csv(path: Path, limit: int | None) -> Iterator[Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for count, row in enumerate(reader):
            if limit is not None and count >= limit:
                return
            yield {k: v for k, v in row.items() if k is not None}


def _iter_txt(path: Path, limit: int | None) -> Iterator[Any]:
    """Blank-line separated documents; falls back to one row per line."""
    buffer: list[str] = []
    emitted = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                buffer.append(line.rstrip("\n"))
                continue
            if buffer:
                yield {"text": "\n".join(buffer)[:MAX_TEXT_ROW_CHARS]}
                emitted += 1
                buffer.clear()
                if limit is not None and emitted >= limit:
                    return
    if buffer and (limit is None or emitted < limit):
        yield {"text": "\n".join(buffer)[:MAX_TEXT_ROW_CHARS]}


def _iter_parquet(path: Path, limit: int | None) -> Iterator[Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise UnsupportedError("Parquet support needs pyarrow: pip install pyarrow") from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    yield from (rows[:limit] if limit is not None else rows)


def load_hf_dataset(
    repo_id: str, subset: str | None, split: str | None, limit: int | None = None
) -> list[Any]:
    """Load a Hugging Face dataset. Requires the optional ``datasets`` package."""
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise UnsupportedError(
            "Hugging Face dataset import needs the `datasets` package: "
            "pip install -r requirements-optional.txt"
        ) from exc

    dataset = load_dataset(repo_id, subset, split=split or "train")
    rows = list(dataset.select(range(min(limit, len(dataset))))) if limit else list(dataset)
    return rows
