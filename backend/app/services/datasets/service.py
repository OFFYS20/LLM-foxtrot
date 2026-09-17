"""Dataset import, analysis and validation.

Files never leave the data root (see :mod:`app.core.paths`), unknown suffixes
are rejected, and malformed rows are reported per-row instead of aborting the
whole import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.errors import ValidationError
from app.core.paths import (
    ALLOWED_DATASET_SUFFIXES,
    dir_size_bytes,
    resolve_within,
    validate_hf_repo_id,
    validate_suffix,
)
from app.db.models.enums import DatasetFormat, DatasetTemplate
from app.schemas.datasets import DatasetIssue, DatasetValidationReport
from app.services.datasets.formats import detect_format, iter_rows, load_hf_dataset
from app.services.datasets.templates import detect_template, row_to_text, validate_row

PREVIEW_ROWS = 25
MAX_ISSUES = 200
CHARS_PER_TOKEN = 4.0


@dataclass(slots=True)
class DatasetAnalysis:
    rows: int = 0
    tokens: int = 0
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    columns: list[str] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)
    template: DatasetTemplate = DatasetTemplate.RAW
    report: DatasetValidationReport | None = None
    size_bytes: int = 0
    token_count_method: str = "estimated"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def resolve_dataset_path(candidate: str) -> Path:
    path = resolve_within(settings.data_dir, candidate)
    if not path.exists():
        raise ValidationError(f"Dataset file not found: {candidate}")
    if path.is_dir():
        raise ValidationError("Expected a file, got a directory")
    validate_suffix(path, ALLOWED_DATASET_SUFFIXES)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise ValidationError(
            f"File exceeds the {settings.max_upload_mb} MB limit",
            details={"size_bytes": path.stat().st_size},
        )
    return path


def analyze_rows(
    rows: list[Any],
    *,
    template: DatasetTemplate | None = None,
    max_validate: int = 5000,
) -> DatasetAnalysis:
    """Compute stats, preview and a validation report for in-memory rows."""
    analysis = DatasetAnalysis()
    if not rows:
        analysis.report = DatasetValidationReport(template=template or DatasetTemplate.RAW)
        return analysis

    resolved_template = template or detect_template([r for r in rows[:50] if isinstance(r, dict)])
    analysis.template = resolved_template

    issues: list[DatasetIssue] = []
    lengths: list[int] = []
    valid_rows = 0
    checked = min(len(rows), max_validate)

    for index, row in enumerate(rows):
        text = row_to_text(row, resolved_template)
        tokens = estimate_tokens(text)
        analysis.tokens += tokens
        lengths.append(tokens)

        if index < checked:
            if isinstance(row, dict) and "__parse_error__" in row:
                issues.append(DatasetIssue(row=index, message=str(row["__parse_error__"])))
            else:
                row_issues = [i for i in validate_row(row, resolved_template, index)]
                errors = [i for i in row_issues if i.severity == "error"]
                if not errors:
                    valid_rows += 1
                issues.extend(row_issues[: max(0, MAX_ISSUES - len(issues))])

    analysis.rows = len(rows)
    analysis.avg_sequence_length = round(sum(lengths) / len(lengths), 2) if lengths else 0.0
    analysis.max_sequence_length = max(lengths) if lengths else 0

    columns: list[str] = []
    for row in rows[:50]:
        if isinstance(row, dict):
            for key in row:
                if key not in columns and not key.startswith("__"):
                    columns.append(key)
    analysis.columns = columns

    analysis.preview = [
        _jsonable(row) if isinstance(row, dict) else {"value": _jsonable(row)}
        for row in rows[:PREVIEW_ROWS]
    ]

    invalid = checked - valid_rows
    analysis.report = DatasetValidationReport(
        template=resolved_template,
        checked_rows=checked,
        valid_rows=valid_rows,
        invalid_rows=max(0, invalid),
        issues=issues[:MAX_ISSUES],
        truncated=len(rows) > checked or len(issues) > MAX_ISSUES,
    )
    return analysis


def analyze_file(
    path: Path,
    *,
    fmt: DatasetFormat | None = None,
    template: DatasetTemplate | None = None,
    max_rows: int | None = None,
) -> tuple[DatasetAnalysis, DatasetFormat]:
    resolved_format = fmt or detect_format(path)
    rows = list(iter_rows(path, resolved_format, limit=max_rows))
    analysis = analyze_rows(rows, template=template)
    analysis.size_bytes = dir_size_bytes(path)
    return analysis, resolved_format


def analyze_hf_dataset(
    repo_id: str,
    subset: str | None,
    split: str | None,
    *,
    template: DatasetTemplate | None = None,
    max_rows: int | None = 20_000,
) -> DatasetAnalysis:
    repo_id = validate_hf_repo_id(repo_id)
    rows = load_hf_dataset(repo_id, subset, split, limit=max_rows)
    analysis = analyze_rows(rows, template=template)
    analysis.size_bytes = sum(len(json.dumps(_jsonable(r), default=str)) for r in rows[:1000])
    return analysis


def read_preview(
    path: Path, fmt: DatasetFormat, *, offset: int = 0, limit: int = 25
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    for index, row in enumerate(iter_rows(path, fmt, limit=offset + limit)):
        if index < offset:
            continue
        item = _jsonable(row) if isinstance(row, dict) else {"value": _jsonable(row)}
        rows.append(item)
        for key in item:
            if key not in columns:
                columns.append(key)
    return rows, columns


def _jsonable(value: Any) -> Any:
    """Make arbitrary row content JSON-serialisable and bounded in size."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in list(value)[:64]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 4000:
            return value[:4000] + "…"
        return value
    return str(value)[:4000]
