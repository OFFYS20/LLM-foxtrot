"""Filesystem helpers that keep every user-supplied path inside the data root.

The platform accepts paths from the API (model directories, dataset files,
export targets). Each one is resolved and then checked against an allow-listed
root, so ``../../etc/passwd`` and symlink escapes are rejected before any I/O.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from app.core.errors import PathTraversalError, ValidationError

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._@-]+$")
_SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

ALLOWED_DATASET_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".parquet", ".tsv"}
ALLOWED_MODEL_SUFFIXES = {".bin", ".safetensors", ".gguf", ".pt", ".pth", ".json", ".model"}


def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to a single safe path component."""
    candidate = Path(name).name.strip().replace(" ", "_")
    if not candidate or candidate in {".", ".."} or not _SAFE_NAME.match(candidate):
        raise ValidationError(f"Unsafe filename: {name!r}")
    return candidate


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and guarantee the result lives under ``root``."""
    root_resolved = Path(root).expanduser().resolve()
    root_resolved.mkdir(parents=True, exist_ok=True)

    raw = Path(candidate).expanduser()
    target = (root_resolved / raw).resolve() if not raw.is_absolute() else raw.resolve()

    if target != root_resolved and root_resolved not in target.parents:
        raise PathTraversalError(
            f"Path {candidate!r} escapes the allowed root {root_resolved}",
            details={"root": str(root_resolved)},
        )
    return target


def resolve_data_path(candidate: str | Path) -> Path:
    return resolve_within(settings.data_dir, candidate)


def validate_suffix(path: Path, allowed: set[str]) -> None:
    if path.suffix.lower() not in allowed:
        raise ValidationError(
            f"Unsupported file type {path.suffix!r}",
            details={"allowed": sorted(allowed)},
        )


def validate_hf_repo_id(repo_id: str) -> str:
    """Accept only ``owner/name`` style Hugging Face repository identifiers."""
    repo_id = repo_id.strip()
    if not _SAFE_REPO.match(repo_id):
        raise ValidationError(
            "Invalid Hugging Face repository id — expected 'owner/name'",
            details={"value": repo_id},
        )
    return repo_id


def human_bytes(num: float | int | None) -> str:
    if not num:
        return "0 B"
    step = 1024.0
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < step:
            return f"{value:3.1f} {unit}".strip()
        value /= step
    return f"{value:.1f} EB"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
