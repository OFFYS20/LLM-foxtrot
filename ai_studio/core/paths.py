"""Filesystem helpers.

Everything the application writes lives under the storage root. User-supplied
paths are resolved and checked against it so an import or export can never
escape into the wider filesystem.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from ai_studio.core.errors import ValidationError

_UNSAFE = re.compile(r"[^A-Za-z0-9._@ -]")


def slugify(name: str, *, max_length: int = 80) -> str:
    """Turn a display name into a safe single path component."""
    cleaned = _UNSAFE.sub("-", name.strip()).strip("-. ")
    cleaned = re.sub(r"-{2,}", "-", cleaned).replace(" ", "-")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_length]


def safe_filename(name: str) -> str:
    candidate = Path(str(name)).name.strip()
    if not candidate or candidate in {".", ".."}:
        raise ValidationError(f"Unsafe filename: {name!r}")
    return _UNSAFE.sub("_", candidate)


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and require the result to sit under ``root``."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = Path(candidate).expanduser()
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if target != root and root not in target.parents:
        raise ValidationError(
            f"Path {candidate!r} escapes the allowed directory {root}",
            hint="Import files from inside the storage root, or use the upload control.",
        )
    return target


def unique_path(directory: Path, filename: str) -> Path:
    """Return a non-colliding path inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / safe_filename(filename)
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(1, 10_000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValidationError(f"Could not find a free filename for {filename!r}")


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def dir_size(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_bytes(num: float | int | None) -> str:
    if not num:
        return "0 B"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.1f} EB"


def remove_path(path: Path) -> None:
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)
