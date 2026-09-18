"""Where models live on disk.

One folder per model, holding exactly what it needs to be run — the weights,
the architecture and the tokenizer — plus a record of everything it has been
taught. That folder is what the Bench web page opens, with nothing to export.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("TEACHER_HOME", Path.home() / "teacher-models"))
HISTORY = "history.json"
REQUIRED = ("model.safetensors", "config.json", "tokenizer.json")


class TeacherError(Exception):
    """Something the user can fix, reported without a traceback."""


@dataclass
class Model:
    name: str
    path: Path

    # ------------------------------------------------------------- files
    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def tokenizer_path(self) -> Path:
        return self.path / "tokenizer.json"

    @property
    def history_path(self) -> Path:
        return self.path / HISTORY

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    def exists(self) -> bool:
        return all((self.path / name).exists() for name in REQUIRED)

    def missing(self) -> list[str]:
        return [name for name in REQUIRED if not (self.path / name).exists()]

    # ----------------------------------------------------------- history
    def history(self) -> dict:
        if not self.history_path.exists():
            return {"name": self.name, "created_at": None, "lessons": []}
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"name": self.name, "created_at": None, "lessons": []}

    def record(self, entry: dict) -> None:
        """Append one lesson to the model's record. Never rewrites earlier ones."""
        history = self.history()
        history.setdefault("name", self.name)
        history.setdefault("lessons", [])
        if not history.get("created_at"):
            history["created_at"] = time.time()
        history["lessons"].append(entry)
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def taught_characters(self) -> int:
        return sum(int(lesson.get("characters", 0)) for lesson in self.history().get("lessons", []))

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())

    def architecture(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # -------------------------------------------------------------- copy
    def pack(self, destination: Path) -> list[str]:
        """Copy just the three files Bench needs into ``destination``."""
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in REQUIRED:
            source = self.path / name
            if not source.exists():
                raise TeacherError(f"{self.name} has no {name} — teach it something first.")
            shutil.copy2(source, destination / name)
            copied.append(name)
        return copied


def root() -> Path:
    path = DEFAULT_ROOT.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in name.strip().lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned:
        raise TeacherError("That name has no usable characters in it.")
    return cleaned


def get(name: str, *, must_exist: bool = True) -> Model:
    model = Model(name=slug(name), path=root() / slug(name))
    if must_exist and not model.exists():
        if model.path.exists():
            raise TeacherError(
                f"'{model.name}' is missing {', '.join(model.missing())}. "
                f"It may not have finished its first lesson."
            )
        raise TeacherError(f"There is no model called '{model.name}'. Try: teacher list")
    return model


def every() -> list[Model]:
    return [
        Model(name=child.name, path=child)
        for child in sorted(root().iterdir())
        if child.is_dir() and (child / "config.json").exists()
    ]
