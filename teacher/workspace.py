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

    def note(self, **fields) -> None:
        """Store facts about the model itself, beside its lesson history."""
        history = self.history()
        history.setdefault("name", self.name)
        history.setdefault("lessons", [])
        history.update(fields)
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

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

    def kind(self) -> str:
        """"studio" for one built here, "pretrained" for one adopted from the Hub."""
        model_type = self.architecture().get("model_type", "")
        return "studio" if model_type == "ai_studio_transformer" else "pretrained"

    def base_repo(self) -> str | None:
        return self.history().get("base_repo")

    def architecture(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------- saved states
    def earlier_states(self) -> list[Path]:
        """Every saved state from before a lesson, oldest first."""
        if not self.checkpoints.exists():
            return []
        return sorted(p for p in self.checkpoints.glob("before-*") if p.is_dir())

    def saved_states(self) -> list[dict]:
        """The saved states, described: when each was taken and how big it is."""
        described = []
        for path in self.earlier_states():
            stamp = path.name.removeprefix("before-")
            try:
                when = time.mktime(time.strptime(stamp, "%Y%m%d-%H%M%S"))
            except ValueError:
                when = path.stat().st_mtime
            described.append({
                "stamp": stamp,
                "at": when,
                "bytes": sum(f.stat().st_size for f in path.iterdir() if f.is_file()),
                "path": str(path),
            })
        return described

    def saved_state(self, stamp: str) -> Path:
        """One saved state by its stamp, with or without the "before-" prefix."""
        wanted = stamp.strip().removeprefix("before-")
        for path in self.earlier_states():
            if path.name.removeprefix("before-") == wanted:
                return path
        available = [state["stamp"] for state in self.saved_states()]
        raise TeacherError(
            f"{self.name} has no saved state '{stamp}'."
            + (f" It has: {', '.join(available)}." if available else
               " It has none — it has not been taught yet.")
        )

    def rollback(self, to: str | None = None) -> str:
        """Restore the weights from a saved state — by default the newest.

        The lesson history is left alone: it is a record of what happened, and
        the rollback happened too.
        """
        if to:
            chosen = self.saved_state(to)
        else:
            saved = self.earlier_states()
            if not saved:
                raise TeacherError(
                    f"{self.name} has no earlier state saved — it has not been taught yet, "
                    f"or the saved states were removed."
                )
            chosen = saved[-1]

        restored = [item for item in chosen.iterdir() if item.is_file()]
        if not restored:
            raise TeacherError(f"The saved state {chosen.name} is empty.")
        for item in restored:
            shutil.copy2(item, self.path / item.name)
        return chosen.name

    # -------------------------------------------------------------- copy
    def pack(self, destination: Path) -> list[str]:
        """Copy just the three files Bench needs into ``destination``."""
        if self.kind() != "studio":
            raise TeacherError(
                f"{self.name} is built on {self.base_repo() or 'a pretrained model'}, and the "
                f"Bench web page only runs models built here. Chat with it in the Teacher UI "
                f"or with: teacher ask {self.name}"
            )
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


def branch(source: Model, new_name: str, *, at: str | None = None) -> Model:
    """Copy a model — or one of its saved states — into a new model of its own.

    This is how you carry on from a checkpoint without risking what you have:
    the source is not touched at all, so you can train the copy hard, hate the
    result, and still have the original exactly as it was.

    The copy starts with an empty lesson list, because its weights have not had
    those lessons in the form the copy holds them. The source's record is kept
    under ``branched_from_history`` rather than thrown away.
    """
    target = get(new_name, must_exist=False)
    if target.path.exists() and any(target.path.iterdir()):
        raise TeacherError(
            f"'{target.name}' already exists. Choose another name, or delete it with: "
            f"teacher forget {target.name} --yes"
        )
    state = source.saved_state(at) if at else None
    if not source.exists():
        raise TeacherError(f"{source.name} is missing {', '.join(source.missing())}.")

    target.path.mkdir(parents=True, exist_ok=True)
    # Everything the model needs to run, including the extra files a pretrained
    # one carries. The checkpoints and the history belong to the source.
    for item in source.path.iterdir():
        if item.is_file() and item.name != HISTORY:
            shutil.copy2(item, target.path / item.name)
    # Then the chosen state's weights on top, if one was named.
    if state:
        for item in state.iterdir():
            if item.is_file():
                shutil.copy2(item, target.path / item.name)

    was = source.history()
    target.note(
        name=target.name,
        created_at=time.time(),
        lessons=[],
        branched_from=source.name,
        branched_at=state.name if state else "its current weights",
        branched_on=time.time(),
        base_repo=was.get("base_repo"),
        base_loss=was.get("base_loss"),
        branched_from_history=was.get("lessons", []),
    )
    return target


def every() -> list[Model]:
    return [
        Model(name=child.name, path=child)
        for child in sorted(root().iterdir())
        if child.is_dir() and (child / "config.json").exists()
    ]
