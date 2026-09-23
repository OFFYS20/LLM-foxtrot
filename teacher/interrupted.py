"""Surviving a crash in the middle of a lesson.

Teacher saves the weights *between* lessons, which is enough when a lesson is
minutes. It is not enough when one is hours across several cards: a power cut
or an out-of-memory three hours into a round used to lose the whole round.

So a run in progress writes itself down periodically — the weights, the
optimizer's momentum, the schedule's position and which step it reached — and
``teacher resume`` picks it up from there.

Resuming puts all of it back: the weights, the optimizer's state, the place in
the learning-rate schedule, and the place in the data — the shuffle has its
own seeded generator, so the order of every epoch can be drawn again and the
batches already trained on passed over. For a model without dropout, a lesson
interrupted and resumed ends on the same weights as one that ran straight
through; a test holds it to that. What is not put back is the shared random
state that dropout draws on, so a model that uses dropout ends close to, not
exactly on, the same weights.

The record is deleted when a lesson finishes. A record that is there means a
lesson that is not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from teacher.workspace import Model, TeacherError

FOLDER = ".interrupted"
STATE = "training.pt"
PLAN = "run.json"


def place(model: Model) -> Path:
    return model.path / FOLDER


def fingerprint(text: str) -> str:
    """Enough of the material to tell whether it is still the same material."""
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def waiting(model: Model) -> dict | None:
    """The plan of an unfinished lesson, if there is one."""
    plan_path = place(model) / PLAN
    if not plan_path.exists():
        return None
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write(model: Model, network, optimizer, scheduler, *, step: int, plan: dict) -> None:
    """Put the run down on disk so a crash costs minutes, not hours."""
    import torch

    scratch = place(model).with_name(FOLDER + ".writing")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        saved = network.module if hasattr(network, "module") else network
        saved.save_pretrained(str(scratch))
        torch.save(
            {
                "step": int(step),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
            },
            scratch / STATE,
        )
        (scratch / PLAN).write_text(
            json.dumps({**plan, "step": int(step), "at": time.time()}, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - a failed save must not fail the lesson
        shutil.rmtree(scratch, ignore_errors=True)
        return

    # Swap it in whole, so a crash mid-write never leaves half a record.
    final = place(model)
    shutil.rmtree(final, ignore_errors=True)
    scratch.rename(final)


def clear(model: Model) -> None:
    """The lesson finished, so there is nothing to resume."""
    shutil.rmtree(place(model), ignore_errors=True)


def restore(model: Model, network, optimizer=None, scheduler=None) -> int:
    """Load an interrupted run back into a model. Returns the step it reached."""
    import torch

    state_path = place(model) / STATE
    if not state_path.exists():
        raise TeacherError(f"{model.name} has no interrupted lesson to resume.")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and state.get("optimizer"):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as exc:
            raise TeacherError(
                f"The saved optimizer does not fit this model ({exc}). "
                f"The settings must match the ones the run started with."
            ) from exc
    if scheduler is not None and state.get("scheduler"):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except (ValueError, KeyError):
            pass  # a schedule is recoverable from the step alone
    return int(state.get("step", 0))


def weights_path(model: Model) -> Path:
    return place(model) / "model.safetensors"


def describe(plan: dict) -> str:
    """One line about an unfinished lesson."""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(plan.get("at", 0)))
    done, total = plan.get("step", 0), plan.get("total_steps", 0)
    share = f"{done}/{total}" if total else str(done)
    return (f"interrupted at step {share} on {when}, "
            f"{plan.get('epochs', '?')} epoch(s) over {plan.get('characters', 0):,} characters")
