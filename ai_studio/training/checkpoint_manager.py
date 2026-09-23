"""Checkpoints: save and restore everything needed to resume exactly.

A checkpoint directory holds the model weights, the optimizer and scheduler
state, the tokenizer, the training configuration, the step/epoch counters, RNG
state and dataset metadata. Saving is atomic — a new checkpoint is written to a
temporary directory and moved into place, so interrupting a save never corrupts
an existing checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import CheckpointError
from ai_studio.core.paths import dir_size, remove_path, slugify

STATE_FILENAME = "training_state.pt"
META_FILENAME = "checkpoint.json"


@dataclass
class CheckpointRecord:
    id: str
    path: Path
    step: int
    epoch: float
    train_loss: float | None
    val_loss: float | None
    size_bytes: int


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "numpy" in state:
            import numpy as np

            np.random.set_state(state["numpy"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 - RNG restore is best-effort
        log.warning(f"Could not fully restore RNG state: {exc}", source="checkpoint")


def save_checkpoint(
    *,
    experiment_id: str,
    model: Any,
    tokenizer: Any | None,
    optimizer: Any | None,
    scheduler: Any | None,
    step: int,
    epoch: float,
    train_loss: float | None,
    val_loss: float | None,
    training_config: dict[str, Any],
    dataset_meta: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    is_best: bool = False,
    is_peft: bool = False,
    name: str | None = None,
) -> dict[str, Any]:
    """Write a complete, resumable checkpoint. Atomic: temp dir then rename."""
    config = get_config()
    label = name or f"step-{step}"
    base = config.checkpoints_dir / slugify(experiment_id)
    base.mkdir(parents=True, exist_ok=True)
    final = base / slugify(label)
    staging = base / f".{slugify(label)}.tmp-{os.getpid()}"

    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        # 1. weights
        if is_peft and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(staging))            # adapter only
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(str(staging))
        else:
            torch.save(model.state_dict(), staging / "pytorch_model.bin")

        # 2. tokenizer travels with the checkpoint so it is always loadable
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(str(staging))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Tokenizer not saved with checkpoint: {exc}", source="checkpoint")

        # 3. resumable training state
        state: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_config": training_config,
            "dataset_meta": dataset_meta or {},
            "metrics": metrics or {},
            "rng": _rng_state(),
            "saved_at": time.time(),
            "torch_version": torch.__version__,
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, staging / STATE_FILENAME)

        # 4. human-readable manifest
        (staging / META_FILENAME).write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "step": step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "is_best": is_best,
                    "is_peft": is_peft,
                    "training_config": training_config,
                    "dataset_meta": dataset_meta or {},
                    "saved_at": time.time(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        raise CheckpointError(f"Could not save checkpoint at step {step}: {exc}") from exc

    size = dir_size(final)
    record = {
        "id": new_id("ckpt"),
        "experiment_id": experiment_id,
        "model_id": None,
        "name": label,
        "path": str(final),
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "benchmark_score": None,
        "size_bytes": size,
        "is_best": int(is_best),
        "has_optimizer_state": int(optimizer is not None),
        "metrics": metrics or {},
        "meta": {"is_peft": is_peft, "dataset": dataset_meta or {}},
        "created_at": time.time(),
    }
    db = get_db()
    db.insert("checkpoints", record)
    if is_best:
        db.query(
            "UPDATE checkpoints SET is_best = 0 WHERE experiment_id = ? AND id != ?",
            (experiment_id, record["id"]),
        )
        db.update("checkpoints", record["id"], {"is_best": 1})

    log.info(
        f"Saved checkpoint {label} (step {step}, {size / 1024**2:.1f} MB)"
        + ("  [best]" if is_best else ""),
        source="checkpoint",
        context={"experiment_id": experiment_id, "checkpoint_id": record["id"]},
    )
    return db.require("checkpoints", record["id"])


def prune_checkpoints(experiment_id: str, keep_last: int, *, keep_best: bool = True) -> int:
    """Delete old checkpoints, always keeping the best one when asked."""
    db = get_db()
    rows = db.list(
        "checkpoints",
        where="experiment_id = ?",
        params=(experiment_id,),
        order_by="step DESC",
    )
    removed = 0
    for index, row in enumerate(rows):
        if index < keep_last:
            continue
        if keep_best and row.get("is_best"):
            continue
        remove_path(Path(row["path"]))
        db.delete("checkpoints", row["id"])
        removed += 1
    if removed:
        log.debug(f"Pruned {removed} old checkpoint(s)", source="checkpoint")
    return removed


def load_training_state(checkpoint_path: Path | str) -> dict[str, Any]:
    """Read the resumable state written alongside the weights."""
    path = Path(checkpoint_path) / STATE_FILENAME
    if not path.exists():
        raise CheckpointError(
            f"No training state in {checkpoint_path} — this checkpoint cannot be resumed.",
            hint="It can still be loaded for inference or evaluation.",
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(f"Checkpoint state is unreadable: {exc}") from exc


def load_checkpoint_weights(checkpoint_path: Path | str, model: Any) -> str:
    """Put a checkpoint's weights into a model that has already been built.

    Resuming is the checkpoint's weights *and* its training state. A run that
    restored only the state would carry on from step N with whatever weights
    it was built with — the untrained ones — which is a shorter new run, not a
    resumed one. Returns "adapter" or "full", for what was loaded.
    """
    from safetensors.torch import load_file

    path = Path(checkpoint_path)
    adapter = path / "adapter_model.safetensors"
    if adapter.exists():
        if not hasattr(model, "peft_config"):
            raise CheckpointError(
                "This checkpoint holds a LoRA adapter, and the run being resumed has "
                "no adapter to load it into.",
                hint="Resume with the method the checkpoint was trained with (LoRA).",
            )
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, load_file(str(adapter)))
        return "adapter"

    full, pickled = path / "model.safetensors", path / "pytorch_model.bin"
    if full.exists():
        weights = load_file(str(full))
    elif pickled.exists():
        weights = torch.load(pickled, map_location="cpu", weights_only=True)
    else:
        raise CheckpointError(f"The checkpoint {path.name} has no weights to resume from.")

    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    missing, unexpected = target.load_state_dict(weights, strict=False)
    # A tied output layer is saved once, as the embeddings; that one may be absent.
    missing = [key for key in missing if not key.endswith("lm_head.weight")]
    if missing or unexpected:
        raise CheckpointError(
            f"The checkpoint's weights do not fit this model ({len(missing)} missing, "
            f"{len(unexpected)} unexpected).",
            hint="Resume with the model the checkpoint was trained from.",
        )
    return "full"


def restore_training_state(
    state: dict[str, Any],
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(f"Optimizer state does not match this model: {exc}") from exc
    if scheduler is not None and "scheduler" in state:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Scheduler state not restored: {exc}", source="checkpoint")
    if restore_rng:
        _restore_rng(state.get("rng"))
    return {
        "step": int(state.get("step", 0)),
        "epoch": float(state.get("epoch", 0.0)),
        "train_loss": state.get("train_loss"),
        "val_loss": state.get("val_loss"),
        "training_config": state.get("training_config", {}),
        "dataset_meta": state.get("dataset_meta", {}),
    }


def verify_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    """Check a checkpoint is complete before offering Load/Resume."""
    path = Path(checkpoint_path)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "has_weights": False,
        "has_state": (path / STATE_FILENAME).exists(),
        "has_tokenizer": any((path / n).exists() for n in ("tokenizer.json", "tokenizer_config.json")),
        "has_config": (path / "config.json").exists() or (path / "adapter_config.json").exists(),
        "resumable": False,
        "problems": [],
    }
    if not path.exists():
        result["problems"].append("Directory is missing.")
        return result

    weight_files = [
        "model.safetensors", "pytorch_model.bin",
        "adapter_model.safetensors", "adapter_model.bin",
    ]
    result["has_weights"] = any((path / name).exists() for name in weight_files) or bool(
        list(path.glob("*.safetensors"))
    )
    if not result["has_weights"]:
        result["problems"].append("No weight file found.")
    if not result["has_config"]:
        result["problems"].append("No config.json / adapter_config.json.")
    if not result["has_state"]:
        result["problems"].append("No training_state.pt — loadable, but not resumable.")

    result["resumable"] = result["has_weights"] and result["has_state"]
    return result


def list_checkpoints(experiment_id: str | None = None) -> list[dict[str, Any]]:
    db = get_db()
    if experiment_id:
        return db.list(
            "checkpoints", where="experiment_id = ?", params=(experiment_id,), order_by="step DESC"
        )
    return db.list("checkpoints", order_by="created_at DESC")


def delete_checkpoint(checkpoint_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("checkpoints", checkpoint_id)
    if remove_files:
        remove_path(Path(record["path"]))
    db.delete("checkpoints", checkpoint_id)
    log.warning(f"Deleted checkpoint {record['name']}", source="checkpoint")
