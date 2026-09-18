"""Background training jobs.

Training runs on a worker thread so the Gradio UI stays responsive. The manager
owns the experiment record, streams metrics into SQLite, writes checkpoints and
converts failures into stored errors rather than crashes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import (
    NotFoundError,
    OutOfMemoryError,
    StudioError,
    TrainingError,
    ValidationError,
)
from ai_studio.training.checkpoint_manager import (
    load_training_state,
    prune_checkpoints,
    restore_training_state,
    save_checkpoint,
)
from ai_studio.training.config import TrainingConfig, preflight
from ai_studio.training.trainer import StepMetrics, Trainer, TrainingControl

METRIC_FLUSH_EVERY = 5
CONSOLE_LINES = 500


@dataclass
class JobHandle:
    experiment_id: str
    thread: threading.Thread
    control: TrainingControl
    started_at: float = field(default_factory=time.time)
    console: deque[str] = field(default_factory=lambda: deque(maxlen=CONSOLE_LINES))
    latest: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()


class TrainingManager:
    """One manager for the whole process; at most one run at a time by default."""

    def __init__(self, *, allow_concurrent: bool = False) -> None:
        self._jobs: dict[str, JobHandle] = {}
        self._lock = threading.RLock()
        self.allow_concurrent = allow_concurrent

    # ------------------------------------------------------------ lifecycle
    def create_experiment(
        self,
        *,
        name: str,
        model_id: str,
        dataset_id: str,
        config: TrainingConfig,
        tokenizer_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        config.validate()
        db = get_db()
        model = db.require("models", model_id)
        dataset = db.require("datasets", dataset_id)

        record = {
            "id": new_id("exp"),
            "name": name or f"{model['name']} · {config.method} · {dataset['name']}",
            "method": config.method,
            "status": "queued",
            "model_id": model_id,
            "base_model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_hash": dataset.get("content_hash"),
            "tokenizer_id": tokenizer_id or model.get("tokenizer_id") or dataset.get("tokenizer_id"),
            "hyperparameters": config.to_dict(),
            "metrics": {},
            "benchmark_summary": {},
            "notes": notes,
            "created_at": time.time(),
        }
        db.insert("experiments", record)
        log.info(
            f"Created experiment '{record['name']}' ({config.method})",
            source="training",
            context={"experiment_id": record["id"]},
        )
        return db.require("experiments", record["id"])

    def start(self, experiment_id: str, *, resume_from: str | None = None) -> JobHandle:
        with self._lock:
            existing = self._jobs.get(experiment_id)
            if existing and existing.alive:
                raise ValidationError("This experiment is already running.")
            if not self.allow_concurrent:
                for handle in self._jobs.values():
                    if handle.alive:
                        raise ValidationError(
                            "Another training run is active.",
                            hint="Stop it first, or enable concurrent runs in Settings.",
                        )

            control = TrainingControl()
            handle = JobHandle(
                experiment_id=experiment_id,
                thread=threading.Thread(
                    target=self._run,
                    args=(experiment_id, control, resume_from),
                    name=f"train-{experiment_id}",
                    daemon=True,
                ),
                control=control,
            )
            self._jobs[experiment_id] = handle
            handle.thread.start()
            return handle

    def _run(self, experiment_id: str, control: TrainingControl, resume_from: str | None) -> None:
        db = get_db()
        handle = self._jobs[experiment_id]
        pending: list[dict[str, Any]] = []

        def console(message: str, level: str = "info") -> None:
            handle.console.append(f"{time.strftime('%H:%M:%S')} {message}")
            log.log(message, level=level, source="training", persist=level in {"warning", "error"})

        try:
            experiment = db.require("experiments", experiment_id)
            config = TrainingConfig.from_dict(experiment["hyperparameters"])
            db.update("experiments", experiment_id, {"status": "preparing", "started_at": time.time(), "error": None})

            model, tokenizer, train_ds, eval_ds, device, peft_stats = self._prepare(
                experiment, config, console
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            check = preflight(
                config, parameter_count=total_params, trainable_parameters=trainable
            )
            for warning in check.warnings:
                console(f"[WARN ] {warning}", "warning")
            if not check.ok:
                raise TrainingError(
                    "Refusing to start: " + " ".join(check.blockers),
                    hint="; ".join(check.suggestions),
                )

            start_step = 0
            if resume_from:
                start_step = self._resume(resume_from, model, console)

            db.update("experiments", experiment_id, {"status": "running"})

            def on_metrics(metrics: StepMetrics) -> None:
                handle.latest = {
                    "step": metrics.step,
                    "epoch": metrics.epoch,
                    "loss": metrics.loss,
                    "val_loss": metrics.val_loss,
                    "learning_rate": metrics.learning_rate,
                    "grad_norm": metrics.grad_norm,
                    "tokens_per_sec": metrics.tokens_per_sec,
                    "samples_per_sec": metrics.samples_per_sec,
                    "tokens_processed": metrics.tokens_processed,
                    "elapsed_seconds": metrics.elapsed_seconds,
                    "eta_seconds": metrics.eta_seconds,
                }
                pending.append(
                    {
                        "experiment_id": experiment_id,
                        "step": metrics.step,
                        "epoch": metrics.epoch,
                        "ts": time.time(),
                        "loss": metrics.loss,
                        "val_loss": metrics.val_loss,
                        "learning_rate": metrics.learning_rate,
                        "grad_norm": metrics.grad_norm,
                        "tokens_per_sec": metrics.tokens_per_sec,
                        "samples_per_sec": metrics.samples_per_sec,
                    }
                )
                if len(pending) >= METRIC_FLUSH_EVERY:
                    self._flush(pending)
                db.update(
                    "experiments",
                    experiment_id,
                    {
                        "current_step": metrics.step,
                        "current_epoch": metrics.epoch,
                        "tokens_processed": metrics.tokens_processed,
                    },
                )

            def on_checkpoint(step: int, loss: float, val_loss: float | None, is_best: bool) -> None:
                try:
                    save_checkpoint(
                        experiment_id=experiment_id,
                        model=trainer.model,
                        tokenizer=tokenizer,
                        optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler,
                        step=step,
                        epoch=trainer.history[-1]["epoch"] if trainer.history else 0.0,
                        train_loss=loss,
                        val_loss=val_loss,
                        training_config=config.to_dict(),
                        dataset_meta={
                            "dataset_id": experiment["dataset_id"],
                            "dataset_hash": experiment.get("dataset_hash"),
                        },
                        metrics=handle.latest,
                        is_best=is_best,
                        is_peft=bool(peft_stats),
                    )
                    prune_checkpoints(experiment_id, config.keep_last_checkpoints, keep_best=config.save_best)
                except StudioError as exc:
                    console(f"[CKPT ] failed: {exc.message}", "error")

            trainer = Trainer(
                model=model,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                config=config,
                device=device,
                control=control,
                on_metrics=on_metrics,
                on_log=console,
                on_checkpoint=on_checkpoint,
                start_step=start_step,
            )

            result = trainer.train()
            self._flush(pending)

            best = db.query(
                "SELECT id FROM checkpoints WHERE experiment_id = ? AND is_best = 1 LIMIT 1",
                (experiment_id,),
            )
            ended = time.time()
            db.update(
                "experiments",
                experiment_id,
                {
                    "status": result.status,
                    "ended_at": ended,
                    "duration_seconds": result.duration_seconds,
                    "final_train_loss": result.final_train_loss,
                    "best_val_loss": result.best_val_loss,
                    "best_checkpoint_id": best[0]["id"] if best else None,
                    "total_steps": result.steps,
                    "current_step": result.steps,
                    "tokens_processed": result.tokens_processed,
                    "error": result.error,
                    "metrics": {
                        **(handle.latest or {}),
                        "peft": peft_stats or {},
                        "device": str(device),
                    },
                },
            )
            console(f"[TRAIN] {result.status}", "info" if result.status != "failed" else "error")

        except OutOfMemoryError as exc:
            handle.error = exc.message
            console(f"[ERROR] {exc.message}", "error")
            for suggestion in exc.suggestions:
                console(f"        → {suggestion}", "warning")
            db.update("experiments", experiment_id, {"status": "failed", "error": exc.message, "ended_at": time.time()})
        except StudioError as exc:
            handle.error = exc.display()
            console(f"[ERROR] {exc.message}", "error")
            db.update("experiments", experiment_id, {"status": "failed", "error": exc.display(), "ended_at": time.time()})
        except Exception as exc:  # noqa: BLE001 - a crashed run must not kill the app
            handle.error = f"{type(exc).__name__}: {exc}"
            log.exception("Training worker crashed", exc, source="training")
            console(f"[ERROR] {handle.error}", "error")
            db.update("experiments", experiment_id, {"status": "failed", "error": handle.error, "ended_at": time.time()})
        finally:
            self._flush(pending)

    # ------------------------------------------------------------- helpers
    def _prepare(self, experiment: dict[str, Any], config: TrainingConfig, console: Any):
        import torch

        from ai_studio.models.model_manager import load_model_for_training
        from ai_studio.training.data import build_torch_dataset

        db = get_db()
        dataset = db.require("datasets", experiment["dataset_id"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console(f"[PREP ] device={device}")

        model, tokenizer = load_model_for_training(
            experiment["model_id"],
            quantization=config.quantization,
            device=device,
            dtype=config.precision,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        if experiment.get("tokenizer_id"):
            try:
                from ai_studio.models.tokenizer_manager import load_tokenizer

                tokenizer = load_tokenizer(experiment["tokenizer_id"])
            except StudioError as exc:
                console(f"[PREP ] using the model's own tokenizer ({exc.message})", "warning")

        peft_stats: dict[str, Any] = {}
        if config.uses_peft:
            from ai_studio.training.lora_trainer import apply_lora

            model, peft_stats = apply_lora(model, config)
            console(
                f"[PREP ] LoRA r={peft_stats['rank']} on {peft_stats['target_modules']} — "
                f"{peft_stats['trainable_parameters']:,} trainable "
                f"({peft_stats['trainable_percent']:.3f}%)"
            )

        mode = dataset["mode"]
        train_ds, stats = build_torch_dataset(
            experiment["dataset_id"], "train", tokenizer,
            max_length=config.max_sequence_length, mode=mode, packing=config.packing,
        )
        if train_ds is None:
            raise TrainingError("The training split is empty after tokenization.")
        eval_ds, _ = build_torch_dataset(
            experiment["dataset_id"], "validation", tokenizer,
            max_length=config.max_sequence_length, mode=mode, packing=config.packing,
        )
        console(
            f"[PREP ] {stats.sequences:,} training sequences, {stats.tokens:,} tokens "
            f"({'packed' if stats.packed else 'padded'})"
            + (f", {len(eval_ds):,} validation" if eval_ds else ", no validation split")
        )
        return model, tokenizer, train_ds, eval_ds, device, peft_stats

    def _resume(self, checkpoint_id_or_path: str, model: Any, console: Any) -> int:
        db = get_db()
        record = db.get("checkpoints", checkpoint_id_or_path)
        path = Path(record["path"]) if record else Path(checkpoint_id_or_path)
        if not path.exists():
            raise NotFoundError(f"Checkpoint {checkpoint_id_or_path} not found.")

        state = load_training_state(path)
        restored = restore_training_state(state)
        console(f"[RESUME] continuing from step {restored['step']} ({path.name})")
        return restored["step"]

    def _flush(self, pending: list[dict[str, Any]]) -> None:
        if not pending:
            return
        rows, pending[:] = list(pending), []
        db = get_db()
        try:
            with db.transaction() as conn:
                conn.executemany(
                    "INSERT INTO training_metrics (experiment_id, step, epoch, ts, loss, val_loss, "
                    "learning_rate, grad_norm, tokens_per_sec, samples_per_sec) "
                    "VALUES (:experiment_id, :step, :epoch, :ts, :loss, :val_loss, :learning_rate, "
                    ":grad_norm, :tokens_per_sec, :samples_per_sec)",
                    rows,
                )
        except Exception as exc:  # noqa: BLE001 - metrics must never break training
            log.warning(f"Could not persist metrics: {exc}", source="training")

    # -------------------------------------------------------------- control
    def _handle(self, experiment_id: str) -> JobHandle:
        handle = self._jobs.get(experiment_id)
        if handle is None:
            raise NotFoundError("No running job for this experiment.")
        return handle

    def pause(self, experiment_id: str) -> None:
        self._handle(experiment_id).control.pause()
        get_db().update("experiments", experiment_id, {"status": "paused"})

    def resume(self, experiment_id: str) -> None:
        self._handle(experiment_id).control.resume()
        get_db().update("experiments", experiment_id, {"status": "running"})

    def stop(self, experiment_id: str) -> None:
        self._handle(experiment_id).control.stop()
        get_db().update("experiments", experiment_id, {"status": "stopping"})

    def request_checkpoint(self, experiment_id: str) -> None:
        self._handle(experiment_id).control.request_checkpoint()

    def status(self, experiment_id: str) -> dict[str, Any]:
        handle = self._jobs.get(experiment_id)
        record = get_db().get("experiments", experiment_id) or {}
        return {
            "experiment": record,
            "running": bool(handle and handle.alive),
            "paused": bool(handle and handle.control.paused),
            "latest": handle.latest if handle else {},
            "console": list(handle.console) if handle else [],
            "error": handle.error if handle else record.get("error"),
        }

    def active_experiment_id(self) -> str | None:
        for experiment_id, handle in self._jobs.items():
            if handle.alive:
                return experiment_id
        return None

    def recover_orphans(self) -> int:
        """Runs interrupted by a restart are marked, never silently resurrected."""
        db = get_db()
        rows = db.list("experiments", where="status IN ('running', 'preparing', 'stopping', 'paused')")
        for row in rows:
            db.update(
                "experiments",
                row["id"],
                {"status": "interrupted", "error": "Interrupted by an application restart", "ended_at": time.time()},
            )
        if rows:
            log.warning(f"Marked {len(rows)} interrupted training run(s)", source="training")
        return len(rows)


manager = TrainingManager()
