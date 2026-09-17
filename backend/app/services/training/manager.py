"""Training job manager.

Owns the lifecycle of every job: validation, queueing, execution on a
background task (never inside an HTTP request), metric persistence, checkpoint
creation, experiment bookkeeping and clean failure handling.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.core.errors import ConflictError, NotFoundError, OutOfMemoryError, ValidationError
from app.core.events import Topics, bus
from app.core.logging import get_logger
from app.core.paths import resolve_within
from app.db.models.catalog import Dataset, Model
from app.db.models.enums import (
    JobStatus,
    ModelStatus,
    RunProvenance,
)
from app.db.models.training import Checkpoint, Experiment, TrainingJob, TrainingMetric
from app.db.session import session_scope
from app.schemas.training import TrainingConfig, TrainingJobCreate
from app.services.hardware.monitor import monitor
from app.services.logbook import logbook
from app.services.training.base import StepMetrics, TrainingBackend, TrainingContext
from app.services.training.schedules import estimate_total_steps
from app.services.training.simulated import SimulatedTrainingBackend
from app.services.training.torch_backend import TorchTrainingBackend

logger = get_logger("foxtrot.training.manager")

METRIC_FLUSH_EVERY = 5


@dataclass
class RunningJob:
    job_id: str
    ctx: TrainingContext
    task: asyncio.Task[Any]
    backend: str
    pending_metrics: list[dict[str, Any]] = field(default_factory=list)


class TrainingManager:
    def __init__(self) -> None:
        self._jobs: dict[str, RunningJob] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ creation
    def create_job(self, payload: TrainingJobCreate) -> TrainingJob:
        with session_scope() as session:
            model = session.get(Model, payload.model_id)
            if model is None:
                raise NotFoundError(f"Model {payload.model_id} not found")
            dataset = session.get(Dataset, payload.dataset_id)
            if dataset is None:
                raise NotFoundError(f"Dataset {payload.dataset_id} not found")

            config = payload.config
            snapshot = monitor.latest()
            vram = snapshot.gpus[0].memory_total_mb if snapshot.gpus else None
            warnings = config.validate_runtime(gpu_available=snapshot.gpu_available, vram_mb=vram)

            total_steps = estimate_total_steps(
                dataset_rows=max(1, dataset.rows),
                epochs=config.epochs,
                batch_size=config.batch_size,
                grad_accum=config.gradient_accumulation_steps,
            )

            backend_name, provenance = self._select_backend(model, dataset)
            name = payload.name or f"{model.name} · {config.method} · {dataset.name}"

            experiment = Experiment(
                name=name,
                model_id=model.id,
                dataset_id=dataset.id,
                project_id=payload.project_id,
                status=JobStatus.QUEUED,
                provenance=provenance,
                method=config.method,
                hyperparameters=config.model_dump(mode="json"),
                notes=payload.notes,
                is_demo=provenance == RunProvenance.SIMULATED,
            )
            session.add(experiment)
            session.flush()

            job = TrainingJob(
                name=name,
                model_id=model.id,
                dataset_id=dataset.id,
                project_id=payload.project_id,
                experiment_id=experiment.id,
                method=config.method,
                status=JobStatus.QUEUED,
                provenance=provenance,
                backend=backend_name,
                config=config.model_dump(mode="json"),
                total_steps=total_steps,
                total_epochs=config.epochs,
                learning_rate=config.learning_rate,
            )
            session.add(job)
            session.flush()

            experiment.job_id = job.id
            session.flush()

            logbook.info(
                f"Created training job {job.id} ({backend_name} backend, {total_steps} steps)",
                source="training",
                job_id=job.id,
                context={"warnings": warnings, "provenance": provenance.value},
            )
            for warning in warnings:
                logbook.warn(warning, source="training", job_id=job.id)

            session.expunge(job)
            return job

    def _select_backend(self, model: Model, dataset: Dataset) -> tuple[str, RunProvenance]:
        """Real training needs a real stack, real weights and a real dataset file."""
        if (
            TorchTrainingBackend.is_available()
            and model.local_path
            and dataset.local_path
            and not model.is_demo
            and not dataset.is_demo
        ):
            return "torch", RunProvenance.MEASURED
        return "simulated", RunProvenance.SIMULATED

    # ------------------------------------------------------------ execution
    async def start(self, job_id: str) -> TrainingJob:
        async with self._lock:
            if job_id in self._jobs and not self._jobs[job_id].task.done():
                raise ConflictError(f"Job {job_id} is already running")

            with session_scope() as session:
                job = session.get(TrainingJob, job_id)
                if job is None:
                    raise NotFoundError(f"Training job {job_id} not found")
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    raise ConflictError(f"Job {job_id} already finished ({job.status})")

                model = session.get(Model, job.model_id)
                dataset = session.get(Dataset, job.dataset_id)
                config = TrainingConfig.model_validate(job.config)

                job.status = JobStatus.RUNNING
                job.started_at = job.started_at or datetime.now(timezone.utc)
                job.error = None
                if model:
                    model.status = ModelStatus.TRAINING
                experiment = (
                    session.get(Experiment, job.experiment_id) if job.experiment_id else None
                )
                if experiment:
                    experiment.status = JobStatus.RUNNING
                    experiment.started_at = experiment.started_at or job.started_at

                ctx = TrainingContext(
                    job_id=job.id,
                    config=config,
                    total_steps=job.total_steps,
                    dataset_rows=dataset.rows if dataset else 0,
                    dataset_tokens=dataset.tokens if dataset else 0,
                    model_name=model.name if model else "unknown",
                    dataset_name=dataset.name if dataset else "unknown",
                    start_step=job.current_step,
                    extra={"output_dir": str(settings.checkpoints_dir / job.id)},
                )
                backend_name = job.backend
                model_path = model.local_path if model else None
                dataset_path = dataset.local_path if dataset else None
                job_snapshot = job.id
                # Flush before detaching: expunge() drops pending changes, which
                # would otherwise lose the RUNNING status and started_at stamp.
                session.flush()
                session.expunge(job)

            backend = self._build_backend(backend_name, model_path, dataset_path)
            ctx.on_metric = self._make_metric_handler(job_snapshot)
            ctx.on_log = self._make_log_handler(job_snapshot)
            ctx.on_checkpoint = self._make_checkpoint_handler(job_snapshot)

            task = asyncio.create_task(
                self._run(job_snapshot, backend, ctx), name=f"train-{job_snapshot}"
            )
            self._jobs[job_snapshot] = RunningJob(
                job_id=job_snapshot, ctx=ctx, task=task, backend=backend_name
            )
            monitor.set_load_bias(0.85)
            self._publish_status(job_snapshot, JobStatus.RUNNING)

        return self.get_job(job_id)

    def _build_backend(
        self, backend_name: str, model_path: str | None, dataset_path: str | None
    ) -> TrainingBackend:
        if backend_name == "torch" and model_path and dataset_path:
            return TorchTrainingBackend(model_path=model_path, dataset_path=dataset_path)
        return SimulatedTrainingBackend()

    async def _run(self, job_id: str, backend: TrainingBackend, ctx: TrainingContext) -> None:
        result: dict[str, Any] = {}
        status = JobStatus.COMPLETED
        error: str | None = None
        try:
            await backend.prepare(ctx)
            result = await backend.run(ctx)
            if ctx.should_stop:
                status = JobStatus.STOPPED
        except asyncio.CancelledError:
            status = JobStatus.STOPPED
            logbook.warn(f"Job {job_id} cancelled", source="training", job_id=job_id)
            raise
        except OutOfMemoryError as exc:
            status, error = JobStatus.FAILED, exc.message
            logbook.error(f"[OOM] {exc.message}", source="training", job_id=job_id)
        except Exception as exc:  # noqa: BLE001
            status, error = JobStatus.FAILED, f"{type(exc).__name__}: {exc}"
            logger.exception("training job %s failed", job_id)
            logbook.error(f"Job failed: {error}", source="training", job_id=job_id)
        finally:
            try:
                await backend.cleanup(ctx)
            except Exception:  # pragma: no cover
                pass
            self._flush_metrics(job_id, force=True)
            self._finalize(job_id, status, result, error)
            monitor.set_load_bias(0.15)
            self._jobs.pop(job_id, None)

    # --------------------------------------------------------------- control
    async def pause(self, job_id: str) -> TrainingJob:
        running = self._require_running(job_id)
        running.ctx.pause_event.clear()
        self._set_status(job_id, JobStatus.PAUSED)
        self._publish_status(job_id, JobStatus.PAUSED)
        return self.get_job(job_id)

    async def resume(self, job_id: str) -> TrainingJob:
        running = self._jobs.get(job_id)
        if running is None:
            # resume a job whose task ended while paused → start a fresh task
            return await self.start(job_id)
        running.ctx.pause_event.set()
        self._set_status(job_id, JobStatus.RUNNING)
        self._publish_status(job_id, JobStatus.RUNNING)
        return self.get_job(job_id)

    async def stop(self, job_id: str) -> TrainingJob:
        running = self._jobs.get(job_id)
        if running is None:
            self._set_status(job_id, JobStatus.STOPPED)
            return self.get_job(job_id)
        running.ctx.stop_event.set()
        running.ctx.pause_event.set()  # unblock a paused loop so it can exit
        self._set_status(job_id, JobStatus.STOPPING)
        self._publish_status(job_id, JobStatus.STOPPING)
        return self.get_job(job_id)

    async def request_checkpoint(self, job_id: str) -> None:
        running = self._require_running(job_id)
        running.ctx.checkpoint_request.set()
        logbook.info("Manual checkpoint requested", source="training", job_id=job_id)

    def _require_running(self, job_id: str) -> RunningJob:
        running = self._jobs.get(job_id)
        if running is None:
            raise ConflictError(f"Job {job_id} is not running")
        return running

    # ----------------------------------------------------------- persistence
    def _make_metric_handler(self, job_id: str):
        async def handler(metrics: StepMetrics) -> None:
            running = self._jobs.get(job_id)
            payload = {
                "job_id": job_id,
                "step": metrics.step,
                "epoch": metrics.epoch,
                "loss": metrics.loss,
                "val_loss": metrics.val_loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": metrics.grad_norm,
                "tokens_per_sec": metrics.tokens_per_sec,
                "samples_per_sec": metrics.samples_per_sec,
                "tokens_processed": metrics.tokens_processed,
                "gpu_utilization": metrics.gpu_utilization,
                "vram_used_mb": metrics.vram_used_mb,
                "eta_seconds": metrics.eta_seconds,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            bus.publish(Topics.TRAINING_METRIC, payload)
            bus.publish(f"{Topics.TRAINING_METRIC}.{job_id}", payload)

            if running is not None:
                running.pending_metrics.append(payload)
                if len(running.pending_metrics) >= METRIC_FLUSH_EVERY:
                    self._flush_metrics(job_id)
            self._update_progress(job_id, metrics)

        return handler

    def _make_log_handler(self, job_id: str):
        async def handler(message: str, level: str = "info") -> None:
            logbook.log(message, level=level, source="training", job_id=job_id, persist=False)

        return handler

    def _make_checkpoint_handler(self, job_id: str):
        async def handler(step: int, loss: float, val_loss: float | None, is_best: bool) -> None:
            self._create_checkpoint(job_id, step, loss, val_loss, is_best)

        return handler

    def _flush_metrics(self, job_id: str, *, force: bool = False) -> None:
        running = self._jobs.get(job_id)
        if running is None or (not force and not running.pending_metrics):
            return
        rows = running.pending_metrics
        running.pending_metrics = []
        if not rows:
            return
        with session_scope() as session:
            session.add_all(
                [
                    TrainingMetric(
                        job_id=job_id,
                        step=row["step"],
                        epoch=row["epoch"],
                        ts=datetime.fromisoformat(row["ts"]),
                        loss=row["loss"],
                        val_loss=row["val_loss"],
                        learning_rate=row["learning_rate"],
                        grad_norm=row["grad_norm"],
                        tokens_per_sec=row["tokens_per_sec"],
                        samples_per_sec=row["samples_per_sec"],
                        gpu_utilization=row["gpu_utilization"],
                        vram_used_mb=row["vram_used_mb"],
                    )
                    for row in rows
                ]
            )

    def _update_progress(self, job_id: str, metrics: StepMetrics) -> None:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job is None:
                return
            job.current_step = metrics.step
            job.current_epoch = metrics.epoch
            job.loss = metrics.loss
            if metrics.val_loss is not None:
                job.val_loss = metrics.val_loss
                job.best_val_loss = (
                    metrics.val_loss
                    if job.best_val_loss is None
                    else min(job.best_val_loss, metrics.val_loss)
                )
            job.learning_rate = metrics.learning_rate
            job.grad_norm = metrics.grad_norm
            job.tokens_processed = metrics.tokens_processed
            job.tokens_per_sec = metrics.tokens_per_sec
            job.samples_per_sec = metrics.samples_per_sec
            job.gpu_utilization = metrics.gpu_utilization
            job.vram_used_mb = metrics.vram_used_mb
            job.eta_seconds = metrics.eta_seconds

    def _create_checkpoint(
        self, job_id: str, step: int, loss: float, val_loss: float | None, is_best: bool
    ) -> None:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job is None:
                return
            config = TrainingConfig.model_validate(job.config)

            directory = resolve_within(settings.checkpoints_dir, f"{job_id}/step-{step}")
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {
                "job_id": job_id,
                "step": step,
                "epoch": job.current_epoch,
                "train_loss": loss,
                "val_loss": val_loss,
                "provenance": job.provenance.value
                if hasattr(job.provenance, "value")
                else job.provenance,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": job.config,
                "note": (
                    "Simulated checkpoint metadata — no weights were written."
                    if job.provenance == RunProvenance.SIMULATED
                    else "Checkpoint written by the training backend."
                ),
            }
            (directory / "foxtrot-checkpoint.json").write_text(json.dumps(manifest, indent=2))

            checkpoint = Checkpoint(
                model_id=job.model_id,
                job_id=job_id,
                experiment_id=job.experiment_id,
                step=step,
                epoch=job.current_epoch,
                path=str(directory),
                size_bytes=_simulated_ckpt_size(config)
                if job.provenance == RunProvenance.SIMULATED
                else 0,
                train_loss=loss,
                val_loss=val_loss,
                benchmark_score=None,
                is_best=is_best,
                provenance=job.provenance,
                is_demo=job.provenance == RunProvenance.SIMULATED,
            )
            session.add(checkpoint)
            session.flush()

            if is_best:
                session.query(Checkpoint).filter(
                    Checkpoint.job_id == job_id, Checkpoint.id != checkpoint.id
                ).update({Checkpoint.is_best: False})
                if job.experiment_id:
                    experiment = session.get(Experiment, job.experiment_id)
                    if experiment:
                        experiment.best_checkpoint_id = checkpoint.id

            # retention: keep last N (best is always kept)
            keep = config.checkpointing.keep_last
            stale = (
                session.query(Checkpoint)
                .filter(Checkpoint.job_id == job_id, Checkpoint.is_best.is_(False))
                .order_by(Checkpoint.step.desc())
                .offset(keep)
                .all()
            )
            for old in stale:
                session.delete(old)

            logbook.info(
                f"[CKPT ] saved step {step} → {directory.name}" + ("  (best)" if is_best else ""),
                source="training",
                job_id=job_id,
                persist=False,
            )
            bus.publish(
                "training.checkpoint",
                {
                    "job_id": job_id,
                    "checkpoint_id": checkpoint.id,
                    "step": step,
                    "is_best": is_best,
                },
            )

    def _set_status(self, job_id: str, status: JobStatus) -> None:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job:
                job.status = status

    def _finalize(
        self, job_id: str, status: JobStatus, result: dict[str, Any], error: str | None
    ) -> None:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = status
            job.ended_at = datetime.now(timezone.utc)
            job.error = error
            if result.get("final_loss") is not None:
                job.loss = result["final_loss"]
            if result.get("final_val_loss") is not None:
                job.val_loss = result["final_val_loss"]

            model = session.get(Model, job.model_id)
            if model:
                model.status = (
                    ModelStatus.READY if status != JobStatus.FAILED else ModelStatus.ERROR
                )
                if status == JobStatus.FAILED:
                    model.error = error

            if job.experiment_id:
                experiment = session.get(Experiment, job.experiment_id)
                if experiment:
                    experiment.status = status
                    experiment.ended_at = job.ended_at
                    if experiment.started_at:
                        started = experiment.started_at
                        if started.tzinfo is None:
                            started = started.replace(tzinfo=timezone.utc)
                        experiment.duration_seconds = (job.ended_at - started).total_seconds()
                    experiment.final_train_loss = job.loss
                    experiment.final_val_loss = job.val_loss

        logbook.info(
            f"Job {job_id} finished with status {status}",
            source="training",
            job_id=job_id,
            context={"error": error} if error else {},
        )
        self._publish_status(job_id, status, error=error)

    def _publish_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        payload = {"job_id": job_id, "status": str(status), "error": error}
        bus.publish(Topics.TRAINING_STATUS, payload)
        bus.publish(f"{Topics.TRAINING_STATUS}.{job_id}", payload)

    # ------------------------------------------------------------- read side
    def get_job(self, job_id: str) -> TrainingJob:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job is None:
                raise NotFoundError(f"Training job {job_id} not found")
            session.expunge(job)
            return job

    def is_running(self, job_id: str) -> bool:
        running = self._jobs.get(job_id)
        return bool(running and not running.task.done())

    def running_job_ids(self) -> list[str]:
        return [jid for jid, r in self._jobs.items() if not r.task.done()]

    async def shutdown(self) -> None:
        for running in list(self._jobs.values()):
            running.ctx.stop_event.set()
            running.ctx.pause_event.set()
        for running in list(self._jobs.values()):
            running.task.cancel()
            try:
                await running.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._jobs.clear()

    def recover_orphans(self) -> int:
        """Jobs left RUNNING by a crash are not silently resurrected."""
        recovered = 0
        with session_scope() as session:
            orphans = session.scalars(
                select(TrainingJob).where(
                    TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.STOPPING])
                )
            ).all()
            for job in orphans:
                job.status = JobStatus.STOPPED
                job.error = "Interrupted by a server restart"
                job.ended_at = datetime.now(timezone.utc)
                recovered += 1
        if recovered:
            logbook.warn(
                f"Marked {recovered} interrupted training job(s) as stopped after restart",
                source="training",
            )
        return recovered


def _simulated_ckpt_size(config: TrainingConfig) -> int:
    """Plausible artefact size for a simulated adapter/checkpoint (bytes)."""
    if str(config.method) in {"lora", "qlora"}:
        return int(config.lora.rank * 4.2 * 1024 * 1024)
    return int(13.4 * 1024**3)


def validate_config_payload(content: str, fmt: str) -> TrainingConfig:
    """Parse the advanced JSON/YAML editor payload into a validated config."""
    import yaml

    try:
        data = json.loads(content) if fmt == "json" else yaml.safe_load(content)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not parse {fmt.upper()}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("Training config must be an object")
    return TrainingConfig.model_validate(data)


manager = TrainingManager()
