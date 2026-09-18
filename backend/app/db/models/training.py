"""Training jobs, per-step metrics, experiments and checkpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, id_column
from app.db.models.enums import JobStatus, RunProvenance, TrainingMethod


class TrainingJob(Base, TimestampMixin):
    __tablename__ = "training_jobs"

    id: Mapped[str] = id_column("job")
    name: Mapped[str] = mapped_column(String(200))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), default=None)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), default=None)

    method: Mapped[TrainingMethod] = mapped_column(String(40), default=TrainingMethod.LORA)
    status: Mapped[JobStatus] = mapped_column(String(24), default=JobStatus.QUEUED, index=True)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)
    backend: Mapped[str] = mapped_column(String(40), default="simulated")

    config: Mapped[dict[str, Any]] = mapped_column(default=dict)

    # live progress -----------------------------------------------------------
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[float] = mapped_column(Float, default=0.0)
    loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    best_val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    learning_rate: Mapped[float | None] = mapped_column(Float, default=None)
    grad_norm: Mapped[float | None] = mapped_column(Float, default=None)
    tokens_processed: Mapped[int] = mapped_column(Integer, default=0)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    samples_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    gpu_utilization: Mapped[float | None] = mapped_column(Float, default=None)
    vram_used_mb: Mapped[float | None] = mapped_column(Float, default=None)
    eta_seconds: Mapped[float | None] = mapped_column(Float, default=None)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    metrics: Mapped[list[TrainingMetric]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="select"
    )

    @property
    def progress(self) -> float:
        if not self.total_steps:
            return 0.0
        return min(1.0, self.current_step / self.total_steps)


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (Index("ix_training_metrics_job_step", "job_id", "step"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"))
    step: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[float] = mapped_column(Float, default=0.0)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    learning_rate: Mapped[float | None] = mapped_column(Float, default=None)
    grad_norm: Mapped[float | None] = mapped_column(Float, default=None)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    samples_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    gpu_utilization: Mapped[float | None] = mapped_column(Float, default=None)
    vram_used_mb: Mapped[float | None] = mapped_column(Float, default=None)

    job: Mapped[TrainingJob] = relationship(back_populates="metrics")


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[str] = id_column("exp")
    name: Mapped[str] = mapped_column(String(200), index=True)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), default=None)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), default=None)
    job_id: Mapped[str | None] = mapped_column(String(40), default=None)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), default=None)

    status: Mapped[JobStatus] = mapped_column(String(24), default=JobStatus.QUEUED)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)
    method: Mapped[TrainingMethod] = mapped_column(String(40), default=TrainingMethod.LORA)

    hyperparameters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)

    final_train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    final_val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    best_checkpoint_id: Mapped[str | None] = mapped_column(String(40), default=None)
    benchmark_summary: Mapped[dict[str, Any]] = mapped_column(default=dict)

    notes: Mapped[str | None] = mapped_column(Text, default=None)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class Checkpoint(Base, TimestampMixin):
    __tablename__ = "checkpoints"

    id: Mapped[str] = id_column("ckpt")
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(40), default=None, index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(40), default=None)

    step: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[float] = mapped_column(Float, default=0.0)
    path: Mapped[str | None] = mapped_column(String(600), default=None)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    benchmark_score: Mapped[float | None] = mapped_column(Float, default=None)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    model: Mapped[Model] = relationship(back_populates="checkpoints")  # noqa: F821
