"""Benchmark runs, per-item results and saved model comparisons."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, id_column
from app.db.models.enums import JobStatus, RunProvenance


class BenchmarkRun(Base, TimestampMixin):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = id_column("bm")
    name: Mapped[str] = mapped_column(String(200))
    suite: Mapped[str] = mapped_column(String(60), index=True)  # mmlu, gsm8k, custom…
    suite_label: Mapped[str] = mapped_column(String(120), default="")
    category: Mapped[str] = mapped_column(String(48), default="knowledge")

    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(String(40), default=None)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), default=None)

    status: Mapped[JobStatus] = mapped_column(String(24), default=JobStatus.QUEUED, index=True)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)

    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)

    overall_score: Mapped[float | None] = mapped_column(Float, default=None)
    accuracy: Mapped[float | None] = mapped_column(Float, default=None)
    pass_at_1: Mapped[float | None] = mapped_column(Float, default=None)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, default=None)
    avg_tokens_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    category_scores: Mapped[dict[str, Any]] = mapped_column(default=dict)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    items: Mapped[list[BenchmarkItem]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="select"
    )


class BenchmarkItem(Base):
    __tablename__ = "benchmark_items"
    __table_args__ = (Index("ix_benchmark_items_run_index", "run_id", "index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("benchmark_runs.id", ondelete="CASCADE"))
    index: Mapped[int] = mapped_column(Integer)

    category: Mapped[str] = mapped_column(String(48), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    prompt: Mapped[str] = mapped_column(Text, default="")
    expected: Mapped[str] = mapped_column(Text, default="")
    response: Mapped[str] = mapped_column(Text, default="")
    raw_output: Mapped[str] = mapped_column(Text, default="")

    correct: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[BenchmarkRun] = relationship(back_populates="items")


class ModelComparison(Base, TimestampMixin):
    __tablename__ = "model_comparisons"

    id: Mapped[str] = id_column("cmp")
    name: Mapped[str] = mapped_column(String(200))
    model_ids: Mapped[list[Any]] = mapped_column(default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
