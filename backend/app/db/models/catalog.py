"""Projects, models and datasets — the things you configure before training."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, id_column
from app.db.models.enums import (
    DatasetFormat,
    DatasetStatus,
    DatasetTemplate,
    ModelFormat,
    ModelSource,
    ModelStatus,
    Precision,
)


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = id_column("prj")
    name: Mapped[str] = mapped_column(String(160), unique=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    base_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), default=None)
    default_dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), default=None)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    settings: Mapped[dict[str, Any]] = mapped_column(default=dict)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class Model(Base, TimestampMixin):
    __tablename__ = "models"

    id: Mapped[str] = id_column("mdl")
    name: Mapped[str] = mapped_column(String(200), index=True)
    display_name: Mapped[str | None] = mapped_column(String(200), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    source: Mapped[ModelSource] = mapped_column(String(32), default=ModelSource.LOCAL)
    format: Mapped[ModelFormat] = mapped_column(String(32), default=ModelFormat.SAFETENSORS)
    status: Mapped[ModelStatus] = mapped_column(String(32), default=ModelStatus.READY, index=True)

    repo_id: Mapped[str | None] = mapped_column(String(200), default=None)
    revision: Mapped[str | None] = mapped_column(String(80), default=None)
    local_path: Mapped[str | None] = mapped_column(String(600), default=None)

    architecture: Mapped[str | None] = mapped_column(String(120), default=None)
    parameters: Mapped[int | None] = mapped_column(Integer, default=None)  # in millions
    context_length: Mapped[int | None] = mapped_column(Integer, default=None)
    precision: Mapped[Precision] = mapped_column(String(16), default=Precision.BF16)
    quantization: Mapped[str | None] = mapped_column(String(40), default=None)
    size_bytes: Mapped[int | None] = mapped_column(Integer, default=None)
    vram_estimate_mb: Mapped[int | None] = mapped_column(Integer, default=None)

    tokenizer: Mapped[dict[str, Any]] = mapped_column(default=dict)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    tags: Mapped[list[Any]] = mapped_column(default=list)

    parent_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), default=None)
    source_checkpoint_id: Mapped[str | None] = mapped_column(String(40), default=None)
    license: Mapped[str | None] = mapped_column(String(120), default=None)

    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    checkpoints: Mapped[list[Checkpoint]] = relationship(  # noqa: F821
        back_populates="model", cascade="all, delete-orphan", lazy="selectin"
    )


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[str] = id_column("ds")
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    format: Mapped[DatasetFormat] = mapped_column(String(32), default=DatasetFormat.JSONL)
    template: Mapped[DatasetTemplate] = mapped_column(String(32), default=DatasetTemplate.RAW)
    status: Mapped[DatasetStatus] = mapped_column(String(32), default=DatasetStatus.READY)

    source: Mapped[str] = mapped_column(String(40), default="upload")  # upload|path|huggingface
    repo_id: Mapped[str | None] = mapped_column(String(200), default=None)
    local_path: Mapped[str | None] = mapped_column(String(600), default=None)

    rows: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    avg_sequence_length: Mapped[float] = mapped_column(Float, default=0.0)
    max_sequence_length: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    train_split: Mapped[float] = mapped_column(Float, default=0.9)
    validation_split: Mapped[float] = mapped_column(Float, default=0.05)
    test_split: Mapped[float] = mapped_column(Float, default=0.05)

    columns: Mapped[list[Any]] = mapped_column(default=list)
    preview: Mapped[list[Any]] = mapped_column(default=list)
    validation_report: Mapped[dict[str, Any]] = mapped_column(default=dict)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    token_count_method: Mapped[str] = mapped_column(String(32), default="estimated")

    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
