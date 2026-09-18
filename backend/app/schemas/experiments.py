"""Experiment tracking schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models.enums import JobStatus, RunProvenance, TrainingMethod
from app.schemas.common import ORMModel


class ExperimentRead(ORMModel):
    id: str
    name: str
    model_id: str | None = None
    dataset_id: str | None = None
    job_id: str | None = None
    project_id: str | None = None
    status: JobStatus
    provenance: RunProvenance
    method: TrainingMethod
    hyperparameters: dict[str, Any]
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_checkpoint_id: str | None = None
    benchmark_summary: dict[str, Any]
    notes: str | None = None
    tags: list[Any]
    is_demo: bool
    created_at: datetime
    updated_at: datetime


class ExperimentDetail(ExperimentRead):
    model_name: str | None = None
    dataset_name: str | None = None
    checkpoint_count: int = 0
    benchmark_runs: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None


class ExperimentDuplicateRequest(BaseModel):
    name: str | None = None
    start_immediately: bool = False


class ExperimentCompareRequest(BaseModel):
    experiment_ids: list[str] = Field(min_length=2, max_length=8)


class ExperimentCompareRow(BaseModel):
    experiment_id: str
    name: str
    model_name: str | None = None
    dataset_name: str | None = None
    method: str
    hyperparameters: dict[str, Any]
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    duration_seconds: float | None = None
    benchmark_summary: dict[str, Any]
    provenance: str


class ExperimentCompareResponse(BaseModel):
    rows: list[ExperimentCompareRow]
    differing_hyperparameters: list[str]
