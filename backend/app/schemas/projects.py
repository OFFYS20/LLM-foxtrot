"""Project schemas.

A project is the container a workspace is organised around: a base model, a
default dataset, and the training runs/experiments that belong to them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class ProjectRead(ORMModel):
    id: str
    name: str
    description: str | None = None
    base_model_id: str | None = None
    default_dataset_id: str | None = None
    tags: list[Any] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    is_demo: bool
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectRead):
    base_model_name: str | None = None
    default_dataset_name: str | None = None
    experiment_count: int = 0
    training_job_count: int = 0
    running_job_count: int = 0
    checkpoint_count: int = 0
    last_activity_at: datetime | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = None
    base_model_id: str | None = None
    default_dataset_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Project name cannot be empty")
        return trimmed


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    base_model_id: str | None = None
    default_dataset_id: str | None = None
    tags: list[str] | None = None
    settings: dict[str, Any] | None = None
