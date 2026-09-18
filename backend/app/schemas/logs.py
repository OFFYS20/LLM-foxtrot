"""Log stream schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models.enums import LogLevel


class LogEntryRead(BaseModel):
    id: int
    ts: datetime
    level: LogLevel
    source: str
    message: str
    job_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class LogQuery(BaseModel):
    level: LogLevel | None = None
    source: str | None = None
    job_id: str | None = None
    search: str | None = None
    limit: int = Field(default=200, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)
