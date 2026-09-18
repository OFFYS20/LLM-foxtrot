"""Shared response envelopes and primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int = 50
    offset: int = 0


class Ack(BaseModel):
    ok: bool = True
    message: str | None = None
    id: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Capability(BaseModel):
    """What this deployment can actually do right now (drives UI banners)."""

    name: str
    available: bool
    detail: str | None = None


class SystemInfo(BaseModel):
    app: str
    version: str
    demo_mode: bool
    gpu_available: bool
    compute_mode: str = Field(description="cuda | rocm | mps | cpu")
    inference_engine: str
    hardware_provider: str
    capabilities: list[Capability]
    server_time: datetime
