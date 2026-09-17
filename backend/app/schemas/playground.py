"""Side-by-side playground schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import SamplingParams
from app.schemas.common import ORMModel


class PlaygroundRunRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1, max_length=4)
    prompt: str = Field(min_length=1)
    system_prompt: str = ""
    params: SamplingParams = Field(default_factory=SamplingParams)
    persist: bool = True

    @field_validator("model_ids")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("model_ids must be unique")
        return value


class PlaygroundEntry(BaseModel):
    model_id: str
    model_name: str
    content: str
    latency_ms: float
    time_to_first_token_ms: float
    tokens_per_sec: float
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    memory_mb: float | None = None
    finish_reason: str = "stop"
    provenance: str = "simulated"
    engine: str = "demo"
    error: str | None = None


class PlaygroundRunResponse(BaseModel):
    id: str | None = None
    prompt: str
    entries: list[PlaygroundEntry]
    created_at: datetime


class PlaygroundComparisonRead(ORMModel):
    id: str
    prompt: str
    system_prompt: str
    params: dict[str, Any]
    entries: list[Any]
    winner_model_id: str | None = None
    votes: dict[str, Any]
    provenance: str
    is_demo: bool
    created_at: datetime


class PlaygroundVoteRequest(BaseModel):
    winner_model_id: str
