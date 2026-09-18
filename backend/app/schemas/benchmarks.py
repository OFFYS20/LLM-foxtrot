"""Benchmark / evaluation schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models.enums import JobStatus, RunProvenance
from app.schemas.common import ORMModel


class BenchmarkSuiteInfo(BaseModel):
    key: str
    label: str
    category: str
    description: str
    metric: str = Field(description="accuracy | pass@1 | exact_match | truthful_rate")
    default_shots: int = 0
    available_items: int = 0
    data_source: str = Field(
        description="bundled_sample | user_dataset | none — 'bundled_sample' is a small "
        "offline sample, never the official split"
    )
    official: bool = Field(
        default=False,
        description="True only when the full official split is present locally",
    )
    requires_execution: bool = False
    notes: str | None = None


class BenchmarkRunConfig(BaseModel):
    num_examples: int = Field(default=50, ge=1, le=100_000)
    few_shot: int = Field(default=0, ge=0, le=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    batch_size: int = Field(default=1, ge=1, le=64)
    stop: list[str] = Field(default_factory=list)


class BenchmarkRunRequest(BaseModel):
    model_id: str
    suite: str
    checkpoint_id: str | None = None
    dataset_id: str | None = Field(default=None, description="Required when suite == 'custom'")
    name: str | None = None
    config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)


class BenchmarkItemRead(BaseModel):
    index: int
    category: str
    question: str
    prompt: str
    expected: str
    response: str
    raw_output: str
    correct: bool
    score: float
    latency_ms: float
    tokens_used: int

    model_config = {"from_attributes": True}


class BenchmarkRunRead(ORMModel):
    id: str
    name: str
    suite: str
    suite_label: str
    category: str
    model_id: str
    checkpoint_id: str | None = None
    dataset_id: str | None = None
    status: JobStatus
    provenance: RunProvenance
    config: dict[str, Any]
    total_items: int
    completed_items: int
    overall_score: float | None = None
    accuracy: float | None = None
    pass_at_1: float | None = None
    avg_latency_ms: float | None = None
    avg_tokens_per_sec: float | None = None
    total_tokens: int
    runtime_seconds: float | None = None
    category_scores: dict[str, Any]
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    is_demo: bool
    created_at: datetime


class BenchmarkRunDetail(BenchmarkRunRead):
    items: list[BenchmarkItemRead] = Field(default_factory=list)
    model_name: str | None = None


class ModelComparisonRequest(BaseModel):
    model_ids: list[str] = Field(min_length=2, max_length=8)
    name: str | None = None
    persist: bool = False


class ComparisonMetricRow(BaseModel):
    model_id: str
    model_name: str
    label: str | None = None
    parameters: int | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    precision: str | None = None
    benchmark_score: float | None = None
    validation_loss: float | None = None
    inference_tokens_per_sec: float | None = None
    vram_usage_mb: float | None = None
    benchmark_breakdown: dict[str, float] = Field(default_factory=dict)
    measured_metrics: list[str] = Field(
        default_factory=list, description="Metric keys backed by a real measured run"
    )
    provenance: str = "simulated"


class ModelComparisonResponse(BaseModel):
    id: str | None = None
    name: str
    rows: list[ComparisonMetricRow]
    categories: list[str]
    created_at: datetime
