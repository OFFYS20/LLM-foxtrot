"""Model catalog schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.db.models.enums import ModelFormat, ModelSource, ModelStatus, Precision
from app.schemas.common import ORMModel


class TokenizerInfo(BaseModel):
    type: str | None = None
    vocab_size: int | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    chat_template: bool = False
    special_tokens: list[str] = Field(default_factory=list)


class CheckpointRead(ORMModel):
    id: str
    model_id: str
    job_id: str | None = None
    experiment_id: str | None = None
    step: int
    epoch: float
    path: str | None = None
    size_bytes: int
    train_loss: float | None = None
    val_loss: float | None = None
    benchmark_score: float | None = None
    is_best: bool
    provenance: str
    notes: str | None = None
    is_demo: bool
    created_at: datetime


class ModelRead(ORMModel):
    id: str
    name: str
    display_name: str | None = None
    description: str | None = None
    source: ModelSource
    format: ModelFormat
    status: ModelStatus
    repo_id: str | None = None
    revision: str | None = None
    local_path: str | None = None
    architecture: str | None = None
    parameters: int | None = Field(default=None, description="Parameter count in millions")
    context_length: int | None = None
    precision: Precision
    quantization: str | None = None
    size_bytes: int | None = None
    vram_estimate_mb: int | None = None
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[Any] = Field(default_factory=list)
    parent_model_id: str | None = None
    license: str | None = None
    is_demo: bool
    error: str | None = None
    loaded_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ModelDetail(ModelRead):
    checkpoints: list[CheckpointRead] = Field(default_factory=list)


class ModelImportRequest(BaseModel):
    """Import from Hugging Face, a local directory, or a registered custom path."""

    source: Literal["huggingface", "local", "custom_path"] = "huggingface"
    name: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    path: str | None = None
    format: ModelFormat = ModelFormat.SAFETENSORS
    precision: Precision = Precision.BF16
    context_length: int | None = Field(default=None, ge=128, le=10_000_000)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _trim(cls, v: str | None) -> str | None:
        return v.strip() if v else v


class ModelUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    precision: Precision | None = None
    context_length: int | None = Field(default=None, ge=128, le=10_000_000)
    notes: str | None = None


class ModelCloneRequest(BaseModel):
    name: str
    description: str | None = None
    copy_checkpoints: bool = False


class ModelExportRequest(BaseModel):
    format: Literal["safetensors", "pytorch", "gguf", "hf_repo"] = "safetensors"
    destination: str | None = Field(default=None, description="Relative path inside the data root")
    include_tokenizer: bool = True
    quantization: str | None = None


class ModelExportResult(BaseModel):
    ok: bool
    format: str
    destination: str | None = None
    manifest: dict[str, Any]
    message: str


class ModelLoadRequest(BaseModel):
    engine: Literal["auto", "demo", "transformers", "llamacpp", "vllm", "ollama"] = "auto"
    precision: Precision | None = None
    device: str | None = None
