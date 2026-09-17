"""Training configuration + job schemas.

``TrainingConfig`` is the single source of truth for hyperparameters: the
visual form, the raw JSON/YAML editor, and the training backends all validate
against this model before anything is executed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.db.models.enums import (
    JobStatus,
    LRScheduler,
    Optimizer,
    Precision,
    RunProvenance,
    TrainingMethod,
)
from app.schemas.common import ORMModel

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRAConfig(BaseModel):
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, le=0.9)
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    bias: Literal["none", "all", "lora_only"] = "none"


class CheckpointConfig(BaseModel):
    save_every_steps: int = Field(default=250, ge=1, le=1_000_000)
    keep_last: int = Field(default=3, ge=1, le=100)
    save_best: bool = True
    save_optimizer_state: bool = False


class TrainingConfig(BaseModel):
    """Validated before a job is ever queued (see ``validate_runtime``)."""

    method: TrainingMethod = TrainingMethod.LORA

    # hyperparameters
    epochs: float = Field(default=3.0, gt=0, le=1000)
    batch_size: int = Field(default=4, ge=1, le=4096)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0, le=1.0)
    warmup_steps: int = Field(default=100, ge=0, le=1_000_000)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_sequence_length: int = Field(default=2048, ge=16, le=1_048_576)
    gradient_clipping: float = Field(default=1.0, ge=0.0, le=100.0)
    optimizer: Optimizer = Optimizer.ADAMW
    lr_scheduler: LRScheduler = LRScheduler.COSINE
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    # precision / memory
    precision: Precision = Precision.BF16
    gradient_checkpointing: bool = True
    flash_attention: bool = False

    # adapters
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # checkpointing + cadence
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    eval_every_steps: int = Field(default=100, ge=1, le=1_000_000)
    log_every_steps: int = Field(default=10, ge=1, le=100_000)

    # dataset handling
    packing: bool = False
    shuffle: bool = True
    num_workers: int = Field(default=2, ge=0, le=64)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @model_validator(mode="after")
    def _coherent(self) -> TrainingConfig:
        if self.method == TrainingMethod.QLORA and self.precision not in (
            Precision.INT4,
            Precision.INT8,
        ):
            # QLoRA quantizes the frozen base model; keep the config honest.
            self.precision = Precision.INT4
        if self.method == TrainingMethod.FULL_FINETUNE and self.precision in (
            Precision.INT4,
            Precision.INT8,
        ):
            raise ValueError(
                "Full fine-tuning cannot run with INT4/INT8 base weights — "
                "use LoRA/QLoRA or pick FP16/BF16/FP32."
            )
        if self.eval_every_steps < self.log_every_steps:
            raise ValueError("eval_every_steps must be >= log_every_steps")
        return self

    def validate_runtime(self, *, gpu_available: bool, vram_mb: float | None) -> list[str]:
        """Non-fatal warnings shown before the user hits Start."""
        warnings: list[str] = []
        if not gpu_available:
            warnings.append(
                "No GPU detected — training will run on CPU and be dramatically slower."
            )
            if self.precision in (Precision.FP16, Precision.INT4, Precision.INT8):
                warnings.append(f"{self.precision.upper()} is not reliable on CPU; prefer FP32.")
        if vram_mb and self.max_sequence_length >= 8192 and vram_mb < 24_000:
            warnings.append(
                f"max_sequence_length={self.max_sequence_length} with {vram_mb / 1024:.0f} GB "
                "VRAM is likely to OOM; lower the sequence length or enable gradient checkpointing."
            )
        if self.method == TrainingMethod.FULL_FINETUNE and vram_mb and vram_mb < 40_000:
            warnings.append(
                "Full fine-tuning typically needs >= 40 GB VRAM for 7B-class models; "
                "consider LoRA or QLoRA."
            )
        if self.effective_batch_size > 512:
            warnings.append(
                f"Effective batch size is {self.effective_batch_size}; "
                "very large batches can destabilise small datasets."
            )
        return warnings


class TrainingJobCreate(BaseModel):
    name: str | None = None
    model_id: str
    dataset_id: str
    project_id: str | None = None
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    notes: str | None = None
    start_immediately: bool = True


class TrainingMetricRead(BaseModel):
    step: int
    epoch: float
    ts: datetime
    loss: float | None = None
    val_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    gpu_utilization: float | None = None
    vram_used_mb: float | None = None

    model_config = {"from_attributes": True}


class TrainingJobRead(ORMModel):
    id: str
    name: str
    model_id: str
    dataset_id: str
    experiment_id: str | None = None
    project_id: str | None = None
    method: TrainingMethod
    status: JobStatus
    provenance: RunProvenance
    backend: str
    config: dict[str, Any]
    total_steps: int
    current_step: int
    total_epochs: float
    current_epoch: float
    loss: float | None = None
    val_loss: float | None = None
    best_val_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    tokens_processed: int
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    gpu_utilization: float | None = None
    vram_used_mb: float | None = None
    eta_seconds: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def progress(self) -> float:
        return min(1.0, self.current_step / self.total_steps) if self.total_steps else 0.0


class TrainingJobDetail(TrainingJobRead):
    metrics: list[TrainingMetricRead] = Field(default_factory=list)
    model_name: str | None = None
    dataset_name: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrainingConfigValidation(BaseModel):
    ok: bool
    config: TrainingConfig | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimated_steps: int | None = None
    estimated_tokens: int | None = None
    effective_batch_size: int | None = None


class RawConfigRequest(BaseModel):
    """Advanced editor payload — JSON or YAML text."""

    format: Literal["json", "yaml"] = "json"
    content: str
    model_id: str | None = None
    dataset_id: str | None = None


class SaveCheckpointRequest(BaseModel):
    notes: str | None = None
    mark_best: bool = False
