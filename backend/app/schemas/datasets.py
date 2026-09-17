"""Dataset manager schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.db.models.enums import DatasetFormat, DatasetStatus, DatasetTemplate
from app.schemas.common import ORMModel


class SplitConfig(BaseModel):
    train: float = Field(default=0.9, ge=0.0, le=1.0)
    validation: float = Field(default=0.05, ge=0.0, le=1.0)
    test: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0 (got {total:.4f})")
        return self


class DatasetIssue(BaseModel):
    row: int
    field: str | None = None
    severity: Literal["error", "warning"] = "error"
    message: str


class DatasetValidationReport(BaseModel):
    template: DatasetTemplate
    checked_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    issues: list[DatasetIssue] = Field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.invalid_rows == 0


class DatasetRead(ORMModel):
    id: str
    name: str
    description: str | None = None
    format: DatasetFormat
    template: DatasetTemplate
    status: DatasetStatus
    source: str
    repo_id: str | None = None
    local_path: str | None = None
    rows: int
    tokens: int
    avg_sequence_length: float
    max_sequence_length: int
    size_bytes: int
    train_split: float
    validation_split: float
    test_split: float
    columns: list[Any] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    tags: list[Any] = Field(default_factory=list)
    token_count_method: str = "estimated"
    is_demo: bool
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DatasetDetail(DatasetRead):
    preview: list[Any] = Field(default_factory=list)


class DatasetImportRequest(BaseModel):
    """Import from a registered path or a Hugging Face dataset repo.

    File uploads use ``POST /api/datasets/upload`` (multipart) instead.
    """

    source: Literal["path", "huggingface"] = "path"
    name: str
    path: str | None = Field(default=None, description="Path relative to the data root")
    repo_id: str | None = None
    subset: str | None = None
    split: str | None = None
    format: DatasetFormat | None = None
    template: DatasetTemplate = DatasetTemplate.RAW
    description: str | None = None
    splits: SplitConfig = Field(default_factory=SplitConfig)
    tags: list[str] = Field(default_factory=list)


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    template: DatasetTemplate | None = None
    splits: SplitConfig | None = None
    tags: list[str] | None = None


class DatasetPreview(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    offset: int
    limit: int


class DatasetTemplateInfo(BaseModel):
    key: DatasetTemplate
    label: str
    description: str
    schema_example: dict[str, Any]
    required_fields: list[str]


class DatasetValidateRequest(BaseModel):
    template: DatasetTemplate | None = None
    max_rows: int = Field(default=5000, ge=1, le=1_000_000)
