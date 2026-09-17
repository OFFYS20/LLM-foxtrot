"""Dataset manager endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.config import settings
from app.core.errors import ConflictError, ValidationError
from app.core.paths import ALLOWED_DATASET_SUFFIXES, resolve_within, safe_filename, validate_suffix
from app.db.models.catalog import Dataset
from app.db.models.enums import DatasetFormat, DatasetStatus, DatasetTemplate
from app.schemas.common import Ack, Page
from app.schemas.datasets import (
    DatasetDetail,
    DatasetImportRequest,
    DatasetPreview,
    DatasetRead,
    DatasetTemplateInfo,
    DatasetUpdate,
    DatasetValidateRequest,
    DatasetValidationReport,
)
from app.services.datasets.formats import detect_format
from app.services.datasets.service import (
    analyze_file,
    analyze_hf_dataset,
    read_preview,
    resolve_dataset_path,
)
from app.services.datasets.templates import list_templates
from app.services.logbook import logbook

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_CHUNK = 1024 * 1024


@router.get("", response_model=Page[DatasetRead])
def list_datasets(
    session: SessionDep,
    pagination: PaginationDep,
    template: DatasetTemplate | None = None,
    search: str | None = None,
    include_demo: bool = True,
) -> Page[DatasetRead]:
    statement = select(Dataset).order_by(Dataset.created_at.desc())
    if template:
        statement = statement.where(Dataset.template == template)
    if search:
        statement = statement.where(Dataset.name.ilike(f"%{search}%"))
    if not include_demo:
        statement = statement.where(Dataset.is_demo.is_(False))

    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[DatasetRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/templates", response_model=list[DatasetTemplateInfo])
def dataset_templates() -> list[DatasetTemplateInfo]:
    return list_templates()


@router.post("/import", response_model=DatasetDetail, status_code=201)
def import_dataset(payload: DatasetImportRequest, session: SessionDep) -> DatasetDetail:
    if session.scalar(select(Dataset).where(Dataset.name == payload.name)):
        raise ConflictError(f"A dataset named {payload.name!r} already exists")

    template = payload.template if payload.template != DatasetTemplate.RAW else None

    if payload.source == "huggingface":
        if not payload.repo_id:
            raise ValidationError("repo_id is required for Hugging Face imports")
        analysis = analyze_hf_dataset(
            payload.repo_id, payload.subset, payload.split, template=template
        )
        fmt = DatasetFormat.HF_DATASET
        local_path = None
    else:
        if not payload.path:
            raise ValidationError("path is required for path imports")
        path = resolve_dataset_path(payload.path)
        analysis, fmt = analyze_file(path, fmt=payload.format, template=template)
        local_path = str(path)

    dataset = _persist(session, payload, analysis, fmt, local_path)
    return DatasetDetail.model_validate(dataset)


@router.post("/upload", response_model=DatasetDetail, status_code=201)
async def upload_dataset(
    session: SessionDep,
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    template: DatasetTemplate = Form(default=DatasetTemplate.RAW),
    description: str | None = Form(default=None),
) -> DatasetDetail:
    """Upload a dataset file. The name is sanitised and the size is capped."""
    filename = safe_filename(file.filename or "dataset.jsonl")
    target = resolve_within(settings.datasets_dir, filename)
    validate_suffix(target, ALLOWED_DATASET_SUFFIXES)

    if target.exists():
        raise ConflictError(f"A file named {filename!r} already exists in the dataset directory")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise ValidationError(f"Upload exceeds the {settings.max_upload_mb} MB limit")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    dataset_name = name or Path(filename).stem
    if session.scalar(select(Dataset).where(Dataset.name == dataset_name)):
        target.unlink(missing_ok=True)
        raise ConflictError(f"A dataset named {dataset_name!r} already exists")

    analysis, fmt = analyze_file(
        target, template=template if template != DatasetTemplate.RAW else None
    )
    payload = DatasetImportRequest(
        source="path",
        name=dataset_name,
        path=str(target),
        template=template,
        description=description,
    )
    dataset = _persist(session, payload, analysis, fmt, str(target))
    return DatasetDetail.model_validate(dataset)


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str, session: SessionDep) -> DatasetDetail:
    dataset = get_or_404(session, Dataset, dataset_id, "Dataset")
    return DatasetDetail.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetDetail)
def update_dataset(dataset_id: str, payload: DatasetUpdate, session: SessionDep) -> DatasetDetail:
    dataset = get_or_404(session, Dataset, dataset_id, "Dataset")
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    splits = data.pop("splits", None)
    for field, value in data.items():
        setattr(dataset, field, value)
    if splits:
        dataset.train_split = splits["train"]
        dataset.validation_split = splits["validation"]
        dataset.test_split = splits["test"]
    session.flush()
    return DatasetDetail.model_validate(dataset)


@router.delete("/{dataset_id}", response_model=Ack)
def delete_dataset(dataset_id: str, session: SessionDep, delete_file: bool = False) -> Ack:
    dataset = get_or_404(session, Dataset, dataset_id, "Dataset")
    name = dataset.name
    if delete_file and dataset.local_path:
        try:
            resolve_within(settings.datasets_dir, dataset.local_path).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logbook.warn(f"Could not remove dataset file: {exc}", source="datasets")
    session.delete(dataset)
    logbook.warn(f"Deleted dataset {name}", source="datasets")
    return Ack(ok=True, message=f"Deleted {name}", id=dataset_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str, session: SessionDep, offset: int = 0, limit: int = 25
) -> DatasetPreview:
    dataset = get_or_404(session, Dataset, dataset_id, "Dataset")
    limit = max(1, min(limit, 200))

    if dataset.local_path:
        path = resolve_dataset_path(dataset.local_path)
        rows, columns = read_preview(path, detect_format(path), offset=offset, limit=limit)
    else:
        stored = list(dataset.preview or [])
        rows = stored[offset : offset + limit]
        columns = list(dataset.columns or [])

    return DatasetPreview(
        dataset_id=dataset.id,
        columns=columns or list({k for row in rows for k in row}),
        rows=rows,
        total_rows=dataset.rows,
        offset=offset,
        limit=limit,
    )


@router.post("/{dataset_id}/validate", response_model=DatasetValidationReport)
def validate_dataset(
    dataset_id: str, payload: DatasetValidateRequest, session: SessionDep
) -> DatasetValidationReport:
    dataset = get_or_404(session, Dataset, dataset_id, "Dataset")
    template = payload.template or dataset.template

    if not dataset.local_path:
        stored = dataset.validation_report or {}
        return DatasetValidationReport(
            template=template,
            checked_rows=stored.get("checked_rows", 0),
            valid_rows=stored.get("valid_rows", 0),
            invalid_rows=stored.get("invalid_rows", 0),
            issues=[],
            truncated=True,
        )

    path = resolve_dataset_path(dataset.local_path)
    analysis, _ = analyze_file(path, template=template, max_rows=payload.max_rows)
    report = analysis.report or DatasetValidationReport(template=template)
    dataset.validation_report = report.model_dump(mode="json")
    dataset.status = DatasetStatus.READY if report.invalid_rows == 0 else DatasetStatus.INVALID
    session.flush()
    return report


def _persist(session, payload: DatasetImportRequest, analysis, fmt, local_path) -> Dataset:
    report = analysis.report
    dataset = Dataset(
        name=payload.name,
        description=payload.description,
        format=fmt,
        template=analysis.template,
        status=DatasetStatus.READY
        if not report or report.invalid_rows == 0
        else DatasetStatus.INVALID,
        source=payload.source,
        repo_id=payload.repo_id,
        local_path=local_path,
        rows=analysis.rows,
        tokens=analysis.tokens,
        avg_sequence_length=analysis.avg_sequence_length,
        max_sequence_length=analysis.max_sequence_length,
        size_bytes=analysis.size_bytes,
        train_split=payload.splits.train,
        validation_split=payload.splits.validation,
        test_split=payload.splits.test,
        columns=analysis.columns,
        preview=analysis.preview,
        validation_report=report.model_dump(mode="json") if report else {},
        tags=payload.tags,
        token_count_method=analysis.token_count_method,
        is_demo=False,
    )
    session.add(dataset)
    session.flush()
    logbook.info(
        f"Imported dataset {dataset.name} — {dataset.rows:,} rows / ~{dataset.tokens:,} tokens",
        source="datasets",
        context={"dataset_id": dataset.id, "invalid_rows": report.invalid_rows if report else 0},
    )
    return dataset
