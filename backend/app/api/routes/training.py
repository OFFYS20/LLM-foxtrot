"""Training job endpoints.

Jobs run on background tasks owned by the training manager — these handlers
only create, inspect and signal them, so an HTTP timeout can never kill a run.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.db.models.catalog import Dataset, Model
from app.db.models.enums import JobStatus
from app.db.models.training import TrainingJob, TrainingMetric
from app.schemas.common import Ack, Page
from app.schemas.training import (
    RawConfigRequest,
    SaveCheckpointRequest,
    TrainingConfig,
    TrainingConfigValidation,
    TrainingJobCreate,
    TrainingJobDetail,
    TrainingJobRead,
    TrainingMetricRead,
)
from app.services.hardware.monitor import monitor
from app.services.logbook import logbook
from app.services.training.manager import manager, validate_config_payload
from app.services.training.schedules import estimate_total_steps

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/jobs", response_model=Page[TrainingJobRead])
def list_jobs(
    session: SessionDep,
    pagination: PaginationDep,
    status: JobStatus | None = None,
    model_id: str | None = None,
) -> Page[TrainingJobRead]:
    statement = select(TrainingJob).order_by(TrainingJob.created_at.desc())
    if status:
        statement = statement.where(TrainingJob.status == status)
    if model_id:
        statement = statement.where(TrainingJob.model_id == model_id)
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[TrainingJobRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/jobs", response_model=TrainingJobDetail, status_code=201)
async def create_job(payload: TrainingJobCreate, session: SessionDep) -> TrainingJobDetail:
    job = manager.create_job(payload)
    if payload.start_immediately:
        await manager.start(job.id)
    fresh = get_or_404(session, TrainingJob, job.id, "Training job")
    session.refresh(fresh)
    return _detail(session, fresh)


@router.post("/validate", response_model=TrainingConfigValidation)
def validate_training_config(
    config: TrainingConfig,
    session: SessionDep,
    dataset_id: str | None = None,
) -> TrainingConfigValidation:
    snapshot = monitor.latest()
    vram = snapshot.gpus[0].memory_total_mb if snapshot.gpus else None
    warnings = config.validate_runtime(gpu_available=snapshot.gpu_available, vram_mb=vram)

    steps = None
    tokens = None
    if dataset_id:
        dataset = session.get(Dataset, dataset_id)
        if dataset:
            steps = estimate_total_steps(
                dataset_rows=max(1, dataset.rows),
                epochs=config.epochs,
                batch_size=config.batch_size,
                grad_accum=config.gradient_accumulation_steps,
            )
            tokens = int(dataset.tokens * config.epochs)

    return TrainingConfigValidation(
        ok=True,
        config=config,
        warnings=warnings,
        estimated_steps=steps,
        estimated_tokens=tokens,
        effective_batch_size=config.effective_batch_size,
    )


@router.post("/config/parse", response_model=TrainingConfigValidation)
def parse_raw_config(payload: RawConfigRequest, session: SessionDep) -> TrainingConfigValidation:
    """Validate the advanced JSON/YAML editor content."""
    try:
        config = validate_config_payload(payload.content, payload.format)
    except Exception as exc:  # noqa: BLE001
        return TrainingConfigValidation(ok=False, errors=[str(exc)])
    return validate_training_config(config, session, payload.dataset_id)


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
def get_job(job_id: str, session: SessionDep) -> TrainingJobDetail:
    job = get_or_404(session, TrainingJob, job_id, "Training job")
    return _detail(session, job)


@router.get("/jobs/{job_id}/metrics", response_model=list[TrainingMetricRead])
def job_metrics(job_id: str, session: SessionDep, limit: int = 2000) -> list[TrainingMetricRead]:
    get_or_404(session, TrainingJob, job_id, "Training job")
    rows = session.scalars(
        select(TrainingMetric)
        .where(TrainingMetric.job_id == job_id)
        .order_by(TrainingMetric.step.asc())
        .limit(max(1, min(limit, 20_000)))
    ).all()
    return [TrainingMetricRead.model_validate(row) for row in rows]


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, limit: int = 300) -> dict:
    return {"job_id": job_id, "entries": logbook.tail(limit=limit, job_id=job_id)}


@router.post("/jobs/{job_id}/start", response_model=TrainingJobRead)
async def start_job(job_id: str) -> TrainingJobRead:
    job = await manager.start(job_id)
    return TrainingJobRead.model_validate(job)


@router.post("/jobs/{job_id}/pause", response_model=TrainingJobRead)
async def pause_job(job_id: str) -> TrainingJobRead:
    job = await manager.pause(job_id)
    return TrainingJobRead.model_validate(job)


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobRead)
async def resume_job(job_id: str) -> TrainingJobRead:
    job = await manager.resume(job_id)
    return TrainingJobRead.model_validate(job)


@router.post("/jobs/{job_id}/stop", response_model=TrainingJobRead)
async def stop_job(job_id: str) -> TrainingJobRead:
    job = await manager.stop(job_id)
    return TrainingJobRead.model_validate(job)


@router.post("/jobs/{job_id}/checkpoint", response_model=Ack)
async def save_checkpoint(job_id: str, payload: SaveCheckpointRequest) -> Ack:
    await manager.request_checkpoint(job_id)
    return Ack(ok=True, message="Checkpoint requested — it will be written at the next step")


@router.delete("/jobs/{job_id}", response_model=Ack)
async def delete_job(job_id: str, session: SessionDep) -> Ack:
    job = get_or_404(session, TrainingJob, job_id, "Training job")
    if manager.is_running(job_id):
        await manager.stop(job_id)
    session.delete(job)
    return Ack(ok=True, message="Training job deleted", id=job_id)


@router.get("/active", response_model=list[TrainingJobRead])
def active_jobs(session: SessionDep) -> list[TrainingJobRead]:
    rows = session.scalars(
        select(TrainingJob)
        .where(TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.QUEUED]))
        .order_by(TrainingJob.created_at.desc())
    ).all()
    return [TrainingJobRead.model_validate(row) for row in rows]


def _detail(session, job: TrainingJob) -> TrainingJobDetail:
    model = session.get(Model, job.model_id)
    dataset = session.get(Dataset, job.dataset_id)
    metrics = session.scalars(
        select(TrainingMetric)
        .where(TrainingMetric.job_id == job.id)
        .order_by(TrainingMetric.step.asc())
        .limit(2000)
    ).all()

    snapshot = monitor.latest()
    config = TrainingConfig.model_validate(job.config)
    warnings = config.validate_runtime(
        gpu_available=snapshot.gpu_available,
        vram_mb=snapshot.gpus[0].memory_total_mb if snapshot.gpus else None,
    )

    detail = TrainingJobDetail.model_validate(job)
    detail.metrics = [TrainingMetricRead.model_validate(m) for m in metrics]
    detail.model_name = model.display_name or model.name if model else None
    detail.dataset_name = dataset.name if dataset else None
    detail.warnings = warnings
    return detail
