"""Checkpoint browser endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.config import settings
from app.core.paths import resolve_within
from app.db.models.catalog import Model
from app.db.models.enums import ModelFormat, ModelSource, ModelStatus
from app.db.models.training import Checkpoint
from app.schemas.benchmarks import BenchmarkRunConfig, BenchmarkRunRead, BenchmarkRunRequest
from app.schemas.common import Ack, Page
from app.schemas.models import CheckpointRead, ModelDetail
from app.services.evaluation.runner import runner
from app.services.logbook import logbook

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])


@router.get("", response_model=Page[CheckpointRead])
def list_checkpoints(
    session: SessionDep,
    pagination: PaginationDep,
    model_id: str | None = None,
    job_id: str | None = None,
    experiment_id: str | None = None,
    best_only: bool = False,
) -> Page[CheckpointRead]:
    statement = select(Checkpoint).order_by(Checkpoint.created_at.desc())
    if model_id:
        statement = statement.where(Checkpoint.model_id == model_id)
    if job_id:
        statement = statement.where(Checkpoint.job_id == job_id)
    if experiment_id:
        statement = statement.where(Checkpoint.experiment_id == experiment_id)
    if best_only:
        statement = statement.where(Checkpoint.is_best.is_(True))
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[CheckpointRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{checkpoint_id}", response_model=CheckpointRead)
def get_checkpoint(checkpoint_id: str, session: SessionDep) -> CheckpointRead:
    checkpoint = get_or_404(session, Checkpoint, checkpoint_id, "Checkpoint")
    return CheckpointRead.model_validate(checkpoint)


@router.post("/{checkpoint_id}/load", response_model=ModelDetail)
def load_checkpoint(checkpoint_id: str, session: SessionDep) -> ModelDetail:
    """Register a checkpoint as its own model entry so it can be chatted with."""
    checkpoint = get_or_404(session, Checkpoint, checkpoint_id, "Checkpoint")
    base = get_or_404(session, Model, checkpoint.model_id, "Model")

    name = f"{base.name}@step-{checkpoint.step}"
    existing = session.scalar(select(Model).where(Model.name == name))
    if existing:
        existing.status = ModelStatus.LOADED
        existing.loaded_at = datetime.now(timezone.utc)
        session.flush()
        return ModelDetail.model_validate(existing)

    model = Model(
        name=name,
        display_name=f"{base.display_name or base.name} · step {checkpoint.step}",
        description=f"Checkpoint from job {checkpoint.job_id} at step {checkpoint.step}",
        source=ModelSource.CHECKPOINT,
        format=base.format if base.format != ModelFormat.HF_REPO else ModelFormat.SAFETENSORS,
        status=ModelStatus.LOADED,
        local_path=checkpoint.path,
        architecture=base.architecture,
        parameters=base.parameters,
        context_length=base.context_length,
        precision=base.precision,
        size_bytes=checkpoint.size_bytes or base.size_bytes,
        vram_estimate_mb=base.vram_estimate_mb,
        tokenizer=dict(base.tokenizer or {}),
        config=dict(base.config or {}),
        metrics={"val_loss": checkpoint.val_loss, "train_loss": checkpoint.train_loss},
        tags=[*(base.tags or []), f"step-{checkpoint.step}"],
        parent_model_id=base.id,
        source_checkpoint_id=checkpoint.id,
        loaded_at=datetime.now(timezone.utc),
        is_demo=checkpoint.is_demo,
    )
    session.add(model)
    session.flush()
    logbook.info(f"Loaded checkpoint {checkpoint_id} as model {model.name}", source="checkpoints")
    return ModelDetail.model_validate(model)


@router.post("/{checkpoint_id}/evaluate", response_model=BenchmarkRunRead, status_code=202)
async def evaluate_checkpoint(
    checkpoint_id: str, session: SessionDep, suite: str = "mmlu", num_examples: int = 20
) -> BenchmarkRunRead:
    checkpoint = get_or_404(session, Checkpoint, checkpoint_id, "Checkpoint")
    run = runner.create_run(
        BenchmarkRunRequest(
            model_id=checkpoint.model_id,
            suite=suite,
            checkpoint_id=checkpoint.id,
            name=f"{suite} · checkpoint step {checkpoint.step}",
            config=BenchmarkRunConfig(num_examples=num_examples),
        )
    )
    await runner.start(run.id)
    return BenchmarkRunRead.model_validate(run)


@router.post("/{checkpoint_id}/export", response_model=Ack)
def export_checkpoint(
    checkpoint_id: str, session: SessionDep, destination: str | None = None
) -> Ack:
    checkpoint = get_or_404(session, Checkpoint, checkpoint_id, "Checkpoint")
    target = resolve_within(settings.exports_dir, destination or f"checkpoint-{checkpoint.id}")
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_id": checkpoint.id,
        "model_id": checkpoint.model_id,
        "job_id": checkpoint.job_id,
        "step": checkpoint.step,
        "epoch": checkpoint.epoch,
        "train_loss": checkpoint.train_loss,
        "val_loss": checkpoint.val_loss,
        "source_path": checkpoint.path,
        "provenance": str(checkpoint.provenance),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "weights_written": False,
        "note": "Manifest only — checkpoint weights are referenced at source_path.",
    }
    (target / "foxtrot-checkpoint-export.json").write_text(json.dumps(manifest, indent=2))
    return Ack(ok=True, message=f"Export manifest written to {target}", id=checkpoint.id)


@router.delete("/{checkpoint_id}", response_model=Ack)
def delete_checkpoint(checkpoint_id: str, session: SessionDep) -> Ack:
    checkpoint = get_or_404(session, Checkpoint, checkpoint_id, "Checkpoint")
    session.delete(checkpoint)
    return Ack(ok=True, message="Checkpoint deleted", id=checkpoint_id)
