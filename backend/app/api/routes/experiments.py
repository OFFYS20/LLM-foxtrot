"""Experiment tracking endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.core.errors import ValidationError
from app.db.models.catalog import Dataset, Model
from app.db.models.enums import JobStatus
from app.db.models.evaluation import BenchmarkRun
from app.db.models.training import Checkpoint, Experiment
from app.schemas.common import Ack, Page
from app.schemas.experiments import (
    ExperimentCompareRequest,
    ExperimentCompareResponse,
    ExperimentCompareRow,
    ExperimentDetail,
    ExperimentDuplicateRequest,
    ExperimentRead,
    ExperimentUpdate,
)
from app.schemas.training import TrainingConfig, TrainingJobCreate
from app.services.training.manager import manager

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=Page[ExperimentRead])
def list_experiments(
    session: SessionDep,
    pagination: PaginationDep,
    status: JobStatus | None = None,
    model_id: str | None = None,
    search: str | None = None,
) -> Page[ExperimentRead]:
    statement = select(Experiment).order_by(Experiment.created_at.desc())
    if status:
        statement = statement.where(Experiment.status == status)
    if model_id:
        statement = statement.where(Experiment.model_id == model_id)
    if search:
        statement = statement.where(Experiment.name.ilike(f"%{search}%"))
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[ExperimentRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str, session: SessionDep) -> ExperimentDetail:
    experiment = get_or_404(session, Experiment, experiment_id, "Experiment")
    detail = ExperimentDetail.model_validate(experiment)

    model = session.get(Model, experiment.model_id) if experiment.model_id else None
    dataset = session.get(Dataset, experiment.dataset_id) if experiment.dataset_id else None
    detail.model_name = (model.display_name or model.name) if model else None
    detail.dataset_name = dataset.name if dataset else None
    detail.checkpoint_count = len(
        session.scalars(select(Checkpoint).where(Checkpoint.experiment_id == experiment_id)).all()
    )
    if experiment.model_id:
        runs = session.scalars(
            select(BenchmarkRun)
            .where(
                BenchmarkRun.model_id == experiment.model_id,
                BenchmarkRun.status == JobStatus.COMPLETED,
            )
            .order_by(BenchmarkRun.created_at.desc())
            .limit(10)
        ).all()
        detail.benchmark_runs = [
            {
                "id": run.id,
                "suite": run.suite,
                "suite_label": run.suite_label,
                "overall_score": run.overall_score,
                "accuracy": run.accuracy,
                "provenance": str(run.provenance),
                "official_split": bool((run.config or {}).get("official_split")),
            }
            for run in runs
        ]
    return detail


@router.patch("/{experiment_id}", response_model=ExperimentRead)
def update_experiment(
    experiment_id: str, payload: ExperimentUpdate, session: SessionDep
) -> ExperimentRead:
    experiment = get_or_404(session, Experiment, experiment_id, "Experiment")
    for field, value in payload.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(experiment, field, value)
    session.flush()
    return ExperimentRead.model_validate(experiment)


@router.delete("/{experiment_id}", response_model=Ack)
def delete_experiment(experiment_id: str, session: SessionDep) -> Ack:
    experiment = get_or_404(session, Experiment, experiment_id, "Experiment")
    session.delete(experiment)
    return Ack(ok=True, message="Experiment deleted", id=experiment_id)


@router.post("/{experiment_id}/duplicate", response_model=ExperimentRead, status_code=201)
async def duplicate_experiment(
    experiment_id: str, payload: ExperimentDuplicateRequest, session: SessionDep
) -> ExperimentRead:
    """Re-run an experiment's configuration as a brand new job."""
    source = get_or_404(session, Experiment, experiment_id, "Experiment")
    if not source.model_id or not source.dataset_id:
        raise ValidationError("Experiment is missing a model or dataset reference")

    config = TrainingConfig.model_validate(source.hyperparameters or {})
    job = manager.create_job(
        TrainingJobCreate(
            name=payload.name or f"{source.name} (copy)",
            model_id=source.model_id,
            dataset_id=source.dataset_id,
            project_id=source.project_id,
            config=config,
            notes=f"Duplicated from experiment {source.id}",
            start_immediately=False,
        )
    )
    if payload.start_immediately:
        await manager.start(job.id)

    experiment = session.get(Experiment, job.experiment_id)
    session.refresh(experiment)
    return ExperimentRead.model_validate(experiment)


@router.post("/compare", response_model=ExperimentCompareResponse)
def compare_experiments(
    payload: ExperimentCompareRequest, session: SessionDep
) -> ExperimentCompareResponse:
    rows: list[ExperimentCompareRow] = []
    for experiment_id in payload.experiment_ids:
        experiment = get_or_404(session, Experiment, experiment_id, "Experiment")
        model = session.get(Model, experiment.model_id) if experiment.model_id else None
        dataset = session.get(Dataset, experiment.dataset_id) if experiment.dataset_id else None
        rows.append(
            ExperimentCompareRow(
                experiment_id=experiment.id,
                name=experiment.name,
                model_name=(model.display_name or model.name) if model else None,
                dataset_name=dataset.name if dataset else None,
                method=str(experiment.method),
                hyperparameters=experiment.hyperparameters or {},
                final_train_loss=experiment.final_train_loss,
                final_val_loss=experiment.final_val_loss,
                duration_seconds=experiment.duration_seconds,
                benchmark_summary=experiment.benchmark_summary or {},
                provenance=str(experiment.provenance),
            )
        )

    differing = _differing_keys([row.hyperparameters for row in rows])
    return ExperimentCompareResponse(rows=rows, differing_hyperparameters=differing)


def _differing_keys(configs: list[dict]) -> list[str]:
    keys: set[str] = set()
    for config in configs:
        keys.update(_flatten(config).keys())
    differing = []
    for key in sorted(keys):
        values = {repr(_flatten(config).get(key)) for config in configs}
        if len(values) > 1:
            differing.append(key)
    return differing


def _flatten(data: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in (data or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat
