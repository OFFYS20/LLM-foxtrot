"""Project endpoints — create and configure the container a workspace uses."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.core.errors import ConflictError, ValidationError
from app.db.models.catalog import Dataset, Model, Project
from app.db.models.enums import JobStatus
from app.db.models.training import Checkpoint, Experiment, TrainingJob
from app.schemas.common import Ack, Page
from app.schemas.projects import ProjectCreate, ProjectDetail, ProjectRead, ProjectUpdate
from app.services.logbook import logbook

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=Page[ProjectRead])
def list_projects(
    session: SessionDep,
    pagination: PaginationDep,
    search: str | None = None,
    include_demo: bool = True,
) -> Page[ProjectRead]:
    statement = select(Project).order_by(Project.updated_at.desc())
    if search:
        statement = statement.where(Project.name.ilike(f"%{search}%"))
    if not include_demo:
        statement = statement.where(Project.is_demo.is_(False))

    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[ProjectRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("", response_model=ProjectDetail, status_code=201)
def create_project(payload: ProjectCreate, session: SessionDep) -> ProjectDetail:
    if session.scalar(select(Project).where(Project.name == payload.name)):
        raise ConflictError(f"A project named {payload.name!r} already exists")

    _validate_references(session, payload.base_model_id, payload.default_dataset_id)

    project = Project(
        name=payload.name,
        description=payload.description,
        base_model_id=payload.base_model_id,
        default_dataset_id=payload.default_dataset_id,
        tags=payload.tags,
        settings=payload.settings,
        is_demo=False,
    )
    session.add(project)
    session.flush()

    logbook.info(f"Created project {project.name}", source="projects", context={"project_id": project.id})
    return _detail(session, project)


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str, session: SessionDep) -> ProjectDetail:
    project = get_or_404(session, Project, project_id, "Project")
    return _detail(session, project)


@router.patch("/{project_id}", response_model=ProjectDetail)
def update_project(project_id: str, payload: ProjectUpdate, session: SessionDep) -> ProjectDetail:
    project = get_or_404(session, Project, project_id, "Project")
    data = payload.model_dump(exclude_unset=True)

    if "name" in data and data["name"]:
        name = data["name"].strip()
        existing = session.scalar(select(Project).where(Project.name == name, Project.id != project_id))
        if existing:
            raise ConflictError(f"A project named {name!r} already exists")
        project.name = name
        data.pop("name")

    _validate_references(
        session,
        data.get("base_model_id", project.base_model_id),
        data.get("default_dataset_id", project.default_dataset_id),
    )

    for field, value in data.items():
        setattr(project, field, value)
    session.flush()
    return _detail(session, project)


@router.delete("/{project_id}", response_model=Ack)
def delete_project(project_id: str, session: SessionDep) -> Ack:
    """Delete the project. Runs and experiments survive — they are detached."""
    project = get_or_404(session, Project, project_id, "Project")

    running = session.scalar(
        select(func.count())
        .select_from(TrainingJob)
        .where(
            TrainingJob.project_id == project_id,
            TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.STOPPING]),
        )
    )
    if running:
        raise ConflictError(
            f"{running} training job(s) in this project are still active — stop them first."
        )

    session.query(TrainingJob).filter(TrainingJob.project_id == project_id).update(
        {TrainingJob.project_id: None}
    )
    session.query(Experiment).filter(Experiment.project_id == project_id).update(
        {Experiment.project_id: None}
    )

    name = project.name
    session.delete(project)
    logbook.warn(f"Deleted project {name}", source="projects", context={"project_id": project_id})
    return Ack(ok=True, message=f"Deleted {name}", id=project_id)


def _validate_references(session, model_id: str | None, dataset_id: str | None) -> None:
    if model_id and session.get(Model, model_id) is None:
        raise ValidationError(f"Base model {model_id} does not exist")
    if dataset_id and session.get(Dataset, dataset_id) is None:
        raise ValidationError(f"Dataset {dataset_id} does not exist")


def _detail(session, project: Project) -> ProjectDetail:
    detail = ProjectDetail.model_validate(project)

    if project.base_model_id:
        model = session.get(Model, project.base_model_id)
        detail.base_model_name = (model.display_name or model.name) if model else None
    if project.default_dataset_id:
        dataset = session.get(Dataset, project.default_dataset_id)
        detail.default_dataset_name = dataset.name if dataset else None

    count = lambda stmt: session.scalar(stmt) or 0  # noqa: E731
    detail.experiment_count = count(
        select(func.count()).select_from(Experiment).where(Experiment.project_id == project.id)
    )
    detail.training_job_count = count(
        select(func.count()).select_from(TrainingJob).where(TrainingJob.project_id == project.id)
    )
    detail.running_job_count = count(
        select(func.count())
        .select_from(TrainingJob)
        .where(
            TrainingJob.project_id == project.id,
            TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.PAUSED]),
        )
    )
    detail.checkpoint_count = count(
        select(func.count())
        .select_from(Checkpoint)
        .join(TrainingJob, TrainingJob.id == Checkpoint.job_id)
        .where(TrainingJob.project_id == project.id)
    )
    detail.last_activity_at = session.scalar(
        select(func.max(TrainingJob.updated_at)).where(TrainingJob.project_id == project.id)
    )
    return detail
