"""Benchmark runner + results endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.db.models.catalog import Model
from app.db.models.enums import JobStatus
from app.db.models.evaluation import BenchmarkItem, BenchmarkRun
from app.schemas.benchmarks import (
    BenchmarkItemRead,
    BenchmarkRunDetail,
    BenchmarkRunRead,
    BenchmarkRunRequest,
    BenchmarkSuiteInfo,
)
from app.schemas.common import Ack, Page
from app.services.evaluation.registry import registry as suite_registry
from app.services.evaluation.runner import runner

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("/suites", response_model=list[BenchmarkSuiteInfo])
def list_suites() -> list[BenchmarkSuiteInfo]:
    """Available suites, with an explicit statement of where their data comes from."""
    return suite_registry.list_info()


@router.post("/run", response_model=BenchmarkRunRead, status_code=202)
async def run_benchmark(payload: BenchmarkRunRequest) -> BenchmarkRunRead:
    run = runner.create_run(payload)
    await runner.start(run.id)
    return BenchmarkRunRead.model_validate(run)


@router.get("/results", response_model=Page[BenchmarkRunRead])
def list_results(
    session: SessionDep,
    pagination: PaginationDep,
    model_id: str | None = None,
    suite: str | None = None,
    status: JobStatus | None = None,
) -> Page[BenchmarkRunRead]:
    statement = select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
    if model_id:
        statement = statement.where(BenchmarkRun.model_id == model_id)
    if suite:
        statement = statement.where(BenchmarkRun.suite == suite)
    if status:
        statement = statement.where(BenchmarkRun.status == status)
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[BenchmarkRunRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/results/{run_id}", response_model=BenchmarkRunDetail)
def get_result(
    run_id: str,
    session: SessionDep,
    only_incorrect: bool = False,
    limit: int = 500,
) -> BenchmarkRunDetail:
    run = get_or_404(session, BenchmarkRun, run_id, "Benchmark run")
    statement = (
        select(BenchmarkItem)
        .where(BenchmarkItem.run_id == run_id)
        .order_by(BenchmarkItem.index.asc())
        .limit(max(1, min(limit, 5000)))
    )
    if only_incorrect:
        statement = statement.where(BenchmarkItem.correct.is_(False))
    items = session.scalars(statement).all()

    model = session.get(Model, run.model_id)
    detail = BenchmarkRunDetail.model_validate(run)
    detail.items = [BenchmarkItemRead.model_validate(item) for item in items]
    detail.model_name = (model.display_name or model.name) if model else None
    return detail


@router.post("/results/{run_id}/cancel", response_model=Ack)
async def cancel_run(run_id: str) -> Ack:
    await runner.cancel(run_id)
    return Ack(ok=True, message="Cancellation requested", id=run_id)


@router.delete("/results/{run_id}", response_model=Ack)
def delete_run(run_id: str, session: SessionDep) -> Ack:
    run = get_or_404(session, BenchmarkRun, run_id, "Benchmark run")
    session.delete(run)
    return Ack(ok=True, message="Benchmark run deleted", id=run_id)
