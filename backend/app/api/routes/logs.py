"""Log query endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.enums import LogLevel
from app.db.models.system import LogEntry
from app.schemas.common import Ack, Page
from app.schemas.logs import LogEntryRead
from app.services.logbook import logbook

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=Page[LogEntryRead])
def list_logs(
    session: SessionDep,
    level: LogLevel | None = None,
    source: str | None = None,
    job_id: str | None = None,
    search: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> Page[LogEntryRead]:
    limit = max(1, min(limit, 5000))
    statement = select(LogEntry).order_by(LogEntry.ts.desc())
    if level:
        statement = statement.where(LogEntry.level == level)
    if source:
        statement = statement.where(LogEntry.source == source)
    if job_id:
        statement = statement.where(LogEntry.job_id == job_id)
    if search:
        statement = statement.where(LogEntry.message.ilike(f"%{search}%"))

    rows = session.scalars(statement.limit(limit).offset(offset)).all()
    total = session.scalar(select(LogEntry.id).order_by(LogEntry.id.desc()).limit(1)) or 0
    return Page(
        items=[LogEntryRead.model_validate(r) for r in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get("/tail")
def tail_logs(limit: int = 200, job_id: str | None = None) -> dict:
    """In-memory ring buffer — cheap polling fallback for the terminal panel."""
    return {"entries": logbook.tail(limit=max(1, min(limit, 2000)), job_id=job_id)}


@router.get("/sources")
def log_sources(session: SessionDep) -> list[str]:
    rows = session.execute(select(LogEntry.source).distinct()).scalars().all()
    return sorted({*rows, "system", "training", "benchmark", "models", "datasets", "chat"})


@router.delete("", response_model=Ack)
def clear_logs(session: SessionDep) -> Ack:
    deleted = session.query(LogEntry).delete()
    return Ack(ok=True, message=f"Cleared {deleted} log entries")
