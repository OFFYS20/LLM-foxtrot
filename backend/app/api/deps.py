"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, TypeVar

from fastapi import Depends, Query
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db.session import get_session

T = TypeVar("T")

SessionDep = Annotated[Session, Depends(get_session)]


class Pagination:
    def __init__(
        self,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset


PaginationDep = Annotated[Pagination, Depends(Pagination)]


def paginate(session: Session, statement: Select, pagination: Pagination) -> tuple[list[Any], int]:
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    rows = session.scalars(statement.limit(pagination.limit).offset(pagination.offset)).all()
    return list(rows), total


def get_or_404(session: Session, model: type[T], entity_id: str, label: str | None = None) -> T:
    instance = session.get(model, entity_id)
    if instance is None:
        raise NotFoundError(f"{label or model.__name__} {entity_id} not found")
    return instance


def db_session() -> Iterator[Session]:  # pragma: no cover - re-export for clarity
    yield from get_session()
