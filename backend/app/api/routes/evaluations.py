"""Evaluation overview: leaderboard and side-by-side model comparison."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.catalog import Model
from app.db.models.enums import JobStatus
from app.db.models.evaluation import BenchmarkRun, ModelComparison
from app.schemas.benchmarks import (
    ModelComparisonRequest,
    ModelComparisonResponse,
)
from app.services.evaluation.comparison import compare_models
from app.services.evaluation.registry import registry as suite_registry

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/leaderboard")
def leaderboard(session: SessionDep, suite: str | None = None) -> dict:
    """Per-suite scores by model. Simulated runs are labelled, never merged in silently."""
    statement = select(BenchmarkRun).where(BenchmarkRun.status == JobStatus.COMPLETED)
    if suite:
        statement = statement.where(BenchmarkRun.suite == suite)
    runs = session.scalars(statement.order_by(BenchmarkRun.created_at.desc())).all()

    models = {m.id: m for m in session.scalars(select(Model)).all()}
    rows: dict[str, dict] = {}
    for run in runs:
        model = models.get(run.model_id)
        if model is None or run.overall_score is None:
            continue
        entry = rows.setdefault(
            run.model_id,
            {
                "model_id": model.id,
                "model_name": model.display_name or model.name,
                "parameters": model.parameters,
                "provenance": "simulated" if model.is_demo else "measured",
                "scores": {},
                "runs": 0,
            },
        )
        if run.suite not in entry["scores"]:
            entry["scores"][run.suite] = {
                "score": round(run.overall_score * 100, 2),
                "accuracy": round(run.accuracy * 100, 2) if run.accuracy is not None else None,
                "run_id": run.id,
                "suite_label": run.suite_label,
                "official_split": bool((run.config or {}).get("official_split")),
                "data_source": (run.config or {}).get("data_source", "unknown"),
                "provenance": str(run.provenance),
            }
            entry["runs"] += 1

    return {
        "suites": [info.model_dump() for info in suite_registry.list_info()],
        "rows": list(rows.values()),
        "note": (
            "Scores come only from runs executed on this machine. Suites marked "
            "official_split=false used a bundled sample, not the published benchmark split."
        ),
    }


@router.post("/compare", response_model=ModelComparisonResponse)
def compare(payload: ModelComparisonRequest, session: SessionDep) -> ModelComparisonResponse:
    response = compare_models(payload.model_ids, payload.name)
    if payload.persist:
        record = ModelComparison(
            name=response.name,
            model_ids=payload.model_ids,
            metrics={row.model_id: row.model_dump(mode="json") for row in response.rows},
        )
        session.add(record)
        session.flush()
        response.id = record.id
    return response


@router.get("/comparisons")
def list_saved_comparisons(session: SessionDep) -> list[dict]:
    rows = session.scalars(
        select(ModelComparison).order_by(ModelComparison.created_at.desc()).limit(50)
    ).all()
    return [
        {
            "id": row.id,
            "name": row.name,
            "model_ids": row.model_ids,
            "created_at": row.created_at,
        }
        for row in rows
    ]
