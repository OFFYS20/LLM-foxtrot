"""Model comparison.

Deliberately does not collapse a model into a single number: each metric is
reported on its own, with provenance, and missing measurements stay missing
rather than being invented.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.core.errors import NotFoundError
from app.db.models.catalog import Model
from app.db.models.enums import JobStatus
from app.db.models.evaluation import BenchmarkRun
from app.db.models.training import Experiment
from app.db.session import session_scope
from app.schemas.benchmarks import ComparisonMetricRow, ModelComparisonResponse


def compare_models(model_ids: list[str], name: str | None = None) -> ModelComparisonResponse:
    rows: list[ComparisonMetricRow] = []
    categories: set[str] = set()

    with session_scope() as session:
        for model_id in model_ids:
            model = session.get(Model, model_id)
            if model is None:
                raise NotFoundError(f"Model {model_id} not found")

            runs = session.scalars(
                select(BenchmarkRun)
                .where(
                    BenchmarkRun.model_id == model_id,
                    BenchmarkRun.status == JobStatus.COMPLETED,
                )
                .order_by(BenchmarkRun.created_at.desc())
            ).all()

            breakdown: dict[str, float] = {}
            for run in runs:
                if run.overall_score is None:
                    continue
                breakdown.setdefault(
                    run.suite_label or run.suite, round(run.overall_score * 100, 2)
                )
                for category in run.category_scores or {}:
                    categories.add(category)
            categories.update(breakdown.keys())

            scored = [r.overall_score for r in runs if r.overall_score is not None]
            benchmark_score = round(sum(scored) / len(scored) * 100, 2) if scored else None

            experiment = session.scalars(
                select(Experiment)
                .where(Experiment.model_id == model_id, Experiment.final_val_loss.isnot(None))
                .order_by(Experiment.created_at.desc())
                .limit(1)
            ).first()

            throughput = (model.metrics or {}).get("inference_tokens_per_sec")
            vram = model.vram_estimate_mb

            measured = []
            if scored and not model.is_demo:
                measured.append("benchmark_score")
            if experiment and not experiment.is_demo:
                measured.append("validation_loss")

            rows.append(
                ComparisonMetricRow(
                    model_id=model.id,
                    model_name=model.display_name or model.name,
                    label=(model.tags or [None])[0] if model.tags else None,
                    parameters=model.parameters,
                    size_bytes=model.size_bytes,
                    context_length=model.context_length,
                    precision=str(model.precision),
                    benchmark_score=benchmark_score,
                    validation_loss=experiment.final_val_loss if experiment else None,
                    inference_tokens_per_sec=throughput,
                    vram_usage_mb=vram,
                    benchmark_breakdown=breakdown,
                    measured_metrics=measured,
                    provenance="simulated" if model.is_demo else "measured",
                )
            )

    return ModelComparisonResponse(
        name=name or "Model comparison",
        rows=rows,
        categories=sorted(categories),
        created_at=datetime.now(timezone.utc),
    )
