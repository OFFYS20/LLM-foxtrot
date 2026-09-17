"""Benchmark runner.

Runs a suite against a model through the inference abstraction, grades each
item with the suite's adapter, persists per-item results, and streams progress
over the event bus. Runs execute as background tasks — the HTTP request that
starts a benchmark returns immediately.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from app.core.errors import ConflictError, NotFoundError
from app.core.events import Topics, bus
from app.core.logging import get_logger
from app.db.models.catalog import Dataset, Model
from app.db.models.enums import JobStatus, RunProvenance
from app.db.models.evaluation import BenchmarkItem, BenchmarkRun
from app.db.session import session_scope
from app.schemas.benchmarks import BenchmarkRunRequest
from app.schemas.chat import ChatMessage, SamplingParams
from app.services.evaluation.base import BenchmarkAdapter, BenchmarkItemSpec
from app.services.evaluation.registry import registry as suite_registry
from app.services.hardware.monitor import monitor
from app.services.inference.registry import registry as inference_registry
from app.services.logbook import logbook

logger = get_logger("foxtrot.evaluation")

SYSTEM_PROMPT = (
    "You are being evaluated. Answer accurately and concisely. "
    "For multiple-choice questions reply with the option letter."
)


class BenchmarkRunner:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------- creation
    def create_run(self, payload: BenchmarkRunRequest) -> BenchmarkRun:
        with session_scope() as session:
            model = session.get(Model, payload.model_id)
            if model is None:
                raise NotFoundError(f"Model {payload.model_id} not found")

            dataset = session.get(Dataset, payload.dataset_id) if payload.dataset_id else None
            if payload.suite == "custom" and dataset is None:
                raise NotFoundError("A dataset_id is required for the custom suite")

            adapter = suite_registry.get(payload.suite, dataset=dataset)
            total = min(payload.config.num_examples, max(1, adapter.available_items()))

            run = BenchmarkRun(
                name=payload.name or f"{adapter.label} · {model.name}",
                suite=payload.suite,
                suite_label=adapter.label,
                category=str(adapter.category),
                model_id=model.id,
                checkpoint_id=payload.checkpoint_id,
                dataset_id=payload.dataset_id,
                status=JobStatus.QUEUED,
                provenance=RunProvenance.SIMULATED if model.is_demo else RunProvenance.MEASURED,
                config={
                    **payload.config.model_dump(mode="json"),
                    "data_source": adapter.data_source,
                    "official_split": adapter.official,
                    "suite_notes": adapter.notes,
                },
                total_items=total,
                is_demo=model.is_demo,
            )
            session.add(run)
            session.flush()
            session.expunge(run)
            logbook.info(
                f"Queued benchmark {adapter.label} on {model.name} ({total} items)",
                source="benchmark",
                context={"run_id": run.id, "data_source": adapter.data_source},
            )
            return run

    async def start(self, run_id: str) -> None:
        if run_id in self._tasks and not self._tasks[run_id].done():
            raise ConflictError(f"Benchmark run {run_id} is already running")
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id), name=f"bench-{run_id}")

    async def cancel(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return bool(task and not task.done())

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    # ------------------------------------------------------------ execution
    async def _execute(self, run_id: str) -> None:
        started = time.perf_counter()
        try:
            with session_scope() as session:
                run = session.get(BenchmarkRun, run_id)
                if run is None:
                    return
                model = session.get(Model, run.model_id)
                dataset = session.get(Dataset, run.dataset_id) if run.dataset_id else None
                run.status = JobStatus.RUNNING
                run.started_at = datetime.now(timezone.utc)
                config = dict(run.config)
                suite = run.suite
                total_items = run.total_items
                session.expunge(run)
                if model:
                    session.expunge(model)
                if dataset:
                    session.expunge(dataset)

            self._publish(run_id, "running", {"completed": 0, "total": total_items})
            monitor.set_load_bias(0.6)

            adapter = suite_registry.get(suite, dataset=dataset)
            items = adapter.items(limit=total_items, seed=int(config.get("seed", 42)))
            shots = _pick_shots(
                adapter, items, int(config.get("few_shot", 0)), int(config.get("seed", 42))
            )

            engine = await inference_registry.get_adapter(model)
            params = SamplingParams(
                temperature=float(config.get("temperature", 0.0)),
                max_tokens=int(config.get("max_tokens", 256)),
                seed=int(config.get("seed", 42)),
                top_p=1.0,
                top_k=0,
            )

            graded_scores: list[float] = []
            correct_count = 0
            latencies: list[float] = []
            throughputs: list[float] = []
            category_totals: dict[str, list[float]] = {}
            total_tokens = 0
            execution_graded = True

            for position, item in enumerate(items):
                prompt = adapter.build_prompt(item, shots)
                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ]
                try:
                    result = await engine.complete(messages, params)
                    response_text, error = result.text, None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("benchmark item failed: %s", exc)
                    response_text, error = "", f"{type(exc).__name__}: {exc}"
                    result = None

                scored = adapter.score(item, response_text) if result else None
                if scored and scored.method != "execution":
                    execution_graded = False

                latency = result.latency_ms if result else 0.0
                tokens = result.total_tokens if result else 0
                tps = result.tokens_per_sec if result else 0.0
                total_tokens += tokens
                latencies.append(latency)
                if tps:
                    throughputs.append(tps)

                if scored and scored.graded:
                    graded_scores.append(scored.score)
                    correct_count += int(scored.correct)
                    category_totals.setdefault(item.category or str(adapter.category), []).append(
                        scored.score
                    )

                with session_scope() as session:
                    session.add(
                        BenchmarkItem(
                            run_id=run_id,
                            index=position,
                            category=item.category or str(adapter.category),
                            question=item.question
                            if not item.context
                            else f"{item.context}\n\n{item.question}",
                            prompt=prompt,
                            expected=item.expected,
                            response=response_text if not error else f"[error] {error}",
                            raw_output=response_text,
                            correct=bool(scored.correct) if scored else False,
                            score=float(scored.score) if scored else 0.0,
                            latency_ms=latency,
                            tokens_used=tokens,
                        )
                    )
                    run = session.get(BenchmarkRun, run_id)
                    if run:
                        run.completed_items = position + 1

                self._publish(
                    run_id,
                    "progress",
                    {
                        "completed": position + 1,
                        "total": len(items),
                        "correct": correct_count,
                        "last_score": scored.score if scored else 0.0,
                        "latency_ms": latency,
                    },
                )

            runtime = time.perf_counter() - started
            graded = len(graded_scores)
            with session_scope() as session:
                run = session.get(BenchmarkRun, run_id)
                if run is None:
                    return
                run.status = JobStatus.COMPLETED
                run.ended_at = datetime.now(timezone.utc)
                run.runtime_seconds = round(runtime, 2)
                run.overall_score = round(sum(graded_scores) / graded, 4) if graded else None
                run.accuracy = round(correct_count / graded, 4) if graded else None
                run.pass_at_1 = (
                    round(correct_count / graded, 4) if graded and execution_graded else None
                )
                run.avg_latency_ms = (
                    round(sum(latencies) / len(latencies), 2) if latencies else None
                )
                run.avg_tokens_per_sec = (
                    round(sum(throughputs) / len(throughputs), 2) if throughputs else None
                )
                run.total_tokens = total_tokens
                run.category_scores = {
                    key: round(sum(values) / len(values), 4)
                    for key, values in category_totals.items()
                }
                summary = {
                    "overall_score": run.overall_score,
                    "accuracy": run.accuracy,
                    "runtime": run.runtime_seconds,
                }

            logbook.info(
                f"Benchmark {run_id} completed — score {summary['overall_score']}",
                source="benchmark",
                context=summary,
            )
            self._publish(run_id, "completed", summary)

        except asyncio.CancelledError:
            self._fail(run_id, "Cancelled", status=JobStatus.STOPPED)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark run %s failed", run_id)
            self._fail(run_id, f"{type(exc).__name__}: {exc}")
        finally:
            monitor.set_load_bias(0.15)
            self._tasks.pop(run_id, None)

    def _fail(self, run_id: str, error: str, status: JobStatus = JobStatus.FAILED) -> None:
        with session_scope() as session:
            run = session.get(BenchmarkRun, run_id)
            if run is None:
                return
            run.status = status
            run.error = error
            run.ended_at = datetime.now(timezone.utc)
        logbook.error(f"Benchmark {run_id}: {error}", source="benchmark")
        self._publish(run_id, str(status), {"error": error})

    def _publish(self, run_id: str, phase: str, payload: dict[str, Any]) -> None:
        body = {"run_id": run_id, "phase": phase, **payload}
        bus.publish(Topics.BENCHMARK_PROGRESS, body)
        bus.publish(f"{Topics.BENCHMARK_PROGRESS}.{run_id}", body)


def _pick_shots(
    adapter: BenchmarkAdapter, items: list[BenchmarkItemSpec], few_shot: int, seed: int
) -> list[BenchmarkItemSpec]:
    if few_shot <= 0:
        return []
    pool = adapter.items(limit=few_shot * 3 + len(items), seed=seed + 1)
    chosen: list[BenchmarkItemSpec] = []
    used = {item.question for item in items}
    for candidate in pool:
        if candidate.question in used:
            continue
        chosen.append(candidate)
        if len(chosen) >= few_shot:
            break
    return chosen


runner = BenchmarkRunner()
