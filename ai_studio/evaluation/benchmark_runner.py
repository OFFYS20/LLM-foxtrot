"""Benchmark execution.

Runs a suite against a model, grades each item with the configured evaluator,
and stores every prompt/response pair so results can be inspected. A run that
does not complete stores its error — no score is ever invented.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_studio.core import logging as log
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import NotFoundError, StudioError, ValidationError
from ai_studio.evaluation.evaluators import evaluate
from ai_studio.evaluation.suites import BenchmarkItem, SuiteInfo, load_custom_tests, load_suite
from ai_studio.inference.generator import GenerationSettings, generate
from ai_studio.inference.model_loader import cache


@dataclass
class BenchmarkConfig:
    suite: str = "mmlu"
    num_examples: int = 20
    few_shot: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 256
    seed: int = 42
    allow_download: bool = True
    judge_model_id: str | None = None
    similarity_threshold: float = 0.7
    custom_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BenchmarkProgress:
    completed: int = 0
    total: int = 0
    correct: int = 0
    running: bool = False
    finished: bool = False
    error: str | None = None
    current: str = ""

    @property
    def accuracy(self) -> float:
        return self.correct / self.completed if self.completed else 0.0


def build_prompt(item: BenchmarkItem, shots: list[BenchmarkItem]) -> str:
    """Few-shot prompt. Deterministic so runs are reproducible."""
    def render(entry: BenchmarkItem, include_answer: bool) -> str:
        block = entry.context + "\n\n" if entry.context else ""
        block += entry.question
        if entry.choices:
            block += "\n" + "\n".join(
                f"{chr(65 + index)}. {choice}" for index, choice in enumerate(entry.choices)
            )
        block += "\nAnswer:"
        if include_answer:
            block += f" {entry.expected}"
        return block

    parts = [render(shot, True) for shot in shots]
    parts.append(render(item, False))
    return "\n\n".join(parts)


class BenchmarkRunner:
    """Runs benchmarks on a worker thread and records every item."""

    def __init__(self) -> None:
        self._progress: dict[str, BenchmarkProgress] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def create_run(self, *, model_id: str, config: BenchmarkConfig, name: str | None = None) -> dict[str, Any]:
        db = get_db()
        model = db.require("models", model_id)
        record = {
            "id": new_id("bench"),
            "name": name or f"{config.suite} · {model['name']}",
            "suite": config.suite,
            "model_id": model_id,
            "status": "queued",
            "config": config.to_dict(),
            "total_items": config.num_examples,
            "completed_items": 0,
            "created_at": time.time(),
        }
        db.insert("benchmark_runs", record)
        return db.require("benchmark_runs", record["id"])

    def start(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._threads and self._threads[run_id].is_alive():
                raise ValidationError("This benchmark is already running.")
            self._cancel[run_id] = threading.Event()
            self._progress[run_id] = BenchmarkProgress(running=True)
            thread = threading.Thread(target=self._run, args=(run_id,), daemon=True, name=f"bench-{run_id}")
            self._threads[run_id] = thread
            thread.start()

    def cancel(self, run_id: str) -> None:
        event = self._cancel.get(run_id)
        if event:
            event.set()

    def progress(self, run_id: str) -> BenchmarkProgress:
        return self._progress.get(run_id, BenchmarkProgress())

    def _run(self, run_id: str) -> None:
        db = get_db()
        cancel = self._cancel[run_id]
        progress = self._progress[run_id]
        started = time.perf_counter()

        try:
            run = db.require("benchmark_runs", run_id)
            config = BenchmarkConfig(**{**BenchmarkConfig().to_dict(), **(run.get("config") or {})})
            model = db.require("models", run["model_id"])

            if config.custom_path:
                items = load_custom_tests(config.custom_path)[: config.num_examples]
                info = SuiteInfo(
                    key="custom", label="Custom tests", category="custom", method="exact_match",
                    description="User-provided tests", source="local", available=len(items),
                    note=f"Loaded from {config.custom_path}",
                )
                shots: list[BenchmarkItem] = []
            else:
                pool, info = load_suite(
                    config.suite,
                    limit=config.num_examples + config.few_shot,
                    allow_download=config.allow_download,
                )
                shots = pool[: config.few_shot] if config.few_shot else []
                items = pool[config.few_shot : config.few_shot + config.num_examples]

            if not items:
                raise StudioError(f"No items available for suite {config.suite!r}.")

            progress.total = len(items)
            db.update(
                "benchmark_runs",
                run_id,
                {
                    "status": "running",
                    "started_at": time.time(),
                    "total_items": len(items),
                    "config": {**config.to_dict(), "source": info.source, "source_note": info.note},
                },
            )
            log.info(
                f"Benchmark {info.label} on {model['name']}: {len(items)} items (source={info.source})",
                source="evaluation",
                context={"run_id": run_id},
            )

            loaded = cache.load(run["model_id"])
            settings = GenerationSettings(
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                top_p=1.0,
                top_k=0,
                seed=config.seed,
            )

            latencies: list[float] = []
            token_rates: list[float] = []
            scores: list[float] = []
            correct = 0
            total_tokens = 0
            by_category: dict[str, list[float]] = {}

            for index, item in enumerate(items):
                if cancel.is_set():
                    db.update("benchmark_runs", run_id, {"status": "cancelled", "ended_at": time.time()})
                    progress.running, progress.finished = False, True
                    return

                prompt = build_prompt(item, shots)
                progress.current = item.question[:80]
                response, stats, error = "", None, None
                try:
                    response, stats = generate(prompt, loaded=loaded, settings=settings)
                except StudioError as exc:
                    error = exc.message
                except Exception as exc:  # noqa: BLE001 - one bad item must not kill the run
                    error = f"{type(exc).__name__}: {exc}"

                if error:
                    score = type("S", (), {"score": 0.0, "correct": False, "method": "error", "detail": error})()
                else:
                    score = evaluate(
                        item.method,
                        response,
                        item.expected,
                        choices=item.choices,
                        question=item.question,
                        judge_model_id=config.judge_model_id,
                        threshold=config.similarity_threshold,
                    )

                latency = stats.generation_seconds * 1000 if stats else 0.0
                latencies.append(latency)
                if stats and stats.tokens_per_second:
                    token_rates.append(stats.tokens_per_second)
                total_tokens += stats.total_tokens if stats else 0
                scores.append(float(score.score))
                correct += int(bool(score.correct))
                by_category.setdefault(item.category or info.category, []).append(float(score.score))

                db.insert(
                    "benchmark_items",
                    {
                        "run_id": run_id,
                        "idx": index,
                        "category": item.category or info.category,
                        "prompt": prompt,
                        "question": item.question,
                        "expected": item.expected,
                        "response": response if not error else f"[error] {error}",
                        "correct": int(bool(score.correct)),
                        "score": float(score.score),
                        "method": score.method,
                        "latency_ms": latency,
                        "tokens": stats.completion_tokens if stats else 0,
                        "meta": {"detail": getattr(score, "detail", None), "error": error},
                    },
                )
                progress.completed = index + 1
                progress.correct = correct
                db.update("benchmark_runs", run_id, {"completed_items": index + 1})

            runtime = time.perf_counter() - started
            db.update(
                "benchmark_runs",
                run_id,
                {
                    "status": "completed",
                    "ended_at": time.time(),
                    "accuracy": correct / len(items),
                    "score": sum(scores) / len(scores),
                    "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                    "tokens_per_sec": sum(token_rates) / len(token_rates) if token_rates else None,
                    "total_tokens": total_tokens,
                    "runtime_seconds": runtime,
                    "category_scores": {
                        category: round(sum(values) / len(values), 4)
                        for category, values in by_category.items()
                    },
                },
            )
            progress.running, progress.finished = False, True
            log.info(
                f"Benchmark {run_id} finished: {correct}/{len(items)} correct in {runtime:.1f}s",
                source="evaluation",
            )

        except StudioError as exc:
            progress.running, progress.finished, progress.error = False, True, exc.message
            db.update("benchmark_runs", run_id, {"status": "failed", "error": exc.display(), "ended_at": time.time()})
            log.error(f"Benchmark {run_id} failed: {exc.message}", source="evaluation")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            progress.running, progress.finished, progress.error = False, True, message
            db.update("benchmark_runs", run_id, {"status": "failed", "error": message, "ended_at": time.time()})
            log.exception("Benchmark crashed", exc, source="evaluation")


def get_results(run_id: str, *, only_incorrect: bool = False, limit: int = 500) -> dict[str, Any]:
    db = get_db()
    run = db.require("benchmark_runs", run_id)
    where = "run_id = ?" + (" AND correct = 0" if only_incorrect else "")
    items = db.list("benchmark_items", where=where, params=(run_id,), order_by="idx ASC", limit=limit)
    return {"run": run, "items": items}


def list_runs(model_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    db = get_db()
    if model_id:
        return db.list("benchmark_runs", where="model_id = ?", params=(model_id,),
                       order_by="created_at DESC", limit=limit)
    return db.list("benchmark_runs", order_by="created_at DESC", limit=limit)


def compare_models(model_ids: list[str]) -> dict[str, Any]:
    """Side-by-side comparison. Only real, completed runs contribute scores."""
    db = get_db()
    rows: list[dict[str, Any]] = []
    suites: set[str] = set()

    for model_id in model_ids:
        model = db.require("models", model_id)
        runs = db.list(
            "benchmark_runs",
            where="model_id = ? AND status = 'completed'",
            params=(model_id,),
            order_by="created_at DESC",
        )
        best_by_suite: dict[str, dict[str, Any]] = {}
        for run in runs:
            if run["suite"] not in best_by_suite:
                best_by_suite[run["suite"]] = run
                suites.add(run["suite"])

        experiments = db.list(
            "experiments",
            where="(model_id = ? OR output_model_id = ?) AND best_val_loss IS NOT NULL",
            params=(model_id, model_id),
            order_by="created_at DESC",
            limit=1,
        )
        rows.append(
            {
                "model_id": model_id,
                "name": model["name"],
                "parameters": model.get("parameters"),
                "size_bytes": model.get("size_bytes"),
                "context_length": model.get("context_length"),
                "precision": model.get("precision"),
                "quantization": model.get("quantization"),
                "architecture": model.get("architecture"),
                "validation_loss": experiments[0]["best_val_loss"] if experiments else None,
                "benchmarks": {
                    suite: {
                        "accuracy": run.get("accuracy"),
                        "score": run.get("score"),
                        "source": (run.get("config") or {}).get("source", "unknown"),
                        "items": run.get("total_items"),
                        "run_id": run["id"],
                    }
                    for suite, run in best_by_suite.items()
                },
                "tokens_per_sec": max(
                    (run.get("tokens_per_sec") or 0 for run in runs), default=None
                ) or None,
            }
        )

    return {"rows": rows, "suites": sorted(suites)}


runner = BenchmarkRunner()
