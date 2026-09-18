"""Training backend contract.

A backend never touches the database or the event bus directly — it reports
through :class:`TrainingContext`, which the job manager owns. That keeps the
simulated backend and the real PyTorch backend interchangeable.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.schemas.training import TrainingConfig


@dataclass(slots=True)
class StepMetrics:
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    val_loss: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    tokens_processed: int = 0
    gpu_utilization: float | None = None
    vram_used_mb: float | None = None
    eta_seconds: float | None = None


@dataclass
class TrainingContext:
    """Control surface + reporting callbacks handed to a backend."""

    job_id: str
    config: TrainingConfig
    total_steps: int
    dataset_rows: int
    dataset_tokens: int
    model_name: str
    dataset_name: str
    start_step: int = 0

    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    checkpoint_request: asyncio.Event = field(default_factory=asyncio.Event)

    on_metric: Callable[[StepMetrics], Awaitable[None]] | None = None
    on_log: Callable[[str, str], Awaitable[None]] | None = None
    on_checkpoint: Callable[[int, float, float | None, bool], Awaitable[None]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pause_event.set()  # "set" == running

    async def report(self, metrics: StepMetrics) -> None:
        if self.on_metric:
            await self.on_metric(metrics)

    async def log(self, message: str, level: str = "info") -> None:
        if self.on_log:
            await self.on_log(message, level)

    async def checkpoint(
        self, step: int, train_loss: float, val_loss: float | None, is_best: bool = False
    ) -> None:
        if self.on_checkpoint:
            await self.on_checkpoint(step, train_loss, val_loss, is_best)

    async def wait_if_paused(self) -> None:
        if not self.pause_event.is_set():
            await self.log("Training paused", "warn")
            await self.pause_event.wait()
            await self.log("Training resumed", "info")

    @property
    def should_stop(self) -> bool:
        return self.stop_event.is_set()


class TrainingBackend(abc.ABC):
    """Implemented by the simulated trainer and the PyTorch trainer."""

    name: str = "base"
    provenance: str = "simulated"

    @classmethod
    def is_available(cls) -> bool:
        return False

    @abc.abstractmethod
    async def run(self, ctx: TrainingContext) -> dict[str, Any]:
        """Run until completion, stop, or failure. Returns a result summary."""

    async def prepare(self, ctx: TrainingContext) -> None:  # pragma: no cover - optional
        return None

    async def cleanup(self, ctx: TrainingContext) -> None:  # pragma: no cover - optional
        return None
