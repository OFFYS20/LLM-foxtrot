"""Background sampler: keeps a rolling history and publishes to the bus."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from app.config import settings
from app.core.events import Topics, bus
from app.core.logging import get_logger
from app.schemas.hardware import (
    HardwareHistory,
    HardwareHistoryPoint,
    HardwareSnapshot,
)
from app.services.hardware.base import HardwareProvider
from app.services.hardware.nvml_provider import NvmlHardwareProvider, nvml_available
from app.services.hardware.simulated import SimulatedHardwareProvider

logger = get_logger("foxtrot.hardware")


def build_provider() -> HardwareProvider:
    choice = settings.hardware_provider
    if choice == "simulated":
        return SimulatedHardwareProvider()
    if choice == "nvml":
        return NvmlHardwareProvider()
    # auto
    if nvml_available():
        return NvmlHardwareProvider()
    if settings.demo_mode:
        logger.info("No NVML device found — using simulated telemetry (demo mode).")
        return SimulatedHardwareProvider()
    logger.info("No NVML device found — reporting CPU-only telemetry.")
    return NvmlHardwareProvider()


class HardwareMonitor:
    def __init__(self, provider: HardwareProvider | None = None) -> None:
        self.provider = provider or build_provider()
        self._history: deque[HardwareHistoryPoint] = deque(maxlen=settings.hardware_history_points)
        self._task: asyncio.Task[None] | None = None
        self._latest: HardwareSnapshot | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="hardware-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        self.provider.close()

    async def _loop(self) -> None:
        while True:
            try:
                snapshot = self.sample()
                bus.publish(Topics.HARDWARE, snapshot.model_dump(mode="json"))
            except Exception as exc:  # pragma: no cover - monitor must not die
                logger.warning("hardware sample failed: %s", exc)
            await asyncio.sleep(settings.hardware_sample_interval_s)

    # --------------------------------------------------------------- access
    def sample(self) -> HardwareSnapshot:
        snapshot = self.provider.snapshot()
        self._latest = snapshot
        self._history.append(
            HardwareHistoryPoint(
                ts=snapshot.ts,
                gpu_utilization=[g.utilization for g in snapshot.gpus],
                gpu_memory_used_mb=[g.memory_used_mb for g in snapshot.gpus],
                gpu_temperature_c=[g.temperature_c or 0.0 for g in snapshot.gpus],
                gpu_power_w=[g.power_draw_w or 0.0 for g in snapshot.gpus],
                cpu_percent=snapshot.cpu_percent,
                ram_used_gb=snapshot.ram_used_gb,
            )
        )
        return snapshot

    def latest(self) -> HardwareSnapshot:
        return self._latest or self.sample()

    def history(self, limit: int | None = None) -> HardwareHistory:
        points = list(self._history)
        if limit:
            points = points[-limit:]
        return HardwareHistory(
            points=points,
            provenance=self.provider.provenance,  # type: ignore[arg-type]
            interval_seconds=settings.hardware_sample_interval_s,
        )

    def set_load_bias(self, bias: float) -> None:
        """Let training/benchmark runs influence simulated telemetry."""
        setter = getattr(self.provider, "set_load_bias", None)
        if callable(setter):
            setter(bias)

    def describe(self) -> dict[str, Any]:
        snapshot = self.latest()
        return {
            "provider": self.provider.name,
            "provenance": self.provider.provenance,
            "gpu_available": snapshot.gpu_available,
            "compute_mode": snapshot.compute_mode,
            "gpu_count": len(snapshot.gpus),
        }


monitor = HardwareMonitor()
