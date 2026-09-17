"""Hardware provider interface.

``NvmlHardwareProvider`` reads real counters through pynvml/psutil;
``SimulatedHardwareProvider`` produces plausible telemetry for demo mode and is
always flagged ``provenance="simulated"`` so the UI can badge it.
"""

from __future__ import annotations

import abc

from app.schemas.hardware import HardwareSnapshot


class HardwareProvider(abc.ABC):
    name: str = "base"
    provenance: str = "simulated"

    @abc.abstractmethod
    def snapshot(self) -> HardwareSnapshot:
        """Return one telemetry sample."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    @property
    def gpu_available(self) -> bool:
        return bool(self.snapshot().gpus)
