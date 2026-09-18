"""Hardware telemetry schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    index: int
    name: str
    utilization: float = Field(ge=0, le=100)
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    fan_speed_pct: float | None = None
    clock_mhz: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    processes: int = 0

    @property
    def memory_pct(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb * 100) if self.memory_total_mb else 0.0


class DiskInfo(BaseModel):
    mount: str
    used_gb: float
    total_gb: float
    percent: float


class HardwareSnapshot(BaseModel):
    ts: datetime
    provenance: Literal["measured", "simulated"] = "simulated"
    compute_mode: Literal["cuda", "rocm", "mps", "cpu"] = "cpu"
    gpu_available: bool = False
    gpus: list[GPUInfo] = Field(default_factory=list)
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    cpu_model: str | None = None
    cpu_temperature_c: float | None = None
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0
    disks: list[DiskInfo] = Field(default_factory=list)
    platform: str = ""
    driver_version: str | None = None
    cuda_version: str | None = None
    note: str | None = None


class HardwareHistoryPoint(BaseModel):
    ts: datetime
    gpu_utilization: list[float] = Field(default_factory=list)
    gpu_memory_used_mb: list[float] = Field(default_factory=list)
    gpu_temperature_c: list[float] = Field(default_factory=list)
    gpu_power_w: list[float] = Field(default_factory=list)
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0


class HardwareHistory(BaseModel):
    points: list[HardwareHistoryPoint]
    provenance: Literal["measured", "simulated"] = "simulated"
    interval_seconds: float
