"""Hardware telemetry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.hardware import HardwareHistory, HardwareSnapshot
from app.services.hardware.monitor import monitor

router = APIRouter(prefix="/hardware", tags=["hardware"])


@router.get("", response_model=HardwareSnapshot)
def get_hardware() -> HardwareSnapshot:
    return monitor.sample()


@router.get("/history", response_model=HardwareHistory)
def get_history(limit: int = 240) -> HardwareHistory:
    return monitor.history(limit=max(1, min(limit, 5000)))


@router.get("/devices")
def get_devices() -> dict:
    snapshot = monitor.latest()
    return {
        "provider": monitor.provider.name,
        "provenance": monitor.provider.provenance,
        "compute_mode": snapshot.compute_mode,
        "gpu_available": snapshot.gpu_available,
        "gpus": [gpu.model_dump() for gpu in snapshot.gpus],
        "cpu": {
            "model": snapshot.cpu_model,
            "cores": snapshot.cpu_cores,
            "percent": snapshot.cpu_percent,
            "temperature_c": snapshot.cpu_temperature_c,
        },
        "memory": {
            "ram_used_gb": snapshot.ram_used_gb,
            "ram_total_gb": snapshot.ram_total_gb,
            "swap_used_gb": snapshot.swap_used_gb,
            "swap_total_gb": snapshot.swap_total_gb,
        },
        "disks": [disk.model_dump() for disk in snapshot.disks],
        "note": snapshot.note,
    }
