"""System info, capabilities and settings."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import settings
from app.schemas.common import Ack, Capability, SystemInfo
from app.services.demo.seed import clear_demo_data, demo_data_present, seed_demo_data
from app.services.evaluation.registry import registry as suite_registry
from app.services.hardware.monitor import monitor
from app.services.inference.registry import registry as inference_registry
from app.services.training.torch_backend import torch_stack_available

router = APIRouter(tags=["system"])


@router.get("/system", response_model=SystemInfo)
def get_system_info() -> SystemInfo:
    snapshot = monitor.latest()
    engines = inference_registry.engine_status()
    capabilities = [
        Capability(
            name="gpu",
            available=snapshot.gpu_available,
            detail=None
            if snapshot.gpu_available
            else "No GPU detected — the platform is operating in CPU mode.",
        ),
        Capability(
            name="training.torch",
            available=torch_stack_available(),
            detail=None
            if torch_stack_available()
            else "PyTorch/Transformers not installed — training runs on the simulated backend.",
        ),
        Capability(
            name="hardware.telemetry",
            available=monitor.provider.provenance == "measured",
            detail=f"provider={monitor.provider.name}",
        ),
        *[
            Capability(
                name=f"inference.{info.engine}", available=info.available, detail=info.detail
            )
            for info in engines
        ],
        Capability(
            name="demo.data",
            available=demo_data_present(),
            detail="Demo records are flagged and can be cleared from Settings.",
        ),
    ]
    return SystemInfo(
        app=settings.app_name,
        version=settings.version,
        demo_mode=settings.demo_mode,
        gpu_available=snapshot.gpu_available,
        compute_mode=snapshot.compute_mode,
        inference_engine=inference_registry.resolve_engine(),
        hardware_provider=monitor.provider.name,
        capabilities=capabilities,
        server_time=datetime.now(timezone.utc),
    )


@router.get("/settings")
def read_settings() -> dict:
    return {
        "demo_mode": settings.demo_mode,
        "inference_engine": settings.inference_engine,
        "hardware_provider": settings.hardware_provider,
        "data_dir": str(settings.data_dir),
        "database_url": _redact(settings.database_url),
        "max_upload_mb": settings.max_upload_mb,
        "hardware_sample_interval_s": settings.hardware_sample_interval_s,
        "cors_origins": settings.cors_origin_list,
        "available_engines": [info.engine for info in inference_registry.engine_status()],
        "engine_status": [asdict(info) for info in inference_registry.engine_status()],
        "benchmark_suites": [info.key for info in suite_registry.list_info()],
        "demo_data_present": demo_data_present(),
    }


@router.post("/settings/demo-data/seed", response_model=Ack)
def seed_demo(force: bool = False) -> Ack:
    result = seed_demo_data(force=force)
    return Ack(
        ok=True,
        message="Demo data already present" if result.get("skipped") else "Demo data seeded",
    )


@router.post("/settings/demo-data/clear", response_model=Ack)
def clear_demo() -> Ack:
    removed = clear_demo_data()
    return Ack(ok=True, message=f"Removed demo records: {removed}")


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    _, host = rest.split("@", 1)
    return f"{scheme}://***@{host}"
