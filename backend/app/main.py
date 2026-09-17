"""Foxtrot API application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router, realtime_router
from app.config import settings
from app.core.errors import FoxtrotError
from app.core.events import bus
from app.core.logging import configure_logging, get_logger
from app.db.session import init_db
from app.services.demo.seed import demo_data_present, seed_demo_data
from app.services.evaluation.runner import runner
from app.services.hardware.monitor import monitor
from app.services.inference.registry import registry as inference_registry
from app.services.logbook import logbook
from app.services.training.manager import manager

logger = get_logger("foxtrot.main")

DESCRIPTION = """
Foxtrot — a local LLM experimentation platform: model catalog, dataset manager,
training jobs, chat + playground inference, benchmark runner, experiment
tracking and hardware telemetry.

**Provenance:** every run carries a `provenance` field — `measured` (produced by
a real engine on this machine) or `simulated` (demo mode). Nothing simulated is
ever reported as a measured result.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_directories()
    init_db()
    bus.bind_loop(asyncio.get_running_loop())

    manager.recover_orphans()
    await monitor.start()

    snapshot = monitor.latest()
    logbook.info(
        f"{settings.app_name} {settings.version} started "
        f"(demo_mode={settings.demo_mode}, compute={snapshot.compute_mode}, "
        f"engine={inference_registry.resolve_engine()})",
        source="system",
    )
    if not snapshot.gpu_available:
        logbook.warn(
            "No GPU detected — the platform is running in CPU mode. Training and "
            "inference will be slow; demo backends are used where enabled.",
            source="system",
        )

    if settings.seed_demo_data and settings.demo_mode and not demo_data_present():
        result = seed_demo_data()
        logbook.info(f"Demo data seeded: {result.get('created')}", source="system")

    try:
        yield
    finally:
        await manager.shutdown()
        await runner.shutdown()
        await inference_registry.unload_all()
        await monitor.stop()
        logger.info("shutdown complete")


app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.version,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FoxtrotError)
async def foxtrot_error_handler(_: Request, exc: FoxtrotError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "request_validation_error",
                "message": "Request payload failed validation",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.get("/health", tags=["system"])
def health() -> dict:
    snapshot = monitor.latest()
    return {
        "status": "ok",
        "version": settings.version,
        "demo_mode": settings.demo_mode,
        "compute_mode": snapshot.compute_mode,
        "gpu_available": snapshot.gpu_available,
        "active_training_jobs": manager.running_job_ids(),
        "event_subscribers": bus.subscriber_count,
    }


# Primary, namespaced mount.
app.include_router(api_router, prefix="/api")
# Convenience aliases so the documented paths (GET /models, POST /chat/completions…)
# work without the prefix too.
app.include_router(api_router, include_in_schema=False)
app.include_router(realtime_router)
app.include_router(realtime_router, prefix="/api", include_in_schema=False)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "api": "/api",
        "websocket": "/ws",
        "events": "/api/events/stream",
    }
