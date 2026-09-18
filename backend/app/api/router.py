"""Aggregate API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    benchmarks,
    chat,
    checkpoints,
    datasets,
    evaluations,
    experiments,
    hardware,
    logs,
    models,
    playground,
    projects,
    realtime,
    system,
    training,
)

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(projects.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(chat.router)
api_router.include_router(playground.router)
api_router.include_router(benchmarks.router)
api_router.include_router(evaluations.router)
api_router.include_router(experiments.router)
api_router.include_router(checkpoints.router)
api_router.include_router(hardware.router)
api_router.include_router(logs.router)

# Realtime transports are mounted separately in app.main (no /api prefix on /ws).
realtime_router = realtime.router
