"""Engine selection + loaded-model cache.

``get_adapter`` is the only thing the API layer calls: it picks the configured
engine, falls back to the demo engine only when demo mode is on (never
silently in production), and keeps one adapter per model id.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.core.errors import EngineUnavailableError, NotFoundError
from app.core.logging import get_logger
from app.services.inference.base import EngineInfo, InferenceAdapter
from app.services.inference.demo import DemoInferenceAdapter
from app.services.inference.remote_engines import (
    LlamaCppAdapter,
    OllamaAdapter,
    VLLMAdapter,
)
from app.services.inference.transformers_engine import TransformersAdapter

logger = get_logger("foxtrot.inference.registry")

ENGINES: dict[str, type[InferenceAdapter]] = {
    "demo": DemoInferenceAdapter,
    "transformers": TransformersAdapter,
    "llamacpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "ollama": OllamaAdapter,
}


class InferenceRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, InferenceAdapter] = {}
        self._lock = asyncio.Lock()

    def resolve_engine(self, requested: str | None = None) -> str:
        name = requested or settings.inference_engine
        if name == "auto":
            for candidate in ("transformers", "ollama", "llamacpp", "vllm"):
                if ENGINES[candidate].is_available():
                    return candidate
            return "demo"
        if name not in ENGINES:
            raise NotFoundError(f"Unknown inference engine {name!r}")
        return name

    async def get_adapter(
        self, model_record: Any, *, engine: str | None = None, **options: Any
    ) -> InferenceAdapter:
        name = self.resolve_engine(engine)
        cls = ENGINES[name]

        if not cls.is_available():
            if settings.demo_mode and name != "demo":
                logger.info("engine %s unavailable — using the demo engine (demo mode)", name)
                cls, name = DemoInferenceAdapter, "demo"
            else:
                raise EngineUnavailableError(
                    f"Inference engine {name!r} is not available on this machine."
                )

        key = f"{name}:{getattr(model_record, 'id', 'anon')}"
        async with self._lock:
            adapter = self._adapters.get(key)
            if adapter is None:
                adapter = cls(model_record, **options)
                await adapter.load()
                self._adapters[key] = adapter
            return adapter

    async def unload(self, model_id: str) -> int:
        removed = 0
        async with self._lock:
            for key in [k for k in self._adapters if k.endswith(f":{model_id}")]:
                await self._adapters.pop(key).unload()
                removed += 1
        return removed

    async def unload_all(self) -> None:
        async with self._lock:
            for adapter in list(self._adapters.values()):
                await adapter.unload()
            self._adapters.clear()

    def loaded_model_ids(self) -> list[str]:
        return [key.split(":", 1)[1] for key in self._adapters]

    def engine_status(self) -> list[EngineInfo]:
        infos: list[EngineInfo] = []
        for name, cls in ENGINES.items():
            try:
                available = cls.is_available()
            except Exception:  # pragma: no cover
                available = False
            infos.append(
                EngineInfo(
                    engine=name,
                    available=available,
                    provenance="simulated" if name == "demo" else "measured",
                    detail=None if available else f"{name} not available in this environment",
                )
            )
        return infos


registry = InferenceRegistry()
