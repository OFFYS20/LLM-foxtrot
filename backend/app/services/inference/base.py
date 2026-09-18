"""Inference adapter abstraction.

Every engine (demo, transformers, llama.cpp, vLLM, Ollama) implements the same
small async interface, so the API layer and the UI never learn which engine is
in use — only what it reports about itself.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.schemas.chat import ChatMessage, SamplingParams


@dataclass(slots=True)
class GenerationChunk:
    text: str
    index: int = 0
    finish_reason: str | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    finish_reason: str = "stop"
    provenance: str = "simulated"
    engine: str = "demo"
    memory_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class EngineInfo:
    engine: str
    available: bool
    provenance: str
    loaded_model: str | None = None
    device: str | None = None
    detail: str | None = None
    supports_streaming: bool = True


class InferenceAdapter(abc.ABC):
    """Contract implemented by every inference engine."""

    engine: str = "base"
    provenance: str = "simulated"

    def __init__(self, model_record: Any | None = None, **options: Any) -> None:
        self.model_record = model_record
        self.options = options
        self._loaded = False

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def is_available(cls) -> bool:
        """Whether this engine can run in the current environment."""
        return False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ----------------------------------------------------------- generation
    @abc.abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> AsyncIterator[GenerationChunk]:
        """Yield chunks as they are produced. Must be an async generator."""

    async def complete(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        """Non-streaming convenience built on :meth:`stream`."""
        import time

        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        finish = "stop"

        async for chunk in self.stream(messages, params):
            if chunk.text and first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(chunk.text)
            if chunk.finish_reason:
                finish = chunk.finish_reason

        text = "".join(pieces)
        elapsed = time.perf_counter() - started
        completion_tokens = self.count_tokens(text)
        prompt_tokens = sum(self.count_tokens(m.content) for m in messages)
        gen_seconds = max(1e-6, elapsed - ((first_token_at or started) - started))

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_to_first_token_ms=((first_token_at or started) - started) * 1000,
            latency_ms=elapsed * 1000,
            tokens_per_sec=completion_tokens / gen_seconds,
            finish_reason=finish,
            provenance=self.provenance,
            engine=self.engine,
            memory_mb=self.memory_usage_mb(),
        )

    # ---------------------------------------------------------------- utils
    def count_tokens(self, text: str) -> int:
        """Estimate token count. Engines with a real tokenizer override this."""
        if not text:
            return 0
        # ~4 characters per token is the usual BPE ballpark for English prose.
        return max(1, round(len(text) / 4))

    def token_count_method(self) -> str:
        return "estimated"

    def memory_usage_mb(self) -> float | None:
        return None

    def info(self) -> EngineInfo:
        return EngineInfo(
            engine=self.engine,
            available=self.is_available(),
            provenance=self.provenance,
            loaded_model=getattr(self.model_record, "name", None),
            detail=None,
        )

    @staticmethod
    def render_prompt(messages: list[ChatMessage]) -> str:
        """Generic chat-ML style rendering used by engines without a template."""
        lines = []
        for message in messages:
            lines.append(f"<|{message.role}|>\n{message.content}")
        lines.append("<|assistant|>\n")
        return "\n".join(lines)
