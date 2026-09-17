"""Adapters for engines that live outside this process.

* ``OllamaAdapter``  — fully implemented against the Ollama HTTP API.
* ``LlamaCppAdapter`` — implemented against a llama.cpp ``server`` endpoint.
* ``VLLMAdapter``     — implemented against a vLLM OpenAI-compatible endpoint.

Each one reports ``available=False`` with an actionable message when its
endpoint cannot be reached, so the UI can show "engine unavailable" rather than
silently falling back to synthetic output.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import settings
from app.core.errors import EngineUnavailableError
from app.core.logging import get_logger
from app.schemas.chat import ChatMessage, SamplingParams
from app.services.inference.base import EngineInfo, GenerationChunk, InferenceAdapter

logger = get_logger("foxtrot.inference.remote")


class _HttpAdapter(InferenceAdapter):
    base_url: str = ""
    health_path: str = "/"

    def __init__(self, model_record=None, **options: Any) -> None:
        super().__init__(model_record, **options)
        self.base_url = options.get("base_url") or self.base_url
        self._timeout = httpx.Timeout(options.get("timeout", 300.0), connect=5.0)

    @classmethod
    def _probe(cls, url: str) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                return client.get(url).status_code < 500
        except Exception:
            return False

    def _model_name(self) -> str:
        record = self.model_record
        return (
            getattr(record, "repo_id", None)
            or getattr(record, "local_path", None)
            or getattr(record, "name", None)
            or "unknown"
        )


class OllamaAdapter(_HttpAdapter):
    engine = "ollama"
    provenance = "measured"
    base_url = settings.ollama_base_url

    @classmethod
    def is_available(cls) -> bool:
        return cls._probe(f"{settings.ollama_base_url}/api/tags")

    async def load(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(
                f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve`."
            )
        self._loaded = True

    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:
        payload = {
            "model": self._model_name(),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
                "repeat_penalty": params.repetition_penalty,
                **({"seed": params.seed} if params.seed is not None else {}),
                **({"stop": params.stop} if params.stop else {}),
            },
        }
        index = 0
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:400]
                    raise EngineUnavailableError(f"Ollama error {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (data.get("message") or {}).get("content", "")
                    if piece:
                        yield GenerationChunk(text=piece, index=index)
                        index += 1
                    if data.get("done"):
                        yield GenerationChunk(
                            text="",
                            index=index,
                            finish_reason=data.get("done_reason") or "stop",
                        )
                        return

    def info(self) -> EngineInfo:
        ok = self.is_available()
        return EngineInfo(
            engine=self.engine,
            available=ok,
            provenance="measured",
            loaded_model=self._model_name(),
            device="ollama",
            detail=None if ok else f"No Ollama server at {self.base_url}",
        )


class LlamaCppAdapter(_HttpAdapter):
    """Targets a running ``llama-server`` (llama.cpp) OpenAI-compatible endpoint."""

    engine = "llamacpp"
    provenance = "measured"
    base_url = "http://localhost:8080"

    @classmethod
    def is_available(cls) -> bool:
        return cls._probe("http://localhost:8080/health")

    async def load(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(
                "No llama.cpp server found at http://localhost:8080. Start one with: "
                "`llama-server -m model.gguf --port 8080`"
            )
        self._loaded = True

    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:
        async for chunk in _openai_sse_stream(
            f"{self.base_url}/v1/chat/completions",
            self._model_name(),
            messages,
            params,
            self._timeout,
        ):
            yield chunk

    def info(self) -> EngineInfo:
        ok = self.is_available()
        return EngineInfo(
            engine=self.engine,
            available=ok,
            provenance="measured",
            loaded_model=self._model_name(),
            device="llama.cpp",
            detail=None if ok else "llama.cpp server not running on :8080",
        )


class VLLMAdapter(_HttpAdapter):
    """Targets a vLLM OpenAI-compatible server."""

    engine = "vllm"
    provenance = "measured"
    base_url = "http://localhost:8000/v1"

    @classmethod
    def is_available(cls) -> bool:
        return cls._probe("http://localhost:8000/v1/models")

    async def load(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(
                "No vLLM server found. Start one with: "
                "`python -m vllm.entrypoints.openai.api_server --model <path>`"
            )
        self._loaded = True

    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:
        async for chunk in _openai_sse_stream(
            f"{self.base_url}/chat/completions", self._model_name(), messages, params, self._timeout
        ):
            yield chunk

    def info(self) -> EngineInfo:
        ok = self.is_available()
        return EngineInfo(
            engine=self.engine,
            available=ok,
            provenance="measured",
            loaded_model=self._model_name(),
            device="vllm",
            detail=None if ok else "vLLM OpenAI server not reachable",
        )


async def _openai_sse_stream(
    url: str,
    model: str,
    messages: list[ChatMessage],
    params: SamplingParams,
    timeout: httpx.Timeout,
) -> AsyncIterator[GenerationChunk]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": True,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }
    if params.seed is not None:
        payload["seed"] = params.seed
    if params.stop:
        payload["stop"] = params.stop

    index = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")[:400]
                raise EngineUnavailableError(f"Engine error {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    yield GenerationChunk(text="", index=index, finish_reason="stop")
                    return
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (parsed.get("choices") or [{}])[0]
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    yield GenerationChunk(text=piece, index=index)
                    index += 1
                if choice.get("finish_reason"):
                    yield GenerationChunk(
                        text="", index=index, finish_reason=choice["finish_reason"]
                    )
                    return
