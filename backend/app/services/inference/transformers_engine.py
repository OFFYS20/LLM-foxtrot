"""Hugging Face Transformers inference adapter (real weights).

Imports are deferred so the platform starts without torch installed. CUDA OOM
is translated into :class:`app.core.errors.OutOfMemoryError` and the GPU cache
is released, instead of taking the process down.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from app.core.errors import EngineUnavailableError, OutOfMemoryError
from app.core.logging import get_logger
from app.schemas.chat import ChatMessage, SamplingParams
from app.services.inference.base import EngineInfo, GenerationChunk, InferenceAdapter

logger = get_logger("foxtrot.inference.transformers")

_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def transformers_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


class TransformersAdapter(InferenceAdapter):
    engine = "transformers"
    provenance = "measured"

    def __init__(self, model_record=None, **options: Any) -> None:
        super().__init__(model_record, **options)
        self._model = None
        self._tokenizer = None
        self._device = options.get("device")

    @classmethod
    def is_available(cls) -> bool:
        return transformers_available()

    # ------------------------------------------------------------ lifecycle
    def _model_path(self) -> str:
        record = self.model_record
        path = getattr(record, "local_path", None) or getattr(record, "repo_id", None)
        if not path:
            raise EngineUnavailableError("Model record has neither a local path nor a repo id")
        return path

    async def load(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(
                "transformers/torch are not installed. "
                "Install with: pip install -r requirements-optional.txt"
            )
        await asyncio.to_thread(self._load_blocking)
        self._loaded = True

    def _load_blocking(self) -> None:  # pragma: no cover - requires torch
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self._model_path()
        precision = getattr(self.model_record, "precision", "bf16")
        dtype_name = _DTYPES.get(str(precision), "bfloat16")
        kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, dtype_name)}

        if str(precision) in {"int8", "int4"}:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=str(precision) == "int4",
                    load_in_8bit=str(precision) == "int8",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs.pop("torch_dtype", None)
            except Exception as exc:
                raise EngineUnavailableError(f"{precision} requires bitsandbytes: {exc}") from exc

        device_map = self._device or ("auto" if torch.cuda.is_available() else None)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path, device_map=device_map, **kwargs
            )
            self._model.eval()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise OutOfMemoryError(
                "CUDA ran out of memory while loading the model. Try a lower precision "
                "(INT8/INT4), a smaller model, or free VRAM held by other jobs.",
                details={"model": path},
            ) from exc

    async def unload(self) -> None:  # pragma: no cover - requires torch
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------- generation
    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:  # pragma: no cover - requires torch
        if not self._loaded:
            await self.load()

        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = self._tokenizer, self._model
        assert tokenizer is not None and model is not None

        payload = [{"role": m.role, "content": m.content} for m in messages]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.render_prompt(messages)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if params.seed is not None:
            torch.manual_seed(params.seed)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=params.max_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-4),
            top_p=params.top_p,
            top_k=params.top_k or None,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        error: list[BaseException] = []

        def _worker() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                error.append(
                    OutOfMemoryError(
                        "CUDA out of memory during generation. Reduce max_tokens or "
                        "context length, or unload other models."
                    )
                )
                _ = exc
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        index = 0
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _pump() -> None:
            for piece in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, piece)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield GenerationChunk(text=piece, index=index)
            index += 1

        if error:
            raise error[0]
        yield GenerationChunk(text="", index=index, finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:  # pragma: no cover - requires torch
            return len(self._tokenizer.encode(text))
        return super().count_tokens(text)

    def token_count_method(self) -> str:
        return "tokenizer" if self._tokenizer is not None else "estimated"

    def memory_usage_mb(self) -> float | None:  # pragma: no cover - requires torch
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**2
        except Exception:
            return None
        return None

    def info(self) -> EngineInfo:
        available = self.is_available()
        return EngineInfo(
            engine=self.engine,
            available=available,
            provenance="measured",
            loaded_model=getattr(self.model_record, "name", None),
            device=str(getattr(self._model, "device", None)) if self._model else None,
            detail=None if available else "torch/transformers not installed",
        )
