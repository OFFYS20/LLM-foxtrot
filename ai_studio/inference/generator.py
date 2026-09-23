"""Text generation with real token-by-token streaming and honest statistics."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.errors import InferenceError
from ai_studio.inference.model_loader import LoadedModel, cache


@dataclass
class GenerationSettings:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    seed: int | None = None
    stop_sequences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    generation_seconds: float = 0.0
    context_used: int = 0
    context_limit: int = 0
    finish_reason: str = "stop"
    device: str = "cpu"
    model: str = ""

    @property
    def context_percent(self) -> float:
        return (self.context_used / self.context_limit * 100) if self.context_limit else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_percent"] = round(self.context_percent, 1)
        return data

    def summary(self) -> str:
        return (
            f"{self.completion_tokens} tokens · {self.tokens_per_second:.1f} tok/s · "
            f"TTFT {self.time_to_first_token_ms:.0f} ms · "
            f"context {self.context_used}/{self.context_limit} "
            f"({self.context_percent:.0f}%) · {self.finish_reason}"
        )


class StopSignal:
    """Lets the UI interrupt an in-flight generation."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def stop(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def stopped(self) -> bool:
        return self._event.is_set()


def stream_generate(
    prompt: str,
    *,
    model_id: str | None = None,
    loaded: LoadedModel | None = None,
    settings: GenerationSettings | None = None,
    stop_signal: StopSignal | None = None,
) -> Iterator[tuple[str, GenerationStats | None]]:
    """Yield (text_delta, None) per token, then ("", stats) once at the end."""
    settings = settings or GenerationSettings()
    resident = loaded or (cache.load(model_id) if model_id else cache.current)
    if resident is None:
        raise InferenceError("No model is loaded.", hint="Select a model on the Chat screen.")

    import torch

    tokenizer, model = resident.tokenizer, resident.model
    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(resident.device)
    prompt_tokens = int(input_ids.shape[-1])

    limit = resident.context_length or 2048
    if prompt_tokens >= limit:
        raise InferenceError(
            f"The prompt is {prompt_tokens} tokens but the model's context is {limit}.",
            hint="Shorten the conversation, or enable a context strategy in Settings.",
        )
    budget = min(settings.max_new_tokens, max(1, limit - prompt_tokens))

    started = time.perf_counter()
    first_token_at: float | None = None
    produced = 0
    text_so_far = ""
    finish_reason = "stop"

    try:
        if resident.kind == "studio":
            generated = input_ids
            past = None
            current = input_ids
            for _ in range(budget):
                if stop_signal and stop_signal.stopped:
                    finish_reason = "cancelled"
                    break
                window = current[:, -limit:]
                outputs = model(window, past_key_values=past, use_cache=True)
                past = outputs["past_key_values"]
                logits = outputs["logits"][:, -1, :].float()
                next_token = _sample(logits, generated, settings)
                generated = torch.cat([generated, next_token], dim=1)
                current = next_token
                produced += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()

                decoded = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)
                delta, text_so_far = decoded[len(text_so_far) :], decoded
                if delta:
                    yield delta, None
                if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if _hit_stop_sequence(text_so_far, settings.stop_sequences):
                    finish_reason = "stop_sequence"
                    break
            else:
                finish_reason = "length"
        else:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=budget,
                do_sample=settings.temperature > 0,
                temperature=max(settings.temperature, 1e-5),
                top_p=settings.top_p,
                top_k=settings.top_k or None,
                repetition_penalty=settings.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                streamer=streamer,
            )
            errors: list[BaseException] = []

            def _worker() -> None:
                try:
                    with torch.inference_mode():
                        model.generate(**kwargs)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            for piece in streamer:
                if stop_signal and stop_signal.stopped:
                    finish_reason = "cancelled"
                    break
                if not piece:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                produced += max(1, len(tokenizer(piece, add_special_tokens=False)["input_ids"]))
                text_so_far += piece
                yield piece, None
                if _hit_stop_sequence(text_so_far, settings.stop_sequences):
                    finish_reason = "stop_sequence"
                    break
            thread.join(timeout=1.0)
            if errors:
                raise InferenceError(f"Generation failed: {errors[0]}")
            if produced >= budget:
                finish_reason = "length"
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Generation failed", exc, source="inference")
        raise InferenceError(f"Generation failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    generation_time = elapsed - ((first_token_at or started) - started)
    stats = GenerationStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=produced,
        total_tokens=prompt_tokens + produced,
        tokens_per_second=produced / generation_time if generation_time > 0 else 0.0,
        time_to_first_token_ms=((first_token_at or started) - started) * 1000,
        generation_seconds=elapsed,
        context_used=prompt_tokens + produced,
        context_limit=limit,
        finish_reason=finish_reason,
        device=resident.device,
        model=resident.name,
    )
    yield "", stats


def generate(
    prompt: str,
    *,
    model_id: str | None = None,
    loaded: LoadedModel | None = None,
    settings: GenerationSettings | None = None,
) -> tuple[str, GenerationStats]:
    """Non-streaming convenience wrapper."""
    pieces: list[str] = []
    stats = GenerationStats()
    for delta, final in stream_generate(prompt, model_id=model_id, loaded=loaded, settings=settings):
        if final is not None:
            stats = final
        else:
            pieces.append(delta)
    return "".join(pieces), stats


def _sample(logits: Any, generated: Any, settings: GenerationSettings) -> Any:
    import torch
    import torch.nn.functional as F

    if settings.repetition_penalty and settings.repetition_penalty != 1.0:
        from ai_studio.models.transformer import penalise_repeats

        logits = penalise_repeats(logits, generated, settings.repetition_penalty)

    if settings.temperature and settings.temperature > 0:
        logits = logits / max(settings.temperature, 1e-5)
        if settings.top_k:
            kth = torch.topk(logits, min(settings.top_k, logits.size(-1))).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if settings.top_p and 0 < settings.top_p < 1.0:
            ordered, indices = torch.sort(logits, descending=True)
            probabilities = F.softmax(ordered, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1) - probabilities
            ordered = ordered.masked_fill(cumulative > settings.top_p, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, ordered)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _hit_stop_sequence(text: str, stops: list[str]) -> bool:
    return any(stop and stop in text for stop in stops)
