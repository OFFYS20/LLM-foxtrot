"""Demo inference engine.

Produces plausible, deterministic, *clearly labelled* synthetic completions so
the whole UI is usable with no weights on disk. It never claims to be a model:
``provenance`` is ``"simulated"`` and every completion opens with a marker line.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import AsyncIterator

from app.schemas.chat import ChatMessage, SamplingParams
from app.services.inference.base import EngineInfo, GenerationChunk, InferenceAdapter

MARKER = "[demo output — simulated, no weights loaded]"

_OPENERS = [
    "Here is how I would approach that.",
    "Short answer first, then the detail.",
    "Let me break this down step by step.",
    "There are three things worth separating here.",
]

_BODY = [
    "The key constraint is throughput: every token you generate costs roughly the same "
    "amount of memory bandwidth, so batching dominates the cost curve.",
    "In practice you want to measure before tuning — a profile usually shows that the "
    "bottleneck is not where the intuition says it is.",
    "A reasonable default is to start with the smallest configuration that fits, then "
    "scale one dimension at a time and re-measure.",
    "If the validation curve flattens while the training curve keeps dropping, you are "
    "memorising the dataset rather than learning the task.",
    "Rank and alpha interact: doubling the rank without adjusting alpha changes the "
    "effective update scale, which is a common source of unstable runs.",
]

_CLOSERS = [
    "If you want, I can turn that into a concrete configuration.",
    "Worth validating on a held-out split before you commit to it.",
    "That should be enough to get a first baseline.",
]

_CODE_HINT = re.compile(r"\b(code|function|python|implement|script|bug|class|def)\b", re.I)
_MATH_HINT = re.compile(r"\b(math|calculate|sum|solve|equation|integral|probability)\b", re.I)


class DemoInferenceAdapter(InferenceAdapter):
    engine = "demo"
    provenance = "simulated"

    def __init__(self, model_record=None, **options) -> None:
        super().__init__(model_record, **options)
        name = getattr(model_record, "name", "foxtrot-demo")
        params_m = getattr(model_record, "parameters", None) or 7000
        seed_src = f"{name}:{params_m}".encode()
        self._model_seed = int(hashlib.sha256(seed_src).hexdigest()[:8], 16)
        # Larger models "generate" a little slower but more coherently.
        self._base_tps = max(9.0, 260.0 - (params_m / 1000.0) * 14.0)

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def load(self) -> None:
        await asyncio.sleep(0.05)
        self._loaded = True

    def _compose(self, messages: list[ChatMessage], params: SamplingParams) -> str:
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "").strip()
        seed = params.seed if params.seed is not None else self._model_seed
        rng = random.Random(seed ^ (hash(prompt) & 0xFFFFFFFF))

        topic = " ".join(prompt.split()[:12]) or "the request"
        lines = [MARKER, "", f"{rng.choice(_OPENERS)} You asked about: “{topic}”.", ""]

        if _CODE_HINT.search(prompt):
            lines += [
                "```python",
                "def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:",
                '    """Cheap token estimate used when no tokenizer is loaded."""',
                "    return max(1, round(len(text) / chars_per_token))",
                "```",
                "",
            ]
        elif _MATH_HINT.search(prompt):
            lines += [
                "Working it through:",
                "  1. Restate the quantities involved.",
                "  2. Apply the relation step by step.",
                "  3. Sanity-check the magnitude of the result.",
                "",
            ]

        body_count = 1 if params.max_tokens < 128 else (2 if params.max_tokens < 512 else 3)
        picked = rng.sample(_BODY, k=min(body_count, len(_BODY)))
        lines += [f"- {item}" for item in picked]
        lines += ["", rng.choice(_CLOSERS)]
        return "\n".join(lines)

    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:
        text = self._compose(messages, params)
        tokens = _tokenize(text)

        budget = min(len(tokens), max(1, params.max_tokens))
        jitter = random.Random(self._model_seed)
        # Simulated prefill delay before the first token.
        await asyncio.sleep(0.12 + jitter.random() * 0.25)

        emitted = 0
        for index, token in enumerate(tokens[:budget]):
            tps = self._base_tps * (0.85 + jitter.random() * 0.3)
            await asyncio.sleep(min(0.08, 1.0 / max(tps, 1.0)))
            emitted += 1
            yield GenerationChunk(text=token, index=index)

        reason = "length" if emitted < len(tokens) else "stop"
        yield GenerationChunk(text="", index=emitted, finish_reason=reason)

    def memory_usage_mb(self) -> float | None:
        params_m = getattr(self.model_record, "parameters", None) or 7000
        return round(params_m * 0.55, 1)  # rough BF16 weight footprint, simulated

    def info(self) -> EngineInfo:
        return EngineInfo(
            engine=self.engine,
            available=True,
            provenance="simulated",
            loaded_model=getattr(self.model_record, "name", None),
            device="demo",
            detail="Synthetic responses for UI/demo use. No weights are loaded.",
        )


def _tokenize(text: str) -> list[str]:
    """Split into word-ish pieces so streaming looks like real token streaming."""
    return re.findall(r"\s*\S+", text) or [text]


def simulated_latency_profile(parameters_m: int | None, tokens: int) -> tuple[float, float]:
    """Latency/throughput helper shared by the playground demo path."""
    params_m = parameters_m or 7000
    tps = max(9.0, 260.0 - (params_m / 1000.0) * 14.0)
    ttft = 90 + params_m / 220.0
    return ttft, tps * (0.9 + 0.2 * random.random()) if tokens else tps
