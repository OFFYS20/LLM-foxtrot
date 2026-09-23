"""Loading a model once, and generating from it a piece at a time.

``talk`` used to load the tokenizer and the weights from disk for every reply.
For one question that is invisible; for a benchmark it is the whole model read
twenty times, and for a server it would be every request. So a loaded model is
kept, and reused for as long as the weights on disk are the ones it holds — a
lesson that rewrites them changes the stamp, and the next request loads the
new ones.

Generation reports text as it is decoded, so a caller can stream it, and stops
at the end-of-text token, at any of the caller's stop strings, at the length
limit, or when the caller says to — and says which of those it was.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from teacher.workspace import Model, TeacherError

#: Models held in memory, by folder. One at a time: two models kept side by
#: side double the memory for the benefit of whichever one is not in use.
_HELD: dict[str, "Loaded"] = {}
_HOLDING = threading.Lock()


@dataclass
class Loaded:
    """A model's tokenizer and weights, ready to generate from."""

    model: Model
    tokenizer: object
    network: object
    stamp: str | None
    context: int
    kind: str
    # Generation is not safe to run twice at once on one set of weights.
    busy: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Sampling:
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    seed: int | None = None
    stop: tuple[str, ...] = ()


class _Enough(Exception):
    """Raised inside generation to end it early."""


def load(model: Model, *, keep: bool = True) -> Loaded:
    """The model, loaded — from memory when the weights have not changed since."""
    from teacher import lessons

    stamp = model.weights_stamp()
    key = str(model.path)
    with _HOLDING:
        held = _HELD.get(key)
        if held is not None and held.stamp == stamp:
            return held

    tokenizer = lessons.load_tokenizer(model)
    network = lessons.load_network(model).eval()
    loaded = Loaded(
        model=model, tokenizer=tokenizer, network=network, stamp=stamp,
        context=lessons.context_length(model, network), kind=model.kind(),
    )
    if keep:
        with _HOLDING:
            _HELD.clear()
            _HELD[key] = loaded
    return loaded


def forget() -> None:
    """Let go of any model held in memory — a lesson wants the room."""
    with _HOLDING:
        _HELD.clear()


def encode(loaded: Loaded, prompt: str) -> list[int]:
    """The prompt as token ids, led by the start token when the model has one."""
    ids = loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bos = loaded.tokenizer.bos_token_id
    return ([bos] + ids) if bos is not None else ids


def _cut(text: str, stops: tuple[str, ...]) -> tuple[str, bool]:
    """``text`` up to the first stop string, and whether one was found."""
    found = [text.index(stop) for stop in stops if stop and stop in text]
    return (text[:min(found)], True) if found else (text, False)


def _held_back(text: str, stops: tuple[str, ...]) -> int:
    """How much of the end of ``text`` might be the start of a stop string."""
    longest = 0
    for stop in stops:
        for size in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:size]):
                longest = max(longest, size)
                break
    return longest


def generate(loaded: Loaded, ids: list[int], sampling: Sampling, *,
             on_text=None, cancelled: threading.Event | None = None) -> dict:
    """Generate from ``ids``, calling ``on_text`` with each new piece of text.

    Returns the text, the token counts, why it stopped — "stop" for the
    end-of-text token or a stop string, "length" for the limit, "cancelled"
    when the caller asked — and how long it took.
    """
    tokenizer, network = loaded.tokenizer, loaded.network
    room = loaded.context - len(ids)
    if room <= 0:
        raise TeacherError(
            f"The prompt is {len(ids):,} tokens and this model reads at most "
            f"{loaded.context:,}. Shorten it."
        )
    limit = max(1, min(int(sampling.max_new_tokens), room))
    eos = tokenizer.eos_token_id
    produced: list[int] = []
    state = {"shown": 0, "text": "", "stopped": False}

    def take(token: int) -> None:
        produced.append(int(token))
        if int(token) == eos:
            return
        text = tokenizer.decode(produced, skip_special_tokens=True)
        text, hit = _cut(text, sampling.stop)
        state["text"] = text
        if hit:
            state["stopped"] = True
        # A byte-level tokenizer can split a character across tokens; the
        # half-decoded character shows as U+FFFD until the rest arrives.
        safe = len(text) if hit else len(text) - _held_back(text, sampling.stop)
        while safe > state["shown"] and text[safe - 1] == "�":
            safe -= 1
        if on_text and safe > state["shown"]:
            on_text(text[state["shown"]:safe])
            state["shown"] = safe
        if hit or (cancelled is not None and cancelled.is_set()):
            raise _Enough

    if sampling.seed is not None:
        torch.manual_seed(sampling.seed)

    started = time.time()
    prompt = torch.tensor([ids])
    with loaded.busy, torch.no_grad():
        try:
            if loaded.kind == "studio":
                network.generate(
                    prompt,
                    max_new_tokens=limit,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    repetition_penalty=sampling.repetition_penalty,
                    eos_token_id=eos,
                    streamer=take,
                )
            else:
                network.generate(
                    prompt,
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    max_new_tokens=limit,
                    do_sample=sampling.temperature > 0,
                    temperature=max(sampling.temperature, 1e-5),
                    top_p=sampling.top_p,
                    top_k=sampling.top_k or None,
                    repetition_penalty=sampling.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id or eos,
                    eos_token_id=eos,
                    streamer=_Streamer(take),
                )
        except _Enough:
            pass

    if on_text and not state["stopped"]:
        # Whatever was held back in case it began a stop string, it did not.
        rest = state["text"][state["shown"]:]
        if rest:
            on_text(rest)

    if state["stopped"] or (produced and produced[-1] == eos):
        reason = "stop"
    elif cancelled is not None and cancelled.is_set():
        reason = "cancelled"
    elif len(produced) >= limit:
        reason = "length"
    else:
        reason = "stop"
    elapsed = max(time.time() - started, 1e-6)
    return {
        "text": state["text"],
        "prompt_tokens": len(ids),
        "completion_tokens": len(produced),
        "finish_reason": reason,
        "seconds": elapsed,
        "tokens_per_second": len(produced) / elapsed,
    }


class _Streamer:
    """What Transformers' generate() calls with each token; passes them on.

    The first call carries the prompt, which is not new text.
    """

    def __init__(self, take) -> None:
        self.take = take
        self.prompt_seen = False

    def put(self, value) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        for token in value.reshape(-1).tolist():
            self.take(token)

    def end(self) -> None:
        pass
