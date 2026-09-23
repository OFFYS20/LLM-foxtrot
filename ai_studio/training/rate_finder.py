"""Finding a learning rate by trying the whole range of them, briefly.

A range test (Smith, "Cyclical Learning Rates for Training Neural Networks",
2015) takes a few dozen optimizer steps while the rate climbs exponentially
from far too small to far too large, and watches the training loss. Too small
and nothing moves; too large and it blows up. Somewhere between, the loss falls
fastest — and a usable rate sits below that.

It costs a few dozen steps, and it spoils the weights it runs on: they have
just been trained at rates up to a hundred times too high. The caller reloads
them before the real lesson.

What it measures is how fast the *training* loss falls, which is not the same
as how well the model does on text it has not seen. That difference is why the
suggestion is taken well below the steepest point rather than at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, RandomSampler

from ai_studio.training.data import collate


@dataclass
class RangeTest:
    rates: list[float]
    losses: list[float]          # smoothed, one per step taken
    steepest: float | None       # where the smoothed loss fell fastest
    lowest: float | None         # where the smoothed loss bottomed out
    suggestion: float | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "steepest": self.steepest,
            "lowest": self.lowest,
            "suggestion": self.suggestion,
            "reason": self.reason,
            "steps": len(self.losses),
        }


def range_test(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device | str,
    start: float = 1e-7,
    end: float = 1e-1,
    steps: int = 60,
    weight_decay: float = 0.01,
    smoothing: float = 0.9,
    seed: int = 42,
    divergence: float = 4.0,
    on_step=None,
) -> RangeTest:
    """Run the test on ``model`` (whose weights it changes) and suggest a rate."""
    device = torch.device(device)
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=start, weight_decay=weight_decay)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate,
        sampler=RandomSampler(dataset, replacement=True, num_samples=steps * batch_size,
                              generator=generator),
    )
    growth = (end / start) ** (1 / max(1, steps - 1))
    # On a GPU that has it, bfloat16 — as the lesson itself will run, and in
    # the memory the preflight check allowed for. Anywhere else, full precision.
    half = device.type == "cuda" and torch.cuda.is_bf16_supported()

    rates: list[float] = []
    losses: list[float] = []
    average, best = 0.0, math.inf
    for index, batch in enumerate(loader):
        rate = start * growth ** index
        for group in optimizer.param_groups:
            group["lr"] = rate
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=half):
            outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        value = float(loss.detach())
        if not math.isfinite(value):
            break
        # An average that forgets slowly, corrected for starting at zero.
        average = smoothing * average + (1 - smoothing) * value
        smoothed = average / (1 - smoothing ** (index + 1))
        rates.append(rate)
        losses.append(smoothed)
        best = min(best, smoothed)
        if on_step:
            on_step(index + 1, steps, rate, smoothed)
        if smoothed > divergence * best:
            break

    return _suggest(rates, losses)


def _suggest(rates: list[float], losses: list[float]) -> RangeTest:
    if len(losses) < 8:
        return RangeTest(rates, losses, None, None, None,
                         "too few steps before the loss blew up to read anything")

    # Where the loss fell fastest, per factor of the rate. The first few steps
    # are skipped: the average is still settling there.
    skip = max(2, len(losses) // 10)
    slopes = [
        (losses[i + 1] - losses[i - 1]) / (math.log(rates[i + 1]) - math.log(rates[i - 1]))
        for i in range(skip, len(losses) - 1)
    ]
    steepest_index = skip + min(range(len(slopes)), key=slopes.__getitem__)
    lowest_index = min(range(len(losses)), key=losses.__getitem__)
    steepest, lowest = rates[steepest_index], rates[lowest_index]

    if losses[lowest_index] > losses[0] * 0.98:
        return RangeTest(rates, losses, steepest, lowest, None,
                         "the loss never fell by more than noise, so there is no range to read")

    # A tenth of the rate at the bottom of the curve is the classic reading;
    # never more than the steepest point, which is already fast for a lesson
    # that runs for epochs rather than a few dozen steps.
    suggestion = min(lowest / 10, steepest)
    return RangeTest(rates, losses, steepest, lowest, suggestion,
                     f"the loss fell fastest at {steepest:.1e} and bottomed out at "
                     f"{lowest:.1e}; a tenth of that is {lowest / 10:.1e}")
