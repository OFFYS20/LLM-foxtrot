"""Learning-rate schedules shared by the simulated and real trainers."""

from __future__ import annotations

import math

from app.db.models.enums import LRScheduler


def lr_at_step(
    *,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    scheduler: LRScheduler | str,
    num_cycles: float = 0.5,
    power: float = 1.0,
    min_lr_ratio: float = 0.0,
) -> float:
    """Return the learning rate for ``step`` (1-indexed)."""
    scheduler = LRScheduler(scheduler)
    if total_steps <= 0:
        return base_lr

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    if scheduler == LRScheduler.CONSTANT or scheduler == LRScheduler.CONSTANT_WARMUP:
        return base_lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    floor = base_lr * min_lr_ratio

    if scheduler == LRScheduler.LINEAR:
        return max(floor, base_lr * (1.0 - progress))
    if scheduler == LRScheduler.COSINE:
        return max(floor, base_lr * 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress)))
    if scheduler == LRScheduler.COSINE_RESTARTS:
        cycles = max(1, int(num_cycles * 4))
        local = (progress * cycles) % 1.0
        return max(floor, base_lr * 0.5 * (1.0 + math.cos(math.pi * local)))
    if scheduler == LRScheduler.POLYNOMIAL:
        return max(floor, base_lr * ((1.0 - progress) ** power))
    return base_lr


def estimate_total_steps(
    *, dataset_rows: int, epochs: float, batch_size: int, grad_accum: int
) -> int:
    effective = max(1, batch_size * grad_accum)
    return max(1, math.ceil((dataset_rows * epochs) / effective))


def estimate_tokens(*, dataset_tokens: int, epochs: float) -> int:
    return int(dataset_tokens * epochs)
