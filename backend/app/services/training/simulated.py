"""Simulated training backend (demo mode).

Generates a realistic-looking optimisation trajectory — warmup, noisy descent,
periodic validation, occasional plateaus — at a watchable pace. Every job it
produces is recorded with ``provenance="simulated"`` so no chart, checkpoint or
experiment produced this way can be mistaken for a real run.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any

from app.services.training.base import StepMetrics, TrainingBackend, TrainingContext
from app.services.training.schedules import lr_at_step

STEP_INTERVAL_SECONDS = 0.35  # wall-clock pace of the simulation


class SimulatedTrainingBackend(TrainingBackend):
    name = "simulated"
    provenance = "simulated"

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def run(self, ctx: TrainingContext) -> dict[str, Any]:
        cfg = ctx.config
        rng = random.Random(cfg.seed)
        total_steps = max(1, ctx.total_steps)

        start_loss = 2.6 + rng.uniform(-0.25, 0.35)
        floor_loss = 0.72 + rng.uniform(-0.08, 0.16)
        decay = 3.1 + rng.uniform(-0.5, 0.8)
        overfit_onset = rng.uniform(0.62, 0.9)

        tokens_per_step = max(
            1,
            int(cfg.batch_size * cfg.gradient_accumulation_steps * cfg.max_sequence_length * 0.55),
        )
        base_tps = rng.uniform(11_000, 24_000)

        await ctx.log(
            f"[TRAIN] simulated run — method={cfg.method} steps={total_steps} "
            f"effective_batch={cfg.effective_batch_size} seq_len={cfg.max_sequence_length}",
            "info",
        )
        await ctx.log(
            "[TRAIN] NOTE: simulated backend — numbers are synthetic demo data, not a real run.",
            "warn",
        )

        tokens_processed = ctx.start_step * tokens_per_step
        val_loss: float | None = None
        best_val = math.inf
        last_report = time.perf_counter()

        for step in range(ctx.start_step + 1, total_steps + 1):
            if ctx.should_stop:
                await ctx.log(f"[TRAIN] stop requested at step {step}", "warn")
                return {"stopped_at": step, "loss": _loss_at(step), "val_loss": val_loss}

            await ctx.wait_if_paused()
            await asyncio.sleep(STEP_INTERVAL_SECONDS)

            progress = step / total_steps
            # smooth exponential descent + step noise + a small plateau
            base = floor_loss + (start_loss - floor_loss) * math.exp(-decay * progress)
            plateau = 0.06 * math.sin(progress * math.pi * 3.0)
            noise = rng.gauss(0, 0.045 * (1.25 - progress))
            loss = max(0.05, base + plateau + noise)

            lr = lr_at_step(
                base_lr=cfg.learning_rate,
                step=step,
                total_steps=total_steps,
                warmup_steps=cfg.warmup_steps,
                scheduler=cfg.lr_scheduler,
            )
            grad_norm = max(0.01, rng.gauss(0.85, 0.25) * (1.6 - progress))
            tps = (
                base_tps
                * (0.9 + rng.random() * 0.25)
                * (1.0 if not cfg.gradient_checkpointing else 0.82)
            )
            sps = tps / max(1, cfg.max_sequence_length * 0.55)
            tokens_processed += tokens_per_step

            if step % cfg.eval_every_steps == 0 or step == total_steps:
                gap = 0.04 + max(0.0, progress - overfit_onset) * 1.4
                val_loss = max(0.05, loss + gap + rng.gauss(0, 0.02))
                best_val = min(best_val, val_loss)
                await ctx.log(
                    f"[EVAL ] step {step}/{total_steps}  val_loss: {val_loss:.4f}  "
                    f"best: {best_val:.4f}",
                    "info",
                )

            elapsed_per_step = time.perf_counter() - last_report
            last_report = time.perf_counter()
            eta = (total_steps - step) * max(STEP_INTERVAL_SECONDS, elapsed_per_step)

            await ctx.report(
                StepMetrics(
                    step=step,
                    epoch=round(progress * cfg.epochs, 4),
                    loss=round(loss, 5),
                    learning_rate=lr,
                    grad_norm=round(grad_norm, 4),
                    val_loss=round(val_loss, 5) if val_loss is not None else None,
                    tokens_per_sec=round(tps, 1),
                    samples_per_sec=round(sps, 2),
                    tokens_processed=tokens_processed,
                    gpu_utilization=round(min(99.0, 74 + rng.gauss(0, 6)), 1),
                    vram_used_mb=round(
                        _vram_estimate(cfg.max_sequence_length, cfg.batch_size, cfg.method), 1
                    ),
                    eta_seconds=round(eta, 1),
                )
            )

            if step % cfg.log_every_steps == 0 or step == total_steps:
                vram_gb = (
                    _vram_estimate(cfg.max_sequence_length, cfg.batch_size, cfg.method) / 1024
                )
                await ctx.log(
                    f"[TRAIN] Step {step}/{total_steps}\n"
                    f"  loss: {loss:.3f}\n"
                    f"  lr: {lr:.2e}\n"
                    f"  tokens/sec: {tps:,.0f}\n"
                    f"  gpu_mem: {vram_gb:.1f} / 24 GB",
                    "info",
                )

            save_every = cfg.checkpointing.save_every_steps
            manual = ctx.checkpoint_request.is_set()
            if manual or step % save_every == 0 or step == total_steps:
                ctx.checkpoint_request.clear()
                is_best = val_loss is not None and val_loss <= best_val
                await ctx.checkpoint(step, round(loss, 5), val_loss, is_best)

        await ctx.log(f"[TRAIN] finished {total_steps} steps", "info")
        return {
            "final_loss": round(_loss_at(total_steps), 5),
            "final_val_loss": round(val_loss, 5) if val_loss is not None else None,
            "best_val_loss": round(best_val, 5) if best_val != math.inf else None,
            "tokens_processed": tokens_processed,
        }


def _loss_at(step: int) -> float:
    return max(0.05, 2.4 * math.exp(-0.0012 * step) + 0.7)


def _vram_estimate(seq_len: int, batch_size: int, method: str) -> float:
    """MB — rough, and only ever used by the simulated backend."""
    base = 8_400 if str(method) in {"lora", "qlora"} else 17_600
    activation = seq_len * batch_size * 0.0021 * 1024
    return min(23_900.0, base + activation)
