"""The training loop.

Real PyTorch training for both the from-scratch transformer and HuggingFace
causal models. The loop is deliberately explicit — forward, backward, clip,
step, schedule — so what the UI reports is exactly what happened.

Control flow: the caller owns ``TrainingControl`` (pause/stop/checkpoint
events); the loop checks it every optimizer step, so stopping is immediate and
never corrupts a checkpoint in progress.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from ai_studio.core import logging as log
from ai_studio.core.errors import OutOfMemoryError, TrainingError
from ai_studio.training.config import TrainingConfig
from ai_studio.training.data import collate


@dataclass
class TrainingControl:
    """Pause / resume / stop / checkpoint signals shared with the UI."""

    _pause: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)
    _checkpoint: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._pause.set()  # set == running

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()  # release a paused loop so it can exit

    def request_checkpoint(self) -> None:
        self._checkpoint.set()

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused(self, on_pause: Callable[[], None] | None = None) -> None:
        if not self._pause.is_set():
            if on_pause:
                on_pause()
            self._pause.wait()

    def take_checkpoint_request(self) -> bool:
        if self._checkpoint.is_set():
            self._checkpoint.clear()
            return True
        return False


@dataclass
class StepMetrics:
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    val_loss: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    tokens_processed: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None


@dataclass
class TrainingResult:
    status: str                       # completed | stopped | failed
    steps: int
    epochs: float
    final_train_loss: float | None
    best_val_loss: float | None
    tokens_processed: int
    duration_seconds: float
    error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    """Owns one training run."""

    def __init__(
        self,
        *,
        model: Any,
        train_dataset: Any,
        config: TrainingConfig,
        eval_dataset: Any | None = None,
        device: torch.device | str = "cpu",
        control: TrainingControl | None = None,
        on_metrics: Callable[[StepMetrics], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        on_checkpoint: Callable[[int, float, float | None, bool], None] | None = None,
        start_step: int = 0,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.device = torch.device(device)
        self.control = control or TrainingControl()
        self.on_metrics = on_metrics
        self.on_log = on_log
        self.on_checkpoint = on_checkpoint
        self.start_step = start_step

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.best_val_loss: float | None = None
        self.history: list[dict[str, Any]] = []
        self._tokens_seen = 0

    # ---------------------------------------------------------------- setup
    def _log(self, message: str, level: str = "info") -> None:
        if self.on_log:
            self.on_log(message, level)
        else:
            log.log(message, level=level, source="training", persist=False)

    def _resolve_dtype(self) -> tuple[torch.dtype, bool]:
        """Return (autocast dtype, enabled). CPU keeps FP32 — half precision
        there is slow and numerically unreliable."""
        if self.device.type == "cuda":
            if self.config.precision == "bf16" and torch.cuda.is_bf16_supported():
                return torch.bfloat16, True
            if self.config.precision == "fp16":
                return torch.float16, True
        elif self.device.type == "cpu" and self.config.precision == "bf16":
            return torch.bfloat16, False  # supported but usually slower; keep FP32
        return torch.float32, False

    def build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            # Biases and norm weights are conventionally excluded from decay.
            if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        name = self.config.optimizer
        if name == "adamw_8bit":
            try:
                import bitsandbytes as bnb

                return bnb.optim.AdamW8bit(groups, lr=self.config.learning_rate, betas=(0.9, 0.95))
            except ImportError:
                self._log("bitsandbytes not installed — falling back to AdamW.", "warning")
                name = "adamw"
        if name == "adafactor":
            try:
                from transformers.optimization import Adafactor

                return Adafactor(
                    groups, lr=self.config.learning_rate, scale_parameter=False, relative_step=False
                )
            except ImportError:
                self._log("transformers not installed — falling back to AdamW.", "warning")
                name = "adamw"
        if name == "sgd":
            return torch.optim.SGD(groups, lr=self.config.learning_rate, momentum=0.9)
        if name == "adam":
            return torch.optim.Adam(groups, lr=self.config.learning_rate, betas=(0.9, 0.95))
        return torch.optim.AdamW(groups, lr=self.config.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Any:
        warmup = self.config.warmup_steps
        if not warmup and self.config.warmup_ratio:
            warmup = int(total_steps * self.config.warmup_ratio)
        warmup = max(0, min(warmup, max(0, total_steps - 1)))
        kind = self.config.lr_scheduler

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / max(1, warmup)
            if kind == "constant" or kind == "constant_with_warmup":
                return 1.0
            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(1.0, max(0.0, progress))
            if kind == "linear":
                return max(0.0, 1.0 - progress)
            if kind == "polynomial":
                return max(0.0, (1.0 - progress) ** 2)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _total_steps(self, loader_length: int) -> int:
        if self.config.max_steps > 0:
            return self.config.max_steps
        per_epoch = max(1, loader_length // self.config.gradient_accumulation_steps)
        return max(1, int(per_epoch * self.config.epochs))

    # ------------------------------------------------------------- training
    def train(self) -> TrainingResult:
        torch.manual_seed(self.config.seed)
        started = time.time()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        if len(train_loader) == 0:
            raise TrainingError("The training split produced no batches.")

        total_steps = self._total_steps(len(train_loader))
        self.model.to(self.device)
        if self.config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing(True)

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(self.optimizer, total_steps)
        dtype, autocast_enabled = self._resolve_dtype()
        scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled and dtype is torch.float16)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self._log(
            f"[TRAIN] starting — {total_steps} steps, {trainable:,}/{total_params:,} trainable "
            f"parameters, device={self.device.type}, precision="
            f"{'autocast ' + str(dtype).split('.')[-1] if autocast_enabled else 'fp32'}"
        )

        step = self.start_step
        epoch = 0.0
        accumulated = 0
        running_loss = 0.0
        last_report = time.time()
        tokens_window = 0
        samples_window = 0
        status = "completed"
        error: str | None = None
        final_loss: float | None = None

        try:
            self.model.train()
            epochs_to_run = (
                self.config.epochs
                if self.config.max_steps <= 0
                else math.ceil(total_steps * self.config.gradient_accumulation_steps / len(train_loader))
            )
            epoch_index = 0

            while step < total_steps and not self.control.stopping:
                epoch_index += 1
                if self.config.max_steps <= 0 and epoch_index > math.ceil(self.config.epochs):
                    break

                for batch in train_loader:
                    if self.control.stopping:
                        status = "stopped"
                        break
                    self.control.wait_if_paused(
                        on_pause=lambda: self._log("[TRAIN] paused", "warning")
                    )
                    if self.control.stopping:
                        status = "stopped"
                        break

                    batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

                    try:
                        with torch.autocast(
                            device_type=self.device.type, dtype=dtype, enabled=autocast_enabled
                        ):
                            outputs = self.model(**batch)
                            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                            if loss is None:
                                raise TrainingError("The model returned no loss — are labels present?")
                            loss = loss / self.config.gradient_accumulation_steps
                    except torch.cuda.OutOfMemoryError as exc:
                        raise self._oom_error(exc) from exc
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise self._oom_error(exc) from exc
                        raise

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    running_loss += loss.item() * self.config.gradient_accumulation_steps
                    tokens_window += int(batch["input_ids"].numel())
                    samples_window += int(batch["input_ids"].shape[0])
                    accumulated += 1

                    if accumulated < self.config.gradient_accumulation_steps:
                        continue

                    # ---- optimizer step -------------------------------------
                    grad_norm = None
                    if self.config.gradient_clipping and self.config.gradient_clipping > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(self.optimizer)
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                self.config.gradient_clipping,
                            )
                        )

                    if scaler.is_enabled():
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    step += 1
                    accumulated = 0
                    self._tokens_seen += tokens_window
                    step_loss = running_loss / self.config.gradient_accumulation_steps
                    running_loss = 0.0
                    epoch = step * self.config.gradient_accumulation_steps / max(1, len(train_loader))
                    final_loss = step_loss

                    # ---- evaluation -----------------------------------------
                    val_loss = None
                    if self.eval_dataset is not None and step % max(1, self.config.eval_interval) == 0:
                        val_loss = self.evaluate()
                        if val_loss is not None and (
                            self.best_val_loss is None or val_loss < self.best_val_loss
                        ):
                            self.best_val_loss = val_loss
                        self.model.train()

                    # ---- reporting ------------------------------------------
                    now = time.time()
                    window = max(1e-6, now - last_report)
                    metrics = StepMetrics(
                        step=step,
                        epoch=round(epoch, 4),
                        loss=step_loss,
                        learning_rate=float(self.scheduler.get_last_lr()[0]),
                        grad_norm=grad_norm,
                        val_loss=val_loss,
                        tokens_per_sec=tokens_window / window,
                        samples_per_sec=samples_window / window,
                        tokens_processed=self._tokens_seen,
                        elapsed_seconds=now - started,
                        eta_seconds=(
                            (now - started) / max(1, step - self.start_step) * (total_steps - step)
                        ),
                    )
                    self.history.append(metrics.__dict__.copy())
                    if self.on_metrics:
                        self.on_metrics(metrics)
                    if step % max(1, self.config.log_interval) == 0 or step == total_steps:
                        self._log(
                            f"[TRAIN] step {step}/{total_steps} | loss {step_loss:.4f}"
                            + (f" | val {val_loss:.4f}" if val_loss is not None else "")
                            + f" | lr {metrics.learning_rate:.3e}"
                            + f" | {metrics.tokens_per_sec:,.0f} tok/s"
                        )
                    last_report, tokens_window, samples_window = now, 0, 0

                    # ---- checkpointing --------------------------------------
                    manual = self.control.take_checkpoint_request()
                    scheduled = step % max(1, self.config.checkpoint_interval) == 0
                    if self.on_checkpoint and (manual or scheduled or step >= total_steps):
                        is_best = (
                            self.config.save_best
                            and val_loss is not None
                            and val_loss == self.best_val_loss
                        )
                        self.on_checkpoint(step, step_loss, val_loss, bool(is_best))

                    if step >= total_steps:
                        break

                if status == "stopped" or step >= total_steps:
                    break

            if self.control.stopping and status != "stopped":
                status = "stopped"

        except OutOfMemoryError:
            raise
        except TrainingError as exc:
            status, error = "failed", exc.message
            log.exception("Training failed", exc, source="training")
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("Training failed", exc, source="training")

        duration = time.time() - started
        self._log(
            f"[TRAIN] {status} after {step} steps in {duration:.1f}s"
            + (f" — final loss {final_loss:.4f}" if final_loss is not None else "")
        )
        return TrainingResult(
            status=status,
            steps=step,
            epochs=round(epoch, 4),
            final_train_loss=final_loss,
            best_val_loss=self.best_val_loss,
            tokens_processed=self._tokens_seen,
            duration_seconds=duration,
            error=error,
            history=self.history,
        )

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, dataset: Any | None = None, *, max_batches: int | None = None) -> float | None:
        data = dataset if dataset is not None else self.eval_dataset
        if data is None or len(data) == 0:
            return None

        loader = DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        limit = max_batches if max_batches is not None else self.config.eval_max_batches
        self.model.eval()
        dtype, autocast_enabled = self._resolve_dtype()

        total, batches = 0.0, 0
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enabled):
                outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            if loss is not None:
                total += float(loss)
                batches += 1
        return total / batches if batches else None

    def _oom_error(self, exc: BaseException) -> OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        suggestions = [
            f"Reduce batch size (currently {self.config.batch_size})",
            f"Raise gradient accumulation (currently {self.config.gradient_accumulation_steps})",
            "Enable gradient checkpointing",
            f"Shorten max_sequence_length (currently {self.config.max_sequence_length})",
        ]
        if self.config.method == "finetune":
            suggestions.append("Switch to LoRA or QLoRA")
        log.error(f"Out of memory at step: {exc}", source="training")
        return OutOfMemoryError(
            "Ran out of memory during training. Nothing was corrupted — the last saved "
            "checkpoint is intact.",
            suggestions,
        )


def perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return float(math.exp(min(loss, 20)))
    except (OverflowError, ValueError):
        return None
