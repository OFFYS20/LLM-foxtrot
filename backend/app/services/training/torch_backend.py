"""Real PyTorch / Transformers / PEFT training backend.

Imported lazily: the platform starts (and runs demo mode) with no ML stack
installed. The heavy work happens in a worker thread so the event loop keeps
serving the API and WebSocket clients while a job runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.errors import EngineUnavailableError, OutOfMemoryError
from app.core.logging import get_logger
from app.db.models.enums import TrainingMethod
from app.services.training.base import StepMetrics, TrainingBackend, TrainingContext

logger = get_logger("foxtrot.training.torch")


def torch_stack_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


class TorchTrainingBackend(TrainingBackend):
    """Supervised fine-tuning with optional LoRA/QLoRA adapters."""

    name = "torch"
    provenance = "measured"

    def __init__(self, model_path: str, dataset_path: str, template: str = "instruction") -> None:
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.template = template

    @classmethod
    def is_available(cls) -> bool:
        return torch_stack_available()

    async def run(self, ctx: TrainingContext) -> dict[str, Any]:  # pragma: no cover - needs torch
        if not self.is_available():
            raise EngineUnavailableError(
                "PyTorch/Transformers are not installed. Install the optional training "
                "extras (pip install -r requirements-optional.txt) or run in demo mode."
            )

        loop = asyncio.get_running_loop()
        result: dict[str, Any] = {}

        def _report(metrics: StepMetrics) -> None:
            asyncio.run_coroutine_threadsafe(ctx.report(metrics), loop)

        def _log(message: str, level: str = "info") -> None:
            asyncio.run_coroutine_threadsafe(ctx.log(message, level), loop)

        def _checkpoint(step: int, loss: float, val_loss: float | None, best: bool) -> None:
            asyncio.run_coroutine_threadsafe(ctx.checkpoint(step, loss, val_loss, best), loop)

        try:
            result = await asyncio.to_thread(self._train_blocking, ctx, _report, _log, _checkpoint)
        except Exception as exc:
            if _is_cuda_oom(exc):
                _free_cuda()
                raise OutOfMemoryError(
                    "CUDA out of memory during training. Reduce batch size, sequence "
                    "length, or switch to LoRA/QLoRA with gradient checkpointing.",
                    details={"original_error": str(exc)[:500]},
                ) from exc
            raise
        return result

    # ------------------------------------------------------------------ impl
    def _train_blocking(self, ctx, report, log, checkpoint) -> dict[str, Any]:  # pragma: no cover
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        cfg = ctx.config
        log(f"[TRAIN] loading base model from {self.model_path}", "info")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(str(cfg.precision), torch.bfloat16)

        model_kwargs: dict[str, Any] = {}
        if str(cfg.precision) in {"int4", "int8"}:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=str(cfg.precision) == "int4",
                load_in_8bit=str(cfg.precision) == "int8",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        if cfg.method in (TrainingMethod.LORA, TrainingMethod.QLORA):
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora.rank,
                    lora_alpha=cfg.lora.alpha,
                    lora_dropout=cfg.lora.dropout,
                    target_modules=cfg.lora.target_modules,
                    bias=cfg.lora.bias,
                    task_type="CAUSAL_LM",
                ),
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"[TRAIN] LoRA adapters attached — {trainable:,} trainable parameters", "info")

        dataset = self._build_dataset(tokenizer, cfg.max_sequence_length)

        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                if "loss" not in logs:
                    return
                step = int(state.global_step)
                report(
                    StepMetrics(
                        step=step,
                        epoch=float(state.epoch or 0.0),
                        loss=float(logs["loss"]),
                        learning_rate=float(logs.get("learning_rate", cfg.learning_rate)),
                        grad_norm=float(logs.get("grad_norm", 0.0)) or None,
                        val_loss=float(logs["eval_loss"]) if "eval_loss" in logs else None,
                        tokens_per_sec=logs.get("train_tokens_per_second"),
                        tokens_processed=step * cfg.effective_batch_size * cfg.max_sequence_length,
                        vram_used_mb=(
                            torch.cuda.memory_allocated() / 1024**2
                            if torch.cuda.is_available()
                            else None
                        ),
                    )
                )
                log(
                    f"[TRAIN] Step {step}/{ctx.total_steps}\n"
                    f"  loss: {logs['loss']:.3f}\n"
                    f"  lr: {logs.get('learning_rate', 0):.2e}",
                    "info",
                )

            def on_save(self, args, state, control, **kwargs):
                checkpoint(
                    int(state.global_step),
                    float(state.log_history[-1].get("loss", 0.0)),
                    None,
                    False,
                )

            def on_step_end(self, args, state, control, **kwargs):
                if ctx.should_stop:
                    control.should_training_stop = True
                return control

        args = TrainingArguments(
            output_dir=str(ctx.extra.get("output_dir", "./data/checkpoints")),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.gradient_clipping,
            lr_scheduler_type=str(cfg.lr_scheduler).replace("_", "-")
            if str(cfg.lr_scheduler) == "cosine_with_restarts"
            else str(cfg.lr_scheduler),
            logging_steps=cfg.log_every_steps,
            save_steps=cfg.checkpointing.save_every_steps,
            save_total_limit=cfg.checkpointing.keep_last,
            eval_strategy="steps" if dataset.get("validation") else "no",
            eval_steps=cfg.eval_every_steps,
            bf16=str(cfg.precision) == "bf16",
            fp16=str(cfg.precision) == "fp16",
            gradient_checkpointing=cfg.gradient_checkpointing,
            seed=cfg.seed,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[_Bridge()],
        )

        train_result = trainer.train()
        metrics = train_result.metrics or {}
        eval_metrics = trainer.evaluate() if dataset.get("validation") else {}

        return {
            "final_loss": metrics.get("train_loss"),
            "final_val_loss": eval_metrics.get("eval_loss"),
            "tokens_processed": int(metrics.get("train_tokens", 0) or 0),
        }

    def _build_dataset(self, tokenizer, max_len: int):  # pragma: no cover - needs datasets
        from datasets import load_dataset

        suffix = self.dataset_path.rsplit(".", 1)[-1].lower()
        loader = {
            "jsonl": "json",
            "json": "json",
            "csv": "csv",
            "parquet": "parquet",
            "txt": "text",
        }
        raw = load_dataset(loader.get(suffix, "json"), data_files=self.dataset_path)["train"]

        def _format(example):
            if "messages" in example:
                text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in example["messages"])
            elif "instruction" in example:
                prompt = example["instruction"]
                if example.get("input"):
                    prompt += f"\n\n{example['input']}"
                text = f"<|user|>\n{prompt}\n<|assistant|>\n{example.get('output', '')}"
            else:
                text = example.get("text", "")
            return tokenizer(text, truncation=True, max_length=max_len)

        tokenized = raw.map(_format, remove_columns=raw.column_names)
        split = tokenized.train_test_split(test_size=0.05, seed=42)
        return {"train": split["train"], "validation": split["test"]}


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "OutOfMemory" in name or "CUDA out of memory" in str(exc)


def _free_cuda() -> None:  # pragma: no cover - needs torch
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
