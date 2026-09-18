"""LoRA / QLoRA setup via PEFT.

Wraps a loaded base model with adapters and reports the real trainable
parameter count. When PEFT (or bitsandbytes for QLoRA) is missing, this raises
rather than silently training the full model.
"""

from __future__ import annotations

from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.errors import DependencyMissingError, TrainingError, UnsupportedError
from ai_studio.training.config import TrainingConfig, suggest_lora_targets


def peft_available() -> bool:
    try:
        import peft  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_target_modules(model: Any, config: TrainingConfig) -> list[str]:
    """Use the configured targets when they exist on the model, else suggest."""
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    requested = [module for module in config.lora.target_modules if module in available]
    if requested:
        return requested

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    suggested = [module for module in suggest_lora_targets(model_type) if module in available]
    if suggested:
        log.warning(
            f"Configured LoRA targets {config.lora.target_modules} are not present in this model; "
            f"using {suggested} instead.",
            source="training",
        )
        return suggested

    # Last resort: every linear layer that looks like attention/MLP projection.
    import torch.nn as nn

    fallback = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and "lm_head" not in name
        }
    )
    if not fallback:
        raise UnsupportedError(
            "No linear layers found to attach LoRA adapters to.",
            hint="This architecture may not support LoRA in AI Studio.",
        )
    log.warning(f"Falling back to LoRA targets: {fallback[:6]}", source="training")
    return fallback[:6]


def apply_lora(model: Any, config: TrainingConfig) -> tuple[Any, dict[str, Any]]:
    """Attach LoRA adapters. Returns (wrapped model, stats)."""
    if not peft_available():
        raise DependencyMissingError("peft", "LoRA/QLoRA training", install="peft")

    from peft import LoraConfig, TaskType, get_peft_model

    if config.method == "qlora":
        if config.quantization == "none":
            config.quantization = "4bit"
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise DependencyMissingError("bitsandbytes", "QLoRA (4-bit) training") from exc
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing
            )
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"Could not prepare the model for 4-bit training: {exc}") from exc

    targets = resolve_target_modules(model, config)
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=targets,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    try:
        wrapped = get_peft_model(model, lora_config)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(
            f"Could not attach LoRA adapters: {exc}",
            hint=f"Targets tried: {targets}",
        ) from exc

    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    stats = {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(trainable / max(1, total) * 100, 4),
        "target_modules": targets,
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
        "quantization": config.quantization,
    }
    log.info(
        f"LoRA attached: {trainable:,} trainable of {total:,} "
        f"({stats['trainable_percent']:.3f}%) on {targets}",
        source="training",
    )
    return wrapped, stats


def merge_adapter(base_model: Any, adapter_path: str, *, output_path: str) -> str:
    """Merge LoRA weights into the base model and save a standalone model."""
    if not peft_available():
        raise DependencyMissingError("peft", "Merging LoRA adapters", install="peft")

    from peft import PeftModel

    try:
        merged = PeftModel.from_pretrained(base_model, adapter_path)
        merged = merged.merge_and_unload()
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Could not merge the adapter: {exc}") from exc

    merged.save_pretrained(output_path)
    log.info(f"Merged adapter {adapter_path} → {output_path}", source="training")
    return output_path
