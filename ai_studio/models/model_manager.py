"""Model registry: create from scratch, import from Hugging Face, export, quantize."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import (
    DependencyMissingError,
    NotFoundError,
    StudioError,
    UnsupportedError,
    ValidationError,
)
from ai_studio.core.paths import dir_size, remove_path, slugify
from ai_studio.models.transformer import SIZE_PRESETS, TransformerConfig, TransformerLM

MODEL_KINDS = ("scratch", "huggingface", "checkpoint", "merged", "adapter")


def create_scratch_model(
    name: str,
    config: TransformerConfig,
    *,
    tokenizer_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Instantiate and save a new randomly-initialised transformer."""
    problems = config.validate()
    if problems:
        raise ValidationError("Invalid architecture: " + "; ".join(problems))

    db = get_db()
    if db.get("models", name, key="name"):
        raise ValidationError(f"A model named {name!r} already exists.")

    if tokenizer_id:
        tokenizer_record = db.require("tokenizers", tokenizer_id)
        if tokenizer_record["vocab_size"] and tokenizer_record["vocab_size"] != config.vocab_size:
            log.warning(
                f"Aligning vocab_size {config.vocab_size} → {tokenizer_record['vocab_size']} "
                f"to match tokenizer '{tokenizer_record['name']}'.",
                source="models",
            )
            config.vocab_size = int(tokenizer_record["vocab_size"])

    directory = get_config().models_dir / slugify(name)
    if directory.exists():
        raise ValidationError(f"Directory {directory.name!r} already exists.")

    model = TransformerLM(config)
    parameters = model.num_parameters()
    model.save_pretrained(
        directory,
        metadata={
            "created_by": "ai_studio",
            "kind": "scratch",
            "tokenizer_id": tokenizer_id,
            "parameters": parameters,
        },
    )

    if tokenizer_id:
        try:
            from ai_studio.models.tokenizer_manager import load_tokenizer

            load_tokenizer(tokenizer_id).save_pretrained(str(directory))
        except Exception as exc:  # noqa: BLE001 - model is still usable
            log.warning(f"Could not copy tokenizer into model directory: {exc}", source="models")

    record = {
        "id": new_id("mdl"),
        "name": name,
        "kind": "scratch",
        "architecture": "ai_studio_transformer",
        "path": str(directory),
        "repo_id": None,
        "tokenizer_id": tokenizer_id,
        "parameters": parameters,
        "context_length": config.max_position_embeddings,
        "precision": "fp32",
        "size_bytes": dir_size(directory),
        "license": "Created locally — you own these weights.",
        "status": "ready",
        "config": config.to_dict(),
        "metrics": {},
        "meta": {
            "description": description,
            "untrained": True,
            "parameter_breakdown": config.parameter_count(),
        },
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    get_db().insert("models", record)
    log.info(
        f"Created model '{name}' with {parameters / 1e6:.1f}M parameters (randomly initialised)",
        source="models",
        context={"model_id": record["id"]},
    )
    return get_db().require("models", record["id"])


def import_huggingface_model(
    repo_id: str,
    *,
    name: str | None = None,
    revision: str | None = None,
    download: bool = True,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Register (and optionally download) a Hugging Face causal LM."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise DependencyMissingError("transformers", "Hugging Face import") from exc

    repo_id = repo_id.strip()
    if not repo_id or repo_id.count("/") > 1:
        raise ValidationError(f"Invalid repository id {repo_id!r} — expected 'owner/name'.")

    db = get_db()
    display = name or repo_id.split("/")[-1]
    if db.get("models", display, key="name"):
        raise ValidationError(f"A model named {display!r} already exists.")

    try:
        hf_config = AutoConfig.from_pretrained(
            repo_id, revision=revision, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001
        raise StudioError(
            f"Could not read the configuration for {repo_id!r}: {exc}",
            hint="Check the repository id, your network connection, and whether it is gated.",
        ) from exc

    config_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}
    architectures = config_dict.get("architectures") or []
    model_type = config_dict.get("model_type")
    if architectures and not any("CausalLM" in arch or "LMHead" in arch for arch in architectures):
        raise UnsupportedError(
            f"{repo_id} is a {architectures[0]} model, not a causal language model.",
            hint="AI Studio trains and chats with decoder-only causal LMs.",
        )

    directory = None
    parameters = _estimate_hf_parameters(config_dict)
    size_bytes = 0
    if download:
        try:
            from transformers import AutoModelForCausalLM

            target = get_config().models_dir / slugify(display)
            model = AutoModelForCausalLM.from_pretrained(
                repo_id, revision=revision, trust_remote_code=trust_remote_code
            )
            parameters = sum(p.numel() for p in model.parameters())
            model.save_pretrained(str(target))
            try:
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained(repo_id, revision=revision).save_pretrained(str(target))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"No tokenizer saved for {repo_id}: {exc}", source="models")
            directory = str(target)
            size_bytes = dir_size(target)
            del model
        except Exception as exc:  # noqa: BLE001
            raise StudioError(
                f"Could not download {repo_id!r}: {exc}",
                hint="Registering without weights still works — set download=False.",
            ) from exc

    record = {
        "id": new_id("mdl"),
        "name": display,
        "kind": "huggingface",
        "architecture": architectures[0] if architectures else model_type,
        "path": directory,
        "repo_id": repo_id,
        "revision": revision,
        "tokenizer_id": None,
        "parameters": parameters,
        "context_length": config_dict.get("max_position_embeddings")
        or config_dict.get("n_positions")
        or config_dict.get("max_sequence_length"),
        "precision": str(config_dict.get("torch_dtype") or "unknown"),
        "size_bytes": size_bytes,
        "license": _license_for(repo_id, config_dict),
        "status": "ready",
        "config": config_dict,
        "metrics": {},
        "meta": {
            "model_type": model_type,
            "downloaded": bool(directory),
            "trust_remote_code": trust_remote_code,
            "supported_methods": supported_methods(model_type, downloaded=bool(directory)),
        },
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    db.insert("models", record)
    log.info(f"Imported {repo_id} as '{display}'", source="models", context={"model_id": record["id"]})
    return db.require("models", record["id"])


def _estimate_hf_parameters(config: dict[str, Any]) -> int | None:
    hidden = config.get("hidden_size") or config.get("n_embd")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    vocab = config.get("vocab_size")
    if not (hidden and layers and vocab):
        return None
    intermediate = config.get("intermediate_size") or 4 * hidden
    per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
    return int(vocab * hidden + layers * per_layer)


def _license_for(repo_id: str, config: dict[str, Any]) -> str:
    """Report the licence when the repo declares one — never assume."""
    declared = config.get("license") or config.get("licence")
    if declared:
        return str(declared)
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id)
        tags = getattr(info, "tags", []) or []
        for tag in tags:
            if str(tag).startswith("license:"):
                return str(tag).split(":", 1)[1]
        if getattr(info, "cardData", None) and info.cardData.get("license"):
            return str(info.cardData["license"])
    except Exception:  # noqa: BLE001 - offline or hub unavailable
        pass
    return (
        "Not declared in the model repository — check the model card before use or redistribution."
    )


def supported_methods(model_type: str | None, *, downloaded: bool = True) -> list[str]:
    """Which training methods this architecture can actually use."""
    if not downloaded:
        return []
    methods = ["finetune", "continued_pretraining", "instruction_tuning", "lora"]
    try:
        import bitsandbytes  # noqa: F401
        import torch

        if torch.cuda.is_available():
            methods.append("qlora")
    except ImportError:
        pass
    return methods


def load_model_for_training(
    model_id: str,
    *,
    quantization: str = "none",
    device: str = "cpu",
    dtype: str = "fp32",
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any]:
    """Load a registered model plus its tokenizer, ready for training."""
    db = get_db()
    record = db.require("models", model_id)

    if record["kind"] == "scratch":
        if not record.get("path"):
            raise NotFoundError(f"Model {record['name']} has no weights on disk.")
        model = TransformerLM.from_pretrained(record["path"], device=device)
        tokenizer = _tokenizer_for(record)
        return model, tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyMissingError("transformers", "Loading Hugging Face models") from exc

    source = record.get("path") or record.get("repo_id")
    if not source:
        raise NotFoundError(f"Model {record['name']} has no local path or repository id.")

    kwargs: dict[str, Any] = {}
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        dtype, torch.float32
    )

    if quantization in {"4bit", "8bit"}:
        if not torch.cuda.is_available():
            raise UnsupportedError(
                f"{quantization} quantization requires a CUDA GPU.",
                hint="Train in fp32 on CPU, or use LoRA on an unquantized small model.",
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise DependencyMissingError("bitsandbytes", f"{quantization} quantization") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(
            source, trust_remote_code=bool((record.get("meta") or {}).get("trust_remote_code")), **kwargs
        )
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            from ai_studio.core.errors import OutOfMemoryError

            raise OutOfMemoryError(
                f"Not enough memory to load {record['name']}.",
                ["Use QLoRA (4-bit)", "Choose a smaller model", "Free GPU memory"],
            ) from exc
        raise StudioError(f"Could not load {record['name']}: {exc}") from exc

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quantization == "none":
        model = model.to(device)
    return model, tokenizer


def _tokenizer_for(record: dict[str, Any]) -> Any:
    from ai_studio.models.tokenizer_manager import load_tokenizer

    if record.get("tokenizer_id"):
        return load_tokenizer(record["tokenizer_id"])
    path = Path(record["path"]) if record.get("path") else None
    if path and (path / "tokenizer.json").exists():
        return load_tokenizer(str(path))
    raise NotFoundError(
        f"No tokenizer is attached to {record['name']}.",
        hint="Train or select a tokenizer, then attach it to this model.",
    )


def export_model(
    model_id: str,
    *,
    destination: str | None = None,
    fmt: str = "safetensors",
    include_tokenizer: bool = True,
) -> dict[str, Any]:
    """Copy a model into the exports directory in a Hugging Face-style layout."""
    record = get_db().require("models", model_id)
    if not record.get("path") or not Path(record["path"]).exists():
        raise NotFoundError(
            f"{record['name']} has no local weights to export.",
            hint="Download the model first (Models → Import → download weights).",
        )

    if fmt == "gguf":
        raise UnsupportedError(
            "GGUF export needs llama.cpp's convert script, which is not bundled.",
            hint="Install llama.cpp and run convert_hf_to_gguf.py against the exported folder.",
        )

    target = Path(destination) if destination else get_config().exports_dir / slugify(record["name"])
    target.mkdir(parents=True, exist_ok=True)
    source = Path(record["path"])

    copied: list[str] = []
    for item in source.iterdir():
        if not include_tokenizer and item.name.startswith("tokenizer"):
            continue
        if item.is_file():
            shutil.copy2(item, target / item.name)
            copied.append(item.name)
        elif item.is_dir():
            shutil.copytree(item, target / item.name, dirs_exist_ok=True)
            copied.append(item.name + "/")

    (target / "ai_studio_export.json").write_text(
        json.dumps(
            {
                "model": record["name"],
                "kind": record["kind"],
                "parameters": record["parameters"],
                "architecture": record["architecture"],
                "license": record["license"],
                "exported_at": time.time(),
                "files": copied,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log.info(f"Exported {record['name']} → {target}", source="models")
    return {"path": str(target), "files": copied, "size_bytes": dir_size(target)}


def quantize_model(model_id: str, bits: int = 8, *, name: str | None = None) -> dict[str, Any]:
    """Save a quantized copy. Requires bitsandbytes and a CUDA GPU."""
    if bits not in (4, 8):
        raise ValidationError("Only 4-bit and 8-bit quantization are supported.")
    try:
        import torch

        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise DependencyMissingError(
            "bitsandbytes", f"{bits}-bit quantization", install="bitsandbytes"
        ) from exc
    if not torch.cuda.is_available():
        raise UnsupportedError(
            f"{bits}-bit quantization requires a CUDA GPU; none is available.",
            hint="Quantization is not faked here — run this on a CUDA machine.",
        )

    record = get_db().require("models", model_id)
    model, tokenizer = load_model_for_training(
        model_id, quantization=f"{bits}bit", device="cuda", dtype="bf16"
    )
    display = name or f"{record['name']}-{bits}bit"
    target = get_config().models_dir / slugify(display)
    model.save_pretrained(str(target))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(target))

    new_record = {
        **{k: v for k, v in record.items() if k not in {"id", "created_at", "updated_at"}},
        "id": new_id("mdl"),
        "name": display,
        "path": str(target),
        "quantization": f"{bits}bit",
        "size_bytes": dir_size(target),
        "parent_model_id": record["id"],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    get_db().insert("models", new_record)
    log.info(f"Quantized {record['name']} to {bits}-bit", source="models")
    return get_db().require("models", new_record["id"])


def register_checkpoint_as_model(
    checkpoint_id: str, name: str | None = None, *, base_model_id: str | None = None
) -> dict[str, Any]:
    """Promote a checkpoint into the model registry so it can be chatted with."""
    db = get_db()
    checkpoint = db.require("checkpoints", checkpoint_id)
    experiment = db.get("experiments", checkpoint.get("experiment_id") or "")
    base_id = base_model_id or (experiment or {}).get("base_model_id") or (experiment or {}).get("model_id")
    base = db.get("models", base_id or "") or {}

    display = name or f"{base.get('name', 'model')}-step{checkpoint['step']}"
    if db.get("models", display, key="name"):
        display = f"{display}-{new_id('v')[-4:]}"

    is_peft = bool((checkpoint.get("meta") or {}).get("is_peft"))
    record = {
        "id": new_id("mdl"),
        "name": display,
        "kind": "adapter" if is_peft else ("scratch" if base.get("kind") == "scratch" else "checkpoint"),
        "architecture": base.get("architecture"),
        "path": checkpoint["path"],
        "repo_id": base.get("repo_id"),
        "tokenizer_id": base.get("tokenizer_id"),
        "parameters": base.get("parameters"),
        "context_length": base.get("context_length"),
        "precision": base.get("precision"),
        "size_bytes": checkpoint.get("size_bytes", 0),
        "license": base.get("license"),
        "status": "ready",
        "parent_model_id": base.get("id"),
        "source_checkpoint_id": checkpoint_id,
        "config": base.get("config") or {},
        "metrics": {"train_loss": checkpoint.get("train_loss"), "val_loss": checkpoint.get("val_loss")},
        "meta": {
            "from_checkpoint": checkpoint_id,
            "step": checkpoint["step"],
            "is_peft": is_peft,
            "base_model_id": base.get("id"),
        },
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    db.insert("models", record)
    db.update("checkpoints", checkpoint_id, {"model_id": record["id"]})
    log.info(f"Registered checkpoint {checkpoint['name']} as model '{display}'", source="models")
    return db.require("models", record["id"])


def list_models(kind: str | None = None) -> list[dict[str, Any]]:
    if kind and kind != "all":
        return get_db().list("models", where="kind = ?", params=(kind,), order_by="created_at DESC")
    return get_db().list("models", order_by="created_at DESC")


def delete_model(model_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("models", model_id)
    if remove_files and record.get("path"):
        path = Path(record["path"])
        # Never delete a shared checkpoint directory out from under a checkpoint row.
        if not db.count("checkpoints", "path = ?", (str(path),)):
            remove_path(path)
    db.delete("models", model_id)
    log.warning(f"Deleted model {record['name']}", source="models")


def model_summary(model_id: str) -> dict[str, Any]:
    db = get_db()
    record = db.require("models", model_id)
    checkpoints = db.count("checkpoints", "model_id = ?", (model_id,))
    experiments = db.count("experiments", "model_id = ? OR base_model_id = ?", (model_id, model_id))
    return {**record, "checkpoint_count": checkpoints, "experiment_count": experiments}


__all__ = [
    "SIZE_PRESETS",
    "create_scratch_model",
    "import_huggingface_model",
    "load_model_for_training",
    "export_model",
    "quantize_model",
    "register_checkpoint_as_model",
    "list_models",
    "delete_model",
    "model_summary",
    "supported_methods",
]
