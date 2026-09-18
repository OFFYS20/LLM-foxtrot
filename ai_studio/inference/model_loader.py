"""Loading models for inference, with a small cache.

Loading a model is expensive, so one loaded model is kept resident and swapped
only when a different model is requested.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.database import get_db
from ai_studio.core.errors import DependencyMissingError, InferenceError, NotFoundError, StudioError


@dataclass
class LoadedModel:
    model_id: str
    name: str
    model: Any
    tokenizer: Any
    device: str
    kind: str
    context_length: int
    parameters: int | None = None
    is_peft: bool = False
    loaded_at: float = 0.0

    def token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001
            return max(1, len(text) // 4)


class ModelCache:
    def __init__(self) -> None:
        self._current: LoadedModel | None = None
        self._lock = threading.RLock()

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def unload(self) -> None:
        with self._lock:
            if self._current is None:
                return
            name = self._current.name
            self._current = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            log.info(f"Unloaded {name}", source="inference")

    def load(self, model_id: str, *, device: str | None = None, force: bool = False) -> LoadedModel:
        with self._lock:
            if not force and self._current and self._current.model_id == model_id:
                return self._current

            db = get_db()
            record = db.require("models", model_id)
            self.unload()

            try:
                import torch
            except ImportError as exc:
                raise DependencyMissingError("torch", "Inference") from exc

            target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            started = time.perf_counter()

            try:
                loaded = self._load_record(record, target_device)
            except StudioError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(
                    f"Could not load {record['name']}: {exc}",
                    hint="Check that the weights still exist on disk.",
                ) from exc

            loaded.loaded_at = time.time()
            self._current = loaded
            log.info(
                f"Loaded {record['name']} on {target_device} in {time.perf_counter() - started:.1f}s",
                source="inference",
                context={"model_id": model_id},
            )
            return loaded

    def _load_record(self, record: dict[str, Any], device: str) -> LoadedModel:
        from ai_studio.models.transformer import TransformerLM

        path = record.get("path")
        kind = record["kind"]
        meta = record.get("meta") or {}

        # 1. A LoRA adapter — load the base, then attach the adapter.
        if kind == "adapter" or meta.get("is_peft"):
            return self._load_adapter(record, device)

        # 2. This project's own transformer.
        if kind == "scratch" or (path and (Path(path) / "config.json").exists() and _is_studio_config(path)):
            if not path or not Path(path).exists():
                raise NotFoundError(f"Weights for {record['name']} are missing on disk.")
            model = TransformerLM.from_pretrained(path, device=device)
            model.eval()
            tokenizer = self._tokenizer_for(record, path)
            return LoadedModel(
                model_id=record["id"],
                name=record["name"],
                model=model,
                tokenizer=tokenizer,
                device=device,
                kind="studio",
                context_length=model.config.max_position_embeddings,
                parameters=model.num_parameters(),
            )

        # 3. A HuggingFace causal LM.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError("transformers", "Hugging Face inference") from exc

        source = path or record.get("repo_id")
        if not source:
            raise NotFoundError(f"{record['name']} has no weights or repository id.")

        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=dtype, trust_remote_code=bool(meta.get("trust_remote_code"))
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return LoadedModel(
            model_id=record["id"],
            name=record["name"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            kind="hf",
            context_length=int(record.get("context_length") or getattr(model.config, "max_position_embeddings", 2048)),
            parameters=sum(p.numel() for p in model.parameters()),
        )

    def _load_adapter(self, record: dict[str, Any], device: str) -> LoadedModel:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise DependencyMissingError("peft", "Loading LoRA adapters") from exc

        db = get_db()
        meta = record.get("meta") or {}
        base_id = meta.get("base_model_id") or record.get("parent_model_id")
        if not base_id:
            raise InferenceError(
                f"{record['name']} is a LoRA adapter but its base model is unknown.",
                hint="Re-register the checkpoint and select its base model.",
            )
        base = db.require("models", base_id)
        base_loaded = self._load_record(base, device)
        try:
            merged = PeftModel.from_pretrained(base_loaded.model, record["path"])
            merged.eval()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Could not attach the adapter: {exc}") from exc

        return LoadedModel(
            model_id=record["id"],
            name=record["name"],
            model=merged,
            tokenizer=base_loaded.tokenizer,
            device=device,
            kind=base_loaded.kind,
            context_length=base_loaded.context_length,
            parameters=base_loaded.parameters,
            is_peft=True,
        )

    def _tokenizer_for(self, record: dict[str, Any], path: str) -> Any:
        from ai_studio.models.tokenizer_manager import load_tokenizer

        directory = Path(path)
        if (directory / "tokenizer.json").exists():
            return load_tokenizer(str(directory))
        if record.get("tokenizer_id"):
            return load_tokenizer(record["tokenizer_id"])
        raise InferenceError(
            f"No tokenizer found for {record['name']}.",
            hint="Attach a tokenizer to this model, or re-create it with one selected.",
        )


def _is_studio_config(path: str) -> bool:
    try:
        import json

        data = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        return data.get("model_type") == "ai_studio_transformer"
    except Exception:  # noqa: BLE001
        return False


cache = ModelCache()
