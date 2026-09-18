"""Fine-Tuning screen: adapt an existing model, and merge LoRA adapters.

The run form is the Training form with fine-tuning methods and defaults; the
adapter tools below it only exist here, because they only apply to PEFT runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import StudioError, ValidationError
from ai_studio.core.paths import dir_size, slugify
from ai_studio.models.model_manager import list_models, load_model_for_training
from ai_studio.training.lora_trainer import merge_adapter, peft_available
from ai_studio.ui import theme as t
from ai_studio.ui import training_page


def adapter_choices() -> list[tuple[str, str]]:
    return [
        (f"{record['name']} · {t.fmt_bytes(record.get('size_bytes'))}", record["id"])
        for record in list_models()
        if record.get("kind") == "adapter" or (record.get("meta") or {}).get("is_peft")
    ]


def base_choices() -> list[tuple[str, str]]:
    return [
        (f"{record['name']} · {t.fmt_params(record.get('parameters'))}", record["id"])
        for record in list_models()
        if record.get("kind") != "adapter"
    ]


def do_merge(adapter_id: str, base_id: str, name: str) -> str:
    if not adapter_id or not base_id:
        return t.note("Choose both an adapter and the base model it was trained on.", "warn")
    if not peft_available():
        return t.note(
            "Merging needs the <code>peft</code> package — install it with "
            "<code>pip install peft</code>.",
            "bad",
        )

    db = get_db()
    try:
        adapter = db.require("models", adapter_id)
        base = db.require("models", base_id)
        adapter_path = adapter.get("path")
        if not adapter_path or not Path(adapter_path).exists():
            raise ValidationError(f"Adapter {adapter['name']} has no weights on disk.")

        display = name.strip() or f"{base['name']}-merged"
        if db.get("models", display, key="name"):
            display = f"{display}-{new_id('v')[-4:]}"
        output = get_config().models_dir / slugify(display)
        if output.exists():
            raise ValidationError(f"A model directory named {output.name!r} already exists.")

        base_model, tokenizer = load_model_for_training(base_id)
        merge_adapter(base_model, adapter_path, output_path=str(output))
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(output))

        record = {
            "id": new_id("mdl"),
            "name": display,
            "kind": "merged",
            "architecture": base.get("architecture"),
            "path": str(output),
            "repo_id": base.get("repo_id"),
            "tokenizer_id": base.get("tokenizer_id"),
            "parameters": base.get("parameters"),
            "context_length": base.get("context_length"),
            "precision": base.get("precision"),
            "size_bytes": dir_size(output),
            # The merged weights inherit the base model's licence, not this app's.
            "license": base.get("license"),
            "status": "ready",
            "parent_model_id": base_id,
            "config": base.get("config") or {},
            "metrics": adapter.get("metrics") or {},
            "meta": {"merged_from_adapter": adapter_id, "base_model_id": base_id},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        db.insert("models", record)
    except StudioError as exc:
        return t.error_message(exc)
    except Exception as exc:  # noqa: BLE001 - one failed merge must not kill the app
        return t.error_message(exc)

    return t.note(
        f"Merged <b>{adapter['name']}</b> into <b>{base['name']}</b> → "
        f"standalone model <b>{display}</b> ({t.fmt_bytes(record['size_bytes'])}). "
        f"It carries the base model's licence: {base.get('license') or 'unknown'}.",
        "good",
    )


def render() -> None:
    training_page.render(variant="finetuning")

    gr.Markdown("### Adapter tools")
    gr.HTML(
        t.note(
            "A LoRA/QLoRA run saves a small adapter, not a full model. Merging writes the "
            "adapter into a copy of the base weights so the result can be used, exported or "
            "quantized on its own. The original adapter and base model are left untouched."
        )
    )
    with gr.Row():
        adapter = gr.Dropdown(choices=adapter_choices(), label="Adapter")
        base = gr.Dropdown(choices=base_choices(), label="Base model")
        merged_name = gr.Textbox(label="Name for the merged model", placeholder="auto")
    with gr.Row():
        merge_button = gr.Button("Merge adapter into base", variant="primary")
        refresh = gr.Button("↻ Refresh", size="sm")
    merge_result = gr.HTML()

    merge_button.click(do_merge, [adapter, base, merged_name], merge_result)
    refresh.click(
        lambda: (gr.update(choices=adapter_choices()), gr.update(choices=base_choices())),
        outputs=[adapter, base],
    )
