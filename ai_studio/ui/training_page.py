"""Training screen: configure a run, start it, watch it live, control it."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ai_studio.core.database import get_db
from ai_studio.core.errors import StudioError
from ai_studio.data.dataset_builder import list_datasets
from ai_studio.models.model_manager import list_models
from ai_studio.models.tokenizer_manager import list_tokenizers
from ai_studio.training.config import (
    METHOD_LABELS,
    METHODS,
    OPTIMIZERS,
    PRECISIONS,
    SCHEDULERS,
    LoRASettings,
    TrainingConfig,
    preflight,
    suggest_lora_targets,
)
from ai_studio.training.worker import manager
from ai_studio.ui import theme as t


def model_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {t.fmt_params(r.get('parameters'))} · {r['kind']}", r["id"]) for r in list_models()]


def dataset_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {r['mode']} · {r['rows']:,} rows", r["id"]) for r in list_datasets()]


def tokenizer_choices() -> list[tuple[str, str]]:
    return [("(use the model's tokenizer)", "")] + [
        (f"{r['name']} · {r['vocab_size']:,}", r["id"]) for r in list_tokenizers()
    ]


def checkpoint_choices() -> list[tuple[str, str]]:
    rows = get_db().list("checkpoints", order_by="created_at DESC", limit=100)
    return [("(start fresh)", "")] + [
        (f"{r['name']} · step {r['step']} · loss {t.fmt_number(r.get('train_loss'), 4)}", r["id"])
        for r in rows if r.get("has_optimizer_state")
    ]


def _config_from(values: dict[str, Any]) -> TrainingConfig:
    config = TrainingConfig(
        method=values["method"],
        epochs=float(values["epochs"]),
        max_steps=int(values["max_steps"]),
        batch_size=int(values["batch_size"]),
        gradient_accumulation_steps=int(values["accumulation"]),
        learning_rate=float(values["learning_rate"]),
        warmup_steps=int(values["warmup"]),
        weight_decay=float(values["weight_decay"]),
        max_sequence_length=int(values["sequence_length"]),
        gradient_clipping=float(values["clipping"]),
        optimizer=values["optimizer"],
        lr_scheduler=values["scheduler"],
        seed=int(values["seed"]),
        precision=values["precision"],
        gradient_checkpointing=bool(values["checkpointing"]),
        flash_attention=bool(values["flash"]),
        quantization=values["quantization"],
        packing=bool(values["packing"]),
        eval_interval=int(values["eval_interval"]),
        log_interval=int(values["log_interval"]),
        checkpoint_interval=int(values["checkpoint_interval"]),
        keep_last_checkpoints=int(values["keep_last"]),
        save_best=bool(values["save_best"]),
    )
    config.lora = LoRASettings(
        rank=int(values["lora_rank"]),
        alpha=int(values["lora_alpha"]),
        dropout=float(values["lora_dropout"]),
        target_modules=[m.strip() for m in str(values["lora_targets"]).split(",") if m.strip()],
    )
    return config


def live_status() -> tuple[str, str]:
    experiment_id = manager.active_experiment_id()
    if not experiment_id:
        rows = get_db().list("experiments", order_by="created_at DESC", limit=1)
        if not rows:
            return t.note("No runs yet."), t.console_html([])
        experiment_id = rows[0]["id"]

    status = manager.status(experiment_id)
    experiment = status["experiment"]
    latest = status.get("latest") or {}
    tone = {"running": "good", "paused": "warn", "failed": "bad"}.get(experiment.get("status", ""), "")

    cards = t.stat_grid([
        t.stat("Status", experiment.get("status", "—")),
        t.stat("Step", t.fmt_number(latest.get("step") or experiment.get("current_step"))),
        t.stat("Epoch", t.fmt_number(latest.get("epoch") or experiment.get("current_epoch"), 2)),
        t.stat("Loss", t.fmt_number(latest.get("loss") or experiment.get("final_train_loss"), 4)),
        t.stat("Val loss", t.fmt_number(latest.get("val_loss") or experiment.get("best_val_loss"), 4)),
        t.stat("LR", f"{latest['learning_rate']:.2e}" if latest.get("learning_rate") else "—"),
        t.stat("Grad norm", t.fmt_number(latest.get("grad_norm"), 3)),
        t.stat("Tokens/sec", t.fmt_number(latest.get("tokens_per_sec"), 0)),
        t.stat("Samples/sec", t.fmt_number(latest.get("samples_per_sec"), 2)),
        t.stat("Tokens seen", t.fmt_params(latest.get("tokens_processed") or experiment.get("tokens_processed"))),
        t.stat("Elapsed", t.fmt_duration(latest.get("elapsed_seconds") or experiment.get("duration_seconds"))),
        t.stat("Remaining", t.fmt_duration(latest.get("eta_seconds"))),
    ])
    header = '<div class="studio-pills">' + t.pill(f"<b>{experiment.get('name', '')}</b>") + \
             t.pill(experiment.get("status", "—"), tone) + "</div>"
    if experiment.get("error"):
        header += t.note(experiment["error"], "bad")
    return header + cards, t.console_html(status.get("console", []))


# Two screens share this form. Pretraining builds a model from tokens; fine-tuning
# adapts one that already exists, so each gets its own methods and defaults.
VARIANTS: dict[str, dict[str, Any]] = {
    "pretraining": {
        "title": "Training",
        "subtitle": "Real PyTorch training — from scratch or continued pretraining on raw text",
        "methods": ("scratch", "continued_pretraining"),
        "default_method": "scratch",
        "learning_rate": 3e-4,
        "epochs": 1.0,
        "packing": True,
        "show_lora": False,
    },
    "finetuning": {
        "title": "Fine-Tuning",
        "subtitle": "Adapt an existing model — full fine-tuning, instruction tuning, LoRA or QLoRA",
        "methods": ("finetune", "instruction_tuning", "lora", "qlora"),
        "default_method": "lora",
        "learning_rate": 2e-4,
        "epochs": 3.0,
        "packing": False,
        "show_lora": True,
    },
}


def render(variant: str = "pretraining") -> None:
    spec = VARIANTS.get(variant, VARIANTS["pretraining"])
    methods = [m for m in METHODS if m in spec["methods"]]

    gr.HTML(f'<div class="studio-title">{spec["title"]}</div>'
            f'<div class="studio-sub">{spec["subtitle"]}</div>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Run")
            run_name = gr.Textbox(label="Experiment name", placeholder="auto")
            model = gr.Dropdown(
                choices=model_choices(),
                label="Base model" if variant == "finetuning" else "Model",
            )
            dataset = gr.Dropdown(choices=dataset_choices(), label="Dataset")
            tokenizer = gr.Dropdown(choices=tokenizer_choices(), value="", label="Tokenizer override")
            method = gr.Dropdown([(METHOD_LABELS[m], m) for m in methods],
                                 value=spec["default_method"], label="Method")
            resume_from = gr.Dropdown(choices=checkpoint_choices(), value="", label="Resume from checkpoint")
            notes = gr.Textbox(label="Notes", lines=2)

        with gr.Column(scale=1):
            gr.Markdown("#### Hyperparameters")
            with gr.Row():
                epochs = gr.Number(value=spec["epochs"], label="Epochs")
                max_steps = gr.Number(value=0, label="Max steps (0 = use epochs)", precision=0)
            with gr.Row():
                batch_size = gr.Number(value=2, label="Batch size", precision=0)
                accumulation = gr.Number(value=1, label="Grad accumulation", precision=0)
            with gr.Row():
                learning_rate = gr.Number(value=spec["learning_rate"], label="Learning rate")
                warmup = gr.Number(value=0, label="Warmup steps", precision=0)
            with gr.Row():
                weight_decay = gr.Number(value=0.01, label="Weight decay")
                clipping = gr.Number(value=1.0, label="Gradient clipping")
            with gr.Row():
                sequence_length = gr.Number(value=256, label="Max sequence length", precision=0)
                seed = gr.Number(value=42, label="Seed", precision=0)
            with gr.Row():
                optimizer = gr.Dropdown(list(OPTIMIZERS), value="adamw", label="Optimizer")
                scheduler = gr.Dropdown(list(SCHEDULERS), value="cosine", label="Scheduler")

        with gr.Column(scale=1):
            gr.Markdown("#### Precision & memory")
            precision = gr.Dropdown(list(PRECISIONS), value="fp32", label="Precision")
            quantization = gr.Dropdown(["none", "8bit", "4bit"], value="none", label="Quantized base")
            with gr.Row():
                checkpointing = gr.Checkbox(value=False, label="Gradient checkpointing")
                flash = gr.Checkbox(value=False, label="Flash Attention")
            packing = gr.Checkbox(value=spec["packing"], label="Pack sequences (raw LM)")
            gr.Markdown("#### LoRA" + ("" if spec["show_lora"] else " (used by LoRA/QLoRA runs)"))
            with gr.Row():
                lora_rank = gr.Number(value=16, label="Rank", precision=0)
                lora_alpha = gr.Number(value=32, label="Alpha", precision=0)
                lora_dropout = gr.Number(value=0.05, label="Dropout")
            lora_targets = gr.Textbox(value="q_proj, k_proj, v_proj, o_proj", label="Target modules")
            suggest_button = gr.Button("Suggest targets for this model", size="sm")
            gr.Markdown("#### Cadence")
            with gr.Row():
                eval_interval = gr.Number(value=50, label="Eval every", precision=0)
                log_interval = gr.Number(value=10, label="Log every", precision=0)
            with gr.Row():
                checkpoint_interval = gr.Number(value=200, label="Checkpoint every", precision=0)
                keep_last = gr.Number(value=3, label="Keep last", precision=0)
            save_best = gr.Checkbox(value=True, label="Always keep the best checkpoint")

    config_inputs = {
        "method": method, "epochs": epochs, "max_steps": max_steps, "batch_size": batch_size,
        "accumulation": accumulation, "learning_rate": learning_rate, "warmup": warmup,
        "weight_decay": weight_decay, "sequence_length": sequence_length, "clipping": clipping,
        "optimizer": optimizer, "scheduler": scheduler, "seed": seed, "precision": precision,
        "checkpointing": checkpointing, "flash": flash, "quantization": quantization,
        "packing": packing, "eval_interval": eval_interval, "log_interval": log_interval,
        "checkpoint_interval": checkpoint_interval, "keep_last": keep_last, "save_best": save_best,
        "lora_rank": lora_rank, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "lora_targets": lora_targets,
    }
    order = list(config_inputs)
    config_components = [config_inputs[key] for key in order]

    with gr.Row():
        check_button = gr.Button("Check configuration")
        start_button = gr.Button("Start training", variant="primary")
        refresh_lists = gr.Button("↻ Refresh", size="sm")
    setup_result = gr.HTML()

    gr.Markdown("### Live run")
    status_html = gr.HTML(lambda: live_status()[0])
    with gr.Row():
        pause_button = gr.Button("Pause")
        resume_button = gr.Button("Resume")
        checkpoint_button = gr.Button("Save checkpoint")
        stop_button = gr.Button("Stop", variant="stop")
    control_result = gr.HTML()
    console = gr.HTML(lambda: live_status()[1])
    timer = gr.Timer(2.0)
    timer.tick(live_status, outputs=[status_html, console])

    # -------------------------------------------------------------- handlers
    def do_suggest(model_id):
        if not model_id:
            return gr.update()
        record = get_db().get("models", model_id) or {}
        model_type = (record.get("meta") or {}).get("model_type") or record.get("architecture")
        return ", ".join(suggest_lora_targets(model_type))

    suggest_button.click(do_suggest, model, lora_targets)

    def do_check(model_id, dataset_id, *values):
        if not model_id or not dataset_id:
            return t.note("Select a model and a dataset.", "warn")
        payload = dict(zip(order, values))
        try:
            config = _config_from(payload)
            config.validate()
        except StudioError as exc:
            return t.error_message(exc)
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc)

        record = get_db().get("models", model_id) or {}
        parameters = int(record.get("parameters") or 0)
        check = preflight(config, parameter_count=parameters)
        html = t.stat_grid([
            t.stat("Method", METHOD_LABELS.get(config.method, config.method)),
            t.stat("Effective batch", config.effective_batch_size),
            t.stat("Device", check.device),
            t.stat("Estimated memory", f"{check.estimated_mb/1024:.2f} GB"),
            t.stat("Available", f"{check.available_mb/1024:.2f} GB" if check.available_mb else None,
                   unavailable=check.available_mb is None),
        ])
        for blocker in check.blockers:
            html += t.note(blocker, "bad")
        for warning in check.warnings:
            html += t.note(warning, "warn")
        if check.suggestions:
            html += t.note("Suggestions: " + "; ".join(check.suggestions))
        if check.ok and not check.warnings:
            html += t.note("Configuration looks good.", "good")
        return html

    check_button.click(do_check, [model, dataset, *config_components], setup_result)

    def do_start(name_value, model_id, dataset_id, tokenizer_id, resume_id, notes_value, *values):
        if not model_id or not dataset_id:
            return t.note("Select a model and a dataset.", "warn"), *live_status()
        payload = dict(zip(order, values))
        try:
            config = _config_from(payload)
            experiment = manager.create_experiment(
                name=name_value or "", model_id=model_id, dataset_id=dataset_id,
                config=config, tokenizer_id=tokenizer_id or None, notes=notes_value or None,
            )
            manager.start(experiment["id"], resume_from=resume_id or None)
        except StudioError as exc:
            return t.error_message(exc), *live_status()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), *live_status()
        return (
            t.note(f"Started <b>{experiment['name']}</b>. Watch the live panel below.", "good"),
            *live_status(),
        )

    start_button.click(
        do_start,
        [run_name, model, dataset, tokenizer, resume_from, notes, *config_components],
        [setup_result, status_html, console],
    )

    def _control(action):
        def handler():
            experiment_id = manager.active_experiment_id()
            if not experiment_id:
                return t.note("No run is active.", "warn"), *live_status()
            try:
                getattr(manager, action)(experiment_id)
            except StudioError as exc:
                return t.error_message(exc), *live_status()
            return t.note(f"{action.replace('_', ' ').title()} requested.", "good"), *live_status()

        return handler

    pause_button.click(_control("pause"), outputs=[control_result, status_html, console])
    resume_button.click(_control("resume"), outputs=[control_result, status_html, console])
    stop_button.click(_control("stop"), outputs=[control_result, status_html, console])
    checkpoint_button.click(_control("request_checkpoint"), outputs=[control_result, status_html, console])

    def do_refresh():
        return (gr.update(choices=model_choices()), gr.update(choices=dataset_choices()),
                gr.update(choices=tokenizer_choices()), gr.update(choices=checkpoint_choices()))

    refresh_lists.click(do_refresh, outputs=[model, dataset, tokenizer, resume_from])
