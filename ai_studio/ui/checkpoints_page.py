"""Checkpoints screen: browse, verify, load, resume, export, delete."""

from __future__ import annotations

import gradio as gr

from ai_studio.core.database import get_db
from ai_studio.core.errors import StudioError
from ai_studio.models.model_manager import register_checkpoint_as_model
from ai_studio.training.checkpoint_manager import (
    delete_checkpoint,
    list_checkpoints,
    verify_checkpoint,
)
from ai_studio.ui import theme as t


def checkpoints_table() -> str:
    db = get_db()
    rows = []
    for record in list_checkpoints():
        experiment = db.get("experiments", record.get("experiment_id") or "") or {}
        rows.append([
            record["name"], experiment.get("name", "—")[:28], t.fmt_number(record["step"]),
            t.fmt_number(record["epoch"], 2), t.fmt_number(record.get("train_loss"), 4),
            t.fmt_number(record.get("val_loss"), 4), t.fmt_bytes(record["size_bytes"]),
            "best" if record.get("is_best") else "", "yes" if record.get("has_optimizer_state") else "no",
            t.fmt_time(record["created_at"]),
        ])
    return t.table(
        ["Name", "Experiment", "Step", "Epoch", "Train loss", "Val loss", "Size", "Flag", "Resumable", "Created"],
        rows,
        empty="No checkpoints yet — they are written during training.",
    )


def checkpoint_choices() -> list[tuple[str, str]]:
    return [
        (f"{r['name']} · step {r['step']} · val {t.fmt_number(r.get('val_loss'), 4)}", r["id"])
        for r in list_checkpoints()
    ]


def render() -> None:
    gr.HTML('<div class="studio-title">Checkpoints</div>'
            '<div class="studio-sub">Every checkpoint stores weights, optimizer state, tokenizer '
            'and configuration</div>')
    table = gr.HTML(checkpoints_table)

    with gr.Row():
        target = gr.Dropdown(choices=checkpoint_choices(), label="Checkpoint")
        refresh_button = gr.Button("↻", size="sm", scale=0)
    with gr.Row():
        verify_button = gr.Button("Verify")
        load_button = gr.Button("Register as model", variant="primary")
        delete_button = gr.Button("Delete", variant="stop")
    model_name = gr.Textbox(label="New model name (optional)", placeholder="auto")
    result = gr.HTML()

    def do_verify(checkpoint_id):
        if not checkpoint_id:
            return t.note("Select a checkpoint.", "warn")
        record = get_db().require("checkpoints", checkpoint_id)
        report = verify_checkpoint(record["path"])
        html = t.stat_grid([
            t.stat("Weights", "present" if report["has_weights"] else "MISSING"),
            t.stat("Training state", "present" if report["has_state"] else "missing"),
            t.stat("Tokenizer", "present" if report["has_tokenizer"] else "missing"),
            t.stat("Config", "present" if report["has_config"] else "missing"),
            t.stat("Resumable", "yes" if report["resumable"] else "no"),
        ])
        html += t.note(f"<code>{record['path']}</code>")
        for problem in report["problems"]:
            html += t.note(problem, "warn" if report["has_weights"] else "bad")
        if not report["problems"]:
            html += t.note("Checkpoint is complete.", "good")
        return html

    verify_button.click(do_verify, target, result)

    def do_load(checkpoint_id, name_value):
        if not checkpoint_id:
            return t.note("Select a checkpoint.", "warn"), checkpoints_table()
        try:
            record = register_checkpoint_as_model(checkpoint_id, name_value or None)
        except StudioError as exc:
            return t.error_message(exc), checkpoints_table()
        return (
            t.note(
                f"Registered as model <b>{record['name']}</b> — it is now available in "
                f"<b>Chat</b>, <b>Benchmarks</b> and <b>Training</b>.", "good",
            ),
            checkpoints_table(),
        )

    load_button.click(do_load, [target, model_name], [result, table])

    def do_delete(checkpoint_id):
        if not checkpoint_id:
            return t.note("Select a checkpoint.", "warn"), checkpoints_table(), gr.update()
        try:
            delete_checkpoint(checkpoint_id)
        except StudioError as exc:
            return t.error_message(exc), checkpoints_table(), gr.update()
        return (t.note("Checkpoint deleted.", "good"), checkpoints_table(),
                gr.update(choices=checkpoint_choices(), value=None))

    delete_button.click(do_delete, target, [result, table, target])
    refresh_button.click(
        lambda: (gr.update(choices=checkpoint_choices()), checkpoints_table()),
        outputs=[target, table],
    )
