"""Experiments screen: history, details, comparison and duplication."""

from __future__ import annotations

import json

import gradio as gr

from ai_studio.core.database import get_db
from ai_studio.core.errors import StudioError
from ai_studio.training.config import TrainingConfig
from ai_studio.training.worker import manager
from ai_studio.ui import theme as t


def experiments_table() -> str:
    db = get_db()
    rows = []
    for record in db.list("experiments", order_by="created_at DESC", limit=200):
        model = db.get("models", record.get("model_id") or "") or {}
        dataset = db.get("datasets", record.get("dataset_id") or "") or {}
        rows.append([
            record["name"][:36], record["method"], record["status"],
            model.get("name", "—")[:22], dataset.get("name", "—")[:22],
            t.fmt_number(record.get("final_train_loss"), 4),
            t.fmt_number(record.get("best_val_loss"), 4),
            t.fmt_number(record.get("current_step")),
            t.fmt_duration(record.get("duration_seconds")),
            t.fmt_time(record["created_at"]),
        ])
    return t.table(
        ["Name", "Method", "Status", "Model", "Dataset", "Train loss", "Val loss", "Steps", "Duration", "Created"],
        rows, empty="No experiments yet.",
    )


def experiment_choices() -> list[tuple[str, str]]:
    return [
        (f"{r['name'][:44]} · {r['status']}", r["id"])
        for r in get_db().list("experiments", order_by="created_at DESC", limit=200)
    ]


def detail_html(experiment_id: str) -> str:
    if not experiment_id:
        return t.note("Select an experiment.")
    db = get_db()
    try:
        record = db.require("experiments", experiment_id)
    except StudioError as exc:
        return t.error_message(exc)

    model = db.get("models", record.get("model_id") or "") or {}
    dataset = db.get("datasets", record.get("dataset_id") or "") or {}
    checkpoints = db.count("checkpoints", "experiment_id = ?", (experiment_id,))
    hyper = record.get("hyperparameters") or {}

    html = t.stat_grid([
        t.stat("Status", record["status"]),
        t.stat("Method", record["method"]),
        t.stat("Steps", t.fmt_number(record.get("current_step"))),
        t.stat("Final train loss", t.fmt_number(record.get("final_train_loss"), 4)),
        t.stat("Best val loss", t.fmt_number(record.get("best_val_loss"), 4)),
        t.stat("Tokens", t.fmt_params(record.get("tokens_processed"))),
        t.stat("Duration", t.fmt_duration(record.get("duration_seconds"))),
        t.stat("Checkpoints", checkpoints),
    ])
    html += t.table(
        ["Field", "Value"],
        [
            ["Base model", model.get("name", "—")],
            ["Dataset", dataset.get("name", "—")],
            ["Dataset hash", (record.get("dataset_hash") or "—")[:16]],
            ["Tokenizer", record.get("tokenizer_id") or "—"],
            ["Best checkpoint", record.get("best_checkpoint_id") or "—"],
            ["Started", t.fmt_time(record.get("started_at"))],
            ["Ended", t.fmt_time(record.get("ended_at"))],
            ["Notes", record.get("notes") or "—"],
        ],
    )
    if record.get("error"):
        html += t.note(record["error"], "bad")
    html += "<br>" + t.table(
        ["Hyperparameter", "Value"],
        [[key, json.dumps(value) if isinstance(value, dict) else value]
         for key, value in sorted(hyper.items()) if key != "lora"],
    )
    return html


def render() -> None:
    gr.HTML('<div class="studio-title">Experiments</div>'
            '<div class="studio-sub">Every training run, with its exact configuration</div>')
    table = gr.HTML(experiments_table)

    with gr.Row():
        target = gr.Dropdown(choices=experiment_choices(), label="Experiment")
        refresh_button = gr.Button("↻", size="sm", scale=0)
    detail = gr.HTML()

    with gr.Row():
        notes_box = gr.Textbox(label="Notes", lines=2)
        save_notes = gr.Button("Save notes", size="sm")
        duplicate_button = gr.Button("Duplicate configuration")
        delete_button = gr.Button("Delete", variant="stop")
    action_result = gr.HTML()

    gr.Markdown("### Compare")
    compare_targets = gr.Dropdown(choices=experiment_choices(), label="Experiments", multiselect=True)
    compare_button = gr.Button("Compare")
    compare_out = gr.HTML()

    target.change(detail_html, target, detail)
    refresh_button.click(
        lambda: (gr.update(choices=experiment_choices()), gr.update(choices=experiment_choices()),
                 experiments_table()),
        outputs=[target, compare_targets, table],
    )

    def do_notes(experiment_id, text):
        if not experiment_id:
            return t.note("Select an experiment.", "warn")
        get_db().update("experiments", experiment_id, {"notes": text})
        return t.note("Notes saved.", "good")

    save_notes.click(do_notes, [target, notes_box], action_result)

    def do_duplicate(experiment_id):
        if not experiment_id:
            return t.note("Select an experiment.", "warn"), experiments_table(), gr.update()
        db = get_db()
        try:
            source = db.require("experiments", experiment_id)
            config = TrainingConfig.from_dict(source.get("hyperparameters") or {})
            record = manager.create_experiment(
                name=f"{source['name']} (copy)",
                model_id=source["model_id"], dataset_id=source["dataset_id"],
                config=config, tokenizer_id=source.get("tokenizer_id"),
                notes=f"Duplicated from {experiment_id}",
            )
        except StudioError as exc:
            return t.error_message(exc), experiments_table(), gr.update()
        return (
            t.note(f"Created <b>{record['name']}</b> (queued). Start it from the Training tab.", "good"),
            experiments_table(), gr.update(choices=experiment_choices()),
        )

    duplicate_button.click(do_duplicate, target, [action_result, table, target])

    def do_delete(experiment_id):
        if not experiment_id:
            return t.note("Select an experiment.", "warn"), experiments_table(), gr.update()
        get_db().delete("experiments", experiment_id)
        return (t.note("Experiment deleted (checkpoints are kept).", "good"),
                experiments_table(), gr.update(choices=experiment_choices(), value=None))

    delete_button.click(do_delete, target, [action_result, table, target])

    def do_compare(experiment_ids):
        if not experiment_ids or len(experiment_ids) < 2:
            return t.note("Select at least two experiments.", "warn")
        db = get_db()
        records = [db.require("experiments", experiment_id) for experiment_id in experiment_ids]
        fields = [
            ("Method", lambda r: r["method"]),
            ("Status", lambda r: r["status"]),
            ("Final train loss", lambda r: t.fmt_number(r.get("final_train_loss"), 4)),
            ("Best val loss", lambda r: t.fmt_number(r.get("best_val_loss"), 4)),
            ("Steps", lambda r: t.fmt_number(r.get("current_step"))),
            ("Duration", lambda r: t.fmt_duration(r.get("duration_seconds"))),
            ("Tokens", lambda r: t.fmt_params(r.get("tokens_processed"))),
        ]
        rows = [[label] + [getter(record) for record in records] for label, getter in fields]

        hyper_keys: set[str] = set()
        for record in records:
            hyper_keys.update(
                key for key, value in (record.get("hyperparameters") or {}).items()
                if not isinstance(value, dict)
            )
        for key in sorted(hyper_keys):
            values = [(record.get("hyperparameters") or {}).get(key) for record in records]
            if len({json.dumps(value, default=str) for value in values}) > 1:
                rows.append([f"⚠ {key}"] + [str(value) for value in values])

        return t.table(["Field"] + [r["name"][:22] for r in records], rows) + t.note(
            "Rows marked ⚠ differ between the selected experiments."
        )

    compare_button.click(do_compare, compare_targets, compare_out)
