"""Dataset Builder: turn documents into training datasets."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.data.dataset_builder import (
    DATASET_MODES,
    MODE_LABELS,
    BuildOptions,
    build_from_documents,
    delete_dataset,
    list_datasets,
    preview_dataset,
)
from ai_studio.data.ingestion import list_documents
from ai_studio.models.tokenizer_manager import list_tokenizers
from ai_studio.ui import theme as t

FORMAT_HELP = {
    "instruction": '{"instruction": "...", "input": "", "output": "..."}',
    "chat": '{"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}]}',
    "raw_lm": '{"text": "..."}',
}


def datasets_table() -> str:
    rows = list_datasets()
    return t.table(
        ["Name", "Mode", "Rows", "Train", "Val", "Test", "Tokens", "Tokens from", "Synthetic", "Created"],
        [
            [
                row["name"], MODE_LABELS.get(row["mode"], row["mode"]),
                t.fmt_number(row["rows"]), t.fmt_number(row["train_rows"]),
                t.fmt_number(row["validation_rows"]), t.fmt_number(row["test_rows"]),
                t.fmt_params(row["token_count"]), row["token_method"],
                "yes" if row["is_synthetic"] else "no", t.fmt_time(row["created_at"]),
            ]
            for row in rows
        ],
        empty="No datasets yet — select documents above and build one.",
    )


def dataset_choices() -> list[tuple[str, str]]:
    return [(f"{row['name']} ({row['mode']}, {row['rows']:,} rows)", row["id"]) for row in list_datasets()]


def document_choices() -> list[tuple[str, str]]:
    return [
        (f"{(row.get('title') or row['filename'])[:50]} · {t.fmt_number(row['word_count'])} words", row["id"])
        for row in list_documents(limit=500)
    ]


def tokenizer_choices() -> list[tuple[str, str]]:
    return [("(character estimate — no tokenizer)", "")] + [
        (f"{row['name']} · {row['kind']} · {row['vocab_size']:,}", row["id"]) for row in list_tokenizers()
    ]


def render() -> None:
    gr.HTML('<div class="studio-title">Dataset Builder</div>'
            '<div class="studio-sub">Chunk documents into training sequences</div>')

    with gr.Row():
        with gr.Column(scale=2):
            name = gr.Textbox(label="Dataset name", placeholder="my-corpus-v1")
            documents = gr.Dropdown(
                choices=document_choices(), label="Documents", multiselect=True, interactive=True
            )
            with gr.Row():
                mode = gr.Dropdown(
                    [(MODE_LABELS[m], m) for m in DATASET_MODES], value="raw_lm", label="Mode"
                )
                tokenizer = gr.Dropdown(choices=tokenizer_choices(), value="", label="Tokenizer")
            description = gr.Textbox(label="Description", placeholder="optional")
        with gr.Column(scale=1):
            block_size = gr.Slider(64, 8192, value=1024, step=64, label="Context length (tokens)")
            overlap = gr.Slider(0, 1024, value=128, step=16, label="Overlap")
            with gr.Row():
                train_split = gr.Slider(0.5, 1.0, value=0.90, step=0.01, label="Train")
                val_split = gr.Slider(0.0, 0.4, value=0.05, step=0.01, label="Validation")
                test_split = gr.Slider(0.0, 0.4, value=0.05, step=0.01, label="Test")
            with gr.Row():
                dedupe = gr.Checkbox(value=True, label="Remove duplicates")
                shuffle = gr.Checkbox(value=True, label="Shuffle")
                group = gr.Checkbox(value=True, label="Split by document (no leakage)")
            seed = gr.Number(value=42, label="Seed", precision=0)

    format_hint = gr.HTML()
    with gr.Row():
        refresh_button = gr.Button("↻ Refresh lists", size="sm")
        build_button = gr.Button("Build dataset", variant="primary")
    build_result = gr.HTML()

    gr.Markdown("### Datasets")
    table = gr.HTML(datasets_table)
    with gr.Row():
        preview_target = gr.Dropdown(choices=dataset_choices(), label="Preview dataset")
        split_choice = gr.Dropdown(["train", "validation", "test"], value="train", label="Split")
        preview_button = gr.Button("Preview", size="sm")
        delete_button = gr.Button("Delete", variant="stop", size="sm")
    preview_out = gr.Code(label="Rows", language="json")
    action_result = gr.HTML()

    def show_format(selected_mode: str) -> str:
        example = FORMAT_HELP.get(selected_mode)
        if selected_mode in {"continued_pretraining", "retrieval"}:
            example = FORMAT_HELP["raw_lm"]
        if selected_mode == "qa":
            example = '{"question": "...", "context": "...", "answer": "..."}'
        return t.note(f"Record format: <code>{example}</code>") if example else ""

    mode.change(show_format, mode, format_hint)

    def do_build(name_value, document_ids, mode_value, tokenizer_id, description_value,
                 block, overlap_value, train, validation, test, dedupe_value, shuffle_value,
                 group_value, seed_value, progress=gr.Progress()):
        if not name_value or not name_value.strip():
            return t.note("Give the dataset a name.", "warn"), datasets_table(), gr.update()
        if not document_ids:
            return t.note("Select at least one document.", "warn"), datasets_table(), gr.update()
        try:
            options = BuildOptions(
                mode=mode_value,
                block_size=int(block),
                overlap=int(overlap_value),
                train_split=float(train),
                validation_split=float(validation),
                test_split=float(test),
                shuffle=bool(shuffle_value),
                seed=int(seed_value),
                deduplicate=bool(dedupe_value),
                group_by_document=bool(group_value),
                tokenizer_id=tokenizer_id or None,
            )
            record = build_from_documents(
                name_value.strip(), list(document_ids), options,
                description=description_value or None, progress=progress,
            )
        except StudioError as exc:
            return t.error_message(exc), datasets_table(), gr.update()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), datasets_table(), gr.update()

        stats = record["stats"]
        html = t.note(
            f"Built <b>{record['name']}</b> — {t.fmt_number(record['rows'])} rows "
            f"({t.fmt_number(record['train_rows'])} train / {t.fmt_number(record['validation_rows'])} val / "
            f"{t.fmt_number(record['test_rows'])} test), ~{t.fmt_params(record['token_count'])} tokens "
            f"({record['token_method']}).", "good",
        )
        for warning in stats.get("warnings", [])[:4]:
            html += t.note(warning, "warn")
        return html, datasets_table(), gr.update(choices=dataset_choices())

    build_button.click(
        do_build,
        [name, documents, mode, tokenizer, description, block_size, overlap, train_split,
         val_split, test_split, dedupe, shuffle, group, seed],
        [build_result, table, preview_target],
    )

    def do_preview(dataset_id, split):
        if not dataset_id:
            return "", t.note("Select a dataset.", "warn")
        try:
            preview = preview_dataset(dataset_id, split, limit=5)
        except StudioError as exc:
            return "", t.error_message(exc)
        text = "\n".join(json.dumps(row, ensure_ascii=False, indent=2) for row in preview["rows"])
        return text or "(this split is empty)", t.note(
            f"{preview['dataset']['name']} · {split} · showing {preview['count']} rows"
        )

    preview_button.click(do_preview, [preview_target, split_choice], [preview_out, action_result])

    def do_delete(dataset_id):
        if not dataset_id:
            return t.note("Select a dataset.", "warn"), datasets_table(), gr.update()
        try:
            delete_dataset(dataset_id)
        except StudioError as exc:
            return t.error_message(exc), datasets_table(), gr.update()
        return (t.note("Dataset deleted.", "good"), datasets_table(),
                gr.update(choices=dataset_choices(), value=None))

    delete_button.click(do_delete, preview_target, [action_result, table, preview_target])

    def do_refresh():
        return (gr.update(choices=document_choices()), gr.update(choices=tokenizer_choices()),
                gr.update(choices=dataset_choices()), datasets_table())

    refresh_button.click(do_refresh, outputs=[documents, tokenizer, preview_target, table])
