"""Dataset Builder: turn documents or structured records into training datasets."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import gradio as gr

from ai_studio.core.errors import StudioError, ValidationError
from ai_studio.data.dataset_builder import (
    DATASET_MODES,
    MODE_LABELS,
    BuildOptions,
    build_from_documents,
    build_from_records,
    delete_dataset,
    list_datasets,
    preview_dataset,
    validate_records,
)
from ai_studio.data.ingestion import list_documents
from ai_studio.models.tokenizer_manager import list_tokenizers
from ai_studio.ui import theme as t

FORMAT_HELP = {
    "instruction": '{"instruction": "...", "input": "", "output": "..."}',
    "chat": '{"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}]}',
    "raw_lm": '{"text": "..."}',
}


def read_records(path: str) -> list[Any]:
    """Parse a records file. JSONL, a JSON array, or a CSV with a header row."""
    file = Path(path)
    suffix = file.suffix.lower()
    text = file.read_text(encoding="utf-8", errors="replace")

    if suffix in {".jsonl", ".ndjson"}:
        records: list[Any] = []
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Line {number} is not valid JSON: {exc.msg}") from exc
        return records

    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Not valid JSON: {exc.msg}") from exc
        if isinstance(payload, list):
            return payload
        for key in ("data", "records", "rows", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValidationError(
            "JSON must be an array of records, or an object with a data/records/rows array."
        )

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValidationError("The file has no header row, so its columns cannot be named.")
        return [dict(row) for row in reader]

    raise ValidationError(f"Unsupported records file {file.suffix!r} — use .jsonl, .json or .csv.")


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

    gr.Markdown("### Import records")
    gr.HTML(t.note(
        "Already have structured examples — a JSONL of instruction pairs, a CSV of "
        "questions and answers? Import them directly. The rows keep their structure "
        "instead of being flattened into raw text, which is what instruction tuning needs."))
    with gr.Row():
        records_file = gr.File(
            label="Records file (.jsonl, .json, .csv)",
            file_types=[".jsonl", ".ndjson", ".json", ".csv", ".tsv"],
            type="filepath",
        )
        with gr.Column():
            records_name = gr.Textbox(label="Dataset name", placeholder="my-instructions")
            records_mode = gr.Dropdown(
                [(MODE_LABELS[key], key) for key in DATASET_MODES], value="instruction",
                label="Mode",
            )
            records_description = gr.Textbox(label="Description", lines=2)
    with gr.Row():
        records_check = gr.Button("Check file", size="sm")
        records_build = gr.Button("Import as dataset", variant="primary")
    records_result = gr.HTML()
    records_preview = gr.Code(label="First rows", language="json")

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

    # ------------------------------------------------------- record import
    def _load_records(path, mode_value):
        if not path:
            raise ValidationError("Choose a file first.")
        records = read_records(path)
        if not records:
            raise ValidationError("The file contains no rows.")
        invalid, issues = validate_records(records, mode_value)
        return records, invalid, issues

    def do_check_records(path, mode_value):
        try:
            records, invalid, issues = _load_records(path, mode_value)
        except StudioError as exc:
            return t.error_message(exc), ""
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), ""

        valid = len(records) - invalid
        tone = "good" if invalid == 0 else ("warn" if valid else "bad")
        html = t.note(
            f"{t.fmt_number(len(records))} row(s) read — <b>{t.fmt_number(valid)} valid</b> for "
            f"{MODE_LABELS.get(mode_value, mode_value)}"
            + (f", {t.fmt_number(invalid)} would be skipped." if invalid else "."),
            tone,
        )
        for issue in issues[:5]:
            html += t.note(f"row {issue['row']}: {issue['error']}", "warn")
        if issues[5:]:
            html += t.note(f"…and {len(issues) - 5} more row(s) with problems.")
        return html, json.dumps(records[:5], indent=2, ensure_ascii=False, default=str)

    records_check.click(
        do_check_records, [records_file, records_mode], [records_result, records_preview]
    )

    def do_build_records(path, name_value, mode_value, description_value, train, validation,
                         test, dedupe_value, shuffle_value, seed_value):
        if not name_value or not name_value.strip():
            return t.note("Give the dataset a name.", "warn"), datasets_table(), gr.update()
        try:
            records, _invalid, _issues = _load_records(path, mode_value)
            options = BuildOptions(
                mode=mode_value,
                train_split=float(train),
                validation_split=float(validation),
                test_split=float(test),
                shuffle=bool(shuffle_value),
                seed=int(seed_value),
                deduplicate=bool(dedupe_value),
            )
            record = build_from_records(
                name_value.strip(), records, options, description=description_value or None
            )
        except StudioError as exc:
            return t.error_message(exc), datasets_table(), gr.update()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), datasets_table(), gr.update()

        stats = record["stats"]
        html = t.note(
            f"Imported <b>{record['name']}</b> — {t.fmt_number(record['rows'])} rows "
            f"({t.fmt_number(record['train_rows'])} train / "
            f"{t.fmt_number(record['validation_rows'])} val / "
            f"{t.fmt_number(record['test_rows'])} test).", "good",
        )
        if stats.get("invalid_records"):
            html += t.note(
                f"{t.fmt_number(stats['invalid_records'])} row(s) were skipped as invalid.", "warn")
        for warning in stats.get("warnings", [])[:4]:
            html += t.note(warning, "warn")
        return html, datasets_table(), gr.update(choices=dataset_choices())

    records_build.click(
        do_build_records,
        [records_file, records_name, records_mode, records_description, train_split, val_split,
         test_split, dedupe, shuffle, seed],
        [records_result, table, preview_target],
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
