"""Data Library: import documents, inspect them, preview and apply cleaning."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.data import ingestion
from ai_studio.data.loaders import SUPPORTED_EXTENSIONS
from ai_studio.data.preprocessing import CleaningOptions
from ai_studio.ui import theme as t

CLEAN_FIELDS = [
    ("normalize_unicode", "Normalise Unicode"),
    ("fix_pdf_wrapping", "Fix broken PDF line wrapping"),
    ("remove_page_numbers", "Remove page numbers"),
    ("remove_headers_footers", "Remove repeated headers/footers"),
    ("remove_repeated_whitespace", "Collapse repeated whitespace"),
    ("remove_duplicate_paragraphs", "Remove duplicate paragraphs"),
    ("remove_empty_sections", "Remove empty sections"),
    ("remove_html_tags", "Strip HTML tags"),
    ("preserve_code_blocks", "Preserve code blocks"),
    ("preserve_headings", "Preserve headings"),
]


def documents_table(search: str = "", doc_type: str = "all") -> str:
    rows = ingestion.list_documents(search=search or None, doc_type=doc_type)
    return t.table(
        ["Title", "File", "Type", "Words", "Est. tokens", "Size", "Cleaned", "Imported"],
        [
            [
                (row.get("title") or "—")[:48],
                row["filename"][:36],
                row["doc_type"],
                t.fmt_number(row["word_count"]),
                t.fmt_number(row["token_estimate"]),
                t.fmt_bytes(row["size_bytes"]),
                "yes" if row.get("cleaned_path") else "no",
                t.fmt_time(row["imported_at"]),
            ]
            for row in rows
        ],
        empty="No documents yet — import files, paste text, or import a folder.",
    )


def document_choices() -> list[tuple[str, str]]:
    return [
        (f"{(row.get('title') or row['filename'])[:50]} · {row['doc_type']}", row["id"])
        for row in ingestion.list_documents(limit=500)
    ]


def stats_html() -> str:
    stats = ingestion.library_stats()
    by_type = ", ".join(f"{count} {kind}" for kind, count in sorted(stats["by_type"].items())) or "—"
    return t.stat_grid([
        t.stat("Documents", t.fmt_number(stats["documents"]), by_type),
        t.stat("Cleaned", t.fmt_number(stats["cleaned"])),
        t.stat("Characters", t.fmt_params(stats["characters"])),
        t.stat("Estimated tokens", t.fmt_params(stats["tokens_estimated"])),
        t.stat("On disk", t.fmt_bytes(stats["disk_bytes"])),
    ])


def _options_from(values: list[bool]) -> CleaningOptions:
    return CleaningOptions(**{field: bool(value) for (field, _), value in zip(CLEAN_FIELDS, values)})


def render() -> None:
    gr.HTML('<div class="studio-title">Data Library</div>'
            '<div class="studio-sub">Books, papers, notes and datasets — originals are never modified</div>')
    stats = gr.HTML(stats_html)

    with gr.Tabs():
        with gr.Tab("Import files"):
            gr.Markdown(f"Supported: `{'`, `'.join(sorted(SUPPORTED_EXTENSIONS))}`")
            files = gr.File(label="Documents", file_count="multiple", type="filepath")
            with gr.Row():
                clean_on_import = gr.Checkbox(value=True, label="Clean after import")
                import_button = gr.Button("Import", variant="primary")
            import_result = gr.HTML()

        with gr.Tab("Paste text"):
            paste_title = gr.Textbox(label="Title", placeholder="Article or chapter title")
            paste_body = gr.Textbox(label="Text", lines=12, placeholder="Paste an article, notes, a chapter…")
            paste_button = gr.Button("Import text", variant="primary")
            paste_result = gr.HTML()

        with gr.Tab("Import folder"):
            folder = gr.Textbox(label="Folder path", placeholder="/path/to/my/library")
            with gr.Row():
                recursive = gr.Checkbox(value=True, label="Include subfolders")
                folder_limit = gr.Number(value=200, label="Max files", precision=0)
            folder_button = gr.Button("Import folder", variant="primary")
            folder_result = gr.HTML()

    gr.Markdown("### Cleaning")
    with gr.Row():
        clean_target = gr.Dropdown(choices=document_choices(), label="Document", interactive=True)
        reload_choices = gr.Button("↻", size="sm", scale=0)
    clean_toggles = [gr.Checkbox(value=True, label=label) for _, label in CLEAN_FIELDS]
    with gr.Row():
        preview_button = gr.Button("Preview cleaning")
        apply_button = gr.Button("Apply cleaning", variant="primary")
    clean_report = gr.HTML()
    with gr.Row():
        before = gr.Textbox(label="Before", lines=14, interactive=False, elem_classes="mono")
        after = gr.Textbox(label="After (preview)", lines=14, interactive=False, elem_classes="mono")

    gr.Markdown("### Documents")
    with gr.Row():
        search = gr.Textbox(label="Search", placeholder="title or filename")
        type_filter = gr.Dropdown(
            ["all", "txt", "markdown", "pdf", "docx", "epub", "html", "csv", "json", "jsonl", "paste"],
            value="all", label="Type",
        )
    table = gr.HTML(documents_table)
    with gr.Row():
        delete_target = gr.Dropdown(choices=document_choices(), label="Delete document")
        delete_button = gr.Button("Delete", variant="stop", scale=0)
    delete_result = gr.HTML()

    # ---------------------------------------------------------------- events
    def do_import(paths, clean):
        if not paths:
            return t.note("Choose at least one file.", "warn"), stats_html(), documents_table(), gr.update(), gr.update()
        imported, failed = [], []
        for path in paths:
            try:
                record = ingestion.import_upload(path, clean=bool(clean))
                imported.append(record)
            except StudioError as exc:
                failed.append((str(path).split("/")[-1], exc.message))
            except Exception as exc:  # noqa: BLE001
                failed.append((str(path).split("/")[-1], f"{type(exc).__name__}: {exc}"))

        html = t.note(f"Imported <b>{len(imported)}</b> document(s).", "good" if imported else "warn")
        if imported:
            html += t.table(
                ["Title", "Type", "Words", "Est. tokens"],
                [[r.get("title"), r["doc_type"], t.fmt_number(r["word_count"]),
                  t.fmt_number(r["token_estimate"])] for r in imported],
            )
        if failed:
            html += t.note(
                "Failed:<br>" + "<br>".join(f"<code>{name}</code>: {reason}" for name, reason in failed),
                "bad",
            )
        choices = document_choices()
        return html, stats_html(), documents_table(), gr.update(choices=choices), gr.update(choices=choices)

    import_button.click(
        do_import, [files, clean_on_import],
        [import_result, stats, table, clean_target, delete_target],
    )

    def do_paste(title, body):
        try:
            record = ingestion.import_text(body, title=title or None)
        except StudioError as exc:
            return t.error_message(exc), stats_html(), documents_table(), gr.update(), gr.update()
        choices = document_choices()
        return (
            t.note(f"Imported <b>{record['title']}</b> ({t.fmt_number(record['word_count'])} words).", "good"),
            stats_html(), documents_table(), gr.update(choices=choices), gr.update(choices=choices),
        )

    paste_button.click(
        do_paste, [paste_title, paste_body],
        [paste_result, stats, table, clean_target, delete_target],
    )

    def do_folder(path, recurse, limit):
        try:
            outcome = ingestion.import_folder(path, recursive=bool(recurse), limit=int(limit) or None)
        except StudioError as exc:
            return t.error_message(exc), stats_html(), documents_table(), gr.update(), gr.update()
        html = t.note(f"Folder import: <b>{outcome.summary}</b>.", "good" if outcome.imported else "warn")
        if outcome.failed:
            html += t.table(["File", "Reason"], [[name, reason] for name, reason in outcome.failed[:30]])
        choices = document_choices()
        return html, stats_html(), documents_table(), gr.update(choices=choices), gr.update(choices=choices)

    folder_button.click(
        do_folder, [folder, recursive, folder_limit],
        [folder_result, stats, table, clean_target, delete_target],
    )

    def do_preview(document_id, *toggles):
        if not document_id:
            return t.note("Select a document.", "warn"), "", ""
        try:
            preview = ingestion.preview_cleaning(document_id, _options_from(list(toggles)))
        except StudioError as exc:
            return t.error_message(exc), "", ""
        return t.note(preview["report"].summary()), preview["before"], preview["after"]

    preview_button.click(do_preview, [clean_target, *clean_toggles], [clean_report, before, after])

    def do_apply(document_id, *toggles):
        if not document_id:
            return t.note("Select a document.", "warn"), documents_table(), stats_html()
        try:
            record = ingestion.clean_document(document_id, _options_from(list(toggles)))
        except StudioError as exc:
            return t.error_message(exc), documents_table(), stats_html()
        report = (record.get("meta") or {}).get("cleaning", {}).get("report", {})
        return (
            t.note(
                f"Cleaned <b>{record['title']}</b> — {report.get('original_chars', 0):,} → "
                f"{report.get('cleaned_chars', 0):,} characters. The original file is untouched.",
                "good",
            ),
            documents_table(), stats_html(),
        )

    apply_button.click(do_apply, [clean_target, *clean_toggles], [clean_report, table, stats])

    def do_delete(document_id):
        if not document_id:
            return t.note("Select a document.", "warn"), documents_table(), stats_html(), gr.update(), gr.update()
        try:
            ingestion.delete_document(document_id)
        except StudioError as exc:
            return t.error_message(exc), documents_table(), stats_html(), gr.update(), gr.update()
        choices = document_choices()
        return (t.note("Document deleted.", "good"), documents_table(), stats_html(),
                gr.update(choices=choices, value=None), gr.update(choices=choices, value=None))

    delete_button.click(do_delete, delete_target,
                        [delete_result, table, stats, clean_target, delete_target])

    for control in (search, type_filter):
        control.change(documents_table, [search, type_filter], table)

    def _reload():
        choices = document_choices()
        return gr.update(choices=choices), gr.update(choices=choices), documents_table()

    reload_choices.click(_reload, outputs=[clean_target, delete_target, table])
