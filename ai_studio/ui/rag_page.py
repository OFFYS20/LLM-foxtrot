"""Knowledge / RAG screen: build indexes and test retrieval."""

from __future__ import annotations

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.data.ingestion import list_documents
from ai_studio.rag.embeddings import SUGGESTED_MODELS
from ai_studio.rag.retrieval import build_index, delete_index, list_indexes, search
from ai_studio.rag.vector_store import faiss_available
from ai_studio.ui import theme as t


def indexes_table() -> str:
    return t.table(
        ["Name", "Embedding model", "Chunks", "Docs", "Chunk size", "Dim", "Backend", "Size", "Created"],
        [
            [
                row["name"], row["embedding_model"].split("/")[-1], t.fmt_number(row["chunk_count"]),
                len(row.get("document_ids") or []), row["chunk_size"], row["dimension"],
                (row.get("config") or {}).get("backend", "—"),
                t.fmt_bytes((row.get("stats") or {}).get("size_bytes")),
                t.fmt_time(row["created_at"]),
            ]
            for row in list_indexes()
        ],
        empty="No knowledge indexes yet.",
    )


def index_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {r['chunk_count']:,} chunks", r["id"]) for r in list_indexes()]


def document_choices() -> list[tuple[str, str]]:
    return [((row.get("title") or row["filename"])[:60], row["id"]) for row in list_documents(limit=500)]


def render() -> None:
    gr.HTML('<div class="studio-title">Knowledge / RAG</div>'
            '<div class="studio-sub">Give a model access to your library without retraining</div>')
    gr.HTML(t.note(
        f"Vector backend: <b>{'FAISS' if faiss_available() else 'NumPy exact search'}</b>. "
        "Embeddings come from sentence-transformers; the first build downloads the embedding model."
    ))

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Index name", placeholder="my-library")
            documents = gr.Dropdown(choices=document_choices(), label="Documents",
                                    multiselect=True, interactive=True)
            embedding_model = gr.Dropdown(
                [model for model, _ in SUGGESTED_MODELS],
                value=SUGGESTED_MODELS[0][0], label="Embedding model", allow_custom_value=True,
            )
        with gr.Column():
            chunk_size = gr.Slider(200, 4000, value=800, step=50, label="Chunk size (characters)")
            chunk_overlap = gr.Slider(0, 800, value=120, step=10, label="Chunk overlap")
            gr.HTML(t.table(["Model", "Notes"], [[m, note] for m, note in SUGGESTED_MODELS]))

    with gr.Row():
        refresh_button = gr.Button("↻ Refresh documents", size="sm")
        build_button = gr.Button("Build index", variant="primary")
    build_result = gr.HTML()

    gr.Markdown("### Indexes")
    table = gr.HTML(indexes_table)

    gr.Markdown("### Test retrieval")
    with gr.Row():
        target = gr.Dropdown(choices=index_choices(), label="Index")
        query = gr.Textbox(label="Query", placeholder="What does the book say about…")
        top_k = gr.Slider(1, 10, value=4, step=1, label="Results")
    with gr.Row():
        search_button = gr.Button("Search", variant="primary")
        delete_button = gr.Button("Delete index", variant="stop")
    search_result = gr.HTML()

    def do_build(name_value, document_ids, model_name, size, overlap, progress=gr.Progress()):
        if not name_value or not name_value.strip():
            return t.note("Give the index a name.", "warn"), indexes_table(), gr.update()
        if not document_ids:
            return t.note("Select at least one document.", "warn"), indexes_table(), gr.update()
        try:
            record = build_index(
                name_value.strip(), list(document_ids), embedding_model=model_name,
                chunk_size=int(size), chunk_overlap=int(overlap), progress=progress,
            )
        except StudioError as exc:
            return t.error_message(exc), indexes_table(), gr.update()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), indexes_table(), gr.update()

        stats = record["stats"]
        html = t.note(
            f"Built <b>{record['name']}</b> — {record['chunk_count']:,} chunks from "
            f"{stats['documents']} documents in {stats['embedding_seconds']}s "
            f"({record['dimension']}-dim, {(record.get('config') or {}).get('backend')}).", "good",
        )
        if stats.get("skipped"):
            html += t.note("Skipped: " + "; ".join(str(s) for s in stats["skipped"][:5]), "warn")
        return html, indexes_table(), gr.update(choices=index_choices())

    build_button.click(
        do_build, [name, documents, embedding_model, chunk_size, chunk_overlap],
        [build_result, table, target],
    )

    def do_search(index_id, query_text, k):
        if not index_id:
            return t.note("Select an index.", "warn")
        if not query_text or not query_text.strip():
            return t.note("Enter a query.", "warn")
        try:
            hits = search(index_id, query_text, top_k=int(k))
        except StudioError as exc:
            return t.error_message(exc)
        if not hits:
            return t.note("No matches.", "warn")
        return t.table(
            ["#", "Similarity", "Source", "Excerpt"],
            [[hit.rank, f"{hit.score:.3f}", hit.chunk.source[:30],
              hit.chunk.text[:220].replace("<", "&lt;") + "…"] for hit in hits],
        )

    search_button.click(do_search, [target, query, top_k], search_result)

    def do_delete(index_id):
        if not index_id:
            return t.note("Select an index.", "warn"), indexes_table(), gr.update()
        try:
            delete_index(index_id)
        except StudioError as exc:
            return t.error_message(exc), indexes_table(), gr.update()
        return (t.note("Index deleted.", "good"), indexes_table(),
                gr.update(choices=index_choices(), value=None))

    delete_button.click(do_delete, target, [search_result, table, target])
    refresh_button.click(lambda: gr.update(choices=document_choices()), outputs=documents)
