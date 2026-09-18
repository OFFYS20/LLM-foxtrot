"""Tokenizer screen: train, inspect and test tokenizers."""

from __future__ import annotations

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.data.ingestion import list_documents
from ai_studio.models.tokenizer_manager import (
    TOKENIZER_KINDS,
    TokenizerConfig,
    delete_tokenizer,
    inspect_tokenizer,
    list_tokenizers,
    register_pretrained_tokenizer,
    preview_tokenization,
    train_tokenizer,
)
from ai_studio.ui import theme as t


def tokenizers_table() -> str:
    return t.table(
        ["Name", "Kind", "Vocab", "Trained on", "Source", "Created"],
        [
            [
                row["name"], row["kind"], t.fmt_number(row["vocab_size"]),
                f"{(row.get('stats') or {}).get('documents', 0)} docs / "
                f"{t.fmt_params((row.get('stats') or {}).get('characters'))} chars",
                row["source"], t.fmt_time(row["created_at"]),
            ]
            for row in list_tokenizers()
        ],
        empty="No tokenizers yet.",
    )


def tokenizer_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {r['kind']} · {r['vocab_size']:,}", r["id"]) for r in list_tokenizers()]


def document_choices() -> list[tuple[str, str]]:
    return [((row.get("title") or row["filename"])[:60], row["id"]) for row in list_documents(limit=500)]


def render() -> None:
    gr.HTML('<div class="studio-title">Tokenizer</div>'
            '<div class="studio-sub">Train a vocabulary from your own documents</div>')

    with gr.Tabs():
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="Name", placeholder="my-tokenizer")
                    kind = gr.Dropdown(list(TOKENIZER_KINDS), value="byte_level_bpe", label="Algorithm")
                    documents = gr.Dropdown(choices=document_choices(), label="Documents",
                                            multiselect=True, interactive=True)
                    extra_text = gr.Textbox(label="Additional text (optional)", lines=4)
                with gr.Column():
                    vocab_size = gr.Slider(256, 100_000, value=8000, step=256, label="Vocabulary size")
                    min_frequency = gr.Slider(1, 20, value=2, step=1, label="Minimum frequency")
                    lowercase = gr.Checkbox(value=False, label="Lowercase")
                    with gr.Row():
                        unk = gr.Textbox(value="<unk>", label="Unknown")
                        bos = gr.Textbox(value="<s>", label="BOS")
                    with gr.Row():
                        eos = gr.Textbox(value="</s>", label="EOS")
                        pad = gr.Textbox(value="<pad>", label="Padding")
            with gr.Row():
                refresh_docs = gr.Button("↻ Refresh documents", size="sm")
                train_button = gr.Button("Train tokenizer", variant="primary")
            train_result = gr.HTML()

        with gr.Tab("Import from Hugging Face"):
            repo = gr.Textbox(label="Repository", placeholder="Qwen/Qwen2.5-0.5B")
            hf_name = gr.Textbox(label="Local name (optional)")
            hf_button = gr.Button("Fetch tokenizer", variant="primary")
            hf_result = gr.HTML()

    gr.Markdown("### Tokenizers")
    table = gr.HTML(tokenizers_table)

    gr.Markdown("### Inspect & test")
    with gr.Row():
        target = gr.Dropdown(choices=tokenizer_choices(), label="Tokenizer")
        delete_button = gr.Button("Delete", variant="stop", size="sm", scale=0)
    sample = gr.Textbox(
        label="Sample text",
        value="The transformer architecture uses self-attention over token embeddings.",
        lines=3,
    )
    with gr.Row():
        test_button = gr.Button("Tokenize")
        inspect_button = gr.Button("Inspect vocabulary")
    result = gr.HTML()
    tokens_out = gr.HighlightedText(label="Tokens", show_legend=False)

    # -------------------------------------------------------------- handlers
    def do_train(name_value, kind_value, document_ids, extra, vocab, frequency,
                 lower, unk_v, bos_v, eos_v, pad_v, progress=gr.Progress()):
        try:
            config = TokenizerConfig(
                name=(name_value or "").strip(), kind=kind_value, vocab_size=int(vocab),
                min_frequency=int(frequency), lowercase=bool(lower),
                unk_token=unk_v, bos_token=bos_v, eos_token=eos_v, pad_token=pad_v,
            )
            record = train_tokenizer(
                config, texts=[extra] if extra and extra.strip() else None,
                document_ids=list(document_ids or []), progress=progress,
            )
        except StudioError as exc:
            return t.error_message(exc), tokenizers_table(), gr.update()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), tokenizers_table(), gr.update()

        stats = record["stats"]
        return (
            t.note(
                f"Trained <b>{record['name']}</b> — vocabulary {record['vocab_size']:,} "
                f"(requested {stats['requested_vocab_size']:,}) from "
                f"{t.fmt_params(stats['characters'])} characters in {stats['training_seconds']}s.",
                "good",
            ),
            tokenizers_table(), gr.update(choices=tokenizer_choices()),
        )

    train_button.click(
        do_train,
        [name, kind, documents, extra_text, vocab_size, min_frequency, lowercase, unk, bos, eos, pad],
        [train_result, table, target],
    )

    def do_hf(repo_value, name_value):
        try:
            record = register_pretrained_tokenizer(repo_value.strip(), name_value or None)
        except StudioError as exc:
            return t.error_message(exc), tokenizers_table(), gr.update()
        return (t.note(f"Registered <b>{record['name']}</b> ({record['vocab_size']:,} tokens).", "good"),
                tokenizers_table(), gr.update(choices=tokenizer_choices()))

    hf_button.click(do_hf, [repo, hf_name], [hf_result, table, target])

    def do_test(tokenizer_id, text):
        if not tokenizer_id:
            return t.note("Select a tokenizer.", "warn"), []
        try:
            outcome = preview_tokenization(tokenizer_id, text)
        except StudioError as exc:
            return t.error_message(exc), []
        cards = t.stat_grid([
            t.stat("Tokens", t.fmt_number(outcome["token_count"])),
            t.stat("Characters", t.fmt_number(outcome["character_count"])),
            t.stat("Chars / token", outcome["chars_per_token"]),
            t.stat("Round-trips exactly", "yes" if outcome["roundtrip_ok"] else "no"),
        ])
        highlighted = [(token.replace("Ġ", " ").replace("▁", " "), str(index % 6))
                       for index, token in enumerate(outcome["tokens"])]
        return cards, highlighted

    test_button.click(do_test, [target, sample], [result, tokens_out])

    def do_inspect(tokenizer_id):
        if not tokenizer_id:
            return t.note("Select a tokenizer.", "warn"), []
        try:
            info = inspect_tokenizer(tokenizer_id, limit=60)
        except StudioError as exc:
            return t.error_message(exc), []
        specials = info["special_tokens"]
        html = t.stat_grid([
            t.stat("Vocabulary", t.fmt_number(info["vocab_size"])),
            t.stat("BOS", specials["bos"] or "—"),
            t.stat("EOS", specials["eos"] or "—"),
            t.stat("PAD", specials["pad"] or "—"),
            t.stat("UNK", specials["unk"] or "—"),
        ])
        html += t.table(
            ["ID", "Token", "ID", "Token"],
            [
                [a[1], a[0].replace("Ġ", "·"), b[1], b[0].replace("Ġ", "·")]
                for a, b in zip(info["first_tokens"][:30], info["last_tokens"][-30:])
            ],
        )
        return html, []

    inspect_button.click(do_inspect, target, [result, tokens_out])

    def do_delete(tokenizer_id):
        if not tokenizer_id:
            return t.note("Select a tokenizer.", "warn"), tokenizers_table(), gr.update()
        try:
            delete_tokenizer(tokenizer_id)
        except StudioError as exc:
            return t.error_message(exc), tokenizers_table(), gr.update()
        return (t.note("Tokenizer deleted.", "good"), tokenizers_table(),
                gr.update(choices=tokenizer_choices(), value=None))

    delete_button.click(do_delete, target, [result, table, target])
    refresh_docs.click(lambda: gr.update(choices=document_choices()), outputs=documents)
