"""Models screen: create from scratch, import from Hugging Face, export, quantize."""

from __future__ import annotations

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.models.model_manager import (
    delete_model,
    export_model,
    import_huggingface_model,
    list_models,
    quantize_model,
    create_scratch_model,
)
from ai_studio.models.tokenizer_manager import list_tokenizers
from ai_studio.models.transformer import ACTIVATIONS, NORMS, POSITION_TYPES, SIZE_PRESETS, TransformerConfig, preset_config
from ai_studio.ui import theme as t


def models_table() -> str:
    return t.table(
        ["Name", "Kind", "Architecture", "Parameters", "Context", "Precision", "Size", "License"],
        [
            [
                row["name"], row["kind"], row.get("architecture") or "—",
                t.fmt_params(row.get("parameters")), t.fmt_number(row.get("context_length")),
                row.get("quantization") or row.get("precision") or "—",
                t.fmt_bytes(row.get("size_bytes")),
                (row.get("license") or "—")[:44],
            ]
            for row in list_models()
        ],
        empty="No models yet — create one from scratch or import from Hugging Face.",
    )


def model_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {t.fmt_params(r.get('parameters'))} · {r['kind']}", r["id"]) for r in list_models()]


def tokenizer_choices() -> list[tuple[str, str]]:
    return [("(none)", "")] + [
        (f"{r['name']} · vocab {r['vocab_size']:,}", r["id"]) for r in list_tokenizers()
    ]


def architecture_summary(preset, vocab, hidden, layers, heads, kv_heads, ffn, context,
                         dropout, norm, activation, position, rope_theta, tie) -> str:
    try:
        config = TransformerConfig(
            vocab_size=int(vocab), hidden_size=int(hidden), num_layers=int(layers),
            num_heads=int(heads), num_kv_heads=int(kv_heads) or None,
            intermediate_size=int(ffn) or None, max_position_embeddings=int(context),
            dropout=float(dropout), norm_type=norm, activation=activation,
            position_embedding=position, rope_theta=float(rope_theta), tie_word_embeddings=bool(tie),
        )
    except Exception as exc:  # noqa: BLE001
        return t.error_message(exc)

    problems = config.validate()
    counts = config.parameter_count()
    memory = config.estimate_memory(batch_size=2, sequence_length=min(512, config.max_position_embeddings))
    html = t.stat_grid([
        t.stat("Parameters", t.fmt_params(counts["total"])),
        t.stat("Embeddings", t.fmt_params(counts["embeddings"])),
        t.stat("Per layer", t.fmt_params(counts["per_layer"])),
        t.stat("Head dim", config.head_dim),
        t.stat("FFN size", config.intermediate_size),
        t.stat("Training memory", f"{memory['total_mb']/1024:.2f} GB", "batch 2 · fp32"),
        t.stat("Inference memory", f"{memory['inference_mb']/1024:.2f} GB"),
    ])
    if problems:
        html += t.note("Invalid: " + "; ".join(problems), "bad")
    return html


def render() -> None:
    gr.HTML('<div class="studio-title">Models</div>'
            '<div class="studio-sub">Build a transformer from scratch, or bring one from Hugging Face</div>')

    with gr.Tabs():
        with gr.Tab("Create from scratch"):
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="Model name", placeholder="my-tiny-lm")
                    preset = gr.Dropdown(list(SIZE_PRESETS), value="tiny-10m", label="Size preset")
                    tokenizer = gr.Dropdown(choices=tokenizer_choices(), value="", label="Tokenizer")
                    description = gr.Textbox(label="Description", placeholder="optional")
                with gr.Column():
                    with gr.Row():
                        vocab = gr.Number(value=8192, label="Vocabulary", precision=0)
                        context = gr.Number(value=512, label="Max context", precision=0)
                    with gr.Row():
                        hidden = gr.Number(value=256, label="Hidden dim", precision=0)
                        layers = gr.Number(value=10, label="Layers", precision=0)
                    with gr.Row():
                        heads = gr.Number(value=4, label="Attention heads", precision=0)
                        kv_heads = gr.Number(value=0, label="KV heads (0 = same)", precision=0)
                    with gr.Row():
                        ffn = gr.Number(value=0, label="FFN dim (0 = auto)", precision=0)
                        dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Dropout")
                    with gr.Row():
                        norm = gr.Dropdown(list(NORMS), value="rmsnorm", label="Normalisation")
                        activation = gr.Dropdown(list(ACTIVATIONS), value="swiglu", label="Activation")
                    with gr.Row():
                        position = gr.Dropdown(list(POSITION_TYPES), value="rope", label="Positions")
                        rope_theta = gr.Number(value=10000, label="RoPE theta")
                    tie = gr.Checkbox(value=True, label="Tie embedding and output weights")

            summary = gr.HTML()
            with gr.Row():
                estimate_button = gr.Button("Estimate")
                create_button = gr.Button("Create model", variant="primary")
            create_result = gr.HTML()

        with gr.Tab("Import from Hugging Face"):
            repo = gr.Textbox(label="Repository id", placeholder="Qwen/Qwen2.5-0.5B")
            with gr.Row():
                hf_name = gr.Textbox(label="Local name (optional)")
                revision = gr.Textbox(label="Revision (optional)")
            with gr.Row():
                download = gr.Checkbox(value=True, label="Download weights now")
                trust = gr.Checkbox(value=False, label="Trust remote code")
            gr.HTML(t.note(
                "Licences vary. AI Studio shows the licence the repository declares; it does not "
                "grant you any right to use or redistribute the weights.", "warn"))
            import_button = gr.Button("Import", variant="primary")
            import_result = gr.HTML()

    gr.Markdown("### Registered models")
    table = gr.HTML(models_table)
    with gr.Row():
        target = gr.Dropdown(choices=model_choices(), label="Model")
        refresh_button = gr.Button("↻", size="sm", scale=0)
    with gr.Row():
        export_button = gr.Button("Export")
        quantize_bits = gr.Dropdown(["8", "4"], value="8", label="Quantize to", scale=0)
        quantize_button = gr.Button("Quantize")
        delete_button = gr.Button("Delete", variant="stop")
    action_result = gr.HTML()

    architecture_inputs = [preset, vocab, hidden, layers, heads, kv_heads, ffn, context,
                           dropout, norm, activation, position, rope_theta, tie]

    def apply_preset(preset_name):
        config = preset_config(preset_name)
        return (config.vocab_size, config.hidden_size, config.num_layers, config.num_heads,
                config.max_position_embeddings)

    preset.change(apply_preset, preset, [vocab, hidden, layers, heads, context])
    estimate_button.click(architecture_summary, architecture_inputs, summary)

    def do_create(name_value, tokenizer_id, description_value, *architecture):
        if not name_value or not name_value.strip():
            return t.note("Give the model a name.", "warn"), models_table(), gr.update()
        (_, vocab_v, hidden_v, layers_v, heads_v, kv_v, ffn_v, context_v,
         dropout_v, norm_v, activation_v, position_v, theta_v, tie_v) = architecture
        try:
            config = TransformerConfig(
                vocab_size=int(vocab_v), hidden_size=int(hidden_v), num_layers=int(layers_v),
                num_heads=int(heads_v), num_kv_heads=int(kv_v) or None,
                intermediate_size=int(ffn_v) or None, max_position_embeddings=int(context_v),
                dropout=float(dropout_v), norm_type=norm_v, activation=activation_v,
                position_embedding=position_v, rope_theta=float(theta_v),
                tie_word_embeddings=bool(tie_v),
            )
            record = create_scratch_model(
                name_value.strip(), config,
                tokenizer_id=tokenizer_id or None, description=description_value or None,
            )
        except StudioError as exc:
            return t.error_message(exc), models_table(), gr.update()
        except Exception as exc:  # noqa: BLE001
            return t.error_message(exc), models_table(), gr.update()
        return (
            t.note(
                f"Created <b>{record['name']}</b> — {t.fmt_params(record['parameters'])} parameters, "
                f"randomly initialised. Train it in the <b>Training</b> tab.", "good",
            ),
            models_table(), gr.update(choices=model_choices()),
        )

    create_button.click(
        do_create, [name, tokenizer, description, *architecture_inputs],
        [create_result, table, target],
    )

    def do_import(repo_value, name_value, revision_value, download_value, trust_value,
                  progress=gr.Progress()):
        if not repo_value or not repo_value.strip():
            return t.note("Enter a repository id.", "warn"), models_table(), gr.update()
        progress(0.3, desc="Reading model configuration")
        try:
            record = import_huggingface_model(
                repo_value.strip(), name=name_value or None, revision=revision_value or None,
                download=bool(download_value), trust_remote_code=bool(trust_value),
            )
        except StudioError as exc:
            return t.error_message(exc), models_table(), gr.update()
        methods = (record.get("meta") or {}).get("supported_methods") or []
        html = t.note(
            f"Imported <b>{record['name']}</b> — {record.get('architecture')}, "
            f"{t.fmt_params(record.get('parameters'))} parameters.<br>"
            f"Licence: {record.get('license')}", "good",
        )
        html += t.note(
            f"Supported training methods for this model here: {', '.join(methods) or 'none (weights not downloaded)'}"
        )
        return html, models_table(), gr.update(choices=model_choices())

    import_button.click(
        do_import, [repo, hf_name, revision, download, trust],
        [import_result, table, target],
    )

    def do_export(model_id):
        if not model_id:
            return t.note("Select a model.", "warn")
        try:
            outcome = export_model(model_id)
        except StudioError as exc:
            return t.error_message(exc)
        return t.note(
            f"Exported to <code>{outcome['path']}</code> ({t.fmt_bytes(outcome['size_bytes'])}, "
            f"{len(outcome['files'])} files).", "good",
        )

    export_button.click(do_export, target, action_result)

    def do_quantize(model_id, bits):
        if not model_id:
            return t.note("Select a model.", "warn"), models_table(), gr.update()
        try:
            record = quantize_model(model_id, int(bits))
        except StudioError as exc:
            return t.error_message(exc), models_table(), gr.update()
        return (t.note(f"Created <b>{record['name']}</b>.", "good"),
                models_table(), gr.update(choices=model_choices()))

    quantize_button.click(do_quantize, [target, quantize_bits], [action_result, table, target])

    def do_delete(model_id):
        if not model_id:
            return t.note("Select a model.", "warn"), models_table(), gr.update()
        try:
            delete_model(model_id)
        except StudioError as exc:
            return t.error_message(exc), models_table(), gr.update()
        return (t.note("Model deleted.", "good"), models_table(),
                gr.update(choices=model_choices(), value=None))

    delete_button.click(do_delete, target, [action_result, table, target])

    def do_refresh():
        return gr.update(choices=model_choices()), gr.update(choices=tokenizer_choices()), models_table()

    refresh_button.click(do_refresh, outputs=[target, tokenizer, table])
