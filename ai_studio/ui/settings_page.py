"""Settings screen: paths, defaults, dependency status, tools and licensing."""

from __future__ import annotations

import importlib
import json

import gradio as gr

from ai_studio.core.config import find_config_file, get_config
from ai_studio.core.database import get_db
from ai_studio.tools.registry import install_default_tools, registry
from ai_studio.ui import theme as t

OPTIONAL_PACKAGES = [
    ("torch", "Training and inference"),
    ("transformers", "Hugging Face models"),
    ("tokenizers", "Tokenizer training"),
    ("datasets", "Official benchmark splits, HF datasets"),
    ("accelerate", "Multi-device training helpers"),
    ("peft", "LoRA / QLoRA"),
    ("bitsandbytes", "8-bit / 4-bit quantization (CUDA only)"),
    ("safetensors", "Safe weight serialisation"),
    ("sentence_transformers", "RAG embeddings"),
    ("faiss", "Fast vector search"),
    ("pymupdf", "PDF import (PyMuPDF)"),
    ("docx", "Word import (python-docx)"),
    ("ebooklib", "EPUB import"),
    ("bs4", "HTML import (BeautifulSoup)"),
    ("pandas", "Tabular data"),
    ("psutil", "CPU/RAM telemetry"),
    ("pynvml", "NVIDIA GPU telemetry"),
]


APP_LICENSE = "MIT"


def licensing_table() -> str:
    """What every registered model and dataset declares — never an assumption."""
    db = get_db()
    rows = []
    for record in db.list("models", order_by="created_at DESC"):
        declared = record.get("license")
        source = record.get("repo_id") or record.get("path") or "—"
        rows.append([
            record["name"],
            record.get("kind", "—"),
            f'<code>{source}</code>',
            declared or '<span class="stat-hint">unknown — check the source</span>',
        ])
    return t.table(
        ["Model", "Kind", "Source", "Declared licence"],
        rows,
        empty="No models registered yet.",
    )


def dataset_licensing_table() -> str:
    db = get_db()
    rows = []
    for record in db.list("datasets", order_by="created_at DESC"):
        meta = record.get("meta") or {}
        rows.append([
            record["name"],
            record.get("mode", "—"),
            "synthetic" if record.get("is_synthetic") else "imported / built",
            meta.get("license") or '<span class="stat-hint">unknown — check the source</span>',
        ])
    return t.table(
        ["Dataset", "Mode", "Origin", "Declared licence"],
        rows,
        empty="No datasets built yet.",
    )


def dependency_table() -> str:
    rows = []
    for package, purpose in OPTIONAL_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            rows.append([package, f"<span style='color:#31c48d'>{version}</span>", purpose])
        except Exception:  # noqa: BLE001
            rows.append([package, "<span style='color:#8b93a3'>not installed</span>", purpose])
    return t.table(["Package", "Status", "Used for"], rows)


def paths_table() -> str:
    config = get_config()
    source = find_config_file()
    rows = [
        ["config.yaml", str(source) if source else "(defaults — no config.yaml found)"],
        ["Storage root", str(config.root)],
        ["Models", str(config.models_dir)],
        ["Datasets", str(config.datasets_dir)],
        ["Checkpoints", str(config.checkpoints_dir)],
        ["Documents (originals)", str(config.originals_dir)],
        ["Documents (cleaned)", str(config.cleaned_dir)],
        ["Tokenizers", str(config.tokenizers_dir)],
        ["Indexes", str(config.indexes_dir)],
        ["Exports", str(config.exports_dir)],
        ["Database", str(config.database_path)],
    ]
    return t.table(["Path", "Location"], rows)


def defaults_table() -> str:
    config = get_config()
    return t.table(
        ["Setting", "Value"],
        [
            ["Default precision", config.training.default_precision],
            ["Checkpoint interval", config.training.checkpoint_interval],
            ["Eval interval", config.training.eval_interval],
            ["Keep last checkpoints", config.training.keep_last_checkpoints],
            ["Max sequence length", config.training.max_sequence_length],
            ["Dataset chunk size", config.data.chunk_size],
            ["Train/val/test", f"{config.data.train_split}/{config.data.validation_split}/{config.data.test_split}"],
            ["RAG embedding model", config.rag.embedding_model],
            ["RAG top-k", config.rag.top_k],
            ["Max upload (MB)", config.data.max_upload_mb],
        ],
    )


def tools_table() -> str:
    install_default_tools()
    return t.table(
        ["Tool", "Enabled", "Confirmation", "Description"],
        [
            [tool.name, "yes" if tool.enabled else "no",
             "required" if tool.requires_confirmation else "—", tool.description]
            for tool in registry.list()
        ],
    )


def storage_usage() -> str:
    from ai_studio.core.paths import dir_size

    config = get_config()
    db = get_db()
    return t.stat_grid([
        t.stat("Documents", t.fmt_bytes(dir_size(config.documents_dir)), f"{db.count('documents')} files"),
        t.stat("Datasets", t.fmt_bytes(dir_size(config.datasets_dir)), f"{db.count('datasets')} datasets"),
        t.stat("Models", t.fmt_bytes(dir_size(config.models_dir)), f"{db.count('models')} models"),
        t.stat("Checkpoints", t.fmt_bytes(dir_size(config.checkpoints_dir)), f"{db.count('checkpoints')} checkpoints"),
        t.stat("Indexes", t.fmt_bytes(dir_size(config.indexes_dir)), f"{db.count('rag_indexes')} indexes"),
        t.stat("Database", t.fmt_bytes(config.database_path.stat().st_size if config.database_path.exists() else 0)),
    ])


def render() -> None:
    gr.HTML('<div class="studio-title">Settings</div>'
            '<div class="studio-sub">Paths, defaults and installed capabilities</div>')

    gr.Markdown("### Storage usage")
    usage = gr.HTML(storage_usage)

    gr.Markdown("### Paths")
    gr.HTML(paths_table)
    gr.HTML(t.note(
        "Edit <code>config.yaml</code> to change any of these, or set environment variables such as "
        "<code>AISTUDIO_STORAGE_ROOT</code> / <code>AISTUDIO_UI_PORT</code>, then restart."))

    gr.Markdown("### Defaults")
    gr.HTML(defaults_table)

    gr.Markdown("### Installed packages")
    dependencies = gr.HTML(dependency_table)
    gr.HTML(t.note(
        "Missing packages disable the matching feature and produce a clear message — "
        "nothing is silently skipped or simulated."))

    gr.Markdown("### Tools")
    gr.HTML(tools_table)
    gr.HTML(t.note(
        "The Python sandbox allows no imports, attribute access, filesystem or network. "
        "Model-generated code never reaches the operating system.", "warn"))

    gr.Markdown("### Licensing")
    gr.HTML(t.note(
        f"AI Studio itself is <b>{APP_LICENSE}</b>-licensed. That covers this application only. "
        "Model weights, datasets and benchmark splits carry their own terms — using this software "
        "does <b>not</b> grant permission to use or redistribute them. Check each source's terms "
        "before using or sharing anything you import.", "warn"))
    gr.Markdown("**Models**")
    licenses = gr.HTML(licensing_table)
    gr.Markdown("**Datasets**")
    dataset_licenses = gr.HTML(dataset_licensing_table)
    gr.HTML(t.note(
        "A licence shown as <i>unknown</i> means the source did not declare one — it is never "
        "guessed, and an absent licence is not permission."))

    refresh = gr.Button("Refresh", size="sm")
    refresh.click(
        lambda: (storage_usage(), dependency_table(), licensing_table(), dataset_licensing_table()),
        outputs=[usage, dependencies, licenses, dataset_licenses],
    )
