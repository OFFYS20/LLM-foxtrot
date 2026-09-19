"""AI Studio — a local laboratory for building, training and using language models.

Run with:  python -m ai_studio.app      (or: python ai_studio/app.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python ai_studio/app.py` as well as `python -m ai_studio.app`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db
from ai_studio.hardware.monitor import monitor
from ai_studio.core import gradio_compat as compat
from ai_studio.tools.registry import install_default_tools
from ai_studio.training.worker import manager
from ai_studio.ui import (
    benchmarks_page,
    chat_page,
    checkpoints_page,
    dashboard,
    data_library,
    dataset_page,
    experiments_page,
    finetuning_page,
    hardware_page,
    logs_page,
    models_page,
    rag_page,
    settings_page,
    theme,
    tokenizer_page,
    training_page,
)

VERSION = "0.1.0"


def header_html() -> str:
    snapshot = monitor.snapshot()
    db = get_db()
    gpu = snapshot.primary_gpu

    pills = [
        theme.pill(f"<b>{snapshot.accelerator.upper()}</b>", "good" if snapshot.gpus else "warn"),
        theme.pill(f"torch <b>{snapshot.torch_version or 'missing'}</b>"),
    ]
    if gpu:
        vram = (
            f"{(gpu.used_memory_mb or 0) / 1024:.1f}/{(gpu.total_memory_mb or 0) / 1024:.0f} GB"
            if gpu.total_memory_mb else "VRAM unavailable"
        )
        pills.append(theme.pill(f"{gpu.name[:22]} · {vram}"))
    else:
        pills.append(theme.pill("no GPU — CPU mode", "warn"))
    if snapshot.ram_total_gb:
        pills.append(theme.pill(f"RAM <b>{snapshot.ram_used_gb:.1f}/{snapshot.ram_total_gb:.0f} GB</b>"))
    pills.append(theme.pill(f"models <b>{db.count('models')}</b>"))
    pills.append(theme.pill(f"datasets <b>{db.count('datasets')}</b>"))

    active = manager.active_experiment_id()
    if active:
        record = db.get("experiments", active) or {}
        pills.append(theme.pill(f"training: <b>{record.get('name', '')[:26]}</b>", "good"))

    return (
        '<div class="studio-header">'
        '<div><div class="studio-title">AI Studio</div>'
        f'<div class="studio-sub">Local model laboratory · v{VERSION}</div></div>'
        f'<div class="studio-pills">{"".join(pills)}</div>'
        "</div>"
    )


def build_app() -> gr.Blocks:
    config = get_config()
    config.ensure_directories()

    with compat.blocks(title=config.ui.title, theme=theme.theme(), css=theme.CSS,
                       fill_height=True) as demo:
        header = gr.HTML(header_html)

        with gr.Tabs():
            with gr.Tab("Dashboard"):
                dashboard.render()
            with gr.Tab("Models"):
                models_page.render()
            with gr.Tab("Data Library"):
                data_library.render()
            with gr.Tab("Dataset Builder"):
                dataset_page.render()
            with gr.Tab("Tokenizer"):
                tokenizer_page.render()
            with gr.Tab("Training"):
                training_page.render()
            with gr.Tab("Fine-Tuning"):
                finetuning_page.render()
            with gr.Tab("Chat"):
                chat_page.render()
            with gr.Tab("Knowledge / RAG"):
                rag_page.render()
            with gr.Tab("Benchmarks"):
                benchmarks_page.render()
            with gr.Tab("Experiments"):
                experiments_page.render()
            with gr.Tab("Checkpoints"):
                checkpoints_page.render()
            with gr.Tab("Hardware"):
                hardware_page.render()
            with gr.Tab("Logs"):
                logs_page.render()
            with gr.Tab("Settings"):
                settings_page.render()

        gr.Timer(5.0).tick(header_html, outputs=header)

    return demo


def startup() -> None:
    """One-time initialisation: config, database, hardware probe, orphan recovery."""
    config = get_config()
    log.configure()
    config.ensure_directories()
    get_db()  # creates the schema on first run
    install_default_tools()

    snapshot = monitor.snapshot()
    recovered = manager.recover_orphans()

    log.info(
        f"AI Studio {VERSION} starting — storage at {config.root}",
        source="app",
    )
    log.info(
        f"Hardware: {snapshot.accelerator.upper()}"
        + (f", {len(snapshot.gpus)} GPU(s)" if snapshot.gpus else ", no GPU (CPU mode)")
        + (f", torch {snapshot.torch_version}" if snapshot.torch_version else ", torch missing"),
        source="app",
    )
    for message in snapshot.notes:
        log.warning(message, source="app")
    if recovered:
        log.warning(f"{recovered} interrupted run(s) marked as interrupted", source="app")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI Studio — local LLM laboratory")
    parser.add_argument("--host", default=None, help="Bind address (default from config.yaml)")
    parser.add_argument("--port", type=int, default=None, help="Port (default from config.yaml)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    args = parser.parse_args(argv)

    startup()
    config = get_config()
    demo = build_app()
    compat.launch(
        demo.queue(default_concurrency_limit=4),
        server_name=args.host or config.ui.host,
        server_port=args.port or config.ui.port,
        share=args.share or config.ui.share,
        inbrowser=not args.no_browser,
        quiet=False,
    )


if __name__ == "__main__":
    main()
