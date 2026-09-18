"""Hardware screen: detection, live usage, and a fit estimate for a model."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ai_studio.hardware.monitor import monitor
from ai_studio.models.transformer import SIZE_PRESETS, preset_config
from ai_studio.ui import theme as t


def detection_html() -> str:
    snapshot = monitor.snapshot()
    rows = [
        ["Accelerator", snapshot.accelerator.upper()],
        ["PyTorch", snapshot.torch_version or "not installed"],
        ["CUDA available", "yes" if snapshot.cuda_available else "no"],
        ["CUDA version", snapshot.cuda_version or "—"],
        ["CPU", snapshot.cpu_model or "Unavailable"],
        ["CPU cores", f"{snapshot.cpu_cores_physical or '?'} physical / {snapshot.cpu_cores_logical or '?'} logical"],
        ["System RAM", f"{snapshot.ram_total_gb:.1f} GB" if snapshot.ram_total_gb else "Unavailable"],
        ["Telemetry source", snapshot.gpu_source or "none (no GPU)"],
        ["Platform", snapshot.platform],
    ]
    html = t.table(["Property", "Value"], rows)
    for message in snapshot.notes:
        html += t.note(message, "warn")
    return html


def live_html() -> str:
    snapshot = monitor.snapshot()
    cards = [
        t.stat("CPU", f"{snapshot.cpu_percent:.0f}%" if snapshot.cpu_percent is not None else None,
               unavailable=snapshot.cpu_percent is None),
        t.stat("RAM", f"{snapshot.ram_used_gb:.1f} / {snapshot.ram_total_gb:.0f} GB"
               if snapshot.ram_total_gb else None, unavailable=snapshot.ram_total_gb is None),
        t.stat("Swap", f"{snapshot.swap_used_gb:.1f} / {snapshot.swap_total_gb:.0f} GB"
               if snapshot.swap_total_gb else None, unavailable=snapshot.swap_total_gb is None),
        t.stat("Disk", f"{snapshot.disk_used_gb:.0f} / {snapshot.disk_total_gb:.0f} GB"
               if snapshot.disk_total_gb else None, unavailable=snapshot.disk_total_gb is None),
    ]
    for gpu in snapshot.gpus:
        cards.extend([
            t.stat(f"GPU {gpu.index}", gpu.name[:24]),
            t.stat("Utilisation", f"{gpu.utilization:.0f}%" if gpu.utilization is not None else None,
                   unavailable=gpu.utilization is None),
            t.stat("VRAM", f"{(gpu.used_memory_mb or 0)/1024:.1f} / {(gpu.total_memory_mb or 0)/1024:.0f} GB"
                   if gpu.total_memory_mb else None, unavailable=gpu.total_memory_mb is None),
            t.stat("Temperature", f"{gpu.temperature_c:.0f} °C" if gpu.temperature_c is not None else None,
                   unavailable=gpu.temperature_c is None),
            t.stat("Power", f"{gpu.power_draw_w:.0f} / {gpu.power_limit_w:.0f} W"
                   if gpu.power_draw_w and gpu.power_limit_w else None,
                   unavailable=gpu.power_draw_w is None),
        ])
    if not snapshot.gpus:
        cards.append(t.stat("GPU", "None detected", "CPU mode", unavailable=True))
    return t.stat_grid(cards)


def fit_estimate(preset: str, batch_size: int, sequence_length: int, precision: str, checkpointing: bool) -> str:
    try:
        config = preset_config(preset)
        estimate = config.estimate_memory(
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            precision=precision,
            gradient_checkpointing=bool(checkpointing),
        )
    except Exception as exc:  # noqa: BLE001
        return t.error_message(exc)

    snapshot = monitor.snapshot()
    gpu = snapshot.primary_gpu
    if gpu and gpu.total_memory_mb:
        available, where = gpu.total_memory_mb, f"{gpu.name} VRAM"
    elif snapshot.ram_available_gb:
        available, where = snapshot.ram_available_gb * 1024, "system RAM (CPU training)"
    else:
        available, where = None, "unknown"

    cards = t.stat_grid([
        t.stat("Parameters", t.fmt_params(estimate["parameters"])),
        t.stat("Weights", f"{estimate['weights_mb'] / 1024:.2f} GB"),
        t.stat("Optimizer", f"{estimate['optimizer_mb'] / 1024:.2f} GB"),
        t.stat("Activations", f"{estimate['activations_mb'] / 1024:.2f} GB"),
        t.stat("Total (training)", f"{estimate['total_mb'] / 1024:.2f} GB"),
        t.stat("Inference only", f"{estimate['inference_mb'] / 1024:.2f} GB"),
    ])

    if available is None:
        return cards + t.note("Available memory could not be read, so no fit verdict is possible.", "warn")
    ratio = estimate["total_mb"] / available
    if ratio > 1.0:
        verdict = t.note(
            f"<b>Will not fit.</b> Needs {estimate['total_mb']/1024:.1f} GB but only "
            f"{available/1024:.1f} GB of {where} is available. Try: smaller batch size, "
            f"gradient checkpointing, LoRA/QLoRA, shorter sequences, or a smaller model.",
            "bad",
        )
    elif ratio > 0.85:
        verdict = t.note(
            f"<b>Very tight.</b> Needs {estimate['total_mb']/1024:.1f} GB of "
            f"{available/1024:.1f} GB {where} — an out-of-memory error is likely.", "warn",
        )
    else:
        verdict = t.note(
            f"<b>Fits.</b> Needs about {estimate['total_mb']/1024:.1f} GB of "
            f"{available/1024:.1f} GB {where}.", "good",
        )
    return cards + verdict


def render() -> None:
    gr.HTML('<div class="studio-title">Hardware</div>'
            '<div class="studio-sub">Detected devices and live utilisation</div>')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Detected")
            detection = gr.HTML(detection_html)
        with gr.Column(scale=1):
            gr.Markdown("### Live")
            live = gr.HTML(live_html)
    refresh = gr.Button("Refresh", size="sm")
    timer = gr.Timer(3.0)
    timer.tick(lambda: (detection_html(), live_html()), outputs=[detection, live])
    refresh.click(lambda: (detection_html(), live_html()), outputs=[detection, live])

    gr.Markdown("### Will this model fit?")
    with gr.Row():
        preset = gr.Dropdown(list(SIZE_PRESETS), value="tiny-10m", label="Model size")
        batch = gr.Number(value=2, label="Batch size", precision=0)
        seq = gr.Number(value=512, label="Sequence length", precision=0)
        precision = gr.Dropdown(["fp32", "bf16", "fp16"], value="fp32", label="Precision")
        checkpointing = gr.Checkbox(value=False, label="Gradient checkpointing")
    estimate_out = gr.HTML()
    gr.Button("Estimate", variant="primary").click(
        fit_estimate, [preset, batch, seq, precision, checkpointing], estimate_out
    )
