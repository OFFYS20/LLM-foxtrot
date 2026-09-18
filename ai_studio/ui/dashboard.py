"""Dashboard: current model, active run, live metrics and charts."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ai_studio.core.database import get_db
from ai_studio.hardware.monitor import monitor
from ai_studio.training.worker import manager
from ai_studio.ui import theme as t

REFRESH_SECONDS = 2.0


def _active_experiment() -> dict[str, Any] | None:
    experiment_id = manager.active_experiment_id()
    db = get_db()
    if experiment_id:
        return db.get("experiments", experiment_id)
    rows = db.list("experiments", order_by="created_at DESC", limit=1)
    return rows[0] if rows else None


def training_cards() -> str:
    experiment = _active_experiment()
    if experiment is None:
        return t.note(
            "No training runs yet. Import documents in <b>Data Library</b>, build a dataset in "
            "<b>Dataset Builder</b>, then start a run in <b>Training</b>.",
        )

    status = manager.status(experiment["id"])
    latest = status.get("latest") or {}
    metrics = experiment.get("metrics") or {}
    db = get_db()
    model = db.get("models", experiment.get("model_id") or "") or {}
    dataset = db.get("datasets", experiment.get("dataset_id") or "") or {}

    if not latest:  # not running now — show the stored outcome
        latest = {
            "step": experiment.get("current_step"),
            "epoch": experiment.get("current_epoch"),
            "loss": experiment.get("final_train_loss"),
            "val_loss": experiment.get("best_val_loss"),
            "learning_rate": metrics.get("learning_rate"),
            "tokens_processed": experiment.get("tokens_processed"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "elapsed_seconds": experiment.get("duration_seconds"),
            "eta_seconds": None,
        }

    tone = {"running": "good", "paused": "warn", "failed": "bad", "interrupted": "warn"}.get(
        experiment["status"], ""
    )
    name = experiment["name"]
    header = (
        '<div class="studio-pills">'
        + t.pill(f"<b>{name}</b>")
        + t.pill(experiment["status"], tone)
        + t.pill("method: " + experiment["method"])
        + t.pill("model: " + (model.get("name") or "—"))
        + t.pill("dataset: " + (dataset.get("name") or "—"))
        + "</div>"
    )

    total = experiment.get("total_steps") or (experiment.get("hyperparameters") or {}).get("max_steps") or 0
    step = latest.get("step") or 0
    progress = f"{step:,} / {total:,}" if total else f"{step:,}"

    cards = [
        t.stat("Status", experiment["status"]),
        t.stat("Step", progress),
        t.stat("Epoch", t.fmt_number(latest.get("epoch"), 2)),
        t.stat("Training loss", t.fmt_number(latest.get("loss"), 4)),
        t.stat("Validation loss", t.fmt_number(latest.get("val_loss"), 4)),
        t.stat(
            "Learning rate",
            f"{latest['learning_rate']:.2e}" if latest.get("learning_rate") else "—",
        ),
        t.stat("Tokens processed", t.fmt_params(latest.get("tokens_processed"))),
        t.stat("Tokens/sec", t.fmt_number(latest.get("tokens_per_sec"), 0)),
        t.stat("Elapsed", t.fmt_duration(latest.get("elapsed_seconds"))),
        t.stat("Remaining (est.)", t.fmt_duration(latest.get("eta_seconds"))),
    ]
    return header + "<br>" + t.stat_grid(cards)


def hardware_cards() -> str:
    snapshot = monitor.snapshot()
    gpu = snapshot.primary_gpu
    cards = [
        t.stat(
            "Accelerator",
            snapshot.accelerator.upper(),
            snapshot.cpu_model[:34] if snapshot.cpu_model else "",
        ),
        t.stat("CPU usage", f"{snapshot.cpu_percent:.0f}%" if snapshot.cpu_percent is not None else None,
               f"{snapshot.cpu_cores_logical or '?'} threads",
               unavailable=snapshot.cpu_percent is None),
        t.stat("System RAM",
               f"{snapshot.ram_used_gb:.1f} / {snapshot.ram_total_gb:.0f} GB" if snapshot.ram_total_gb else None,
               unavailable=snapshot.ram_total_gb is None),
        t.stat("Disk free",
               f"{snapshot.disk_free_gb:.0f} GB" if snapshot.disk_free_gb else None,
               unavailable=snapshot.disk_free_gb is None),
    ]
    if gpu:
        cards.extend([
            t.stat("GPU", gpu.name[:22], f"driver {gpu.driver_version or '—'}"),
            t.stat("GPU usage", f"{gpu.utilization:.0f}%" if gpu.utilization is not None else None,
                   unavailable=gpu.utilization is None),
            t.stat("VRAM",
                   f"{(gpu.used_memory_mb or 0) / 1024:.1f} / {(gpu.total_memory_mb or 0) / 1024:.0f} GB"
                   if gpu.total_memory_mb else None,
                   unavailable=gpu.total_memory_mb is None),
            t.stat("GPU temp", f"{gpu.temperature_c:.0f} °C" if gpu.temperature_c is not None else None,
                   unavailable=gpu.temperature_c is None),
        ])
    else:
        cards.append(
            t.stat("GPU", "None detected", "CPU mode — small models only", unavailable=True)
        )
    return t.stat_grid(cards)


def library_cards() -> str:
    db = get_db()
    return t.stat_grid([
        t.stat("Documents", t.fmt_number(db.count("documents"))),
        t.stat("Datasets", t.fmt_number(db.count("datasets"))),
        t.stat("Tokenizers", t.fmt_number(db.count("tokenizers"))),
        t.stat("Models", t.fmt_number(db.count("models"))),
        t.stat("Checkpoints", t.fmt_number(db.count("checkpoints"))),
        t.stat("Experiments", t.fmt_number(db.count("experiments"))),
        t.stat("Knowledge indexes", t.fmt_number(db.count("rag_indexes"))),
        t.stat("Conversations", t.fmt_number(db.count("conversations"))),
    ])


def _metric_series(experiment_id: str | None, limit: int = 400) -> list[dict[str, Any]]:
    if not experiment_id:
        return []
    return get_db().query(
        "SELECT step, loss, val_loss, learning_rate, tokens_per_sec FROM training_metrics "
        "WHERE experiment_id = ? ORDER BY step ASC LIMIT ?",
        (experiment_id, limit),
        table="training_metrics",
    )


def _frame(records: list[dict[str, Any]], columns: list[str]):
    """Always return a typed DataFrame — an untyped empty frame breaks the plot."""
    import pandas as pd

    if not records:
        return pd.DataFrame({column: pd.Series(dtype="float64" if column != "series" else "object")
                             for column in columns})
    return pd.DataFrame(records, columns=columns)


def loss_frame():
    experiment = _active_experiment()
    rows = _metric_series(experiment["id"] if experiment else None)
    records = [
        {"step": r["step"], "value": r["loss"], "series": "train"}
        for r in rows if r.get("loss") is not None
    ] + [
        {"step": r["step"], "value": r["val_loss"], "series": "validation"}
        for r in rows if r.get("val_loss") is not None
    ]
    return _frame(records, ["step", "value", "series"])


def lr_frame():
    experiment = _active_experiment()
    rows = _metric_series(experiment["id"] if experiment else None)
    return _frame(
        [{"step": r["step"], "value": r["learning_rate"]}
         for r in rows if r.get("learning_rate") is not None],
        ["step", "value"],
    )


def throughput_frame():
    experiment = _active_experiment()
    rows = _metric_series(experiment["id"] if experiment else None)
    return _frame(
        [{"step": r["step"], "value": r["tokens_per_sec"]}
         for r in rows if r.get("tokens_per_sec") is not None],
        ["step", "value"],
    )


def hardware_frame():
    records: list[dict[str, Any]] = []
    for index, snapshot in enumerate(monitor.history(limit=120)):
        if snapshot.cpu_percent is not None:
            records.append({"sample": index, "value": snapshot.cpu_percent, "series": "CPU %"})
        if snapshot.ram_total_gb and snapshot.ram_used_gb:
            records.append({"sample": index,
                            "value": snapshot.ram_used_gb / snapshot.ram_total_gb * 100,
                            "series": "RAM %"})
        gpu = snapshot.primary_gpu
        if gpu and gpu.utilization is not None:
            records.append({"sample": index, "value": gpu.utilization, "series": "GPU %"})
        if gpu and gpu.used_memory_mb and gpu.total_memory_mb:
            records.append({"sample": index,
                            "value": gpu.used_memory_mb / gpu.total_memory_mb * 100,
                            "series": "VRAM %"})
    return _frame(records, ["sample", "value", "series"])


def render() -> dict[str, Any]:
    with gr.Column():
        gr.HTML('<div class="studio-title">Dashboard</div>'
                '<div class="studio-sub">Live state of the workspace</div>')
        training = gr.HTML(training_cards)
        gr.HTML('<div class="stat-label" style="margin-top:14px">Hardware</div>')
        hardware = gr.HTML(hardware_cards)
        gr.HTML('<div class="stat-label" style="margin-top:14px">Library</div>')
        library = gr.HTML(library_cards)

        gr.HTML('<div class="stat-label" style="margin-top:14px">Training curves</div>')
        with gr.Row():
            loss = gr.LinePlot(value=loss_frame, x="step", y="value", color="series",
                               height=240, label="Training & validation loss")
            hardware_chart = gr.LinePlot(value=hardware_frame, x="sample", y="value", color="series",
                                         height=240, label="Hardware utilisation (%)")
        with gr.Row():
            lr = gr.LinePlot(value=lr_frame, x="step", y="value", height=200, label="Learning rate")
            throughput = gr.LinePlot(value=throughput_frame, x="step", y="value",
                                     height=200, label="Tokens / sec")

        refresh = gr.Button("Refresh", size="sm")
        timer = gr.Timer(REFRESH_SECONDS)

        outputs = [training, hardware, library, loss, lr, throughput, hardware_chart]

        def _refresh():
            return (
                training_cards(), hardware_cards(), library_cards(),
                loss_frame(), lr_frame(), throughput_frame(), hardware_frame(),
            )

        timer.tick(_refresh, outputs=outputs)
        refresh.click(_refresh, outputs=outputs)

    return {"refresh": _refresh, "outputs": outputs}
