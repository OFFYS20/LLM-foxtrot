"""Benchmarks screen: run suites, inspect every item, compare models."""

from __future__ import annotations

import gradio as gr

from ai_studio.core.database import get_db
from ai_studio.core.errors import StudioError
from ai_studio.evaluation.benchmark_runner import (
    BenchmarkConfig,
    compare_models,
    get_results,
    list_runs,
    runner,
)
from ai_studio.evaluation.suites import SUITES, list_suites
from ai_studio.models.model_manager import list_models
from ai_studio.ui import theme as t


def model_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {t.fmt_params(r.get('parameters'))}", r["id"]) for r in list_models()]


def suites_table() -> str:
    rows = []
    for info in list_suites():
        rows.append([
            info.key, info.label, info.category, info.method,
            info.source, info.available,
            "official split" if info.source in {"official", "local"} else "sample only",
        ])
    return t.table(["Key", "Suite", "Category", "Scoring", "Source", "Items", "Status"], rows)


def runs_table() -> str:
    db = get_db()
    rows = []
    for record in list_runs(limit=60):
        model = db.get("models", record["model_id"]) or {}
        source = (record.get("config") or {}).get("source", "—")
        rows.append([
            record["name"][:34], model.get("name", "—")[:22], record["status"],
            f"{record['completed_items']}/{record['total_items']}",
            f"{record['accuracy']*100:.1f}%" if record.get("accuracy") is not None else "—",
            t.fmt_number(record.get("avg_latency_ms"), 0),
            t.fmt_number(record.get("tokens_per_sec"), 1),
            t.fmt_duration(record.get("runtime_seconds")),
            source, t.fmt_time(record["created_at"]),
        ])
    return t.table(
        ["Run", "Model", "Status", "Progress", "Accuracy", "Latency ms", "Tok/s", "Runtime", "Source", "Created"],
        rows, empty="No benchmark runs yet.",
    )


def run_choices() -> list[tuple[str, str]]:
    return [(f"{r['name'][:40]} · {r['status']}", r["id"]) for r in list_runs(limit=60)]


def render() -> None:
    gr.HTML('<div class="studio-title">Benchmarks</div>'
            '<div class="studio-sub">Measured results only — a score appears after the run completes</div>')
    gr.HTML(t.note(
        "Suites marked <b>sample</b> use a small bundled item set in the right format, for "
        "smoke-testing. Install <code>datasets</code> (and be online), or drop the real split at "
        "<code>storage/benchmarks/&lt;suite&gt;.jsonl</code>, to run the official benchmark.", "warn"))
    gr.HTML(suites_table)

    with gr.Row():
        model = gr.Dropdown(choices=model_choices(), label="Model")
        suite = gr.Dropdown(list(SUITES) + ["custom"], value="mmlu", label="Suite")
        num_examples = gr.Slider(1, 500, value=10, step=1, label="Examples")
        few_shot = gr.Slider(0, 16, value=0, step=1, label="Few-shot")
    with gr.Row():
        temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature")
        max_tokens = gr.Slider(8, 1024, value=128, step=8, label="Max new tokens")
        seed = gr.Number(value=42, label="Seed", precision=0)
        allow_download = gr.Checkbox(value=True, label="Use official split when available")
    custom_path = gr.Textbox(
        label="Custom test file (JSONL with prompt/expected/method)", placeholder="optional"
    )
    with gr.Row():
        run_button = gr.Button("Run benchmark", variant="primary")
        cancel_button = gr.Button("Cancel")
        refresh_button = gr.Button("↻ Refresh", size="sm")
    run_state = gr.State(None)
    run_status = gr.HTML()

    gr.Markdown("### Runs")
    table = gr.HTML(runs_table)

    gr.Markdown("### Inspect a run")
    with gr.Row():
        run_picker = gr.Dropdown(choices=run_choices(), label="Run")
        only_incorrect = gr.Checkbox(value=False, label="Only incorrect")
        inspect_button = gr.Button("Load results")
    results_html = gr.HTML()

    gr.Markdown("### Compare models")
    compare_models_picker = gr.Dropdown(choices=model_choices(), label="Models", multiselect=True)
    compare_button = gr.Button("Compare")
    compare_html = gr.HTML()

    # -------------------------------------------------------------- handlers
    def do_run(model_id, suite_value, examples, shots, temperature_v, max_tokens_v, seed_v,
               allow, custom):
        if not model_id:
            return t.note("Select a model.", "warn"), None, runs_table(), gr.update()
        config = BenchmarkConfig(
            suite=suite_value, num_examples=int(examples), few_shot=int(shots),
            temperature=float(temperature_v), max_new_tokens=int(max_tokens_v), seed=int(seed_v),
            allow_download=bool(allow), custom_path=custom or None,
        )
        try:
            record = runner.create_run(model_id=model_id, config=config)
            runner.start(record["id"])
        except StudioError as exc:
            return t.error_message(exc), None, runs_table(), gr.update()
        return (t.note(f"Running <b>{record['name']}</b>…"), record["id"], runs_table(),
                gr.update(choices=run_choices()))

    run_button.click(
        do_run,
        [model, suite, num_examples, few_shot, temperature, max_tokens, seed, allow_download, custom_path],
        [run_status, run_state, table, run_picker],
    )

    def do_cancel(run_id):
        if not run_id:
            return t.note("No run in progress.", "warn")
        runner.cancel(run_id)
        return t.note("Cancellation requested.", "warn")

    cancel_button.click(do_cancel, run_state, run_status)

    def poll(run_id):
        if not run_id:
            return gr.update(), gr.update()
        progress = runner.progress(run_id)
        if progress.error:
            return t.note(progress.error, "bad"), runs_table()
        if progress.finished:
            record = get_db().get("benchmark_runs", run_id) or {}
            if record.get("status") == "completed":
                return (
                    t.note(
                        f"Finished: <b>{record['accuracy']*100:.1f}%</b> accuracy on "
                        f"{record['total_items']} items "
                        f"({(record.get('config') or {}).get('source', '—')} source).", "good"),
                    runs_table(),
                )
            return t.note(f"Run {record.get('status', 'finished')}.", "warn"), runs_table()
        return (
            t.note(
                f"Running… {progress.completed}/{progress.total} items, "
                f"{progress.correct} correct so far. <span class='stat-hint'>{progress.current}</span>"),
            gr.update(),
        )

    gr.Timer(2.0).tick(poll, run_state, [run_status, table])

    def do_inspect(run_id, incorrect_only):
        if not run_id:
            return t.note("Select a run.", "warn")
        try:
            payload = get_results(run_id, only_incorrect=bool(incorrect_only))
        except StudioError as exc:
            return t.error_message(exc)
        record, items = payload["run"], payload["items"]
        config = record.get("config") or {}

        html = t.stat_grid([
            t.stat("Accuracy", f"{record['accuracy']*100:.1f}%" if record.get("accuracy") is not None else None,
                   unavailable=record.get("accuracy") is None),
            t.stat("Mean score", t.fmt_number(record.get("score"), 3)),
            t.stat("Items", t.fmt_number(record.get("total_items"))),
            t.stat("Avg latency", f"{record['avg_latency_ms']:.0f} ms" if record.get("avg_latency_ms") else None,
                   unavailable=not record.get("avg_latency_ms")),
            t.stat("Tokens/sec", t.fmt_number(record.get("tokens_per_sec"), 1)),
            t.stat("Runtime", t.fmt_duration(record.get("runtime_seconds"))),
        ])
        if config.get("source_note"):
            html += t.note(config["source_note"], "warn" if config.get("source") == "sample" else "")
        if record.get("error"):
            html += t.note(record["error"], "bad")
        if record.get("category_scores"):
            html += t.table(
                ["Category", "Score"],
                [[key, f"{value*100:.1f}%"] for key, value in sorted(record["category_scores"].items())],
            )
        html += "<br>" + t.table(
            ["#", "Question", "Expected", "Model answer", "Result", "Score", "ms", "Tokens"],
            [
                [
                    item["idx"] + 1,
                    (item["question"] or "")[:90].replace("<", "&lt;"),
                    (item["expected"] or "")[:40].replace("<", "&lt;"),
                    (item["response"] or "")[:90].replace("<", "&lt;"),
                    "✓" if item["correct"] else "✗",
                    f"{item['score']:.2f}",
                    f"{item['latency_ms']:.0f}",
                    item["tokens"],
                ]
                for item in items
            ],
            empty="No items match this filter.",
        )
        return html

    inspect_button.click(do_inspect, [run_picker, only_incorrect], results_html)

    def do_compare(model_ids):
        if not model_ids or len(model_ids) < 2:
            return t.note("Select at least two models.", "warn")
        payload = compare_models(list(model_ids))
        rows, suites_seen = payload["rows"], payload["suites"]
        header = ["Metric"] + [row["name"][:20] for row in rows]
        body = [
            ["Parameters"] + [t.fmt_params(row["parameters"]) for row in rows],
            ["Model size"] + [t.fmt_bytes(row["size_bytes"]) for row in rows],
            ["Context length"] + [t.fmt_number(row["context_length"]) for row in rows],
            ["Precision"] + [row.get("quantization") or row.get("precision") or "—" for row in rows],
            ["Architecture"] + [row.get("architecture") or "—" for row in rows],
            ["Best val loss"] + [t.fmt_number(row["validation_loss"], 4) for row in rows],
            ["Inference tok/s"] + [t.fmt_number(row["tokens_per_sec"], 1) for row in rows],
        ]
        for suite_key in suites_seen:
            cells = []
            for row in rows:
                entry = row["benchmarks"].get(suite_key)
                if not entry or entry["accuracy"] is None:
                    cells.append("not run")
                else:
                    marker = "" if entry["source"] in {"official", "local"} else " *"
                    cells.append(f"{entry['accuracy']*100:.1f}%{marker}")
            body.append([f"{suite_key} accuracy"] + cells)

        html = t.table(header, body)
        html += t.note(
            "Blank cells mean the benchmark was never run for that model — no score is inferred. "
            "<b>*</b> marks a bundled-sample run, which is not an official benchmark score."
        )
        return html

    compare_button.click(do_compare, compare_models_picker, compare_html)

    def do_refresh():
        return (gr.update(choices=model_choices()), gr.update(choices=run_choices()),
                gr.update(choices=model_choices()), runs_table())

    refresh_button.click(do_refresh, outputs=[model, run_picker, compare_models_picker, table])
