"""Logs screen: structured, filterable application log."""

from __future__ import annotations

import gradio as gr

from ai_studio.core import logging as log
from ai_studio.core.database import get_db
from ai_studio.ui import theme as t

LEVEL_CLASS = {"debug": "", "info": "", "warning": "warn", "error": "err"}


def load_logs(level: str, source: str, search: str, limit: int) -> str:
    clauses, params = [], []
    if level and level != "all":
        clauses.append("level = ?")
        params.append(level)
    if source and source != "all":
        clauses.append("source = ?")
        params.append(source)
    if search:
        clauses.append("message LIKE ?")
        params.append(f"%{search}%")

    rows = get_db().list(
        "logs",
        where=" AND ".join(clauses) or None,
        params=params,
        order_by="ts DESC",
        limit=int(limit),
    )
    if not rows:
        return t.note("No log entries match these filters.")

    import datetime

    lines = []
    for row in reversed(rows):
        stamp = datetime.datetime.fromtimestamp(row["ts"]).strftime("%H:%M:%S")
        css = LEVEL_CLASS.get(row["level"], "")
        message = row["message"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = f"{stamp} {row['level'][:4].upper():5} {row['source']:12} {message}"
        lines.append(f'<span class="{css}">{text}</span>' if css else text)
    return f'<div class="console">{"<br>".join(lines)}</div>'


def sources() -> list[str]:
    rows = get_db().query("SELECT DISTINCT source FROM logs ORDER BY source")
    return ["all"] + [row["source"] for row in rows]


def render() -> None:
    gr.HTML('<div class="studio-title">Logs</div>'
            '<div class="studio-sub">Everything the application records</div>')
    with gr.Row():
        level = gr.Dropdown(["all", "debug", "info", "warning", "error"], value="all", label="Level")
        source = gr.Dropdown(sources(), value="all", label="Source", allow_custom_value=True)
        search = gr.Textbox(label="Search", placeholder="filter messages…")
        limit = gr.Number(value=300, label="Lines", precision=0)
    output = gr.HTML(lambda: load_logs("all", "all", "", 300))

    with gr.Row():
        refresh = gr.Button("Refresh", size="sm")
        follow = gr.Checkbox(value=True, label="Auto-refresh")
        clear = gr.Button("Clear log history", size="sm", variant="stop")

    inputs = [level, source, search, limit]
    refresh.click(load_logs, inputs, output)
    for control in inputs:
        control.change(load_logs, inputs, output)

    timer = gr.Timer(4.0)

    def _tick(is_following, *args):
        return load_logs(*args) if is_following else gr.update()

    timer.tick(_tick, [follow, *inputs], output)

    def _clear():
        get_db().query("DELETE FROM logs")
        log.info("Log history cleared", source="app")
        return load_logs("all", "all", "", 300), gr.update(choices=sources())

    clear.click(_clear, outputs=[output, source])
