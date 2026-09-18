"""Dark theme and shared UI helpers."""

from __future__ import annotations

from typing import Any

import gradio as gr

CSS = """
:root, .dark {
    --studio-bg: #0b0d10;
    --studio-panel: #14171c;
    --studio-panel-2: #1a1e25;
    --studio-border: #262b34;
    --studio-text: #e7e9ee;
    --studio-muted: #8b93a3;
    --studio-accent: #4f8cff;
    --studio-good: #31c48d;
    --studio-warn: #f0b429;
    --studio-bad: #f26d6d;
}
.gradio-container { background: var(--studio-bg) !important; max-width: 100% !important; }
footer { display: none !important; }

.studio-header {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 10px 16px; margin-bottom: 8px;
    background: var(--studio-panel); border: 1px solid var(--studio-border); border-radius: 10px;
}
.studio-title { font-size: 18px; font-weight: 650; letter-spacing: -0.2px; color: var(--studio-text); }
.studio-sub { font-size: 12px; color: var(--studio-muted); }

.studio-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.studio-pill {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
    padding: 4px 9px; border-radius: 6px;
    background: var(--studio-panel-2); border: 1px solid var(--studio-border); color: var(--studio-muted);
}
.studio-pill b { color: var(--studio-text); font-weight: 600; }
.pill-good { border-color: rgba(49,196,141,.45); color: var(--studio-good); }
.pill-warn { border-color: rgba(240,180,41,.45); color: var(--studio-warn); }
.pill-bad  { border-color: rgba(242,109,109,.45); color: var(--studio-bad); }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.stat-card {
    background: var(--studio-panel); border: 1px solid var(--studio-border);
    border-radius: 10px; padding: 12px 14px;
}
.stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--studio-muted); }
.stat-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 20px; font-weight: 600; color: var(--studio-text); margin-top: 4px;
}
.stat-hint { font-size: 11px; color: var(--studio-muted); margin-top: 2px; }
.stat-unavailable .stat-value { color: var(--studio-muted); font-size: 15px; }

.console {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.55;
    background: #07090c; border: 1px solid var(--studio-border); border-radius: 8px;
    padding: 12px; max-height: 340px; overflow-y: auto; white-space: pre-wrap; color: #c9d1d9;
}
.console .warn { color: var(--studio-warn); }
.console .err  { color: var(--studio-bad); }

.note {
    border-left: 3px solid var(--studio-accent); background: var(--studio-panel);
    padding: 10px 12px; border-radius: 0 8px 8px 0; font-size: 13px; color: var(--studio-text);
}
.note-warn { border-left-color: var(--studio-warn); }
.note-bad  { border-left-color: var(--studio-bad); }
.note-good { border-left-color: var(--studio-good); }
.note code { background: var(--studio-panel-2); padding: 1px 5px; border-radius: 4px; font-size: 12px; }

.mono, .mono input, .mono textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
}
.tabitem { padding-top: 10px !important; }
table.studio { width: 100%; border-collapse: collapse; font-size: 12px; }
table.studio th {
    text-align: left; color: var(--studio-muted); font-weight: 500; text-transform: uppercase;
    font-size: 10px; letter-spacing: .08em; padding: 6px 8px; border-bottom: 1px solid var(--studio-border);
}
table.studio td { padding: 6px 8px; border-bottom: 1px solid var(--studio-border); color: var(--studio-text); }
"""


def theme() -> Any:
    # Fonts are set in CSS rather than here: Gradio's theme comparison expects
    # font objects, and plain strings break it.
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0b0d10",
        body_background_fill_dark="#0b0d10",
        block_background_fill="#14171c",
        block_background_fill_dark="#14171c",
        block_border_color="#262b34",
        block_border_color_dark="#262b34",
        block_label_text_color="#8b93a3",
        block_title_text_color="#e7e9ee",
        input_background_fill="#1a1e25",
        input_background_fill_dark="#1a1e25",
        body_text_color="#e7e9ee",
        body_text_color_dark="#e7e9ee",
        button_primary_background_fill="#2f6fe0",
        button_primary_background_fill_dark="#2f6fe0",
    )


# ----------------------------------------------------------------- fragments
def stat(label: str, value: Any, hint: str = "", *, unavailable: bool = False) -> str:
    classes = "stat-card stat-unavailable" if unavailable else "stat-card"
    shown = "Unavailable" if unavailable or value is None else value
    hint_html = f'<div class="stat-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="{classes}"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{shown}</div>{hint_html}</div>'
    )


def stat_grid(cards: list[str]) -> str:
    return f'<div class="stat-grid">{"".join(cards)}</div>'


def pill(text: str, tone: str = "") -> str:
    tone_class = f" pill-{tone}" if tone else ""
    return f'<span class="studio-pill{tone_class}">{text}</span>'


def note(text: str, tone: str = "") -> str:
    tone_class = f" note-{tone}" if tone else ""
    return f'<div class="note{tone_class}">{text}</div>'


def table(headers: list[str], rows: list[list[Any]], *, empty: str = "Nothing yet.") -> str:
    if not rows:
        return f'<div class="stat-hint">{empty}</div>'
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{'' if cell is None else cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<table class="studio"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def console_html(lines: list[str]) -> str:
    rendered = []
    for line in lines[-400:]:
        css = "err" if "[ERROR]" in line else ("warn" if "[WARN" in line or "paused" in line else "")
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rendered.append(f'<span class="{css}">{escaped}</span>' if css else escaped)
    return f'<div class="console">{"<br>".join(rendered) or "waiting for output…"}</div>'


def fmt_bytes(value: Any) -> str:
    if not value:
        return "—"
    number = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(number) < 1024:
            return f"{number:.1f} {unit}" if unit != "B" else f"{number:.0f} B"
        number /= 1024
    return f"{number:.1f} PB"


def fmt_number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_params(value: Any) -> str:
    if not value:
        return "—"
    number = float(value)
    if number >= 1e9:
        return f"{number / 1e9:.2f}B"
    if number >= 1e6:
        return f"{number / 1e6:.1f}M"
    return f"{number:,.0f}"


def fmt_duration(seconds: Any) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def fmt_time(ts: Any) -> str:
    if not ts:
        return "—"
    import datetime

    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")


def error_message(exc: BaseException) -> str:
    from ai_studio.core.errors import StudioError

    if isinstance(exc, StudioError):
        text = exc.message + (f"<br><span class='stat-hint'>{exc.hint}</span>" if exc.hint else "")
        return note(text, "bad")
    return note(f"{type(exc).__name__}: {exc}", "bad")
