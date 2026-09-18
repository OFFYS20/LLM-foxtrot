"""Structured logging.

Every notable event goes to stdout, to the ``logs`` table (so it survives a
restart) and to an in-memory ring buffer the Logs screen tails.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from typing import Any

LEVELS = ("debug", "info", "warning", "error")
_STDLIB = {"debug": 10, "info": 20, "warning": 30, "error": 40}

_configured = False
_buffer: deque[dict[str, Any]] = deque(maxlen=4000)
_lock = threading.Lock()
_listeners: list[Any] = []


def configure(level: str = "info") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)-22s %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger("ai_studio")
    root.handlers = [handler]
    root.setLevel(_STDLIB.get(level, 20))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"ai_studio.{name}")


def log(
    message: str,
    *,
    level: str = "info",
    source: str = "app",
    context: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Record one event. Never raises — logging must not break a workflow."""
    entry = {
        "ts": time.time(),
        "level": level if level in LEVELS else "info",
        "source": source,
        "message": message,
        "context": context or {},
    }
    with _lock:
        _buffer.append(entry)

    try:
        get_logger(source).log(_STDLIB.get(entry["level"], 20), message)
    except Exception:  # noqa: BLE001
        pass

    if persist:
        try:
            from ai_studio.core.database import get_db

            get_db().insert("logs", dict(entry))
        except Exception:  # noqa: BLE001 - never let persistence break the caller
            pass

    for listener in list(_listeners):
        try:
            listener(entry)
        except Exception:  # noqa: BLE001
            pass
    return entry


def debug(message: str, **kw: Any) -> dict[str, Any]:
    return log(message, level="debug", **kw)


def info(message: str, **kw: Any) -> dict[str, Any]:
    return log(message, level="info", **kw)


def warning(message: str, **kw: Any) -> dict[str, Any]:
    return log(message, level="warning", **kw)


def error(message: str, **kw: Any) -> dict[str, Any]:
    return log(message, level="error", **kw)


def exception(message: str, exc: BaseException, **kw: Any) -> dict[str, Any]:
    context = dict(kw.pop("context", {}) or {})
    context["exception"] = f"{type(exc).__name__}: {exc}"
    return log(message, level="error", context=context, **kw)


def tail(limit: int = 300, *, level: str | None = None, source: str | None = None) -> list[dict[str, Any]]:
    with _lock:
        entries = list(_buffer)
    if level:
        entries = [e for e in entries if e["level"] == level]
    if source:
        entries = [e for e in entries if e["source"] == source]
    return entries[-limit:]


def subscribe(listener: Any) -> Any:
    _listeners.append(listener)
    return lambda: _listeners.remove(listener) if listener in _listeners else None
