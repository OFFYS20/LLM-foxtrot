"""Central log book: persists entries, keeps a ring buffer, publishes to the bus.

Everything that shows up on the Logs page and in the training console goes
through here, so the terminal panel, the database and the WebSocket stream can
never disagree.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.core.events import Topics, bus
from app.core.logging import get_logger
from app.db.models.enums import LogLevel
from app.db.session import session_scope

_logger = get_logger("foxtrot.logbook")


class LogBook:
    def __init__(self, capacity: int | None = None) -> None:
        self._buffer: deque[dict[str, Any]] = deque(
            maxlen=capacity or settings.log_ring_buffer_size
        )
        self._lock = threading.Lock()
        self._seq = 0

    def log(
        self,
        message: str,
        *,
        level: LogLevel | str = LogLevel.INFO,
        source: str = "system",
        job_id: str | None = None,
        context: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        level = LogLevel(level) if not isinstance(level, LogLevel) else level
        ts = datetime.now(timezone.utc)

        with self._lock:
            self._seq += 1
            entry = {
                "id": self._seq,
                "ts": ts.isoformat(),
                "level": level.value,
                "source": source,
                "message": message,
                "job_id": job_id,
                "context": context or {},
            }
            self._buffer.append(entry)

        _logger.log(_LEVEL_MAP.get(level, 20), "[%s] %s", source, message)

        if persist:
            try:
                from app.db.models.system import LogEntry

                with session_scope() as session:
                    session.add(
                        LogEntry(
                            ts=ts,
                            level=level,
                            source=source,
                            message=message,
                            job_id=job_id,
                            context=context or {},
                        )
                    )
            except Exception as exc:  # pragma: no cover - logging must never crash callers
                _logger.warning("log persistence failed: %s", exc)

        bus.publish(Topics.LOG, entry)
        if job_id:
            bus.publish(f"{Topics.TRAINING_LOG}.{job_id}", entry)
        return entry

    def debug(self, message: str, **kw: Any) -> dict[str, Any]:
        return self.log(message, level=LogLevel.DEBUG, **kw)

    def info(self, message: str, **kw: Any) -> dict[str, Any]:
        return self.log(message, level=LogLevel.INFO, **kw)

    def warn(self, message: str, **kw: Any) -> dict[str, Any]:
        return self.log(message, level=LogLevel.WARN, **kw)

    def error(self, message: str, **kw: Any) -> dict[str, Any]:
        return self.log(message, level=LogLevel.ERROR, **kw)

    def tail(self, limit: int = 200, *, job_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._buffer)
        if job_id:
            entries = [e for e in entries if e.get("job_id") == job_id]
        return entries[-limit:]


_LEVEL_MAP = {
    LogLevel.DEBUG: 10,
    LogLevel.INFO: 20,
    LogLevel.WARN: 30,
    LogLevel.ERROR: 40,
}

logbook = LogBook()
