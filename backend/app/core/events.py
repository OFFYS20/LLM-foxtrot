"""In-process pub/sub used by the WebSocket and SSE transports.

Producers (training manager, hardware monitor, benchmark runner, log book) call
:meth:`EventBus.publish` from any thread; consumers get an ``asyncio.Queue`` per
subscription and are never blocked by a slow peer — a full queue drops its
oldest event instead of stalling the producer.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

MAX_QUEUE = 512
REPLAY_BUFFER = 200


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "ts": self.ts, "seq": self.seq, "data": self.payload}


class _Subscription:
    def __init__(self, topics: set[str]) -> None:
        self.topics = topics
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=MAX_QUEUE)

    def matches(self, topic: str) -> bool:
        if "*" in self.topics:
            return True
        if topic in self.topics:
            return True
        # prefix wildcards: "training.*" matches "training.job.42"
        return any(t.endswith("*") and topic.startswith(t[:-1]) for t in self.topics)


class EventBus:
    def __init__(self) -> None:
        self._subs: set[_Subscription] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seq = 0
        self._recent: deque[Event] = deque(maxlen=REPLAY_BUFFER)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the serving loop so worker threads can publish safely."""
        self._loop = loop

    # ------------------------------------------------------------------ pub
    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        event = Event(topic=topic, payload=payload, seq=self._seq)
        self._recent.append(event)

        running: asyncio.AbstractEventLoop | None
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            self._dispatch(event)
        elif self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: Event) -> None:
        for sub in list(self._subs):
            if not sub.matches(event.topic):
                continue
            if sub.queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(event)

    # ------------------------------------------------------------------ sub
    @contextlib.asynccontextmanager
    async def subscribe(self, topics: Iterable[str] | None = None):
        sub = _Subscription(set(topics or {"*"}))
        self._subs.add(sub)
        try:
            yield sub.queue
        finally:
            self._subs.discard(sub)

    async def stream(
        self, topics: Iterable[str] | None = None, *, heartbeat: float = 15.0
    ) -> AsyncIterator[Event]:
        async with self.subscribe(topics) as queue:
            while True:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except TimeoutError:
                    yield Event(topic="system.heartbeat", payload={"ok": True})

    def recent(self, topics: Iterable[str] | None = None, limit: int = 50) -> list[Event]:
        wanted = _Subscription(set(topics or {"*"}))
        return [e for e in list(self._recent) if wanted.matches(e.topic)][-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


bus = EventBus()


class Topics:
    """Canonical topic names (also consumed by the frontend WebSocket client)."""

    TRAINING = "training"
    TRAINING_METRIC = "training.metric"
    TRAINING_STATUS = "training.status"
    TRAINING_LOG = "training.log"
    HARDWARE = "hardware.sample"
    BENCHMARK_PROGRESS = "benchmark.progress"
    BENCHMARK_STATUS = "benchmark.status"
    CHAT_TOKEN = "chat.token"
    LOG = "log.entry"
    MODEL_STATUS = "model.status"
    SYSTEM = "system.notice"
