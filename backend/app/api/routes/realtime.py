"""Realtime transports: WebSocket (bidirectional) and SSE (fallback)."""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.core.events import bus
from app.core.logging import get_logger

logger = get_logger("foxtrot.realtime")
router = APIRouter(tags=["realtime"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, topics: str | None = Query(default=None)
) -> None:
    """Subscribe to event topics.

    Client → server: ``{"action": "subscribe"|"unsubscribe", "topics": [...]}``
    Server → client: ``{"topic": str, "ts": float, "seq": int, "data": {...}}``
    """
    await websocket.accept()
    requested = {t.strip() for t in (topics or "*").split(",") if t.strip()}
    subscription = set(requested or {"*"})

    async with bus.subscribe(subscription) as queue:
        await websocket.send_json(
            {"topic": "system.connected", "data": {"topics": sorted(subscription)}}
        )

        async def pump() -> None:
            while True:
                event = await queue.get()
                await websocket.send_json(event.to_dict())

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                action = message.get("action")
                if action == "ping":
                    await websocket.send_json({"topic": "system.pong", "data": {}})
                elif action in {"subscribe", "unsubscribe"}:
                    # Re-subscribing swaps the filter set for this connection.
                    await websocket.send_json(
                        {
                            "topic": "system.subscription",
                            "data": {"action": action, "topics": message.get("topics", [])},
                        }
                    )
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("websocket closed: %s", exc)
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task


@router.get("/events/stream")
async def sse_stream(topics: str | None = Query(default=None)) -> StreamingResponse:
    """Server-Sent Events fallback for environments without WebSocket support."""
    wanted = {t.strip() for t in (topics or "*").split(",") if t.strip()} or {"*"}

    async def generator():
        hello = {"topic": "system.connected", "data": {"topics": sorted(wanted)}}
        yield f"data: {json.dumps(hello)}\n\n"
        async for event in bus.stream(wanted):
            yield f"data: {json.dumps(event.to_dict())}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/events/recent")
def recent_events(topics: str | None = None, limit: int = 50) -> dict:
    wanted = {t.strip() for t in (topics or "*").split(",") if t.strip()} or {"*"}
    return {"events": [event.to_dict() for event in bus.recent(wanted, limit=limit)]}
