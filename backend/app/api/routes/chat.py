"""Chat, conversations and streaming generation."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.core.errors import ValidationError
from app.db.models.catalog import Model
from app.db.models.chat import Conversation, Message
from app.db.session import session_scope
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationUpdate,
    GenerationUsage,
    MessageRead,
    MessageUpdate,
    TokenCountRequest,
    TokenCountResponse,
)
from app.schemas.common import Ack, Page
from app.services.inference.registry import registry as inference_registry
from app.services.logbook import logbook

router = APIRouter(tags=["chat"])


# --------------------------------------------------------------- completions
@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, session: SessionDep):
    """Streaming (SSE) or buffered chat completion.

    Streaming frames: ``{"type": "token"|"usage"|"error", ...}`` — one JSON
    object per SSE ``data:`` line, terminated by ``data: [DONE]``.
    """
    model = get_or_404(session, Model, payload.model_id, "Model")
    adapter = await inference_registry.get_adapter(model)

    messages: list[ChatMessage] = []
    if payload.system_prompt:
        messages.append(ChatMessage(role="system", content=payload.system_prompt))
    messages.extend(payload.messages)
    if not any(m.role == "user" for m in messages):
        raise ValidationError("At least one user message is required")

    conversation_id = payload.conversation_id
    if payload.persist and conversation_id:
        _append_user_message(conversation_id, payload.messages[-1], adapter)

    if not payload.stream:
        result = await adapter.complete(messages, payload.params)
        usage = GenerationUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            tokens_per_sec=round(result.tokens_per_sec, 2),
            latency_ms=round(result.latency_ms, 2),
            time_to_first_token_ms=round(result.time_to_first_token_ms, 2),
            finish_reason=result.finish_reason,
            provenance=result.provenance,
            engine=result.engine,
        )
        message_id = None
        if payload.persist and conversation_id:
            message_id = _append_assistant_message(conversation_id, model.id, result.text, usage)
        return ChatCompletionResponse(
            id=message_id or "msg_transient",
            conversation_id=conversation_id,
            model_id=model.id,
            content=result.text,
            usage=usage,
            created_at=datetime.now(timezone.utc),
        )

    async def event_stream():
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        finish_reason = "stop"
        try:
            async for chunk in adapter.stream(messages, payload.params):
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(chunk.text)
                    yield _sse({"type": "token", "text": chunk.text, "index": chunk.index})
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
        except asyncio.CancelledError:
            finish_reason = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            logbook.error(f"Generation failed: {exc}", source="chat")
            yield _sse({"type": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        text = "".join(pieces)
        elapsed = time.perf_counter() - started
        ttft = ((first_token_at or started) - started) * 1000
        completion_tokens = adapter.count_tokens(text)
        prompt_tokens = sum(adapter.count_tokens(m.content) for m in messages)
        gen_seconds = max(1e-6, elapsed - ttft / 1000)

        usage = GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tokens_per_sec=round(completion_tokens / gen_seconds, 2),
            latency_ms=round(elapsed * 1000, 2),
            time_to_first_token_ms=round(ttft, 2),
            finish_reason=finish_reason,
            provenance=adapter.provenance,
            engine=adapter.engine,
        )
        message_id = None
        if payload.persist and conversation_id:
            message_id = _append_assistant_message(conversation_id, model.id, text, usage)

        yield _sse({"type": "usage", "usage": usage.model_dump(), "message_id": message_id})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat/tokenize", response_model=TokenCountResponse)
async def count_tokens(payload: TokenCountRequest, session: SessionDep) -> TokenCountResponse:
    model = session.get(Model, payload.model_id) if payload.model_id else None
    if model is None:
        estimate = max(1, round(len(payload.text) / 4)) if payload.text else 0
        return TokenCountResponse(tokens=estimate, characters=len(payload.text), method="estimated")
    adapter = await inference_registry.get_adapter(model)
    return TokenCountResponse(
        tokens=adapter.count_tokens(payload.text),
        characters=len(payload.text),
        method=adapter.token_count_method(),  # type: ignore[arg-type]
    )


# -------------------------------------------------------------- conversations
@router.get("/conversations", response_model=Page[ConversationRead])
def list_conversations(session: SessionDep, pagination: PaginationDep) -> Page[ConversationRead]:
    statement = select(Conversation).order_by(
        Conversation.pinned.desc(), Conversation.updated_at.desc()
    )
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[ConversationRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/conversations", response_model=ConversationDetail, status_code=201)
def create_conversation(payload: ConversationCreate, session: SessionDep) -> ConversationDetail:
    conversation = Conversation(
        title=payload.title or "New conversation",
        model_id=payload.model_id,
        system_prompt=payload.system_prompt,
        params=payload.params.model_dump(mode="json"),
    )
    session.add(conversation)
    session.flush()
    return ConversationDetail.model_validate(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, session: SessionDep) -> ConversationDetail:
    conversation = get_or_404(session, Conversation, conversation_id, "Conversation")
    return ConversationDetail.model_validate(conversation)


@router.patch("/conversations/{conversation_id}", response_model=ConversationDetail)
def update_conversation(
    conversation_id: str, payload: ConversationUpdate, session: SessionDep
) -> ConversationDetail:
    conversation = get_or_404(session, Conversation, conversation_id, "Conversation")
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    params = data.pop("params", None)
    for field, value in data.items():
        setattr(conversation, field, value)
    if params is not None:
        conversation.params = params
    session.flush()
    return ConversationDetail.model_validate(conversation)


@router.delete("/conversations/{conversation_id}", response_model=Ack)
def delete_conversation(conversation_id: str, session: SessionDep) -> Ack:
    conversation = get_or_404(session, Conversation, conversation_id, "Conversation")
    session.delete(conversation)
    return Ack(ok=True, message="Conversation deleted", id=conversation_id)


@router.post(
    "/conversations/{conversation_id}/messages", response_model=MessageRead, status_code=201
)
def add_message(conversation_id: str, payload: ChatMessage, session: SessionDep) -> MessageRead:
    conversation = get_or_404(session, Conversation, conversation_id, "Conversation")
    message = Message(
        conversation_id=conversation.id,
        role=payload.role,
        content=payload.content,
        prompt_tokens=max(1, round(len(payload.content) / 4)),
    )
    session.add(message)
    session.flush()
    return MessageRead.model_validate(message)


@router.patch("/messages/{message_id}", response_model=MessageRead)
def edit_message(message_id: str, payload: MessageUpdate, session: SessionDep) -> MessageRead:
    message = get_or_404(session, Message, message_id, "Message")
    message.content = payload.content
    session.flush()
    return MessageRead.model_validate(message)


@router.delete("/messages/{message_id}", response_model=Ack)
def delete_message(message_id: str, session: SessionDep, and_after: bool = False) -> Ack:
    message = get_or_404(session, Message, message_id, "Message")
    removed = 1
    if and_after:
        later = session.scalars(
            select(Message).where(
                Message.conversation_id == message.conversation_id,
                Message.created_at > message.created_at,
            )
        ).all()
        removed += len(later)
        for item in later:
            session.delete(item)
    session.delete(message)
    return Ack(ok=True, message=f"Deleted {removed} message(s)", id=message_id)


# -------------------------------------------------------------------- helpers
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _append_user_message(conversation_id: str, message: ChatMessage, adapter) -> None:
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return
        session.add(
            Message(
                conversation_id=conversation_id,
                role=message.role,
                content=message.content,
                prompt_tokens=adapter.count_tokens(message.content),
                total_tokens=adapter.count_tokens(message.content),
            )
        )
        if conversation.title in ("New conversation", "") and message.role == "user":
            conversation.title = message.content[:60] + ("…" if len(message.content) > 60 else "")


def _append_assistant_message(
    conversation_id: str, model_id: str, text: str, usage: GenerationUsage
) -> str | None:
    with session_scope() as session:
        if session.get(Conversation, conversation_id) is None:
            return None
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=text,
            model_id=model_id,
            provenance=usage.provenance,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            tokens_per_sec=usage.tokens_per_sec,
            latency_ms=usage.latency_ms,
            time_to_first_token_ms=usage.time_to_first_token_ms,
            finish_reason=usage.finish_reason,
        )
        session.add(message)
        session.flush()
        return message.id
