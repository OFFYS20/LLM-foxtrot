"""Conversations, messages and playground comparisons."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, id_column
from app.db.models.enums import RunProvenance


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = id_column("conv")
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), default=None)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Message.created_at",
    )


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = id_column("msg")
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # system | user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    model_id: Mapped[str | None] = mapped_column(String(40), default=None)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, default=None)
    latency_ms: Mapped[float | None] = mapped_column(Float, default=None)
    time_to_first_token_ms: Mapped[float | None] = mapped_column(Float, default=None)
    finish_reason: Mapped[str | None] = mapped_column(String(32), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class PlaygroundComparison(Base, TimestampMixin):
    __tablename__ = "playground_comparisons"

    id: Mapped[str] = id_column("pg")
    prompt: Mapped[str] = mapped_column(Text)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    entries: Mapped[list[Any]] = mapped_column(default=list)
    winner_model_id: Mapped[str | None] = mapped_column(String(40), default=None)
    votes: Mapped[dict[str, Any]] = mapped_column(default=dict)
    provenance: Mapped[RunProvenance] = mapped_column(String(24), default=RunProvenance.SIMULATED)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
