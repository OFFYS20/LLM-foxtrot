"""Conversations: history, prompt assembly, context management, persistence.

When a conversation outgrows the model's context the chosen strategy is applied
and *reported* — history is never silently discarded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import NotFoundError, ValidationError

CONTEXT_STRATEGIES = ("trim_oldest", "summarize", "new_context", "retrieve")

SYSTEM_PRESETS: dict[str, str] = {
    "General Assistant": (
        "You are a helpful, accurate assistant. Answer clearly and concisely, and say when "
        "you are unsure."
    ),
    "Coding Assistant": (
        "You are a careful programming assistant. Prefer correct, readable code. Explain "
        "trade-offs briefly and point out edge cases."
    ),
    "Research Assistant": (
        "You are a research assistant. Be precise, cite the source material you were given, "
        "and clearly separate what the sources say from your own inference."
    ),
    "Writing Assistant": (
        "You are a writing assistant. Improve clarity and flow while preserving the author's "
        "voice and meaning."
    ),
    "Document Assistant": (
        "You answer questions using the retrieved document excerpts provided. If the excerpts "
        "do not contain the answer, say so instead of guessing."
    ),
}

ROLE_TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


@dataclass
class Message:
    role: str
    content: str
    id: str | None = None
    created_at: float = field(default_factory=time.time)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "stats": self.stats,
        }


@dataclass
class ContextReport:
    """What had to be done to fit the conversation into the context window."""

    strategy: str = "trim_oldest"
    prompt_tokens: int = 0
    context_limit: int = 0
    messages_dropped: int = 0
    messages_summarized: int = 0
    truncated: bool = False
    note: str | None = None

    @property
    def percent(self) -> float:
        return (self.prompt_tokens / self.context_limit * 100) if self.context_limit else 0.0


def create_conversation(
    *,
    title: str | None = None,
    model_id: str | None = None,
    system_prompt: str = "",
    params: dict[str, Any] | None = None,
    rag_index_id: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": new_id("conv"),
        "title": title or "New conversation",
        "model_id": model_id,
        "system_prompt": system_prompt,
        "rag_index_id": rag_index_id,
        "params": params or {},
        "meta": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    get_db().insert("conversations", record)
    return get_db().require("conversations", record["id"])


def add_message(
    conversation_id: str, role: str, content: str, *, stats: dict[str, Any] | None = None
) -> dict[str, Any]:
    if role not in {"system", "user", "assistant"}:
        raise ValidationError(f"Invalid role {role!r}")
    db = get_db()
    db.require("conversations", conversation_id)
    record = {
        "id": new_id("msg"),
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "stats": stats or {},
        "created_at": time.time(),
    }
    db.insert("messages", record)
    updates: dict[str, Any] = {"updated_at": time.time()}
    conversation = db.require("conversations", conversation_id)
    if role == "user" and conversation["title"] in {"New conversation", ""}:
        updates["title"] = content.strip()[:60] + ("…" if len(content.strip()) > 60 else "")
    db.update("conversations", conversation_id, updates)
    return record


def get_messages(conversation_id: str) -> list[Message]:
    rows = get_db().list(
        "messages", where="conversation_id = ?", params=(conversation_id,), order_by="created_at ASC"
    )
    return [
        Message(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            stats=row.get("stats") or {},
        )
        for row in rows
    ]


def update_message(message_id: str, content: str) -> dict[str, Any]:
    db = get_db()
    db.require("messages", message_id)
    db.update("messages", message_id, {"content": content})
    return db.require("messages", message_id)


def delete_message(message_id: str, *, and_after: bool = False) -> int:
    db = get_db()
    record = db.require("messages", message_id)
    removed = 1
    if and_after:
        later = db.list(
            "messages",
            where="conversation_id = ? AND created_at > ?",
            params=(record["conversation_id"], record["created_at"]),
        )
        for row in later:
            db.delete("messages", row["id"])
        removed += len(later)
    db.delete("messages", message_id)
    return removed


def clear_conversation(conversation_id: str) -> int:
    db = get_db()
    rows = db.list("messages", where="conversation_id = ?", params=(conversation_id,))
    for row in rows:
        db.delete("messages", row["id"])
    return len(rows)


def delete_conversation(conversation_id: str) -> None:
    db = get_db()
    clear_conversation(conversation_id)
    db.delete("conversations", conversation_id)


def list_conversations(limit: int = 100) -> list[dict[str, Any]]:
    return get_db().list("conversations", order_by="updated_at DESC", limit=limit)


# ------------------------------------------------------------ prompt building
def build_prompt(
    messages: list[Message],
    *,
    system_prompt: str = "",
    tokenizer: Any = None,
    context_limit: int = 2048,
    reserve_for_response: int = 256,
    strategy: str = "trim_oldest",
    retrieved_context: str = "",
    summarizer: Any = None,
) -> tuple[str, ContextReport]:
    """Assemble the prompt, applying a context strategy when it does not fit."""
    report = ContextReport(strategy=strategy, context_limit=context_limit)

    def count(text: str) -> int:
        if tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001
            return max(1, len(text) // 4)

    system_parts = [part for part in (system_prompt.strip(), retrieved_context.strip()) if part]
    system_block = "\n\n".join(system_parts)
    budget = max(64, context_limit - reserve_for_response)

    def render(history: list[Message], summary: str = "") -> str:
        lines: list[str] = []
        if system_block:
            lines.append(f"{ROLE_TAGS['system']}\n{system_block}")
        if summary:
            lines.append(f"{ROLE_TAGS['system']}\nSummary of earlier conversation:\n{summary}")
        for message in history:
            lines.append(f"{ROLE_TAGS.get(message.role, '<|user|>')}\n{message.content}")
        lines.append(f"{ROLE_TAGS['assistant']}\n")
        return "\n".join(lines)

    history = list(messages)
    prompt = render(history)
    tokens = count(prompt)

    if tokens <= budget:
        report.prompt_tokens = tokens
        return prompt, report

    if strategy == "new_context":
        keep = history[-1:] if history else []
        report.messages_dropped = len(history) - len(keep)
        prompt = render(keep)
        report.prompt_tokens = count(prompt)
        report.note = (
            f"Context limit reached — started a fresh context and kept only the latest message "
            f"({report.messages_dropped} earlier messages are still stored in the conversation)."
        )
        return prompt, report

    if strategy == "summarize" and summarizer is not None and len(history) > 2:
        head, tail = history[:-4], history[-4:]
        if head:
            try:
                summary = summarizer(head)
                report.messages_summarized = len(head)
                prompt = render(tail, summary=summary)
                report.prompt_tokens = count(prompt)
                if report.prompt_tokens <= budget:
                    report.note = (
                        f"Summarised {len(head)} earlier messages to fit the context window."
                    )
                    return prompt, report
            except Exception as exc:  # noqa: BLE001 - fall through to trimming
                log.warning(f"Summarisation failed, trimming instead: {exc}", source="inference")

    # Default: drop the oldest turns, always keeping the most recent user message.
    dropped = 0
    while history and count(render(history)) > budget:
        history.pop(0)
        dropped += 1
    if not history and messages:
        history = messages[-1:]
        prompt = render(history)
        tokens = count(prompt)
        if tokens > budget:  # a single message longer than the window
            report.truncated = True
            content = history[0].content
            while count(render([Message(role=history[0].role, content=content)])) > budget and len(content) > 200:
                content = content[: int(len(content) * 0.8)]
            history = [Message(role=history[0].role, content=content + "\n[truncated]")]
    prompt = render(history)
    report.messages_dropped = dropped
    report.prompt_tokens = count(prompt)
    report.note = (
        f"Context limit reached — dropped the {dropped} oldest message(s) from the prompt. "
        f"They remain in the saved conversation."
        if dropped
        else None
    )
    if report.truncated:
        report.note = (report.note or "") + " The remaining message was truncated to fit."
    return prompt, report


def conversation_token_usage(conversation_id: str, tokenizer: Any = None) -> dict[str, Any]:
    messages = get_messages(conversation_id)
    def count(text: str) -> int:
        if tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001
            return max(1, len(text) // 4)

    total = sum(count(message.content) for message in messages)
    return {"messages": len(messages), "tokens": total}
