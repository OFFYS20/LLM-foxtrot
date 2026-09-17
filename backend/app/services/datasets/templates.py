"""Dataset formatting templates and row validation rules."""

from __future__ import annotations

from typing import Any

from app.db.models.enums import DatasetTemplate
from app.schemas.datasets import DatasetIssue, DatasetTemplateInfo

INSTRUCTION_EXAMPLE: dict[str, Any] = {"instruction": "", "input": "", "output": ""}

CHAT_EXAMPLE: dict[str, Any] = {
    "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
}

PLAIN_TEXT_EXAMPLE: dict[str, Any] = {"text": ""}

VALID_ROLES = {"system", "user", "assistant", "tool"}

TEMPLATES: dict[DatasetTemplate, DatasetTemplateInfo] = {
    DatasetTemplate.INSTRUCTION: DatasetTemplateInfo(
        key=DatasetTemplate.INSTRUCTION,
        label="Instruction tuning",
        description="Alpaca-style supervised pairs. `input` may be empty for open prompts.",
        schema_example=INSTRUCTION_EXAMPLE,
        required_fields=["instruction", "output"],
    ),
    DatasetTemplate.CHAT: DatasetTemplateInfo(
        key=DatasetTemplate.CHAT,
        label="Chat",
        description="Multi-turn conversations with role-tagged messages.",
        schema_example=CHAT_EXAMPLE,
        required_fields=["messages"],
    ),
    DatasetTemplate.PLAIN_TEXT: DatasetTemplateInfo(
        key=DatasetTemplate.PLAIN_TEXT,
        label="Plain text",
        description="Raw documents for continued pretraining.",
        schema_example=PLAIN_TEXT_EXAMPLE,
        required_fields=["text"],
    ),
    DatasetTemplate.RAW: DatasetTemplateInfo(
        key=DatasetTemplate.RAW,
        label="Raw / custom",
        description="No schema enforced — map the columns at training time.",
        schema_example={"<column>": "<value>"},
        required_fields=[],
    ),
}


def list_templates() -> list[DatasetTemplateInfo]:
    return list(TEMPLATES.values())


def detect_template(rows: list[dict[str, Any]]) -> DatasetTemplate:
    """Best-effort template detection from the first rows."""
    if not rows:
        return DatasetTemplate.RAW
    sample = rows[0]
    if not isinstance(sample, dict):
        return DatasetTemplate.PLAIN_TEXT
    keys = set(sample.keys())
    if {"instruction", "output"} <= keys:
        return DatasetTemplate.INSTRUCTION
    if "messages" in keys:
        return DatasetTemplate.CHAT
    if keys == {"text"} or ("text" in keys and len(keys) <= 2):
        return DatasetTemplate.PLAIN_TEXT
    return DatasetTemplate.RAW


def validate_row(row: Any, template: DatasetTemplate, index: int) -> list[DatasetIssue]:
    """Return issues for a single row (empty list == valid)."""
    issues: list[DatasetIssue] = []

    if template == DatasetTemplate.RAW:
        if not isinstance(row, dict):
            issues.append(DatasetIssue(row=index, message="Row is not an object"))
        return issues

    if not isinstance(row, dict):
        issues.append(
            DatasetIssue(row=index, message=f"Expected an object, got {type(row).__name__}")
        )
        return issues

    if template == DatasetTemplate.INSTRUCTION:
        for field in ("instruction", "output"):
            value = row.get(field)
            if value is None:
                issues.append(DatasetIssue(row=index, field=field, message=f"Missing `{field}`"))
            elif not isinstance(value, str):
                issues.append(
                    DatasetIssue(row=index, field=field, message=f"`{field}` must be a string")
                )
            elif not value.strip():
                issues.append(DatasetIssue(row=index, field=field, message=f"`{field}` is empty"))
        if "input" in row and not isinstance(row["input"], str):
            issues.append(
                DatasetIssue(row=index, field="input", message="`input` must be a string")
            )

    elif template == DatasetTemplate.CHAT:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            issues.append(
                DatasetIssue(
                    row=index, field="messages", message="`messages` must be a non-empty array"
                )
            )
            return issues
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                issues.append(
                    DatasetIssue(
                        row=index,
                        field=f"messages[{position}]",
                        message="Message must be an object",
                    )
                )
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in VALID_ROLES:
                issues.append(
                    DatasetIssue(
                        row=index,
                        field=f"messages[{position}].role",
                        message=f"Invalid role {role!r} (expected one of {sorted(VALID_ROLES)})",
                    )
                )
            if not isinstance(content, str) or not content.strip():
                issues.append(
                    DatasetIssue(
                        row=index,
                        field=f"messages[{position}].content",
                        message="`content` must be a non-empty string",
                    )
                )
        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if "assistant" not in roles:
            issues.append(
                DatasetIssue(
                    row=index,
                    field="messages",
                    severity="warning",
                    message="Conversation has no assistant turn — nothing to learn from",
                )
            )

    elif template == DatasetTemplate.PLAIN_TEXT:
        text = row.get("text")
        if not isinstance(text, str):
            issues.append(DatasetIssue(row=index, field="text", message="`text` must be a string"))
        elif not text.strip():
            issues.append(DatasetIssue(row=index, field="text", message="`text` is empty"))

    return issues


def row_to_text(row: Any, template: DatasetTemplate) -> str:
    """Flatten a row to the string that would actually be tokenised."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)

    if template == DatasetTemplate.INSTRUCTION:
        parts = [
            str(row.get("instruction", "")),
            str(row.get("input", "") or ""),
            str(row.get("output", "")),
        ]
        return "\n".join(p for p in parts if p)
    if template == DatasetTemplate.CHAT:
        messages = row.get("messages") or []
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in messages if isinstance(m, dict)
        )
    if template == DatasetTemplate.PLAIN_TEXT:
        return str(row.get("text", ""))
    return " ".join(str(v) for v in row.values())
