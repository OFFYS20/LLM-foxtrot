"""Conversations: persistence, deletion, and context management that is reported.

The rule these tests enforce is that history is never lost silently — whatever a
strategy does to fit the window is visible in the ContextReport.
"""

from __future__ import annotations

import pytest

from ai_studio.core.errors import NotFoundError
from ai_studio.inference.conversation import (
    CONTEXT_STRATEGIES,
    Message,
    add_message,
    build_prompt,
    clear_conversation,
    conversation_token_usage,
    create_conversation,
    delete_conversation,
    delete_message,
    get_messages,
    list_conversations,
    update_message,
)


def conversation_with(turns: int) -> str:
    record = create_conversation(title=f"{turns}-turn")
    for index in range(turns):
        add_message(record["id"], "user", f"question {index}")
        add_message(record["id"], "assistant", f"answer {index}")
    return record["id"]


# ------------------------------------------------------------- persistence
def test_messages_round_trip():
    conversation_id = conversation_with(2)
    messages = get_messages(conversation_id)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0].content == "question 0"


def test_clear_keeps_the_conversation_but_drops_messages():
    conversation_id = conversation_with(2)
    removed = clear_conversation(conversation_id)
    assert removed == 4
    assert get_messages(conversation_id) == []
    assert conversation_id in {row["id"] for row in list_conversations()}


def test_delete_removes_the_conversation_and_its_messages():
    conversation_id = conversation_with(2)
    delete_conversation(conversation_id)
    assert conversation_id not in {row["id"] for row in list_conversations()}
    assert get_messages(conversation_id) == []


def test_deleting_an_unknown_conversation_is_an_error():
    with pytest.raises(NotFoundError):
        delete_conversation("conv-does-not-exist")


def test_editing_a_message_rewrites_only_that_message():
    conversation_id = conversation_with(1)
    first = get_messages(conversation_id)[0]
    update_message(first.id, "edited question")
    contents = [m.content for m in get_messages(conversation_id)]
    assert contents == ["edited question", "answer 0"]


def test_delete_message_and_after_truncates_the_branch():
    conversation_id = conversation_with(3)
    messages = get_messages(conversation_id)
    removed = delete_message(messages[2].id, and_after=True)
    assert removed == 4
    assert [m.content for m in get_messages(conversation_id)] == ["question 0", "answer 0"]


def test_token_usage_counts_every_message():
    conversation_id = conversation_with(2)
    usage = conversation_token_usage(conversation_id)
    assert usage["messages"] == 4
    assert usage["tokens"] > 0


# --------------------------------------------------------- context handling
def test_a_conversation_that_fits_is_left_alone():
    messages = [Message(role="user", content="hello")]
    prompt, report = build_prompt(messages, context_limit=2048)
    assert "hello" in prompt
    assert report.messages_dropped == 0
    assert report.note is None, "nothing happened, so nothing is reported"


@pytest.mark.parametrize("strategy", CONTEXT_STRATEGIES)
def test_every_declared_strategy_is_implemented(strategy):
    """Each name in CONTEXT_STRATEGIES must do something and say what it did."""
    messages = [
        Message(role="user" if index % 2 == 0 else "assistant", content=f"turn {index} " + "w" * 400)
        for index in range(12)
    ]
    prompt, report = build_prompt(messages, context_limit=256, strategy=strategy)

    assert report.strategy == strategy
    assert prompt, "a strategy must still produce a prompt"
    assert report.note, f"{strategy} reduced the context without reporting it"
    assert report.prompt_tokens > 0


def test_trim_oldest_keeps_the_most_recent_turn():
    messages = [Message(role="user", content=f"turn {index} " + "w" * 400) for index in range(10)]
    prompt, report = build_prompt(messages, context_limit=256, strategy="trim_oldest")
    assert "turn 9" in prompt
    assert report.messages_dropped > 0
    assert "still remain in the saved conversation" in report.note or "remain in the saved" in report.note


def test_new_context_keeps_only_the_latest_message():
    messages = [Message(role="user", content=f"turn {index} " + "w" * 400) for index in range(10)]
    prompt, report = build_prompt(messages, context_limit=256, strategy="new_context")
    assert "turn 9" in prompt
    assert "turn 0" not in prompt
    assert report.messages_dropped == 9


def test_retrieve_keeps_the_relevant_turn_rather_than_the_newest():
    filler = "unrelated chatter about the weather and lunch plans " * 12
    messages = [
        Message(role="user", content="How do I configure the gradient accumulation steps?"),
        Message(role="assistant", content="Set gradient_accumulation_steps in the training config."),
        *[Message(role="user", content=f"{filler} {index}") for index in range(6)],
        Message(role="assistant", content="Understood."),
        Message(role="user", content="Remind me about gradient accumulation steps again."),
    ]
    prompt, report = build_prompt(messages, context_limit=900, strategy="retrieve")

    assert "Remind me about gradient accumulation" in prompt, "the question must survive"
    assert report.messages_retrieved >= 1
    assert "gradient_accumulation_steps" in prompt, (
        "the relevant older turn should be retrieved ahead of newer filler"
    )
    assert "relevant" in report.note and "Nothing was deleted" in report.note


def test_retrieve_preserves_conversation_order():
    messages = [
        Message(role="user", content=f"alpha beta gamma message {index}") for index in range(10)
    ]
    prompt, _ = build_prompt(messages, context_limit=400, strategy="retrieve")
    indexes = [
        int(line.split()[-1])
        for line in prompt.splitlines()
        if line.startswith("alpha beta gamma message")
    ]
    assert indexes == sorted(indexes), "retrieved turns must stay in chronological order"


def test_a_single_oversized_message_is_truncated_and_says_so():
    messages = [Message(role="user", content="w" * 40_000)]
    prompt, report = build_prompt(messages, context_limit=256, strategy="trim_oldest")
    assert report.truncated is True
    assert "truncated" in prompt
    assert "truncated" in (report.note or "")


def test_summarize_falls_back_to_trimming_when_the_summariser_fails():
    messages = [Message(role="user", content=f"turn {index} " + "w" * 400) for index in range(10)]

    def broken_summarizer(_history):
        raise RuntimeError("no summariser model loaded")

    prompt, report = build_prompt(
        messages, context_limit=256, strategy="summarize", summarizer=broken_summarizer
    )
    assert prompt, "a failed summariser must not break the chat"
    assert report.messages_dropped > 0


def test_retrieved_context_is_placed_in_the_prompt():
    messages = [Message(role="user", content="What is the warranty?")]
    prompt, _ = build_prompt(
        messages, context_limit=2048, retrieved_context="Warranty lasts 24 months."
    )
    assert "Warranty lasts 24 months." in prompt
