"""Chunking, dataset building, splitting and validation."""

from __future__ import annotations

import pytest

from ai_studio.core.errors import ValidationError
from ai_studio.data.chunking import chunk_by_characters, deduplicate
from ai_studio.data.dataset_builder import (
    BuildOptions,
    build_from_documents,
    build_from_records,
    load_split,
    record_to_text,
    validate_record,
    validate_records,
)
from ai_studio.data.ingestion import import_text


def test_character_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph number {i} with some filler content." for i in range(40))
    chunks = chunk_by_characters(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 400 for chunk in chunks)


def test_deduplicate_removes_identical_chunks():
    # Each paragraph is long enough to survive min_chars and short enough that
    # chunk_size holds exactly one, so the chunks really are byte-identical.
    paragraph = "The quick brown fox jumps over the lazy dog beside the wide river at dawn."
    chunks = chunk_by_characters("\n\n".join([paragraph] * 6), chunk_size=80, overlap=0)
    assert len(chunks) == 6
    kept, removed = deduplicate(chunks)
    assert removed == 5
    assert len(kept) == 1
    assert [chunk.index for chunk in kept] == [0], "indexes are renumbered after dedup"


@pytest.mark.parametrize(
    "mode,record,valid",
    [
        ("raw_lm", {"text": "hello"}, True),
        ("raw_lm", {"text": ""}, False),
        ("raw_lm", {"nope": 1}, False),
        ("instruction", {"instruction": "Do", "output": "Done"}, True),
        ("instruction", {"instruction": "Do"}, False),
        ("chat", {"messages": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}]}, True),
        ("chat", {"messages": [{"role": "user", "content": "hi"}]}, False),
        ("chat", {"messages": []}, False),
    ],
)
def test_record_validation(mode, record, valid):
    assert (validate_record(record, mode) is None) is valid


def test_validate_records_reports_row_numbers():
    invalid, issues = validate_records([{"text": "ok"}, {"bad": 1}], "raw_lm")
    assert invalid == 1
    assert issues[0]["row"] == 1


def test_build_from_documents_splits_without_leakage():
    ids = [
        import_text(f"Document {i}. " + f"Unique body for document {i}. " * 60, title=f"doc{i}")["id"]
        for i in range(10)
    ]
    dataset = build_from_documents(
        "split-test", ids, BuildOptions(mode="raw_lm", block_size=120, overlap=20)
    )
    assert dataset["train_rows"] > 0
    assert dataset["validation_rows"] > 0
    assert dataset["test_rows"] > 0

    train = {row["text"] for row in load_split(dataset["id"], "train")}
    validation = {row["text"] for row in load_split(dataset["id"], "validation")}
    assert not (train & validation), "identical rows leaked across splits"


def test_build_from_records_rejects_all_invalid():
    with pytest.raises(ValidationError):
        build_from_records("bad-records", [{"nope": 1}], BuildOptions(mode="instruction"))


def test_build_from_records_counts_invalid_rows():
    dataset = build_from_records(
        "mixed-records",
        [{"instruction": "A", "output": "B"}, {"instruction": "", "output": "B"}, {"x": 1}],
        BuildOptions(mode="instruction"),
    )
    assert dataset["stats"]["invalid_records"] == 2
    assert dataset["rows"] == 1, "only the one valid record may be stored"


def test_splits_never_share_a_row():
    records = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
    dataset = build_from_records("disjoint-records", records, BuildOptions(mode="instruction"))
    rows = {split: load_split(dataset["id"], split) for split in ("train", "validation", "test")}
    assert sum(len(part) for part in rows.values()) == 10
    seen: set[str] = set()
    for part in rows.values():
        keys = {record["instruction"] for record in part}
        assert not (keys & seen), "a record appears in more than one split"
        seen |= keys


def test_single_record_stays_in_train_only():
    dataset = build_from_records(
        "one-record", [{"instruction": "A", "output": "B"}], BuildOptions(mode="instruction")
    )
    assert dataset["rows"] == 1
    assert len(load_split(dataset["id"], "train")) == 1
    assert load_split(dataset["id"], "test") == []
    assert any("empty" in warning for warning in dataset["stats"]["warnings"])


def test_splits_must_sum_to_one():
    with pytest.raises(ValidationError):
        BuildOptions(train_split=0.9, validation_split=0.3, test_split=0.3).validate()


def test_record_to_text_formats_each_mode():
    assert record_to_text({"text": "raw"}, "raw_lm") == "raw"
    instruction = record_to_text({"instruction": "Explain", "input": "X", "output": "Y"}, "instruction")
    assert "### Instruction:" in instruction and "### Response:" in instruction
    chat = record_to_text({"messages": [{"role": "user", "content": "hi"}]}, "chat")
    assert "<|user|>" in chat
