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


# ------------------------------------------------- masking the prompt, exactly
class _Toy:
    """A tokenizer that wraps every encoding in <s> … </s>, as a real one does."""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, add_special_tokens=False, **_):
        body = [10 + (ord(ch) % 50) for ch in text]
        return {"input_ids": ([1] + body + [2]) if add_special_tokens else body}


def test_the_prompt_mask_does_not_swallow_the_first_token_of_the_answer():
    """The bug this guards against is silent and total: mask one token too many
    and the model never learns which token *starts* an answer, so it emits
    end-of-sequence instead and returns nothing at all."""
    from ai_studio.data.dataset_builder import record_to_text
    from ai_studio.training.data import _prompt_length

    record = {"question": "Where is Paris?", "answer": "Paris is in France."}
    tokenizer = _Toy()

    whole = tokenizer(record_to_text(record, "qa"), add_special_tokens=True)["input_ids"]
    masked = _prompt_length(record, "qa", tokenizer, "")

    assert masked < len(whole), "the whole example must not be masked"
    prompt_text = "### Question:\nWhere is Paris?\n\n### Answer:\n"
    expected = 1 + len(tokenizer(prompt_text)["input_ids"])   # BOS + the prompt
    assert masked == expected, (
        f"masked {masked} positions where the prompt occupies {expected}; "
        f"the difference is answer tokens the model never learns"
    )


def test_the_answer_is_what_is_left_unmasked():
    from ai_studio.data.dataset_builder import record_to_text
    from ai_studio.training.data import IGNORE_INDEX, SequenceDataset, _prompt_length

    record = {"question": "Where is Paris?", "answer": "Paris is in France."}
    tokenizer = _Toy()
    ids = tokenizer(record_to_text(record, "qa"), add_special_tokens=True)["input_ids"]
    prompt = _prompt_length(record, "qa", tokenizer, "")

    dataset = SequenceDataset([ids], tokenizer.pad_token_id, len(ids), [prompt])
    labels = dataset[0]["labels"]

    learned = [int(value) for value in labels if int(value) != IGNORE_INDEX]
    assert learned, "something must be learned from every pair"
    assert learned == ids[prompt:], "exactly the answer, and all of it"


def test_an_instruction_record_masks_its_prompt_too():
    from ai_studio.training.data import _prompt_length

    record = {"instruction": "Add two numbers.", "input": "2 and 3", "output": "5"}
    assert _prompt_length(record, "instruction", _Toy(), "") > 0


def test_a_plain_text_record_masks_nothing():
    """Continuing text means learning every token, including the first."""
    from ai_studio.training.data import _prompt_length

    assert _prompt_length({"text": "hello"}, "raw_lm", _Toy(), "") == 0
