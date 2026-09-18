"""Importing structured records: parsing, validation and building a dataset.

Structured examples (a JSONL of instruction pairs, a CSV of Q&A) must keep their
shape — flattening them into raw text is what instruction tuning cannot use.
"""

from __future__ import annotations

import json

import pytest

from ai_studio.core.errors import ValidationError
from ai_studio.data.dataset_builder import (
    BuildOptions,
    build_from_records,
    load_split,
    validate_records,
)
from ai_studio.ui.dataset_page import read_records

PAIRS = [
    {"instruction": "Explain LoRA", "input": "", "output": "Low-rank adapters freeze the base."},
    {"instruction": "Explain QLoRA", "input": "", "output": "LoRA on a 4-bit base."},
]


def write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ parsing
def test_reads_jsonl(tmp_path):
    path = write(tmp_path, "pairs.jsonl", "\n".join(json.dumps(row) for row in PAIRS))
    assert read_records(path) == PAIRS


def test_blank_lines_in_jsonl_are_skipped(tmp_path):
    path = write(tmp_path, "gappy.jsonl", json.dumps(PAIRS[0]) + "\n\n\n" + json.dumps(PAIRS[1]))
    assert read_records(path) == PAIRS


def test_reads_a_json_array(tmp_path):
    assert read_records(write(tmp_path, "pairs.json", json.dumps(PAIRS))) == PAIRS


@pytest.mark.parametrize("key", ["data", "records", "rows", "examples"])
def test_reads_a_wrapped_json_object(tmp_path, key):
    path = write(tmp_path, f"{key}.json", json.dumps({key: PAIRS}))
    assert read_records(path) == PAIRS


def test_reads_csv_with_a_header(tmp_path):
    path = write(tmp_path, "pairs.csv", "instruction,input,output\nExplain LoRA,,Adapters\n")
    assert read_records(path) == [
        {"instruction": "Explain LoRA", "input": "", "output": "Adapters"}
    ]


def test_reads_tsv(tmp_path):
    path = write(tmp_path, "pairs.tsv", "question\tanswer\nWhat is LoRA?\tAdapters\n")
    assert read_records(path) == [{"question": "What is LoRA?", "answer": "Adapters"}]


def test_malformed_jsonl_names_the_line(tmp_path):
    path = write(tmp_path, "bad.jsonl", json.dumps(PAIRS[0]) + "\n{not json}\n")
    with pytest.raises(ValidationError) as excinfo:
        read_records(path)
    assert "Line 2" in str(excinfo.value)


def test_malformed_json_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        read_records(write(tmp_path, "bad.json", "{"))


def test_json_that_is_not_records_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        read_records(write(tmp_path, "object.json", json.dumps({"instruction": "not a list"})))


def test_csv_without_a_header_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        read_records(write(tmp_path, "headerless.csv", ""))


def test_unsupported_extensions_are_rejected(tmp_path):
    with pytest.raises(ValidationError) as excinfo:
        read_records(write(tmp_path, "notes.parquet", "x"))
    assert ".jsonl" in str(excinfo.value), "the error should name what is supported"


# --------------------------------------------------------------- validation
def test_validation_counts_and_locates_bad_rows():
    records = [PAIRS[0], {"instruction": "", "output": "x"}, {"nope": 1}]
    invalid, issues = validate_records(records, "instruction")
    assert invalid == 2
    assert [issue["row"] for issue in issues] == [1, 2]
    assert all(issue["error"] for issue in issues)


# ------------------------------------------------------------------ building
def test_imported_records_keep_their_structure(tmp_path):
    records = [
        {"instruction": f"Question {index}", "input": "", "output": f"Answer {index}"}
        for index in range(12)
    ]
    dataset = build_from_records(
        "imported-pairs", records, BuildOptions(mode="instruction"), description="from jsonl"
    )
    assert dataset["mode"] == "instruction"
    assert dataset["rows"] == 12

    rows = load_split(dataset["id"], "train")
    assert rows, "the train split must not be empty"
    assert set(rows[0]) >= {"instruction", "output"}, "structure is preserved, not flattened"


def test_a_csv_of_questions_becomes_a_qa_dataset(tmp_path):
    path = write(
        tmp_path,
        "qa.csv",
        "question,answer\n" + "\n".join(f"Q{i},A{i}" for i in range(10)) + "\n",
    )
    dataset = build_from_records("imported-qa", read_records(path), BuildOptions(mode="qa"))
    assert dataset["rows"] == 10
    assert set(load_split(dataset["id"], "train")[0]) >= {"question", "answer"}
