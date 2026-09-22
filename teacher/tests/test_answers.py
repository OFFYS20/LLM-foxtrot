"""Teaching a model to answer: reading the pairs, and asking the right way."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-answer-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import answers, workspace  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


def write(folder: Path, name: str, rows: list[dict]) -> Path:
    path = folder / name
    if name.endswith(".jsonl"):
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    elif name.endswith(".json"):
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        keys = sorted({key for row in rows for key in row})
        lines = [",".join(keys)]
        lines += [",".join(str(row.get(key, "")) for key in keys) for row in rows]
        path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------ reading them
def test_question_and_answer_columns_are_understood(tmp_path):
    write(tmp_path, "pairs.jsonl", [{"question": "Where?", "answer": "There."}] * 3)
    records, style, _ = answers.read_pairs([str(tmp_path)])
    assert style == "qa"
    assert records[0] == {"question": "Where?", "answer": "There.", "context": ""}


def test_instruction_and_output_columns_are_understood(tmp_path):
    write(tmp_path, "pairs.jsonl", [{"instruction": "Add.", "output": "5"}])
    records, style, _ = answers.read_pairs([str(tmp_path)])
    assert style == "instruction"
    assert records[0]["output"] == "5"


def test_a_chat_export_is_understood(tmp_path):
    write(tmp_path, "chat.jsonl", [{"messages": [
        {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}])
    records, style, _ = answers.read_pairs([str(tmp_path)])
    assert style == "chat"
    assert len(records[0]["messages"]) == 2


def test_the_common_synonyms_work_too(tmp_path):
    write(tmp_path, "pairs.jsonl", [{"prompt": "Do it.", "response": "Done."}])
    records, style, _ = answers.read_pairs([str(tmp_path)])
    assert style == "instruction"
    assert records[0]["instruction"] == "Do it." and records[0]["output"] == "Done."


def test_csv_works(tmp_path):
    write(tmp_path, "pairs.csv", [{"question": "Where?", "answer": "There."}])
    records, style, _ = answers.read_pairs([str(tmp_path)])
    assert style == "qa" and len(records) == 1


def test_a_folder_of_files_is_read_through(tmp_path):
    write(tmp_path, "one.jsonl", [{"question": "A?", "answer": "1"}])
    write(tmp_path, "two.jsonl", [{"question": "B?", "answer": "2"}])
    records, _style, _ = answers.read_pairs([str(tmp_path)])
    assert len(records) == 2


def test_half_a_pair_is_not_a_pair(tmp_path):
    """A question with no answer teaches nothing about answering."""
    write(tmp_path, "pairs.jsonl", [{"question": "Where?"}, {"answer": "There."}])
    with pytest.raises(TeacherError, match="No question-and-answer pairs"):
        answers.read_pairs([str(tmp_path)])


def test_the_error_shows_the_shapes_that_would_work(tmp_path):
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(TeacherError) as excinfo:
        answers.read_pairs([str(tmp_path)])
    message = str(excinfo.value)
    assert "instruction" in message and "question" in message and "messages" in message


def test_one_unreadable_file_does_not_lose_the_others(tmp_path):
    write(tmp_path, "good.jsonl", [{"question": "A?", "answer": "1"}])
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    records, _style, skipped = answers.read_pairs([str(tmp_path)])
    assert len(records) == 1
    assert any("bad.json" in path for path, _why in skipped)


def test_a_missing_path_is_reported_not_crashed(tmp_path):
    write(tmp_path, "good.jsonl", [{"question": "A?", "answer": "1"}])
    _records, _style, skipped = answers.read_pairs([str(tmp_path), "/nowhere/at/all"])
    assert any("no such file" in why for _path, why in skipped)


# --------------------------------------------------------- asking it back
def test_a_model_taught_to_answer_is_asked_the_way_it_was_taught():
    """Ask a "### Question:" model something bare and it continues a document
    instead of replying, which looks like the training failed."""
    model = workspace.Model(name="asked", path=TMP_HOME / "asked")
    model.path.mkdir(parents=True, exist_ok=True)
    model.note(answer_style="qa")

    wrapped = answers.opener(model, "Where is Paris?")
    assert wrapped.startswith("### Question:")
    assert wrapped.rstrip().endswith("### Answer:")
    assert "Where is Paris?" in wrapped


def test_each_style_has_its_own_opener():
    for style, marker in (("qa", "### Question:"), ("instruction", "### Instruction:"),
                          ("chat", "<|user|>")):
        model = workspace.Model(name=f"s-{style}", path=TMP_HOME / f"s-{style}")
        model.path.mkdir(parents=True, exist_ok=True)
        model.note(answer_style=style)
        assert answers.opener(model, "x").startswith(marker)


def test_a_model_never_taught_to_answer_is_asked_plainly():
    model = workspace.Model(name="plain", path=TMP_HOME / "plain")
    model.path.mkdir(parents=True, exist_ok=True)
    assert answers.opener(model, "once upon a") == "once upon a"
