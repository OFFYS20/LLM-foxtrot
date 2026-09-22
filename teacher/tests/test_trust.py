"""The three things that decide whether the numbers can be believed.

Repeated material inflates a score by teaching the model to recite. A held-out
set taken from the same text flatters a model that memorised it. And a run that
loses three hours to a crash is a number you never get to see at all.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-trust-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import interrupted, workspace  # noqa: E402
from teacher.material import MIN_REPEAT, drop_repeats, gather  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402

LONG = "A passage long enough to be worth checking for repetition, said twice. " * 4


# ------------------------------------------------------------- duplicates
def test_a_passage_said_twice_is_learned_once():
    text, dropped, characters = drop_repeats(LONG + "\n\n" + LONG)
    assert dropped == 1
    assert characters > MIN_REPEAT
    assert text.count("worth checking") == 4, "one copy, with its four sentences"


def test_short_lines_repeat_for_honest_reasons():
    """A heading, a name, a refrain. Dropping those edits the text."""
    text, dropped, _ = drop_repeats("Chapter One\n\nChapter One\n\nChapter One")
    assert dropped == 0
    assert text.count("Chapter One") == 3


def test_different_passages_are_both_kept():
    other = LONG.replace("passage", "section")
    text, dropped, _ = drop_repeats(LONG + "\n\n" + other)
    assert dropped == 0
    assert "passage" in text and "section" in text


def test_the_same_book_in_two_folders_is_read_once(tmp_path):
    for name in ("copy-one", "copy-two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "book.txt").write_text(LONG, encoding="utf-8")

    material = gather([str(tmp_path)])
    assert material.repeats == 1
    assert "repeat(s) dropped" in material.summary()


def test_duplicates_can_be_kept_on_purpose(tmp_path):
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "book.txt").write_text(LONG, encoding="utf-8")

    assert gather([str(tmp_path)], dedupe=False).repeats == 0
    assert gather([str(tmp_path)], dedupe=True).repeats == 1


def test_within_one_file_the_cleaner_already_handles_it(tmp_path):
    """Worth knowing where the line is: clean_text drops repeated paragraphs
    inside a document, so what this adds is catching them *across* sources."""
    (tmp_path / "a.txt").write_text(LONG + "\n\n" + LONG, encoding="utf-8")
    material = gather([str(tmp_path)], dedupe=False)
    assert material.text.count("worth checking") == 4, "the cleaner already deduped it"


# ------------------------------------------------------- an unfinished run
def model_at(name: str) -> workspace.Model:
    model = workspace.Model(name=name, path=TMP_HOME / name)
    model.path.mkdir(parents=True, exist_ok=True)
    return model


def test_a_model_with_nothing_unfinished_says_so():
    assert interrupted.waiting(model_at("clean")) is None


def test_a_record_is_written_whole_or_not_at_all():
    """A crash during the save must not leave half a record to resume from."""
    model = model_at("saving")

    class Weights:
        def save_pretrained(self, path):
            (Path(path) / "model.safetensors").write_bytes(b"w")

    class Broken:
        def state_dict(self):
            raise RuntimeError("the optimizer is in a bad way")

    interrupted.write(model, Weights(), Broken(), None, step=5, plan={})
    assert interrupted.waiting(model) is None, "a failed save leaves nothing behind"
    assert not interrupted.place(model).exists()
    assert not interrupted.place(model).with_name(interrupted.FOLDER + ".writing").exists()


def test_a_saved_run_records_where_it_got_to():
    import torch

    model = model_at("midway")

    class Weights:
        def save_pretrained(self, path):
            (Path(path) / "model.safetensors").write_bytes(b"w")

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    interrupted.write(model, Weights(), optimizer, None, step=312,
                      plan={"epochs": 40.0, "total_steps": 1040, "characters": 207173})

    plan = interrupted.waiting(model)
    assert plan["step"] == 312 and plan["total_steps"] == 1040
    assert "step 312/1040" in interrupted.describe(plan)


def test_finishing_a_lesson_leaves_nothing_to_resume():
    import torch

    model = model_at("finished")

    class Weights:
        def save_pretrained(self, path):
            (Path(path) / "model.safetensors").write_bytes(b"w")

    interrupted.write(model, Weights(),
                      torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1),
                      None, step=1, plan={})
    assert interrupted.waiting(model) is not None
    interrupted.clear(model)
    assert interrupted.waiting(model) is None, "a record that is there means a lesson that is not"


def test_resuming_nothing_is_refused():
    with pytest.raises(TeacherError, match="no interrupted lesson"):
        interrupted.restore(model_at("empty"), None)


def test_material_that_changed_is_a_different_run():
    """Resuming onto different text would train on something the run never saw."""
    first = interrupted.fingerprint("the original material")
    assert interrupted.fingerprint("the original material") == first
    assert interrupted.fingerprint("the original material.") != first


def test_the_last_step_does_not_write_a_record_that_is_about_to_be_deleted(monkeypatch):
    """Saving weights and optimizer state at the final step, moments before the
    finished lesson clears them, is gigabytes written for nothing."""
    import inspect

    from teacher import lessons

    source = inspect.getsource(lessons.teach)
    assert 'step >= held["total"]' in source, "the final-step save must be skipped"
