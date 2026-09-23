"""Model cards: written from the record, and honest about what the weights carry.

The lineage tests work on records alone. The rest train a tiny model for real,
because what a card must get right — which lessons are in the weights on disk —
depends on files being written, copied aside and put back.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-card-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import card, exams, lessons, workspace  # noqa: E402
from teacher.material import gather  # noqa: E402
from teacher.workspace import TeacherError, lineage, style_in_effect  # noqa: E402


def step(before: str, after: str, **extra) -> dict:
    return {"weights_before": before, "weights": after, "characters": 100, **extra}


def stamps(lessons_: list[dict]) -> list[str]:
    return [lesson["weights"] for lesson in lessons_]


# ------------------------------------------------------------------ lineage
def test_a_new_model_carries_no_lessons():
    assert lineage({"lessons": [], "origin_weights": "w0"}, "w0") == {
        "in_effect": [], "undone": [], "certain": True}


def test_lessons_that_built_on_each_other_are_all_in_effect():
    got = lineage({"origin_weights": "w0", "lessons": [step("w0", "w1"), step("w1", "w2")]}, "w2")
    assert stamps(got["in_effect"]) == ["w1", "w2"] and not got["undone"] and got["certain"]


def test_a_rolled_back_lesson_is_undone_but_still_in_the_record():
    got = lineage({"origin_weights": "w0", "lessons": [step("w0", "w1"), step("w1", "w2")]}, "w1")
    assert stamps(got["in_effect"]) == ["w1"]
    assert stamps(got["undone"]) == ["w2"]
    assert got["certain"]


def test_teaching_after_a_rollback_leaves_the_undone_lesson_out():
    history = {"origin_weights": "w0",
               "lessons": [step("w0", "w1"), step("w1", "w2"), step("w1", "w3")]}
    got = lineage(history, "w3")
    assert stamps(got["in_effect"]) == ["w1", "w3"]
    assert stamps(got["undone"]) == ["w2"]


def test_weights_the_record_cannot_explain_are_not_vouched_for():
    assert lineage({"origin_weights": "w0", "lessons": [step("w0", "w1")]}, "w9")["certain"] is False


def test_old_records_are_listed_as_taught_but_not_vouched_for():
    got = lineage({"lessons": [{"characters": 1}, {"characters": 2}]}, "w5")
    assert len(got["in_effect"]) == 2 and got["certain"] is False


def test_a_branch_carries_only_what_it_was_branched_with():
    history = {"origin_weights": "w0", "lessons": [],
               "branched_from_history": [step("w0", "w1"), step("w1", "w2")]}
    got = lineage(history, "w1")
    assert stamps(got["in_effect"]) == ["w1"] and got["in_effect"][0]["inherited"]
    assert stamps(got["undone"]) == ["w2"]


def test_a_rollback_into_the_middle_of_a_run_of_rounds_is_placed_exactly():
    """Teaching until best is one entry in the record but several saved states."""
    run = step("w0", "w3", round_weights=["w1", "w2", "w3"], rounds=3, rounds_kept=3)
    got = lineage({"origin_weights": "w0", "lessons": [run]}, "w2")
    assert got["certain"] and len(got["in_effect"]) == 1
    assert got["in_effect"][0]["rounds_in_effect"] == 2


def test_rolling_back_past_an_answer_lesson_forgets_its_template():
    history = {"origin_weights": "w0", "answer_style": "qa",
               "lessons": [step("w0", "w1", style="text"), step("w1", "w2", style="qa")]}
    assert style_in_effect(history, "w2") == "qa"
    assert style_in_effect(history, "w1") is None


# ------------------------------------------------------------ real weights
def prose(words: list[str], repeats: int = 40) -> str:
    return " ".join(f"{a} {b} {c} ." for _ in range(repeats)
                    for a in words for b in reversed(words) for c in words[:2])


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    folder = tmp_path_factory.mktemp("card-material")
    (folder / "01-lighthouse.txt").write_text(
        "Lighthouse - Wikipedia\nSource: https://en.wikipedia.org/wiki/Lighthouse\n"
        "Retrieved: 2026-09-19\n\n" + prose(["the", "lamp", "tower", "keeper", "sea"]),
        encoding="utf-8")
    (folder / "private-notes.txt").write_text(
        prose(["a", "bird", "sang", "over", "hills"]), encoding="utf-8")
    return folder


def fresh(name: str, material: Path) -> workspace.Model:
    model = workspace.get(name, must_exist=False)
    lessons.create(model, gather([str(material)]), "tiny", context=64)
    return model


def teach(model: workspace.Model, material: Path) -> dict:
    return lessons.teach(model, gather([str(material)]), epochs=1.0, batch_size=8)


@pytest.fixture(scope="module")
def taught(material):
    model = fresh("carded", material)
    teach(model, material)
    return model


def test_every_lesson_notes_the_weights_it_began_from_and_saved(taught):
    history = taught.history()
    only = history["lessons"][0]
    assert only["weights_before"] == history["origin_weights"]
    assert only["weights"] == taught.weights_stamp()
    assert only["learning_rate"] and only["batch_size"]


def test_a_card_names_web_pages_and_keeps_local_paths_private(taught, material):
    text = card.render(card.facts(taught, look_up_licence=False))
    assert "(https://en.wikipedia.org/wiki/Lighthouse)" in text
    assert "retrieved 2026-09-19" in text
    assert "`private-notes.txt` — a local file" in text
    assert str(material) not in text, "a card is for sharing; this machine's folders are not"


def test_a_card_chooses_no_licence_for_the_model(taught):
    text = card.render(card.facts(taught, look_up_licence=False))
    assert "\nlicense:" not in text, "no licence field is filled in on anyone's behalf"
    assert "No licence has been chosen for this model" in text
    assert "Apache 2.0" in text and "does not cover these weights" in text


def test_a_rolled_back_lesson_leaves_the_card_but_not_the_record(material):
    model = fresh("rolled", material)
    teach(model, material)
    teach(model, material)
    assert len(card.facts(model, look_up_licence=False)["lessons"]) == 2

    model.rollback()
    known = card.facts(model, look_up_licence=False)
    assert len(known["lessons"]) == 1 and len(known["undone"]) == 1 and known["certain"]
    assert len(model.history()["lessons"]) == 2, "the rollback happened; so did the lesson"
    assert model.history()["rollbacks"][-1]["restored"].startswith("before-")
    assert "not in these weights" in card.render(known)


def test_a_score_is_shown_only_while_its_weights_are(material):
    model = fresh("scored", material)
    teach(model, material)
    exams.keep(model, {
        "suite": "arc", "label": "ARC", "method": "multiple_choice",
        "items": 20, "shots": 0, "correct": 5, "accuracy": 0.25, "chance": 0.25,
        "beats_chance": False, "source": "official", "official": True, "note": "",
        "seconds": 1.0, "results": [{"question": "q"}],
    })
    known = card.facts(model, look_up_licence=False)
    assert [exam["correct"] for exam in known["exams"]] == [5]
    assert "results" not in model.history()["exams"][0], "the score is kept, not every answer"
    assert "Indistinguishable from guessing" in card.render(known)

    teach(model, material)
    known = card.facts(model, look_up_licence=False)
    assert known["exams"] == [] and len(known["older_exams"]) == 1


def test_someone_elses_readme_is_left_alone(taught):
    readme = taught.path / "README.md"
    readme.write_text("my own notes", encoding="utf-8")
    with pytest.raises(TeacherError, match="was not written by Teacher"):
        card.write(taught, look_up_licence=False)
    assert readme.read_text(encoding="utf-8") == "my own notes"

    readme.unlink()
    written = card.write(taught, look_up_licence=False)
    assert card.written_by_teacher(written)
    card.write(taught, look_up_licence=False)  # its own card it may rewrite


def test_an_export_gets_its_card_beside_it(taught, tmp_path):
    exported = tmp_path / "carded-f16.gguf"
    exported.write_bytes(b"GGUF" + b"\0" * 64)
    entry, written, why_not = card.beside_export(taught, {"path": str(exported), "precision": "f16"})
    assert why_not is None and written == tmp_path / "carded-f16.md"
    assert entry["sha256"] in written.read_text(encoding="utf-8")
    assert taught.history()["exports"][-1]["file"] == "carded-f16.gguf"

    other = tmp_path / "theirs.gguf"
    other.write_bytes(b"GGUF")
    (tmp_path / "theirs.md").write_text("not Teacher's", encoding="utf-8")
    _entry, written, why_not = card.beside_export(taught, {"path": str(other), "precision": "f16"})
    assert written is None and "left alone" in why_not
    assert (tmp_path / "theirs.md").read_text(encoding="utf-8") == "not Teacher's"


def test_making_a_model_over_another_keeps_the_old_record_apart(material):
    model = fresh("remade", material)
    teach(model, material)
    lessons.create(model, gather([str(material)]), "tiny", context=64)
    history = model.history()
    assert history["lessons"] == [], "the new weights were never taught those lessons"
    assert history["replaced"][0]["lessons"], "and the old record is kept, not dropped"


def test_the_card_command_reports_what_it_wrote(taught, capsys):
    from teacher.__main__ import main

    (taught.path / "README.md").unlink(missing_ok=True)
    assert main(["--json", "card", taught.name, "--offline"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "card" and payload["lessons"] == 1
    assert Path(payload["path"]).name == "README.md"
