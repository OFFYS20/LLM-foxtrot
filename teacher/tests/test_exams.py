"""Benchmarks: what the number means, and what it must never be allowed to imply.

The arithmetic and the labelling are tested here without running a model. What
a given model scores is not a property of this code; what *is* a property of
this code is refusing to dress six bundled example items up as a benchmark
result, and printing the chance rate next to every score.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-exam-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from ai_studio.evaluation.suites import BenchmarkItem  # noqa: E402
from teacher import exams  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


def choices(count: int, how_many: int = 4) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(question="q", expected="A", choices=[f"c{n}" for n in range(how_many)])
        for _ in range(count)
    ]


# ------------------------------------------------------------- chance
def test_four_choices_is_a_quarter():
    assert exams.chance_rate(choices(10)) == pytest.approx(0.25)


def test_ten_choices_is_a_tenth():
    assert exams.chance_rate(choices(10, how_many=10)) == pytest.approx(0.1)


def test_a_mixed_suite_averages_its_chances():
    items = choices(1, 4) + choices(1, 2)
    assert exams.chance_rate(items) == pytest.approx((0.25 + 0.5) / 2)


def test_a_question_with_no_choices_has_no_chance_rate():
    """There is no number to guess your way to on an open answer."""
    assert exams.chance_rate([BenchmarkItem(question="q", expected="42")]) is None


def test_one_open_question_disqualifies_the_whole_rate():
    assert exams.chance_rate(choices(3) + [BenchmarkItem(question="q", expected="x")]) is None


# ------------------------------------------------- is it more than guessing
def test_scoring_at_chance_is_not_a_result():
    assert exams.beats_chance(correct=5, total=20, chance=0.25) is False


def test_a_little_above_chance_is_still_not_a_result():
    """Seven of twenty against a quarter is one standard deviation. Noise."""
    assert exams.beats_chance(correct=7, total=20, chance=0.25) is False


def test_a_lot_above_chance_is_a_result():
    assert exams.beats_chance(correct=15, total=20, chance=0.25) is True


def test_more_items_make_a_smaller_lead_meaningful():
    """The same 40% is noise over 20 items and a result over 400."""
    assert exams.beats_chance(correct=8, total=20, chance=0.25) is False
    assert exams.beats_chance(correct=160, total=400, chance=0.25) is True


def test_with_no_chance_rate_there_is_nothing_to_beat():
    assert exams.beats_chance(correct=5, total=10, chance=None) is None


# --------------------------------------------------------- what it says
def test_a_sample_run_is_never_called_a_benchmark_score():
    verdict = exams.verdict({"official": False, "beats_chance": True, "source": "sample"})
    assert "plumbing" in verdict
    assert "benchmark score" not in verdict.lower().replace("not a benchmark score", "")


def test_scoring_at_chance_is_explained_not_just_reported():
    verdict = exams.verdict({"official": True, "beats_chance": False, "source": "official"})
    assert "guessing" in verdict
    assert "not a fault" in verdict, "a small model at chance is expected, not broken"


def test_a_real_result_is_allowed_to_be_a_real_result():
    verdict = exams.verdict({"official": True, "beats_chance": True, "source": "official"})
    assert "Better than guessing" in verdict


# ------------------------------------------------------------- the catalogue
def test_every_suite_says_where_its_items_would_come_from():
    for row in exams.catalogue():
        assert row["suite"] and row["label"] and row["method"]
        assert row["source"] in ("official", "local", "sample")
        assert row["available"] >= 0


def test_the_catalogue_covers_every_suite_ai_studio_has():
    from ai_studio.evaluation.suites import SUITES

    assert {row["suite"] for row in exams.catalogue()} == set(SUITES)


def test_an_unknown_suite_is_refused_by_name():
    from teacher import workspace

    model = workspace.Model(name="nobody", path=TMP_HOME / "nobody")
    with pytest.raises(TeacherError, match="Unknown benchmark suite"):
        exams.sit(model, "phrenology", limit=1)


def test_asking_for_more_examples_than_items_says_so(monkeypatch):
    from ai_studio.evaluation import suites as suite_module

    from teacher import workspace

    info = suite_module.SUITES["arc"]
    monkeypatch.setattr(
        exams, "load_suite",
        lambda key, **kw: (choices(2), suite_module.SuiteInfo(**{**info.__dict__})))

    model = workspace.Model(name="nobody", path=TMP_HOME / "nobody")
    with pytest.raises(TeacherError, match="no questions left"):
        exams.sit(model, "arc", limit=5, few_shot=2)


def test_generation_room_is_sized_to_the_kind_of_answer():
    """A letter needs a few tokens; a function needs room. 256 for a
    multiple-choice question is most of the runtime for none of the benefit."""
    assert exams.ROOM["multiple_choice"] < exams.ROOM["numeric"] < exams.ROOM["contains"]
