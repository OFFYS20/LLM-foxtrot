"""Learning rates, and whether the number a lesson reports is the model on disk.

A lesson used to report the best held-out loss it saw part-way through, while
saving the weights from the end. When a lesson overshoots, those are different
models — and the number described the one that was thrown away.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-rate-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import lessons, workspace  # noqa: E402
from teacher.material import gather  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402

TEXT = " ".join(f"the {a} {b} the {c} ." for a in ("keeper", "lamp", "ship", "gull")
                for b in ("saw", "lit", "passed", "crossed") for c in ("rock", "tower", "sea")) * 40


@pytest.fixture(scope="module")
def made():
    model = workspace.get("rated", must_exist=False)
    lessons.create(model, gather([], raw_text=TEXT), "tiny", context=64)
    return model


def test_each_kind_of_lesson_has_its_own_starting_rate(made, monkeypatch):
    assert lessons.default_rate(made) == lessons.RATES["scratch"]
    assert lessons.default_rate(made, lora=True) == lessons.RATES["lora"]
    monkeypatch.setattr(type(made), "kind", lambda self: "pretrained")
    assert lessons.default_rate(made) == lessons.RATES["pretrained"]


def test_a_lesson_says_which_rate_it_used_and_why(made):
    lesson = lessons.teach(made, gather([], raw_text=TEXT), epochs=1.0, batch_size=8)
    assert lesson["learning_rate"] == lessons.RATES["scratch"]
    assert lesson["rate_from"] == "the default for this kind of lesson"

    given = lessons.teach(made, gather([], raw_text=TEXT), epochs=1.0, batch_size=8,
                          learning_rate=1e-3)
    assert given["learning_rate"] == 1e-3 and given["rate_from"] == "given"


def test_the_held_out_number_is_the_model_that_was_saved(made):
    """Measured again from disk, the saved weights score what the lesson said."""
    import torch

    from ai_studio.training.config import TrainingConfig
    from ai_studio.training.data import PackedLMDataset
    from ai_studio.training.trainer import Trainer

    lesson = lessons.teach(made, gather([], raw_text=TEXT), epochs=1.0, batch_size=8)
    assert lesson["held_out_before"] is not None

    tokenizer = lessons.load_tokenizer(made)
    ids = tokenizer(gather([], raw_text=TEXT).text, add_special_tokens=False)["input_ids"]
    block = 64
    tail = ids[-max(block, int(len(ids) * 0.05)):]
    saved = lessons.load_network(made)
    again = Trainer(model=saved, train_dataset=PackedLMDataset(tail, block),
                    eval_dataset=PackedLMDataset(tail, block),
                    config=TrainingConfig(batch_size=8, max_sequence_length=block),
                    device=torch.device("cpu")).evaluate()
    assert again == pytest.approx(lesson["held_out_loss"], abs=1e-4)


def test_a_rate_that_is_not_one_is_refused(made):
    with pytest.raises(TeacherError, match="makes no sense"):
        lessons.teach(made, gather([], raw_text=TEXT), epochs=1.0, learning_rate=3.0)


def test_teaching_until_best_keeps_the_best_round_not_the_last(made, monkeypatch):
    """A round that makes it worse is undone, and the record says so."""
    losses = iter([2.0, 1.8, 1.9])
    rolled_back = []

    def fake_lesson(model, material, **options):
        return {"held_out_loss": next(losses), "held_out_before": 2.2, "final_loss": 1.0,
                "steps": 10, "tokens": 100, "epochs": options["epochs"],
                "perplexity": 7.0, "device": "cpu", "status": "completed", "style": "text"}

    monkeypatch.setattr(lessons, "teach", fake_lesson)
    monkeypatch.setattr(made, "rollback", lambda to=None: rolled_back.append(to) or "before-x")
    summary = lessons.teach_until(made, gather([], raw_text=TEXT), target="best",
                                  epochs_per_round=1.0, max_rounds=5)
    assert rolled_back == [None], "the round that made it worse was undone"
    assert summary["rounds"] == 3 and summary["rounds_kept"] == 2
    assert summary["held_out_loss"] == 1.8, "the number reported is the round that was kept"
    assert "made it worse" in summary["reason"]


def test_a_first_round_that_makes_it_worse_keeps_nothing(made, monkeypatch):
    rolled_back = []

    def fake_lesson(model, material, **options):
        return {"held_out_loss": 2.5, "held_out_before": 2.2, "final_loss": 1.0,
                "steps": 10, "tokens": 100, "epochs": 1.0, "perplexity": 12.0,
                "device": "cpu", "status": "completed", "style": "text"}

    monkeypatch.setattr(lessons, "teach", fake_lesson)
    monkeypatch.setattr(made, "rollback", lambda to=None: rolled_back.append(to) or "before-x")
    summary = lessons.teach_until(made, gather([], raw_text=TEXT), target="best",
                                  epochs_per_round=1.0, max_rounds=5)
    assert rolled_back == [None]
    assert summary["rounds_kept"] == 0 and summary["held_out_loss"] == 2.2


def test_a_bigger_model_from_scratch_takes_smaller_steps(made, monkeypatch):
    """Measured at 1M and 10M; above that the rate falls with size, to a floor."""
    history = made.history()
    measured = lessons.SCRATCH_MEASURED_UP_TO
    for size, expected in [(1_000_000, lessons.RATES["scratch"]),
                           (measured, lessons.RATES["scratch"]),
                           (measured * 2, lessons.RATES["scratch"] / 2),
                           (measured * 1000, lessons.SCRATCH_FLOOR)]:
        monkeypatch.setattr(type(made), "history", lambda self, size=size: {**history,
                                                                            "parameters": size})
        assert lessons.default_rate(made) == pytest.approx(expected)
