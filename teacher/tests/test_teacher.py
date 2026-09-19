"""Teacher: gathering material, building a model, and teaching it for real.

The training test runs actual PyTorch steps on a tiny model — if the lesson were
simulated, the loss assertion would not hold.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import lessons, workspace  # noqa: E402
from teacher.lessons import verdict  # noqa: E402
from teacher.material import gather  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


def corpus(repeats: int = 400) -> str:
    subjects = ["the cat", "the dog", "a bird", "the fox"]
    verbs = ["sat on", "jumped over", "ran past", "looked at"]
    objects = ["the mat", "the fence", "the river", "the moon"]
    return " ".join(
        f"{s} {v} {o} ." for _ in range(repeats) for s in subjects for v in verbs for o in objects
    )


@pytest.fixture(scope="module")
def material_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("material")
    (folder / "lesson.txt").write_text(corpus(), encoding="utf-8")
    (folder / "notes.md").write_text("# Notes\n\n" + corpus(20), encoding="utf-8")
    (folder / "ignored.bin").write_bytes(b"\x00\x01\x02")
    return folder


# ------------------------------------------------------------------ material
def test_gathers_a_folder(material_dir):
    found = gather([str(material_dir)])
    assert len(found.sources) == 2, "only readable file types are picked up"
    assert found.characters > 10_000
    assert "the cat" in found.text


def test_gathers_a_single_file(material_dir):
    found = gather([str(material_dir / "lesson.txt")])
    assert len(found.sources) == 1


def test_raw_text_is_accepted():
    found = gather([], raw_text="hello there")
    assert found.text == "hello there"
    assert found.sources == ["(text given on the command line)"]


def test_missing_paths_are_reported_not_fatal(material_dir):
    found = gather([str(material_dir), "/no/such/place"])
    assert found.characters > 0, "one bad path must not lose the good material"
    assert any("no such file" in why for _path, why in found.skipped)


def test_summary_counts_what_it_read(material_dir):
    summary = gather([str(material_dir)]).summary()
    assert "2 source(s)" in summary and "characters" in summary


# -------------------------------------------------------------------- naming
@pytest.mark.parametrize(
    "given,expected",
    [("My Model", "my-model"), ("  spaced  ", "spaced"), ("a/b\\c", "a-b-c"), ("UPPER", "upper")],
)
def test_names_become_safe_folder_names(given, expected):
    assert workspace.slug(given) == expected


def test_an_unusable_name_is_refused():
    with pytest.raises(TeacherError):
        workspace.slug("///")


def test_asking_for_an_unknown_model_is_refused():
    with pytest.raises(TeacherError) as excinfo:
        workspace.get("never-made")
    assert "teacher list" in str(excinfo.value)


# ------------------------------------------------------------------- lessons
def test_too_little_material_is_refused(material_dir):
    model = workspace.get("too-small", must_exist=False)
    with pytest.raises(TeacherError) as excinfo:
        lessons.create(model, gather([], raw_text="tiny"), "tiny")
    assert "too little" in str(excinfo.value)


def test_unknown_size_is_refused(material_dir):
    model = workspace.get("bad-size", must_exist=False)
    with pytest.raises(TeacherError):
        lessons.create(model, gather([str(material_dir)]), "enormous")


@pytest.fixture(scope="module")
def taught(material_dir):
    """Build a model and teach it once — the fixture the rest depends on."""
    model = workspace.get("pupil", must_exist=False)
    built = lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    lesson = lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8)
    return model, built, lesson


def test_building_writes_the_three_files(taught):
    model, built, _lesson = taught
    assert model.exists(), f"missing {model.missing()}"
    assert built["parameters"] > 0
    assert built["context"] == 64
    assert (model.path / "model.safetensors").stat().st_size > 0


def test_the_vocabulary_is_capped_by_how_much_text_there_is(taught):
    _model, built, _lesson = taught
    # 4096 is the preset's vocabulary; the corpus is far too small to support it.
    assert built["vocab_size"] < 4096


def test_teaching_actually_lowers_the_loss(taught):
    import math

    _model, built, lesson = taught
    assert lesson["status"] == "completed"
    assert lesson["steps"] > 0
    assert lesson["final_loss"] is not None

    # An untrained model sits near ln(vocab); one epoch must beat that clearly.
    untrained = math.log(built["vocab_size"])
    assert lesson["final_loss"] < untrained * 0.9, (
        f"loss {lesson['final_loss']:.3f} is no better than an untrained model ({untrained:.3f})"
    )


def test_the_held_out_loss_is_measured(taught):
    _model, _built, lesson = taught
    assert lesson["held_out_loss"] is not None
    assert lesson["perplexity"] > 1.0


def test_the_lesson_is_recorded(taught):
    model, _built, lesson = taught
    history = model.history()
    assert len(history["lessons"]) == 1
    assert history["lessons"][0]["characters"] == lesson["characters"]
    assert history["lessons"][0]["sources"]


def test_teaching_again_keeps_the_earlier_record(taught, material_dir):
    model, _built, _lesson = taught
    before = len(model.history()["lessons"])
    lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8)
    after = model.history()["lessons"]
    assert len(after) == before + 1, "a second lesson must be appended, not replace the first"
    assert after[0]["at"] < after[-1]["at"]


def test_the_previous_weights_are_kept_before_being_overwritten(taught):
    model, _built, _lesson = taught
    saved = sorted(p for p in model.checkpoints.iterdir() if p.is_dir())
    assert saved, "the state before a lesson must be recoverable"
    assert (saved[-1] / "model.safetensors").exists()


def test_talking_produces_tokens(taught):
    model, _built, _lesson = taught
    text, stats = lessons.talk(model, "the cat", max_new_tokens=16, seed=1)
    assert stats["generated"] > 0
    assert stats["tokens_per_second"] > 0
    assert isinstance(text, str)


def test_a_seed_repeats_the_same_answer(taught):
    model, _built, _lesson = taught
    first, _ = lessons.talk(model, "the cat", max_new_tokens=16, seed=42)
    second, _ = lessons.talk(model, "the cat", max_new_tokens=16, seed=42)
    assert first == second


# -------------------------------------------------------------------- sizes
def test_the_sizes_climb_and_match_their_advertised_scale():
    """tiny -> small -> medium -> large -> huge is 1M, 10M, 100M, 500M, 1B."""
    from ai_studio.models.transformer import preset_config

    order = ("tiny", "small", "medium", "large", "huge")
    counts = [preset_config(lessons.SIZES[key][0]).parameter_count()["total"] for key in order]
    assert counts == sorted(counts), "each tier must be bigger than the last"

    expected = [1e6, 1e7, 1e8, 5e8, 1e9]
    for key, actual, target in zip(order, counts, expected):
        assert 0.5 * target <= actual <= 2.0 * target, (
            f"{key} is {actual:,} parameters, not near {target:,.0f}"
        )


def test_every_size_names_a_real_preset():
    from ai_studio.models.transformer import SIZE_PRESETS

    for key, (preset, blurb) in lessons.SIZES.items():
        assert preset in SIZE_PRESETS, f"{key} points at a preset that does not exist"
        assert blurb, f"{key} has no description"


def test_describe_sizes_reads_billions_as_billions():
    listing = lessons.describe_sizes()
    assert "1.0B" in listing, listing
    assert "1019M" not in listing, "a billion-parameter model should not be shown in millions"


# ------------------------------------------------------------------ verdict
# Each pair below was measured from a real run, so the labels stay tied to what
# a model at that loss actually writes.
@pytest.mark.parametrize(
    "loss,vocab,expected,observed",
    [
        (4.87, 329, "barely started", "noise and broken characters"),
        (3.41, 329, "learning the alphabet", "word fragments"),
        (2.89, 329, "learning words", "whole words in no order"),
        (0.95, 369, "learning sentences", "full sentences"),
        (0.53, 346, "has the shape of your text", "fluent in the source style"),
    ],
)
def test_the_verdict_matches_what_the_model_writes(loss, vocab, expected, observed):
    label, note, _share = verdict(loss, vocab)
    assert label == expected, f"at loss {loss} the model writes {observed}, not '{label}'"
    assert note


def test_the_verdict_improves_as_the_loss_falls():
    shares = [verdict(loss, 329)[2] for loss in (4.9, 3.4, 2.0, 1.0, 0.3)]
    assert shares == sorted(shares, reverse=True)


def test_a_bigger_vocabulary_moves_the_baseline():
    # The same loss is further along for a small vocabulary than a large one.
    _label, _note, small = verdict(3.0, 300)
    _label, _note, large = verdict(3.0, 30000)
    assert small > large


def test_no_held_out_set_is_reported_as_unmeasured():
    label, note, _share = verdict(None, 329)
    assert label == "unmeasured"
    assert "nothing to judge" in note


def test_the_verdict_reads_the_vocabulary_off_the_model(taught):
    model, built, _lesson = taught
    assert lessons.model_vocab(model) == built["vocab_size"]


# -------------------------------------------------------------- teach until
def test_an_unknown_target_is_refused(taught, material_dir):
    model, _built, _lesson = taught
    with pytest.raises(TeacherError):
        lessons.teach_until(model, gather([str(material_dir)]), target="perfection")


def test_it_stops_once_it_reaches_the_target(material_dir):
    model = workspace.get("auto-target", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    summary = lessons.teach_until(
        model, gather([str(material_dir)]),
        target="words", epochs_per_round=2.0, max_rounds=12, batch_size=8,
    )
    assert summary["rounds"] >= 1
    assert "reached" in summary["reason"]

    _label, _note, share = lessons.verdict(summary["held_out_loss"], lessons.model_vocab(model))
    assert share < lessons.TARGETS["words"], "it stopped before actually getting there"
    # On this corpus one round can already overshoot, so only require no regression.
    assert summary["first_loss"] >= summary["held_out_loss"]
    assert summary["held_out_loss"] < math.log(lessons.model_vocab(model)) * 0.28


def test_a_model_already_past_the_target_is_left_alone(material_dir):
    model = workspace.get("auto-target", must_exist=True)
    before = len(model.history()["lessons"])
    summary = lessons.teach_until(
        model, gather([str(material_dir)]), target="words", epochs_per_round=1.0,
    )
    assert summary["rounds"] == 0
    assert "already past" in summary["reason"]
    assert len(model.history()["lessons"]) == before, "a no-op must not write a lesson"


def test_max_rounds_is_honoured(material_dir):
    model = workspace.get("auto-capped", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    summary = lessons.teach_until(
        model, gather([str(material_dir)]),
        target="best", epochs_per_round=1.0, max_rounds=2, batch_size=8,
    )
    assert summary["rounds"] <= 2


def test_an_automatic_run_records_one_lesson_not_one_per_round(material_dir):
    model = workspace.get("auto-record", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    summary = lessons.teach_until(
        model, gather([str(material_dir)]),
        target="words", epochs_per_round=2.0, max_rounds=10, batch_size=8,
    )
    entries = model.history()["lessons"]
    assert len(entries) == 1, "the whole run is one entry in the record"
    assert entries[0]["rounds"] == summary["rounds"]
    assert entries[0]["epochs"] == summary["rounds"] * 2.0
    assert len(entries[0]["lessons"]) == summary["rounds"], "each round's loss is kept"


def test_every_round_is_saved_so_stopping_early_keeps_the_work(material_dir):
    model = workspace.get("auto-saved", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    lessons.teach_until(
        model, gather([str(material_dir)]),
        target="best", epochs_per_round=1.0, max_rounds=2, batch_size=8,
    )
    assert model.exists(), "the weights must be on disk after an automatic run"
    text, stats = lessons.talk(model, "the cat", max_new_tokens=8, seed=1)
    assert stats["generated"] > 0


# --------------------------------------------------------------- pretrained
def test_a_scratch_model_is_recognised_as_built_here(taught):
    model, _built, _lesson = taught
    assert model.kind() == "studio"
    assert model.base_repo() is None


def test_the_verdict_uses_the_right_baseline_for_each_kind():
    # From scratch the baseline is ln(vocab); adopted, it is where it started.
    scratch, _n, scratch_share = lessons.verdict(3.0, 300)
    fitted, _n, fitted_share = lessons.verdict(3.0, 49152, baseline=4.0)
    assert scratch in [row[1] for row in lessons.STAGES]
    assert fitted in [row[1] for row in lessons.FITTING]
    assert scratch_share != fitted_share


def test_every_suggested_base_has_a_repo_and_a_note():
    for shortcut, (repo, note) in lessons.BASES.items():
        assert "/" in repo, f"{shortcut} should name a Hugging Face repo"
        assert note


def test_a_pretrained_model_cannot_be_packed_for_the_browser(tmp_path):
    """Bench only runs the studio architecture, so pack must refuse, not mislead."""
    model = workspace.Model(name="borrowed", path=tmp_path / "borrowed")
    model.path.mkdir()
    (model.path / "config.json").write_text('{"model_type": "gpt2"}')
    (model.path / "model.safetensors").write_bytes(b"x")
    (model.path / "tokenizer.json").write_text("{}")

    assert model.kind() == "pretrained"
    with pytest.raises(TeacherError) as excinfo:
        model.pack(tmp_path / "out")
    assert "only runs models built here" in str(excinfo.value)


def test_notes_are_kept_beside_the_lessons(taught):
    model, _built, _lesson = taught
    before = len(model.history()["lessons"])
    model.note(base_repo="somewhere/else")
    assert model.history()["base_repo"] == "somewhere/else"
    assert len(model.history()["lessons"]) == before, "a note must not disturb the lessons"


# ----------------------------------------------------------------- rollback
def test_rollback_restores_the_weights_from_before_the_last_lesson(material_dir):
    model = workspace.get("undoer", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8)

    after_first = (model.path / "model.safetensors").read_bytes()
    lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8)
    after_second = (model.path / "model.safetensors").read_bytes()
    assert after_first != after_second, "the second lesson should have changed the weights"

    model.rollback()
    assert (model.path / "model.safetensors").read_bytes() == after_first


def test_rollback_leaves_the_history_alone(material_dir):
    """The record says what happened, and the lesson did happen."""
    model = workspace.get("undoer", must_exist=True)
    before = len(model.history()["lessons"])
    model.rollback()
    assert len(model.history()["lessons"]) == before


def test_rollback_without_a_saved_state_is_refused(tmp_path):
    model = workspace.Model(name="fresh", path=tmp_path / "fresh")
    model.path.mkdir()
    with pytest.raises(TeacherError) as excinfo:
        model.rollback()
    assert "no earlier state" in str(excinfo.value)


# --------------------------------------------------------------------- pack
def test_pack_copies_exactly_what_the_web_page_needs(taught, tmp_path):
    model, _built, _lesson = taught
    copied = model.pack(tmp_path / "out")
    assert sorted(copied) == ["config.json", "model.safetensors", "tokenizer.json"]
    for name in copied:
        assert (tmp_path / "out" / name).stat().st_size > 0

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["model_type"] == "ai_studio_transformer"


def test_pack_refuses_an_incomplete_model(tmp_path):
    empty = workspace.Model(name="hollow", path=tmp_path / "hollow")
    empty.path.mkdir()
    with pytest.raises(TeacherError):
        empty.pack(tmp_path / "out")


def test_listing_finds_the_models_it_made(taught):
    names = {model.name for model in workspace.every()}
    assert "pupil" in names
