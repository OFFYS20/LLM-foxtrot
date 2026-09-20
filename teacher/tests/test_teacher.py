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
def test_a_shortcut_is_exactly_the_number_it_is_named_after():
    """A name that quietly meant "near enough" would be the one case where the
    number you wrote is not the number you get."""
    for key, (target, _blurb) in lessons.SIZES.items():
        written = float(key[:-1]) * (1e9 if key.endswith("b") else 1e6)
        assert target == written, f"'{key}' asks for {target:,}, not {written:,.0f}"


def test_the_shortcuts_climb():
    targets = [target for target, _ in lessons.SIZES.values()]
    assert targets == sorted(targets), "each shortcut must be bigger than the last"


def test_a_shortcut_goes_through_the_same_search_as_a_typed_number(material_dir):
    """--size 1m and --size 1000000 must build the same model."""
    named = workspace.get("by-name", must_exist=False)
    typed = workspace.get("by-number", must_exist=False)
    first = lessons.create(named, gather([str(material_dir)]), "1m", context=128)
    second = lessons.create(typed, gather([str(material_dir)]), "1000000", context=128)

    assert first["parameters"] == second["parameters"]
    assert first["hidden_size"] == second["hidden_size"]
    assert first["layers"] == second["layers"]


def test_the_older_size_names_still_work():
    """Someone's script or notes should not break because the ladder was renamed."""
    for old_name, current in lessons.SIZE_ALIASES.items():
        assert lessons.resolve_size(old_name) == lessons.SIZES[current][0]
    assert lessons.resolve_size("200M") == 200_000_000, "case should not matter"


def test_an_unknown_size_names_what_is_on_offer():
    with pytest.raises(TeacherError) as excinfo:
        lessons.resolve_size("enormous")
    assert "200m" in str(excinfo.value) and "tiny" in str(excinfo.value)


def test_every_shortcut_says_what_it_needs():
    for key, (target, blurb) in lessons.SIZES.items():
        assert target > 0, f"{key} asks for nothing"
        assert blurb, f"{key} has no description"


def test_describe_sizes_reads_billions_as_billions():
    listing = lessons.describe_sizes()
    assert "1B" in listing, listing
    assert "1000M" not in listing, "a billion-parameter model should not be shown in millions"


def test_describe_sizes_says_a_number_can_be_typed():
    """The shortcuts are a convenience, not the menu."""
    assert "any count you name" in lessons.describe_sizes()


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


# ------------------------------------------------------- saved states
def test_every_lesson_leaves_a_state_behind(material_dir):
    model = workspace.get("stateful", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    for _ in range(3):
        lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8,
                      keep_checkpoints=5)

    states = model.saved_states()
    assert len(states) == 3
    assert [state["stamp"] for state in states] == sorted(state["stamp"] for state in states), \
        "oldest first, so the newest is what rollback takes"
    for state in states:
        assert state["bytes"] > 0 and state["at"] > 0


def test_only_as_many_states_as_asked_for_are_kept(material_dir):
    """Each state is a full copy of the weights, so they cannot all be kept."""
    model = workspace.get("thrifty", must_exist=False)
    lessons.create(model, gather([str(material_dir)]), "tiny", context=64)
    for _ in range(4):
        lessons.teach(model, gather([str(material_dir)]), epochs=1.0, batch_size=8,
                      keep_checkpoints=2)
    assert len(model.saved_states()) == 2


def test_a_state_can_be_found_by_its_stamp(material_dir):
    model = workspace.get("stateful", must_exist=True)
    stamp = model.saved_states()[0]["stamp"]
    assert model.saved_state(stamp).name == f"before-{stamp}"
    assert model.saved_state(f"before-{stamp}").name == f"before-{stamp}"


def test_an_unknown_stamp_lists_the_ones_that_exist(material_dir):
    model = workspace.get("stateful", must_exist=True)
    with pytest.raises(TeacherError) as excinfo:
        model.saved_state("19990101-000000")
    assert model.saved_states()[0]["stamp"] in str(excinfo.value)


def test_rolling_back_to_a_named_state_goes_further_than_one_lesson(material_dir):
    model = workspace.get("stateful", must_exist=True)
    states = model.saved_states()
    oldest = states[0]["stamp"]
    expected = (model.saved_state(oldest) / "model.safetensors").read_bytes()

    assert (model.path / "model.safetensors").read_bytes() != expected
    model.rollback(to=oldest)
    assert (model.path / "model.safetensors").read_bytes() == expected


# ---------------------------------------------------------------- branching
def test_branching_copies_the_weights_as_they_are(material_dir):
    source = workspace.get("stateful", must_exist=True)
    copy = workspace.branch(source, "stateful-now")
    assert copy.exists()
    assert (copy.path / "model.safetensors").read_bytes() == \
        (source.path / "model.safetensors").read_bytes()


def test_branching_at_a_state_takes_that_state_not_the_current_weights(material_dir):
    source = workspace.get("stateful", must_exist=True)
    stamp = source.saved_states()[1]["stamp"]
    copy = workspace.branch(source, "stateful-then", at=stamp)
    assert (copy.path / "model.safetensors").read_bytes() == \
        (source.saved_state(stamp) / "model.safetensors").read_bytes()


def test_training_a_branch_leaves_the_original_exactly_as_it_was(material_dir):
    """The whole point: experiment on the copy without risking the original."""
    source = workspace.get("stateful", must_exist=True)
    before = (source.path / "model.safetensors").read_bytes()
    before_lessons = len(source.history()["lessons"])

    copy = workspace.branch(source, "stateful-risky")
    lessons.teach(copy, gather([str(material_dir)]), epochs=1.0, batch_size=8)

    assert (source.path / "model.safetensors").read_bytes() == before
    assert len(source.history()["lessons"]) == before_lessons
    assert (copy.path / "model.safetensors").read_bytes() != before


def test_a_branch_carries_its_provenance_and_keeps_the_old_record(material_dir):
    source = workspace.get("stateful", must_exist=True)
    copy = workspace.branch(source, "stateful-traced", at=source.saved_states()[0]["stamp"])
    history = copy.history()

    assert history["branched_from"] == source.name
    assert history["branched_at"].startswith("before-")
    assert history["lessons"] == [], "its weights have not had those lessons in this form"
    assert history["branched_from_history"] == source.history()["lessons"], \
        "the original's record is kept, not discarded"


def test_a_branch_does_not_carry_the_originals_saved_states(material_dir):
    """They belong to the original, and copying them would double the disk cost."""
    source = workspace.get("stateful", must_exist=True)
    copy = workspace.branch(source, "stateful-clean")
    assert copy.saved_states() == []


def test_branching_onto_an_existing_name_is_refused(material_dir):
    source = workspace.get("stateful", must_exist=True)
    with pytest.raises(TeacherError, match="already exists"):
        workspace.branch(source, "stateful-now")


def test_branching_a_pretrained_model_takes_the_files_it_needs_to_run(tmp_path):
    """A pretrained model carries more than the three files a scratch one does."""
    source = workspace.get("adopted-ish", must_exist=False)
    source.path.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json",
                 "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        (source.path / name).write_text("{}" if name.endswith(".json") else "weights")
    source.note(base_repo="somewhere/small", base_loss=2.5)

    copy = workspace.branch(source, "adopted-copy")
    assert sorted(f.name for f in copy.path.iterdir() if f.is_file()) == sorted(
        ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
         "generation_config.json", "special_tokens_map.json", "history.json"])
    assert copy.history()["base_loss"] == 2.5, "its stage scale must come with it"


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


# ------------------------------------------------------- row-oriented files
def records_file(folder, count: int, name: str = "rows.jsonl"):
    path = folder / name
    path.write_text(
        "\n".join(
            '{"first_name": "Name%d", "city": "City%d", "note": "a line of prose %d"}' % (n, n, n)
            for n in range(count)
        ),
        encoding="utf-8",
    )
    return path


def test_every_record_of_a_jsonl_file_is_learned_from(tmp_path):
    """Training on a preview of the data would be the worst kind of bug: the
    lesson still looks like it worked."""
    from ai_studio.data.loaders import MAX_PREVIEW_ROWS

    records_file(tmp_path, MAX_PREVIEW_ROWS + 500)
    material = gather([str(tmp_path)])
    assert material.text.count("first_name:") == MAX_PREVIEW_ROWS + 500
    assert material.partial == [], "nothing was left out, so nothing to report"


def test_a_csv_is_read_in_full_too(tmp_path):
    from ai_studio.data.loaders import MAX_PREVIEW_ROWS

    rows = MAX_PREVIEW_ROWS + 300
    (tmp_path / "rows.csv").write_text(
        "name,city\n" + "\n".join(f"Name{n},City{n}" for n in range(rows)), encoding="utf-8")
    assert gather([str(tmp_path)]).text.count("name:") == rows


def test_a_previewed_file_says_how_much_was_left_out(tmp_path):
    """The Data Library still previews; what it must not do is stay quiet."""
    from ai_studio.data.loaders import MAX_PREVIEW_ROWS, load_document

    path = records_file(tmp_path, MAX_PREVIEW_ROWS + 42)
    document = load_document(path)
    assert document.meta["records"] == MAX_PREVIEW_ROWS + 42
    assert document.meta["used_records"] == MAX_PREVIEW_ROWS
    assert document.meta["truncated"] == 42


def test_material_reports_a_file_it_could_only_read_in_part(tmp_path, monkeypatch):
    import teacher.material as material_module
    from ai_studio.data.loaders import load_document

    records_file(tmp_path, 50)
    # Force a ceiling, as the Data Library's default would.
    monkeypatch.setattr(material_module, "load_document",
                        lambda path, **kw: load_document(path, max_rows=10))

    material = gather([str(tmp_path)])
    assert material.partial and material.partial[0][1] == 40
    assert "read only in part" in material.summary()


# ---------------------------------------------------------------- comparing
def test_comparing_runs_the_same_prompt_through_each_model(taught):
    model, _built, _lesson = taught
    other = workspace.branch(model, "rival")
    result = lessons.compare([model, other], "the cat", max_new_tokens=8, seed=3)

    assert [entry["model"] for entry in result["models"]] == [model.name, "rival"]
    assert all(isinstance(entry["reply"], str) for entry in result["models"])
    assert result["prompt"] == "the cat"
    assert result["seed"] == 3


def test_two_copies_of_one_model_answer_identically_at_the_same_seed(taught):
    """If the dice differed, a comparison would say nothing about the models."""
    model, _built, _lesson = taught
    twin = workspace.branch(model, "twin")
    result = lessons.compare([model, twin], "the cat", max_new_tokens=12, seed=7)
    first, second = result["models"]
    assert first["reply"] == second["reply"]


def test_comparing_says_when_the_losses_line_up(taught):
    model, _built, _lesson = taught
    other = workspace.branch(model, "samevocab")
    result = lessons.compare([model, other], "the cat", max_new_tokens=4, seed=1)
    assert result["same_vocabulary"] is True


def test_losses_from_different_vocabularies_are_not_put_side_by_side():
    """A loss is an average over the tokens a model has. Different tokens,
    different scale, no comparison."""
    same, comparable = lessons.comparability([
        {"vocab_size": 4096, "held_out_loss": 2.1},
        {"vocab_size": 49152, "held_out_loss": 2.6},
    ])
    assert same is False and comparable is False


def test_losses_line_up_when_the_vocabulary_matches():
    same, comparable = lessons.comparability([
        {"vocab_size": 4096, "held_out_loss": 2.1},
        {"vocab_size": 4096, "held_out_loss": 1.8},
    ])
    assert same is True and comparable is True


def test_a_model_never_measured_cannot_be_compared_by_number():
    same, comparable = lessons.comparability([
        {"vocab_size": 4096, "held_out_loss": 2.1},
        {"vocab_size": 4096, "held_out_loss": None},
    ])
    assert same is True, "they do share a vocabulary"
    assert comparable is False, "but one of them has no number"


def test_comparing_needs_at_least_two(taught):
    model, _built, _lesson = taught
    with pytest.raises(TeacherError, match="at least two"):
        lessons.compare([model], "the cat")


# -------------------------------------------------------------------- LoRA
def test_lora_is_refused_on_a_model_built_from_noise(taught, material_dir):
    """There is nothing to adapt yet; the whole model is what needs training."""
    model, _built, _lesson = taught
    with pytest.raises(TeacherError) as excinfo:
        lessons.teach(model, gather([str(material_dir)]), epochs=0.1, lora=True)
    assert "built here" in str(excinfo.value)
    assert "--lora" in str(excinfo.value)


# ------------------------------------------------- a size of your own choosing
def test_a_shortcut_resolves_to_the_number_it_stands_for():
    """Every size becomes a number, so every size takes the same path."""
    assert lessons.resolve_size("200m") == 200_000_000
    assert lessons.resolve_size("tiny") == 1_000_000


def test_a_number_resolves_to_that_many_parameters():
    assert lessons.resolve_size("50M") == 50_000_000
    assert lessons.resolve_size("70k") == 70_000
    assert lessons.resolve_size("1.5B") == 1_500_000_000
    assert lessons.resolve_size("250000") == 250_000


def test_a_size_that_is_neither_lists_both_kinds():
    with pytest.raises(TeacherError) as excinfo:
        lessons.resolve_size("enormous")
    message = str(excinfo.value)
    assert "70K" in message, "it should show the shape of a number"
    assert "200m" in message and "tiny" in message, "and the ready-made rungs"


def test_building_an_asked_for_size_lands_near_it(material_dir):
    model = workspace.get("five-hundred-k", must_exist=False)
    built = lessons.create(model, gather([str(material_dir)]), "500K", context=128)

    assert built["asked_for"] == 500_000
    assert abs(built["parameters"] / 500_000 - 1) < 0.1, built
    assert model.exists(), "it has to be a real model on disk, not a number"


def test_what_was_built_is_reported_not_what_was_asked_for(material_dir):
    """Reporting the number asked for would be a small lie that compounds."""
    model = workspace.get("honest-count", must_exist=False)
    built = lessons.create(model, gather([str(material_dir)]), "333K", context=128)

    from ai_studio.models.transformer import TransformerConfig

    arch = model.architecture()
    known = {k: v for k, v in arch.items() if k in TransformerConfig.__dataclass_fields__}
    real = TransformerConfig(**known).parameter_count()["total"]
    assert built["parameters"] == real, "the reported count must be the built count"


def test_a_size_this_machine_cannot_hold_is_refused_before_it_is_built(material_dir):
    model = workspace.get("impossible", must_exist=False)
    with pytest.raises(TeacherError) as excinfo:
        lessons.create(model, gather([str(material_dir)]), "100T")

    message = str(excinfo.value)
    assert "cannot be built here" in message
    assert "GB at four bytes each" in message, "say the arithmetic, not just no"
    assert "largest that would fit" in message, "and say what would work"
    assert not model.exists(), "nothing should have been written"
