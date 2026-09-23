"""The window: the pieces that shape what a person sees, without launching it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-ui-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import lessons, workspace  # noqa: E402
from teacher import ui  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


def test_the_window_builds():
    """Gradio validates every component and wiring as it constructs the page."""
    assert ui.build() is not None


def test_an_empty_workspace_says_so():
    assert "No model selected" in ui.describe("")


def test_a_missing_model_is_explained_not_crashed():
    assert "no model called" in ui.describe("never-made").lower()


def test_errors_a_person_can_fix_are_shown_plainly():
    friendly = ui.friendly(TeacherError("Give the model a name."))
    assert "Stopped" in friendly and "Give the model a name." in friendly
    assert "Traceback" not in friendly


def test_unexpected_errors_still_name_the_type():
    friendly = ui.friendly(ValueError("something odd"))
    assert "ValueError" in friendly


@pytest.mark.parametrize("value,expected", [(950, "950"), (12_500, "12.5K"), (2_400_000, "2.4M"),
                                            (1_200_000_000, "1.20B"), (None, "—")])
def test_counts_are_readable(value, expected):
    assert ui.fmt(value) == expected


def test_the_card_never_runs_paragraphs_together(tmp_path):
    """Markdown needs blank lines, or the layers line joins the vocabulary line."""
    model = workspace.get("carded", must_exist=False)
    model.path.mkdir(parents=True, exist_ok=True)
    (model.path / "config.json").write_text(
        '{"model_type": "ai_studio_transformer", "vocab_size": 315, "num_layers": 4,'
        ' "hidden_size": 128, "num_heads": 4, "max_position_embeddings": 256}'
    )
    # A card is only drawn for a complete model, so give it the other two files.
    (model.path / "model.safetensors").write_bytes(b"stub")
    (model.path / "tokenizer.json").write_text("{}")

    card = ui.describe("carded")
    assert "vocabulary 315\n\n4 layers" in card, card


def test_the_picker_offers_every_model_that_exists(tmp_path):
    names = ui.model_names()
    assert "carded" in names


def test_refresh_returns_an_update_for_each_picker_and_the_card():
    *updates, card = ui.refresh_everything(None)
    assert len(updates) == 3, "the three picker outputs must each get an update"
    assert isinstance(card, str)


def test_material_from_a_folder_and_pasted_text_combine(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta gamma " * 50, encoding="utf-8")
    material = ui.collect(None, "pasted words here", str(tmp_path))
    assert material.characters > 0
    assert "pasted words here" in material.text
    assert "alpha beta" in material.text


# ------------------------------------------------------- gathering from the web
def test_the_web_panel_says_what_it_needs_when_given_nothing():
    message, _ = next(ui.do_web("", 5, ""))
    assert "search for" in message


def test_a_finished_harvest_points_the_folder_box_at_what_it_kept(monkeypatch, tmp_path):
    from ai_studio.data.web import Page
    from teacher import websearch

    harvest = websearch.Harvest(query="lighthouses", pages=[
        Page(url="https://a.test/1", title="A page", text="word " * 300),
    ])
    monkeypatch.setattr(websearch, "harvest", lambda *a, **kw: harvest)
    monkeypatch.setattr(websearch, "folder_for", lambda query, root: tmp_path / "kept")

    message, folder = list(ui.do_web("lighthouses", 3, ""))[-1]
    assert folder == str(tmp_path / "kept")
    assert "1 page(s)" in message
    assert "https://a.test/1" in message
    assert (tmp_path / "kept" / "01-a-page.txt").exists()


def test_a_failed_search_is_reported_not_raised(monkeypatch):
    from teacher import websearch

    def blocked(*args, **kwargs):
        raise TeacherError("The search found nothing.")

    monkeypatch.setattr(websearch, "harvest", blocked)
    message, _ = list(ui.do_web("lighthouses", 3, ""))[-1]
    assert "Stopped" in message and "found nothing" in message


# ----------------------------------------------- Gradio 5 and 6 both work
def test_the_chat_box_is_built_for_whichever_gradio_is_installed():
    """Gradio 6 removed Chatbot(type=...); 5 needs it. Both must open."""
    import gradio as gr

    from ai_studio.core import gradio_compat as compat

    with gr.Blocks():
        box = compat.chatbot(height=380, label=None)
    assert box is not None


def test_arguments_this_gradio_does_not_take_are_dropped():
    from ai_studio.core import gradio_compat as compat

    def sample(alpha, beta=2):
        return alpha, beta

    assert compat.accepted(sample, {"alpha": 1, "gamma": 3}) == {"alpha": 1}


def test_a_function_taking_anything_keeps_everything():
    from ai_studio.core import gradio_compat as compat

    def sample(**kwargs):
        return kwargs

    given = {"anything": 1, "at": 2, "all": 3}
    assert compat.accepted(sample, given) == given


def test_styling_goes_where_this_version_wants_it():
    from ai_studio.core import gradio_compat as compat

    app = compat.blocks(title="T", css="body {}", theme=None)
    deferred = getattr(app, "_teacher_deferred_styling", {})
    if compat.STYLE_AT_LAUNCH:
        assert "css" in deferred, "Gradio 6 takes css at launch()"
    else:
        assert deferred == {}, "Gradio 5 takes css on the constructor"


def test_one_gpu_or_none_is_not_offered_as_a_slider(monkeypatch):
    """Gradio 6 refuses a slider from 1 to 1, and that stopped the window opening."""
    import gradio as gr

    monkeypatch.setattr(ui, "hardware", lambda: {"gpus": 1, "names": ["one card"]})
    with gr.Blocks():
        assert not isinstance(ui.gpu_picker(), gr.Slider)
    monkeypatch.setattr(ui, "hardware", lambda: {"gpus": 2, "names": ["a", "b"]})
    with gr.Blocks():
        assert isinstance(ui.gpu_picker(), gr.Slider)


# ------------------------------------------- what the terminal can, the window can
@pytest.mark.parametrize("typed,expected", [("auto", "auto"), ("16", 16), (8, 8), ("", 8)])
def test_the_batch_box_takes_a_number_or_auto(typed, expected):
    assert ui.parse_batch(typed) == expected


@pytest.mark.parametrize("typed,expected", [("", None), ("default", None), ("auto", "auto"),
                                            ("5e-5", 5e-5)])
def test_the_rate_box_takes_nothing_a_number_or_auto(typed, expected):
    assert ui.parse_rate(typed) == expected


@pytest.mark.parametrize("typed", ["fast", "2", "-1e-4"])
def test_a_rate_that_is_not_one_is_refused_in_words(typed):
    with pytest.raises(TeacherError):
        ui.parse_rate(typed)


def test_comparing_needs_two_models():
    assert "at least two" in ui.do_compare(["carded"], "hello", 8, 0.8)


def test_the_resume_button_hides_when_there_is_nothing_to_resume():
    note, button = ui.resume_state("carded")
    assert note["visible"] is False and button["visible"] is False


def test_the_model_card_is_written_from_the_window():
    status, text = ui.do_card("carded")
    assert "Model card written" in status
    assert "# carded" in text and "No licence has been chosen" in text


def test_serving_starts_and_stops_from_the_window():
    import json
    import socket
    import urllib.request

    from teacher.material import gather

    model = workspace.get("served-ui", must_exist=False)
    lessons.create(model, gather([], raw_text="the lamp lit the sea . " * 400), "tiny",
                   context=64)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    started = ui.do_serve(model.name, port, "")
    try:
        assert "Serving" in started, started
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as reply:
            assert json.load(reply)["data"][0]["id"] == model.name
        assert "Already serving" in ui.do_serve(model.name, port, "")
    finally:
        assert "Stopped" in ui.do_stop_serving()
    assert ui.do_stop_serving() == "Nothing is being served."
