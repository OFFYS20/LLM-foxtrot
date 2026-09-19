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
