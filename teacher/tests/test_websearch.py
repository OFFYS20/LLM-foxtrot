"""Gathering material from the web.

Offline: the engine underneath is replaced with recorded pages, so what is
tested is Teacher's part — reading several addresses without letting one dead
link cost the rest, and writing every page down with where it came from.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-web-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from ai_studio.core.errors import ValidationError, WebError  # noqa: E402
from ai_studio.data import web as engine  # noqa: E402
from teacher import websearch  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """The pause between fetches is politeness, not behaviour under test."""
    monkeypatch.setattr(websearch, "PAUSE", 0)


def page(url: str, title: str = "A page", words: int = 300) -> engine.Page:
    return engine.Page(url=url, title=title, text=" ".join(["word"] * words))


# ------------------------------------------------------------------ harvest
def test_a_search_is_read_and_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(websearch, "search", lambda query, *, limit=5: [
        engine.Result(title="One", url="https://a.test/1"),
        engine.Result(title="Two", url="https://a.test/2"),
    ])
    monkeypatch.setattr(websearch, "read_page", lambda url: page(url))

    collected = websearch.harvest("lighthouses", limit=2)
    assert len(collected.pages) == 2
    assert collected.characters == sum(p.characters for p in collected.pages)
    assert "2 page(s)" in collected.summary()


def test_addresses_given_directly_need_no_search(monkeypatch):
    monkeypatch.setattr(websearch, "read_page", lambda url: page(url))
    collected = websearch.harvest(urls=["https://a.test/1", "https://a.test/2"])
    assert [p.url for p in collected.pages] == ["https://a.test/1", "https://a.test/2"]
    assert collected.results == [], "nothing was searched for"


def test_one_dead_link_does_not_cost_the_others(monkeypatch):
    def read(url):
        if url.endswith("2"):
            raise TeacherError("the site answered 404")
        return page(url)

    monkeypatch.setattr(websearch, "read_page", read)
    collected = websearch.harvest(urls=[f"https://a.test/{n}" for n in (1, 2, 3)])
    assert [p.url for p in collected.pages] == ["https://a.test/1", "https://a.test/3"]
    assert collected.skipped == [("https://a.test/2", "the site answered 404")]
    assert "1 skipped" in collected.summary()


def test_when_nothing_could_be_read_it_says_why(monkeypatch):
    monkeypatch.setattr(websearch, "read_page",
                        lambda url: (_ for _ in ()).throw(TeacherError("the site answered 403")))
    with pytest.raises(TeacherError, match="403"):
        websearch.harvest(urls=["https://a.test/1"])


def test_harvesting_nothing_at_all_is_refused():
    with pytest.raises(TeacherError, match="search for"):
        websearch.harvest("")


def test_progress_is_reported_as_it_goes(monkeypatch):
    monkeypatch.setattr(websearch, "read_page", lambda url: page(url))
    steps: list[tuple] = []
    websearch.harvest(urls=["https://a.test/1", "https://a.test/2"],
                      on_step=lambda i, total, url: steps.append((i, total, url)))
    assert steps == [(1, 2, "https://a.test/1"), (2, 2, "https://a.test/2")]


def test_a_page_with_no_title_falls_back_to_the_search_result(monkeypatch):
    monkeypatch.setattr(websearch, "search", lambda query, *, limit=5: [
        engine.Result(title="From the search", url="https://a.test/1"),
    ])
    monkeypatch.setattr(websearch, "read_page", lambda url: page(url, title=""))
    assert websearch.harvest("x").pages[0].title == "From the search"


# ------------------------------------------------------- errors a person reads
def test_engine_errors_come_back_as_something_a_person_can_act_on(monkeypatch):
    monkeypatch.setattr(engine, "search", lambda query, *, limit=5: (_ for _ in ()).throw(
        WebError("The search found nothing.", hint="wait a minute")))
    with pytest.raises(TeacherError) as raised:
        websearch.search("lighthouses")
    assert "wait a minute" in str(raised.value)


def test_a_refused_address_is_a_teacher_error_not_a_traceback(monkeypatch):
    monkeypatch.setattr(engine, "fetch_page", lambda url: (_ for _ in ()).throw(
        ValidationError("127.0.0.1 is this machine or its local network, not the web.")))
    with pytest.raises(TeacherError, match="not the web"):
        websearch.read_page("http://127.0.0.1/")


# ------------------------------------------------------------------- saving
def test_every_saved_page_records_where_it_came_from(tmp_path):
    collected = websearch.Harvest(query="lighthouses", pages=[
        page("https://a.test/keeper", "The Keeper"),
    ])
    written = websearch.save(collected, tmp_path / "material")
    assert len(written) == 1

    text = written[0].read_text(encoding="utf-8")
    assert text.startswith("The Keeper\n")
    assert "Source: https://a.test/keeper" in text
    assert "Retrieved: " in text, "a corpus with no date cannot be checked later"
    assert "word word" in text


def test_saved_files_are_named_after_their_page(tmp_path):
    collected = websearch.Harvest(query="x", pages=[
        page("https://a.test/1", "The Keeper & the Lamp"),
        page("https://a.test/2", "Another/Page: here"),
    ])
    names = [path.name for path in websearch.save(collected, tmp_path)]
    assert names == ["01-the-keeper-the-lamp.txt", "02-another-page-here.txt"]


def test_a_page_with_no_usable_title_still_gets_a_file(tmp_path):
    collected = websearch.Harvest(query="x", pages=[page("https://a.test/1", "!!!")])
    assert websearch.save(collected, tmp_path)[0].name == "01-page.txt"


def test_saving_creates_the_folder_it_was_given(tmp_path):
    destination = tmp_path / "deep" / "deeper"
    websearch.save(websearch.Harvest(query="x", pages=[page("https://a.test/1")]), destination)
    assert destination.is_dir()


def test_two_harvests_of_the_same_query_do_not_overwrite_each_other(monkeypatch):
    """Material already gathered is never quietly replaced."""
    clock = iter(["20260101-120000", "20260101-120500"])
    monkeypatch.setattr(websearch.time, "strftime", lambda fmt: next(clock))

    first = websearch.folder_for("lighthouse keepers", TMP_HOME)
    second = websearch.folder_for("lighthouse keepers", TMP_HOME)
    assert first.parent.name == "web-material"
    assert first.name == "lighthouse-keepers-20260101-120000"
    assert second.name == "lighthouse-keepers-20260101-120500"
    assert first != second, "the second run lands beside the first, not on top of it"


def test_a_folder_name_made_from_an_address_drops_the_scheme():
    assert websearch.folder_for("https://www.example.com/a", TMP_HOME).name.startswith(
        "example-com-a-")
