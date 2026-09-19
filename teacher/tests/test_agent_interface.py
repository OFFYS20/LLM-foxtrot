"""The contract an assistant drives Teacher through.

Every command, on every path, must put exactly one parseable object on stdout —
an empty workspace and a failure included. Silence is what breaks an agent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
HOME = Path(tempfile.mkdtemp(prefix="teacher-agent-tests-"))
TEXT = HOME / "material"


def setup_module(module):
    TEXT.mkdir(parents=True, exist_ok=True)
    subjects = ["the cat", "the dog", "a bird", "the fox"]
    verbs = ["sat on", "ran past", "looked at", "dreamed of"]
    objects = ["the mat", "the river", "the moon", "the garden"]
    corpus = " ".join(
        f"{s} {v} {o} ." for _ in range(90) for s in subjects for v in verbs for o in objects
    )
    (TEXT / "notes.txt").write_text(corpus, encoding="utf-8")


def teacher(*args, expect_ok: bool = True) -> dict:
    """Run one command the way an assistant would, and parse its reply."""
    env = {**os.environ, "TEACHER_HOME": str(HOME / "models")}
    proc = subprocess.run(
        [sys.executable, "-m", "teacher", "--json", *args],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=900,
    )
    assert proc.stdout.strip(), (
        f"'{' '.join(args)}' printed nothing — an assistant has no result to read.\n"
        f"stderr: {proc.stderr[-300:]}"
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - the failure message is the point
        pytest.fail(f"'{' '.join(args)}' did not print one JSON object: {exc}\n"
                    f"stdout: {proc.stdout[:400]}")
    assert payload.get("ok") is expect_ok, payload
    return payload


# ------------------------------------------------------------- the contract
def test_an_empty_workspace_still_answers():
    payload = teacher("list")
    assert payload["models"] == []
    assert payload["home"]


def test_the_bases_are_listed_with_their_repos():
    payload = teacher("bases")
    assert payload["bases"]
    assert all("/" in row["repo"] for row in payload["bases"])


def test_the_guide_carries_the_paths_an_assistant_needs():
    payload = teacher("guide")
    assert payload["repo"] and payload["home"] and payload["python"]
    for command in ("--json list", "--json teach", "--json ask", "--json new"):
        assert command in payload["guide"], f"the guide never mentions {command}"
    assert "ok" in payload["guide"], "the guide must explain the reply shape"


def test_a_failure_is_reported_as_data_not_silence():
    payload = teacher("show", "never-made", expect_ok=False)
    assert "error" in payload and payload["error"]
    assert payload["command"] == "show"


def test_making_and_teaching_reports_a_stage():
    made = teacher("new", "agent-made", "--from", str(TEXT), "--size", "tiny",
                   "--epochs", "2", "--until", "words")
    assert made["model"] == "agent-made"
    assert made["stage"]
    assert made["held_out_loss"] > 0


def test_show_reports_enough_to_decide_what_to_do_next():
    payload = teacher("show", "agent-made")
    for key in ("stage", "held_out_loss", "lessons", "kind", "architecture"):
        assert key in payload, f"an assistant cannot judge progress without {key}"


def test_teaching_again_says_why_it_stopped():
    payload = teacher("teach", "agent-made", "--from", str(TEXT),
                      "--until", "best", "--max-rounds", "2", "--epochs", "1")
    assert payload["reason"], "an assistant must be able to explain why it stopped"


def test_asking_returns_the_reply_and_the_speed():
    payload = teacher("ask", "agent-made", "the cat")
    assert isinstance(payload["reply"], str)
    assert payload["generated"] > 0
    assert payload["tokens_per_second"] > 0


def test_a_model_that_was_built_but_not_taught_still_answers():
    payload = teacher("new", "agent-untaught", "--from", str(TEXT),
                      "--size", "tiny", "--no-teach")
    assert payload["taught"] is False
    assert payload["parameters"] > 0


def test_nothing_but_the_object_reaches_stdout():
    """Training logs and progress must not pollute what an assistant parses."""
    env = {**os.environ, "TEACHER_HOME": str(HOME / "models")}
    proc = subprocess.run(
        [sys.executable, "-m", "teacher", "--json", "teach", "agent-made",
         "--from", str(TEXT), "--epochs", "1"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=900,
    )
    payload = json.loads(proc.stdout)
    assert proc.stdout.strip() == json.dumps(payload, indent=2, default=str).strip()
