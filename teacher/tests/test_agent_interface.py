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


def test_the_guide_covers_every_command_it_tells_an_assistant_to_run():
    """A command named in the guide must exist; one that exists should be known."""
    import re

    payload = teacher("guide")
    named = set(re.findall(r"-m teacher --json (\w+)", payload["guide"]))
    real = set(re.findall(r'add_parser\(\s*"(\w+)"',
                          (REPO / "teacher" / "__main__.py").read_text()))
    assert named <= real, f"the guide names commands that do not exist: {named - real}"
    for essential in ("list", "new", "teach", "ask", "show", "rollback"):
        assert essential in named, f"an assistant is never told about '{essential}'"


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


# ------------------------------------------------------------- the web command
def test_asking_for_nothing_from_the_web_is_refused_as_data():
    payload = teacher("web", expect_ok=False)
    assert "error" in payload and payload["command"] == "web"


def test_an_address_on_this_machine_is_refused_without_a_request_leaving():
    payload = teacher("web", "--url", "http://127.0.0.1:9/secrets", expect_ok=False)
    assert "not the web" in payload["error"], payload


def test_the_guide_tells_an_assistant_the_web_is_there():
    guide = teacher("guide")["guide"]
    assert "web" in guide
    assert "--web" in guide or "--json web" in guide


def test_the_readme_does_not_overstate_how_much_holds_this_contract():
    """A count in prose goes stale the moment a test is added or removed."""
    import re

    units = {"": 0, "-one": 1, "-two": 2, "-three": 3, "-four": 4, "-five": 5,
             "-six": 6, "-seven": 7, "-eight": 8, "-nine": 9}
    words = {"Ten": 10, "Eleven": 11, "Twelve": 12, "Thirteen": 13, "Fourteen": 14,
             "Fifteen": 15, "Sixteen": 16, "Seventeen": 17, "Eighteen": 18,
             "Nineteen": 19}
    for tens, base in (("Twenty", 20), ("Thirty", 30), ("Forty", 40), ("Fifty", 50)):
        words.update({f"{tens}{suffix}": base + value for suffix, value in units.items()})
    readme = (REPO / "README.md").read_text()
    claimed = re.search(r"([\w-]+) tests hold that contract in place", readme)
    assert claimed, "the README no longer makes the claim this test checks"

    written = claimed.group(1)
    count = words.get(written) or int(written)
    # This test checks the README, not the contract, so it does not count itself.
    here = re.findall(r"^def (test_\w+)", Path(__file__).read_text(), re.M)
    real = len([name for name in here if "readme" not in name])
    assert count == real, (
        f"README.md says {written.lower()} tests hold the JSON contract; "
        f"this file has {real}"
    )


# --------------------------------------------------------- saved states
def test_checkpoints_answers_even_with_none_saved():
    payload = teacher("checkpoints", "agent-untaught")
    assert payload["states"] == []


def test_checkpoints_lists_what_a_taught_model_can_go_back_to():
    payload = teacher("checkpoints", "agent-made")
    assert payload["states"], "a taught model must have something to roll back to"
    for state in payload["states"]:
        assert state["stamp"] and state["bytes"] > 0


def test_branching_reports_where_the_copy_came_from():
    payload = teacher("branch", "agent-made", "agent-branch")
    assert payload["branched_from"] == "agent-made"
    assert payload["model"] == "agent-branch"
    assert Path(payload["path"]).is_dir()


def test_branching_onto_a_name_in_use_is_reported_as_data():
    payload = teacher("branch", "agent-made", "agent-branch", expect_ok=False)
    assert "already exists" in payload["error"]


def test_a_branch_can_be_taught_without_disturbing_its_original():
    before = teacher("show", "agent-made")
    teacher("teach", "agent-branch", "--from", str(TEXT), "--epochs", "1")
    after = teacher("show", "agent-made")
    assert after["lessons"] == before["lessons"]
    assert after["held_out_loss"] == before["held_out_loss"]


def test_an_unknown_saved_state_is_refused_with_the_real_ones_named():
    payload = teacher("rollback", "agent-made", "--to", "19990101-000000", expect_ok=False)
    assert "no saved state" in payload["error"]


# ------------------------------------------------- the paste-into-an-assistant doc
DOC = REPO / "docs" / "OPERATING.md"


def test_the_operating_document_exists():
    assert DOC.is_file(), "docs/OPERATING.md is what people paste into an assistant"


def test_it_names_no_command_that_does_not_exist():
    """An assistant will run exactly what this file tells it to."""
    import re

    named = set(re.findall(r"python -m teacher (?:--json )?([a-z]+)", DOC.read_text()))
    real = set(re.findall(r'add_parser\(\s*"(\w+)"',
                          (REPO / "teacher" / "__main__.py").read_text()))
    assert named <= real, f"the document names commands that do not exist: {named - real}"


def test_it_names_no_flag_that_does_not_exist():
    import re

    text = DOC.read_text()
    source = (REPO / "teacher" / "__main__.py").read_text()
    real = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', source))
    real |= set(re.findall(r'"(--[\w-]+)"', source))

    # Flags as they appear in prose and code blocks, ignoring the ones that
    # belong to pip, venv and ai_studio.
    theirs = {"--json", "--help"}
    elsewhere = {"--host", "--port", "--share", "--no-browser"}
    named = set(re.findall(r"(?<![\w-])(--[a-z][\w-]+)", text))
    unknown = named - real - theirs - elsewhere
    assert not unknown, f"the document names flags Teacher does not have: {sorted(unknown)}"


def test_it_covers_every_command_someone_would_need():
    text = DOC.read_text()
    for command in ("new", "teach", "ask", "list", "show", "compare", "branch",
                    "rollback", "checkpoints", "web", "pack", "bases", "forget", "ui"):
        assert f"teacher {command}" in text, f"an assistant is never told about '{command}'"


def test_it_names_every_size_that_exists():
    from teacher import lessons

    text = DOC.read_text()
    for size in lessons.SIZES:
        assert f"`{size}`" in text, f"the size '{size}' is not in the document"
