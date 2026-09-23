"""docs/MANUAL.md: the manual people read when something breaks.

A manual that has fallen behind the program is worse than none: it sends the
reader to fix the wrong thing. So these tests read the program and check the
manual against it — every message the program can stop with, every command,
and the numbers it states.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANUAL = REPO / "docs" / "MANUAL.md"


def flat(text: str) -> str:
    return " ".join(text.split())


def fragments(node) -> list[str]:
    """The fixed parts of a message: the text around its {placeholders}."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts, current = [], ""
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                current += piece.value
            else:
                parts.append(current)
                current = ""
        return parts + [current]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return fragments(node.left) + fragments(node.right)
    return []


def messages() -> dict[str, str]:
    """Each TeacherError's longest fixed fragment, and where it is raised."""
    found: dict[str, str] = {}
    for path in sorted((REPO / "teacher").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                    and getattr(node.exc.func, "id", "") == "TeacherError" and node.exc.args):
                pieces = [flat(piece) for piece in fragments(node.exc.args[0])]
                longest = max(pieces, key=len, default="")
                # Trimmed to the words: the punctuation at the edges belongs to
                # whatever was filled in beside it.
                core = re.sub(r"^[^\w-]+|[^\w.]+$", "", longest)
                if len(core) >= 15:
                    found.setdefault(core, f"{path.name}:{node.lineno}")
    return found


def test_the_manual_exists():
    assert MANUAL.is_file()


def test_every_message_the_program_can_stop_with_is_explained():
    text = flat(MANUAL.read_text(encoding="utf-8"))
    missing = [f"{where}: {message!r}" for message, where in messages().items()
               if message not in text]
    assert not missing, "the manual does not explain:\n  " + "\n  ".join(missing)


def test_the_messages_found_are_the_real_ones():
    """A reader that found nothing would pass the test above by default."""
    found = messages()
    assert len(found) > 50
    assert "This lesson would not fit in memory" in found


def test_every_command_is_in_the_manual():
    text = MANUAL.read_text(encoding="utf-8")
    source = (REPO / "teacher" / "__main__.py").read_text(encoding="utf-8")
    for command in re.findall(r'add_parser\(\s*"(\w+)"', source):
        assert f"teacher {command}" in text, f"the manual never mentions 'teacher {command}'"


def test_the_numbers_it_states_are_the_programs():
    from ai_studio.training import config
    from teacher import lessons, material, serve

    text = flat(MANUAL.read_text(encoding="utf-8"))
    stated = {
        "the smallest lesson": f"at least **{lessons.MIN_CHARACTERS:,}** characters",
        "the repeat threshold": f"**{material.MIN_REPEAT} characters or more**",
        "the batch ladder": ", ".join(str(size) for size in config.BATCH_LADDER),
        "the fewest steps an epoch": f"at least {config.MIN_STEPS_PER_EPOCH} optimizer steps",
        "the safety margin": f"a safety margin** of {config.SAFETY_MARGIN}",
        "the port": f"http://127.0.0.1:{serve.DEFAULT_PORT}/v1",
        "the longest reply": f"at most {serve.MOST_TOKENS:,}",
        "the plateau": f"less than {lessons.PLATEAU:.0%} of the loss",
    }
    for what, phrase in stated.items():
        assert flat(phrase) in text, f"the manual's {what} does not match the program: {phrase!r}"


def test_the_manual_names_each_default_rate():
    from teacher import lessons

    text = MANUAL.read_text(encoding="utf-8")
    for kind, rate in lessons.RATES.items():
        assert f"{rate:.0e}".replace("e-0", "e-") in text, f"the {kind} default is not stated"
