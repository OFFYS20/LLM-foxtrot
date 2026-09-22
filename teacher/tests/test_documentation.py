"""The combined documentation has to say what its sources say."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def builder():
    spec = importlib.util.spec_from_file_location(
        "build_documentation", REPO / "docs" / "build_documentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_combined_documentation_is_up_to_date():
    """A hand-edited copy starts going stale the first time a source changes.
    If this fails: python docs/build_documentation.py"""
    expected = builder().build()
    actual = (REPO / "DOCUMENTATION.md").read_text(encoding="utf-8")
    assert actual == expected, (
        "DOCUMENTATION.md is out of step with its sources. "
        "Run: python docs/build_documentation.py")


def test_every_documentation_file_in_the_project_is_in_it():
    """A new README that nobody added to the list would be silently missing."""
    import subprocess

    included = {name for name, _title, _description in builder().PARTS}
    # What git tracks is what the project's documentation is — not a cache's
    # README, not a dependency's.
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()
    found = {name for name in tracked
             if "node_modules" not in name and name != "DOCUMENTATION.md"}
    missing = found - included
    assert not missing, f"documentation files left out of DOCUMENTATION.md: {sorted(missing)}"


def test_every_section_of_every_source_is_there():
    module = builder()
    combined = (REPO / "DOCUMENTATION.md").read_text(encoding="utf-8")
    for name, _title, _description in module.PARTS:
        _lines, headings = module.section(REPO / name, _title)
        for _level, text in headings:
            assert text in combined, f"'{text}' from {name} is missing"


def test_every_link_in_the_contents_points_at_a_heading():
    module = builder()
    combined = (REPO / "DOCUMENTATION.md").read_text(encoding="utf-8")

    seen: dict[str, int] = {}
    anchors = set()
    inside_code = False
    for line in combined.splitlines():
        if module.FENCE.match(line):
            inside_code = not inside_code
            continue
        match = None if inside_code else module.HEADING.match(line)
        if match:
            anchors.add(module.slug(match.group(2), seen))

    contents = combined.split("## Contents", 1)[1].split("\n---", 1)[0]
    for target in re.findall(r"\]\(#([^)]+)\)", contents):
        assert target in anchors, f"the contents link #{target} goes nowhere"


def test_links_still_resolve_from_the_repository_root():
    """A relative link from teacher/README.md means something else from here."""
    combined = (REPO / "DOCUMENTATION.md").read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", combined):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        if not (REPO / target).exists():
            broken.append(target)
    assert not broken, f"links that point nowhere from the root: {sorted(set(broken))[:8]}"
