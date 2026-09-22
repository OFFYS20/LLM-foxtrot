"""Build DOCUMENTATION.md — every documentation file in the project, in one place.

    python docs/build_documentation.py

The combined file is generated rather than written, because a hand-merged copy
of three thousand lines starts going stale the first time anyone edits one of
its sources. This keeps the sources as the truth and the combined file as a
view of them. A test fails when the two disagree, so the combined file cannot
quietly fall behind.

What it does to each source:

* its title is replaced by a part heading, and its sections keep their levels,
  so each part reads exactly as its source does;
* headings inside code blocks are left alone — a line starting with `#` in a
  shell example is a comment, not a heading;
* links written relative to the source's own folder are rewritten relative to
  the repository root, so they still resolve from here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "DOCUMENTATION.md"

#: In reading order: what it is, then the program most people use, then the
#: others, then the material for assistants and for people extending it.
PARTS = (
    ("README.md", "Overview",
     "The four programs, and how to start."),
    ("teacher/README.md", "Teacher",
     "Making, teaching, testing and exporting a model — the main program."),
    ("web_chat/README.md", "Bench",
     "Running a model in a browser, from a single HTML file."),
    ("ai_studio/README.md", "AI Studio",
     "The full workbench: datasets, tokenizers, LoRA, RAG and benchmarks."),
    ("docs/OPERATING.md", "Operating it through an assistant",
     "Written to be pasted into ChatGPT or Claude so it can drive everything."),
    ("docs/EXTENDING.md", "Extending Foxtrot",
     "Adding an inference engine, a benchmark or a training backend."),
    ("docs/SCHEMA.md", "Foxtrot database schema",
     "Every table the Foxtrot backend keeps."),
)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def slug(text: str, seen: dict[str, int]) -> str:
    """The anchor GitHub gives a heading, counting repeats the way it does."""
    plain = re.sub(r"[`*_\[\]()]", "", text).strip().lower()
    base = re.sub(r"[^\w\- ]", "", plain).replace(" ", "-")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def rebase(target: str, source: Path) -> str:
    """Rewrite a link relative to the source file so it works from the root."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
        return target
    path, _, anchor = target.partition("#")
    if not path:
        return target
    resolved = (source.parent / path).resolve()
    try:
        relative = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return target
    return relative + (f"#{anchor}" if anchor else "")


def section(source: Path, title: str) -> tuple[list[str], list[tuple[int, str]]]:
    """One source with its title removed. Returns its lines and its headings."""
    lines = source.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    headings: list[tuple[int, str]] = []
    inside_code = False
    replaced_title = False

    for line in lines:
        if FENCE.match(line):
            inside_code = not inside_code
            out.append(line)
            continue
        if inside_code:
            out.append(line)
            continue

        match = HEADING.match(line)
        if match:
            level, text = len(match.group(1)), match.group(2)
            if level == 1 and not replaced_title:
                # The file's own title is replaced by the part heading above it.
                replaced_title = True
                continue
            headings.append((level, text))
            out.append(line)
            continue

        out.append(LINK.sub(lambda m: m.group(1) + rebase(m.group(2), source) + m.group(3), line))

    while out and not out[0].strip():
        out.pop(0)
    return out, headings


def build() -> str:
    """The whole document, as text."""
    missing = [name for name, _t, _d in PARTS if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Documentation sources are missing: {', '.join(missing)}")

    bodies: list[tuple[str, str, str, list[str], list[tuple[int, str]]]] = []
    for name, title, description in PARTS:
        lines, headings = section(ROOT / name, title)
        bodies.append((name, title, description, lines, headings))

    seen: dict[str, int] = {}
    slug("LLM Foxtrot — complete documentation", seen)
    slug("Contents", seen)

    contents: list[str] = []
    for number, (name, title, description, _lines, headings) in enumerate(bodies, start=1):
        part = f"Part {number} — {title}"
        contents.append(f"{number}. **[{part}](#{slug(part, seen)})** — {description}")
        for level, text in headings:
            anchor = slug(text, seen)
            if level == 2:
                contents.append(f"   - [{text}](#{anchor})")

    document = [
        "# LLM Foxtrot — complete documentation",
        "",
        "Every documentation file in the project, in one place and in reading order.",
        "",
        "> **Generated — do not edit this file by hand.** It is assembled from the files",
        "> listed below by `python docs/build_documentation.py`, and a test fails if it",
        "> falls out of step with them. Change the source, then rebuild.",
        "",
        "| Part | Source |",
        "|---|---|",
    ]
    for number, (name, title, _description, _lines, _headings) in enumerate(bodies, start=1):
        document.append(f"| {number}. {title} | [`{name}`]({name}) |")

    document += ["", "## Contents", "", *contents, ""]

    for number, (name, title, description, lines, _headings) in enumerate(bodies, start=1):
        document += [
            "---",
            "",
            f"# Part {number} — {title}",
            "",
            f"*{description} From [`{name}`]({name}).*",
            "",
            *lines,
            "",
        ]
    return "\n".join(document).rstrip() + "\n"


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("DOCUMENTATION.md is out of date. Run: python docs/build_documentation.py")
            return 1
        print("DOCUMENTATION.md is up to date.")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    parts = len(PARTS)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} — {parts} parts, "
          f"{text.count(chr(10)):,} lines, {len(text):,} characters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
