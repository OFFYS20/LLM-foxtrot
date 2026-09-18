"""Document cleaning.

Cleaning is opt-in per step and always non-destructive: the original file is
kept byte-for-byte and the cleaned text is written alongside it, so a bad
cleaning run is never irreversible.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INDENTED_CODE = re.compile(r"(?:^(?: {4}|\t).*\n?)+", re.MULTILINE)
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
HTML_TAG = re.compile(r"<[^>]{1,400}>")
ZERO_WIDTH = re.compile(r"[​-‏  ﻿­]")

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$"),
    re.compile(r"^\s*page\s+\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
    re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE),
]

SENTENCE_END = re.compile(r"[.!?:;\"')\]]$")


@dataclass
class CleaningOptions:
    normalize_unicode: bool = True
    fix_pdf_wrapping: bool = True
    remove_page_numbers: bool = True
    remove_headers_footers: bool = True
    remove_repeated_whitespace: bool = True
    remove_duplicate_paragraphs: bool = True
    remove_empty_sections: bool = True
    remove_html_tags: bool = False
    preserve_code_blocks: bool = True
    preserve_headings: bool = True
    min_paragraph_chars: int = 0

    @classmethod
    def for_type(cls, doc_type: str) -> "CleaningOptions":
        """Sensible defaults per source format."""
        options = cls()
        if doc_type == "pdf":
            return options
        if doc_type in {"html", "epub"}:
            options.fix_pdf_wrapping = False
            options.remove_html_tags = True
            options.remove_page_numbers = False
            return options
        if doc_type in {"markdown", "md"}:
            options.fix_pdf_wrapping = False
            options.remove_page_numbers = False
            options.remove_headers_footers = False
            return options
        if doc_type in {"csv", "json", "jsonl"}:
            return cls(
                normalize_unicode=True,
                fix_pdf_wrapping=False,
                remove_page_numbers=False,
                remove_headers_footers=False,
                remove_repeated_whitespace=True,
                remove_duplicate_paragraphs=False,
                remove_empty_sections=False,
            )
        options.fix_pdf_wrapping = False
        options.remove_page_numbers = False
        options.remove_headers_footers = False
        return options

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningReport:
    original_chars: int = 0
    cleaned_chars: int = 0
    steps_applied: list[str] = field(default_factory=list)
    page_numbers_removed: int = 0
    header_footer_lines_removed: int = 0
    duplicate_paragraphs_removed: int = 0
    empty_sections_removed: int = 0
    wrapped_lines_joined: int = 0
    html_tags_removed: int = 0
    code_blocks_preserved: int = 0

    @property
    def chars_removed(self) -> int:
        return max(0, self.original_chars - self.cleaned_chars)

    @property
    def reduction_percent(self) -> float:
        if not self.original_chars:
            return 0.0
        return round(self.chars_removed / self.original_chars * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chars_removed"] = self.chars_removed
        data["reduction_percent"] = self.reduction_percent
        return data

    def summary(self) -> str:
        parts = [
            f"{self.original_chars:,} → {self.cleaned_chars:,} chars "
            f"(−{self.reduction_percent:.1f}%)"
        ]
        if self.page_numbers_removed:
            parts.append(f"{self.page_numbers_removed} page numbers")
        if self.header_footer_lines_removed:
            parts.append(f"{self.header_footer_lines_removed} header/footer lines")
        if self.duplicate_paragraphs_removed:
            parts.append(f"{self.duplicate_paragraphs_removed} duplicate paragraphs")
        if self.wrapped_lines_joined:
            parts.append(f"{self.wrapped_lines_joined} wrapped lines joined")
        if self.empty_sections_removed:
            parts.append(f"{self.empty_sections_removed} empty sections")
        if self.html_tags_removed:
            parts.append(f"{self.html_tags_removed} HTML tags")
        if self.code_blocks_preserved:
            parts.append(f"{self.code_blocks_preserved} code blocks preserved")
        return " · ".join(parts)


@dataclass
class CleaningResult:
    text: str
    report: CleaningReport
    options: CleaningOptions


def clean_text(
    text: str,
    options: CleaningOptions | None = None,
    *,
    sections: list[str] | None = None,
) -> CleaningResult:
    """Run the enabled cleaning steps over ``text``.

    ``sections`` (PDF pages, EPUB chapters) enables repeated header/footer
    detection, which needs to compare the same line position across pages.
    """
    options = options or CleaningOptions()
    report = CleaningReport(original_chars=len(text))
    working = text

    placeholders: dict[str, str] = {}
    if options.preserve_code_blocks:
        working, placeholders = _protect_code_blocks(working)
        report.code_blocks_preserved = len(placeholders)
        if placeholders:
            report.steps_applied.append("preserve_code_blocks")

    if options.normalize_unicode:
        working = _normalize_unicode(working)
        report.steps_applied.append("normalize_unicode")

    if options.remove_html_tags:
        working, removed = _strip_html(working)
        report.html_tags_removed = removed
        report.steps_applied.append("remove_html_tags")

    if options.remove_headers_footers and sections and len(sections) >= 3:
        repeated = _find_repeated_lines(sections)
        if repeated:
            working, removed = _drop_lines(working, repeated)
            report.header_footer_lines_removed = removed
            report.steps_applied.append("remove_headers_footers")

    if options.remove_page_numbers:
        working, removed = _remove_page_numbers(working)
        report.page_numbers_removed = removed
        report.steps_applied.append("remove_page_numbers")

    if options.fix_pdf_wrapping:
        working, joined = _fix_wrapping(working, preserve_headings=options.preserve_headings)
        report.wrapped_lines_joined = joined
        report.steps_applied.append("fix_pdf_wrapping")

    if options.remove_repeated_whitespace:
        working = _collapse_whitespace(working)
        report.steps_applied.append("remove_repeated_whitespace")

    if options.remove_duplicate_paragraphs:
        working, removed = _dedupe_paragraphs(working, min_chars=options.min_paragraph_chars)
        report.duplicate_paragraphs_removed = removed
        report.steps_applied.append("remove_duplicate_paragraphs")

    if options.remove_empty_sections:
        working, removed = _remove_empty_sections(working)
        report.empty_sections_removed = removed
        report.steps_applied.append("remove_empty_sections")

    if placeholders:
        working = _restore_code_blocks(working, placeholders)

    working = working.strip() + "\n" if working.strip() else ""
    report.cleaned_chars = len(working)
    return CleaningResult(text=working, report=report, options=options)


# --------------------------------------------------------------------- steps
def _protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def _stash(match: re.Match[str]) -> str:
        key = f"\x00CODE{len(placeholders):05d}\x00"
        placeholders[key] = match.group(0)
        return key

    return CODE_FENCE.sub(_stash, text), placeholders


def _restore_code_blocks(text: str, placeholders: dict[str, str]) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    replacements = {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "—", "−": "-",
        " ": " ", "…": "...",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_html(text: str) -> tuple[str, int]:
    matches = HTML_TAG.findall(text)
    return HTML_TAG.sub(" ", text), len(matches)


def _find_repeated_lines(sections: list[str], *, threshold: float = 0.6) -> set[str]:
    """Lines appearing at the top/bottom of most sections are running heads."""
    counter: Counter[str] = Counter()
    for section in sections:
        lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
        if not lines:
            continue
        for line in {*lines[:2], *lines[-2:]}:
            if 3 <= len(line) <= 120:
                counter[line] += 1
    minimum = max(3, int(len(sections) * threshold))
    return {line for line, count in counter.items() if count >= minimum}


def _drop_lines(text: str, unwanted: set[str]) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if line.strip() in unwanted:
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _remove_page_numbers(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(pattern.match(stripped) for pattern in PAGE_NUMBER_PATTERNS):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _fix_wrapping(text: str, *, preserve_headings: bool = True) -> tuple[str, int]:
    """Re-join lines broken mid-sentence by PDF extraction."""
    lines = text.split("\n")
    output: list[str] = []
    joined = 0

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        is_heading = preserve_headings and bool(HEADING.match(line))
        is_list = bool(re.match(r"^\s*([-*•]|\d+[.)])\s+", line))

        if not output or not stripped or is_heading or is_list:
            output.append(line)
            continue

        previous = output[-1]
        previous_stripped = previous.strip()
        if not previous_stripped or (preserve_headings and HEADING.match(previous)):
            output.append(line)
            continue

        # "exam-\nple" → "example"
        if previous_stripped.endswith("-") and not previous_stripped.endswith("--"):
            output[-1] = previous_stripped[:-1] + stripped
            joined += 1
            continue

        # A line that does not end a sentence and continues in lower case is a wrap.
        if not SENTENCE_END.search(previous_stripped) and (
            stripped[0].islower() or stripped[0] in ",;)"
        ):
            output[-1] = f"{previous_stripped} {stripped}"
            joined += 1
            continue

        output.append(line)

    return "\n".join(output), joined


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _dedupe_paragraphs(text: str, *, min_chars: int = 0) -> tuple[str, int]:
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            continue
        if min_chars and len(stripped) < min_chars and not HEADING.match(stripped):
            removed += 1
            continue
        # Headings legitimately repeat; only dedupe substantial prose.
        if HEADING.match(stripped) or len(stripped) < 40:
            kept.append(paragraph)
            continue
        fingerprint = hashlib.sha1(
            re.sub(r"\W+", " ", stripped.lower()).strip().encode()
        ).hexdigest()
        if fingerprint in seen:
            removed += 1
            continue
        seen.add(fingerprint)
        kept.append(paragraph)
    return "\n\n".join(kept), removed


def _remove_empty_sections(text: str) -> tuple[str, int]:
    """Drop headings with no body before the next heading of the same/higher level."""
    lines = text.split("\n")
    output: list[str] = []
    removed = 0
    index = 0

    while index < len(lines):
        match = HEADING.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue

        level = len(match.group(1))
        lookahead = index + 1
        has_body = False
        while lookahead < len(lines):
            next_heading = HEADING.match(lines[lookahead])
            if next_heading and len(next_heading.group(1)) <= level:
                break
            if lines[lookahead].strip():
                has_body = True
                break
            lookahead += 1

        if has_body:
            output.append(lines[index])
        else:
            removed += 1
        index += 1

    return "\n".join(output), removed


def estimate_tokens(text: str) -> int:
    """Rough token estimate used before a tokenizer is attached (~4 chars/token)."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def preview_pair(original: str, cleaned: str, *, limit: int = 2400) -> tuple[str, str]:
    """Before/after snippets for the UI."""
    return original[:limit], cleaned[:limit]
