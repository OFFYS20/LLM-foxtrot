"""Benchmark adapter abstraction.

Adding a benchmark means implementing one class and registering it — the API,
the runner and the UI need no changes. Adapters are explicit about where their
items come from (``data_source``) and whether the numbers they produce can be
called official (``official``), so a bundled 12-item sample is never presented
as "MMLU".
"""

from __future__ import annotations

import abc
import random
import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas.benchmarks import BenchmarkSuiteInfo


@dataclass(slots=True)
class BenchmarkItemSpec:
    index: int
    question: str
    expected: str
    category: str = ""
    choices: list[str] = field(default_factory=list)
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreResult:
    correct: bool
    score: float
    graded: bool = True
    method: str = "exact_match"
    note: str | None = None


class BenchmarkAdapter(abc.ABC):
    key: str = "base"
    label: str = "Base"
    category: str = "knowledge"
    metric: str = "accuracy"
    description: str = ""
    default_shots: int = 0
    requires_execution: bool = False
    data_source: str = "bundled_sample"
    official: bool = False
    notes: str | None = None

    @abc.abstractmethod
    def items(self, *, limit: int, seed: int = 42) -> list[BenchmarkItemSpec]:
        """Return up to ``limit`` items in a deterministic order for ``seed``."""

    def available_items(self) -> int:
        return len(self.items(limit=10_000))

    def build_prompt(
        self, item: BenchmarkItemSpec, shots: list[BenchmarkItemSpec] | None = None
    ) -> str:
        parts: list[str] = []
        for shot in shots or []:
            parts.append(f"{self._render_question(shot)}\nAnswer: {shot.expected}\n")
        parts.append(f"{self._render_question(item)}\nAnswer:")
        return "\n".join(parts)

    def _render_question(self, item: BenchmarkItemSpec) -> str:
        block = item.question if not item.context else f"{item.context}\n\n{item.question}"
        if item.choices:
            options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item.choices))
            block = f"{block}\n{options}"
        return block

    @abc.abstractmethod
    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        """Grade one response."""

    def info(self) -> BenchmarkSuiteInfo:
        return BenchmarkSuiteInfo(
            key=self.key,
            label=self.label,
            category=self.category,
            description=self.description,
            metric=self.metric,
            default_shots=self.default_shots,
            available_items=self.available_items(),
            data_source=self.data_source,
            official=self.official,
            requires_execution=self.requires_execution,
            notes=self.notes,
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def normalize(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"^(the\s+)?answer\s*(is)?\s*[:\-]?\s*", "", text)
        text = re.sub(r"[^\w\s./-]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def extract_choice(response: str, choices: list[str]) -> str | None:
        """Pull an A/B/C/D style answer (or a verbatim option) out of free text."""
        if not choices:
            return None
        letters = [chr(65 + i) for i in range(len(choices))]
        match = re.search(r"\b([A-Z])\b[\).:]?", response.strip()[:120])
        if match and match.group(1) in letters:
            return choices[letters.index(match.group(1))]
        lowered = response.lower()
        for choice in choices:
            if choice.lower() in lowered:
                return choice
        return None

    @staticmethod
    def extract_number(response: str) -> str | None:
        matches = re.findall(r"-?\d[\d,]*\.?\d*", response.replace("$", ""))
        if not matches:
            return None
        return matches[-1].replace(",", "").rstrip(".")

    @staticmethod
    def sample(items: list[BenchmarkItemSpec], limit: int, seed: int) -> list[BenchmarkItemSpec]:
        if limit >= len(items):
            picked = list(items)
        else:
            rng = random.Random(seed)
            picked = rng.sample(items, limit)
        return [
            BenchmarkItemSpec(
                index=i,
                question=item.question,
                expected=item.expected,
                category=item.category,
                choices=item.choices,
                context=item.context,
                metadata=item.metadata,
            )
            for i, item in enumerate(picked)
        ]
