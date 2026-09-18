"""Benchmark adapter backed by a user-imported dataset."""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.db.models.enums import BenchmarkCategory, DatasetTemplate
from app.services.datasets.formats import detect_format, iter_rows
from app.services.datasets.service import resolve_dataset_path
from app.services.evaluation.base import BenchmarkAdapter, BenchmarkItemSpec, ScoreResult

QUESTION_KEYS = ("question", "instruction", "prompt", "input", "q")
ANSWER_KEYS = ("answer", "output", "expected", "response", "a", "label")


class CustomDatasetAdapter(BenchmarkAdapter):
    key = "custom"
    label = "Custom dataset"
    category = BenchmarkCategory.CUSTOM
    metric = "exact_match"
    description = "Evaluate against any imported dataset with question/answer columns."
    data_source = "user_dataset"
    official = False

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.label = f"Custom · {getattr(dataset, 'name', 'dataset')}"
        self.notes = (
            "Scores come from your own dataset; comparisons to public leaderboards do not apply."
        )
        self._rows: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        path_value = getattr(self.dataset, "local_path", None)
        if not path_value:
            raise ValidationError(
                "This dataset has no local file to evaluate against.",
                details={"dataset_id": getattr(self.dataset, "id", None)},
            )
        path = resolve_dataset_path(path_value)
        rows = [
            r for r in iter_rows(path, detect_format(path), limit=20_000) if isinstance(r, dict)
        ]
        self._rows = rows
        return rows

    def items(self, *, limit: int, seed: int = 42) -> list[BenchmarkItemSpec]:
        template = str(getattr(self.dataset, "template", DatasetTemplate.RAW))
        specs: list[BenchmarkItemSpec] = []
        for index, row in enumerate(self._load()):
            question, expected = _extract_pair(row, template)
            if not question or not expected:
                continue
            specs.append(
                BenchmarkItemSpec(
                    index=index,
                    question=question,
                    expected=expected,
                    category=str(row.get("category") or "custom"),
                )
            )
        if not specs:
            raise ValidationError(
                "No question/answer pairs found. Expected columns like "
                "question/answer, instruction/output, or chat messages."
            )
        return self.sample(specs, limit, seed)

    def available_items(self) -> int:
        try:
            return len(self.items(limit=100_000))
        except Exception:
            return 0

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        expected = self.normalize(item.expected)
        got = self.normalize(response)
        if not expected:
            return ScoreResult(False, 0.0, graded=False, method="none", note="No reference answer")
        if got == expected:
            return ScoreResult(True, 1.0, method="exact_match")
        if expected in got:
            return ScoreResult(True, 0.9, method="contains")
        expected_tokens = set(expected.split())
        overlap = len(expected_tokens & set(got.split())) / max(1, len(expected_tokens))
        return ScoreResult(overlap >= 0.8, round(overlap, 3), method="token_overlap")


def _extract_pair(row: dict[str, Any], template: str) -> tuple[str, str]:
    if "messages" in row and isinstance(row["messages"], list):
        user = next(
            (
                m.get("content")
                for m in row["messages"]
                if isinstance(m, dict) and m.get("role") == "user"
            ),
            "",
        )
        assistant = next(
            (
                m.get("content")
                for m in reversed(row["messages"])
                if isinstance(m, dict) and m.get("role") == "assistant"
            ),
            "",
        )
        return str(user or ""), str(assistant or "")

    question = next((str(row[k]) for k in QUESTION_KEYS if row.get(k)), "")
    if "instruction" in row and row.get("input"):
        question = f"{row['instruction']}\n\n{row['input']}"
    expected = next((str(row[k]) for k in ANSWER_KEYS if row.get(k)), "")
    return question, expected
