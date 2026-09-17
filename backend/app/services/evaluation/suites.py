"""Built-in benchmark adapters.

Each adapter prefers a locally provided split at
``<data_dir>/benchmarks/<key>.jsonl`` (fields: ``question``/``answer`` plus an
optional ``choices`` array). When such a file exists the run is reported as
using a local dataset; otherwise the bundled sample pool is used and the run is
explicitly labelled as a sample, never as the official benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.db.models.enums import BenchmarkCategory
from app.services.evaluation import samples
from app.services.evaluation.base import BenchmarkAdapter, BenchmarkItemSpec, ScoreResult


def _local_split(key: str) -> list[dict[str, Any]]:
    path = Path(settings.data_dir) / "benchmarks" / f"{key}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class _PoolAdapter(BenchmarkAdapter):
    """Common behaviour for the bundled-sample suites."""

    pool: list[dict[str, Any]] = []

    def __init__(self) -> None:
        local = _local_split(self.key)
        self._local = local
        if local:
            self.data_source = "local_split"
            self.official = True
            self.notes = (
                f"Using the local split at data/benchmarks/{self.key}.jsonl ({len(local)} items)."
            )
        else:
            self.data_source = "bundled_sample"
            self.official = False
            self.notes = (
                "Bundled offline sample — representative of the task format, but NOT the "
                f"official {self.label} split. Drop the real split at "
                f"data/benchmarks/{self.key}.jsonl to run it for real."
            )

    def _raw_items(self) -> list[BenchmarkItemSpec]:
        if self._local:
            return [
                BenchmarkItemSpec(
                    index=i,
                    question=str(row.get("question") or row.get("q") or ""),
                    expected=str(row.get("answer") or row.get("a") or ""),
                    choices=[str(c) for c in (row.get("choices") or [])],
                    context=row.get("context"),
                    category=str(row.get("category") or self.category),
                )
                for i, row in enumerate(self._local)
            ]
        return self._pool_items()

    def _pool_items(self) -> list[BenchmarkItemSpec]:
        return [
            BenchmarkItemSpec(
                index=i,
                question=str(row.get("q", "")),
                expected=str(row.get("a", "")),
                choices=[str(c) for c in row.get("choices", [])],
                context=row.get("ctx"),
                category=str(row.get("cat") or self.category),
                metadata={
                    k: v for k, v in row.items() if k not in {"q", "a", "choices", "ctx", "cat"}
                },
            )
            for i, row in enumerate(self.pool)
        ]

    def items(self, *, limit: int, seed: int = 42) -> list[BenchmarkItemSpec]:
        return self.sample(self._raw_items(), limit, seed)

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        if item.choices:
            picked = self.extract_choice(response, item.choices)
            correct = picked is not None and self.normalize(picked) == self.normalize(item.expected)
            return ScoreResult(
                correct=correct, score=1.0 if correct else 0.0, method="multiple_choice"
            )
        correct = self.normalize(item.expected) in self.normalize(response)
        return ScoreResult(correct=correct, score=1.0 if correct else 0.0, method="contains")


class MMLUAdapter(_PoolAdapter):
    key, label = "mmlu", "MMLU"
    category = BenchmarkCategory.KNOWLEDGE
    metric = "accuracy"
    default_shots = 5
    description = "Multitask multiple-choice knowledge across 57 academic subjects."
    pool = samples.MMLU


class MMLUProAdapter(_PoolAdapter):
    key, label = "mmlu_pro", "MMLU-Pro"
    category = BenchmarkCategory.REASONING
    metric = "accuracy"
    default_shots = 5
    description = "Harder MMLU variant with 10 options and reasoning-heavy questions."
    pool = samples.MMLU_PRO


class GSM8KAdapter(_PoolAdapter):
    key, label = "gsm8k", "GSM8K"
    category = BenchmarkCategory.MATH
    metric = "exact_match"
    default_shots = 8
    description = "Grade-school word problems requiring multi-step arithmetic."
    pool = samples.GSM8K

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        predicted = self.extract_number(response)
        expected = self.extract_number(item.expected) or item.expected
        if predicted is None:
            return ScoreResult(False, 0.0, method="numeric_match", note="No number found in output")
        try:
            correct = abs(float(predicted) - float(expected)) < 1e-6
        except ValueError:
            correct = predicted.strip() == expected.strip()
        return ScoreResult(correct, 1.0 if correct else 0.0, method="numeric_match")


class ARCAdapter(_PoolAdapter):
    key, label = "arc", "ARC-Challenge"
    category = BenchmarkCategory.REASONING
    metric = "accuracy"
    default_shots = 0
    description = "Grade-school science questions that resist simple retrieval."
    pool = samples.ARC


class HellaSwagAdapter(_PoolAdapter):
    key, label = "hellaswag", "HellaSwag"
    category = BenchmarkCategory.LANGUAGE_UNDERSTANDING
    metric = "accuracy"
    description = "Commonsense sentence completion with adversarial distractors."
    pool = samples.HELLASWAG

    def _pool_items(self) -> list[BenchmarkItemSpec]:
        return [
            BenchmarkItemSpec(
                index=i,
                question="Which continuation is most plausible?",
                context=row["ctx"],
                expected=row["a"],
                choices=list(row["choices"]),
                category=self.category,
            )
            for i, row in enumerate(self.pool)
        ]


class TruthfulQAAdapter(_PoolAdapter):
    key, label = "truthfulqa", "TruthfulQA"
    category = BenchmarkCategory.HALLUCINATION_RESISTANCE
    metric = "truthful_rate"
    description = "Questions where the common answer is a popular misconception."
    pool = samples.TRUTHFULQA

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        normalized = self.normalize(response)
        bad = item.metadata.get("bad")
        key_terms = [t for t in self.normalize(item.expected).split() if len(t) > 4][:4]
        hits = sum(1 for term in key_terms if term in normalized)
        truthful = hits >= max(1, len(key_terms) // 2)
        if bad and self.normalize(str(bad)) in normalized and not truthful:
            return ScoreResult(
                False, 0.0, method="keyword_overlap", note="Matched the misconception"
            )
        score = hits / max(1, len(key_terms))
        return ScoreResult(truthful, round(score, 3), method="keyword_overlap")


class WinograndeAdapter(_PoolAdapter):
    key, label = "winogrande", "Winogrande"
    category = BenchmarkCategory.LANGUAGE_UNDERSTANDING
    metric = "accuracy"
    description = "Pronoun-resolution pairs designed to defeat surface heuristics."
    pool = samples.WINOGRANDE


class BBHAdapter(_PoolAdapter):
    key, label = "bbh", "BIG-Bench Hard"
    category = BenchmarkCategory.REASONING
    metric = "exact_match"
    default_shots = 3
    description = "Tasks where chain-of-thought materially improves accuracy."
    pool = samples.BBH


class LongContextAdapter(_PoolAdapter):
    key, label = "long_context", "Long-context retrieval"
    category = BenchmarkCategory.LONG_CONTEXT
    metric = "exact_match"
    description = "Needle-in-a-haystack style retrieval from long documents."
    pool = samples.LONG_CONTEXT


class InstructionFollowingAdapter(_PoolAdapter):
    key, label = "instruction_following", "Instruction following"
    category = BenchmarkCategory.INSTRUCTION_FOLLOWING
    metric = "constraint_satisfaction"
    description = "Format and constraint compliance — exact output shape matters."
    pool = samples.INSTRUCTION_FOLLOWING

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        rule = item.metadata.get("rule", "exact")
        stripped = response.strip().strip("`").strip()
        if rule == "json":
            try:
                parsed = json.loads(stripped)
                expected = json.loads(item.expected)
                ok = parsed == expected
                return ScoreResult(ok, 1.0 if ok else 0.0, method="json_equality")
            except json.JSONDecodeError:
                return ScoreResult(
                    False, 0.0, method="json_equality", note="Output was not valid JSON"
                )
        ok = stripped == item.expected.strip()
        loose = self.normalize(stripped) == self.normalize(item.expected)
        return ScoreResult(
            ok,
            1.0 if ok else (0.5 if loose else 0.0),
            method="exact_format",
            note=None
            if ok
            else "Content matched but formatting constraint was violated"
            if loose
            else None,
        )


class SafetyAdapter(_PoolAdapter):
    key, label = "safety", "Safety evaluation"
    category = BenchmarkCategory.SAFETY
    metric = "accuracy"
    description = (
        "Policy-judgement items: does the model choose the appropriate handling for a "
        "sensitive request? Not a substitute for a full red-team evaluation."
    )
    pool = samples.SAFETY


class _CodeAdapter(_PoolAdapter):
    """Code suites are graded structurally — never by executing model output."""

    requires_execution = True
    metric = "structural_match"

    def __init__(self) -> None:
        super().__init__()
        self.notes = (
            (self.notes or "")
            + " Execution-based pass@1 is disabled: the platform never runs model-generated "
            "code. Items are graded structurally (signature + key operations), which is a "
            "proxy, not pass@1."
        ).strip()

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        entry = str(item.metadata.get("entry", ""))
        body = response.lower()
        checks = {
            "defines_function": f"def {entry}" in response,
            "returns": "return" in body,
            "no_placeholder": "todo" not in body and "..." not in response,
        }
        reference_tokens = {
            token
            for token in self.normalize(item.expected).split()
            if len(token) > 3 and token not in {"return", "range", "self"}
        }
        overlap = (
            len(reference_tokens & set(self.normalize(response).split())) / len(reference_tokens)
            if reference_tokens
            else 0.0
        )
        score = round(0.6 * (sum(checks.values()) / len(checks)) + 0.4 * min(1.0, overlap), 3)
        return ScoreResult(
            correct=score >= 0.75,
            score=score,
            graded=True,
            method="structural_proxy",
            note="Structural proxy score — not executed pass@1",
        )


class HumanEvalAdapter(_CodeAdapter):
    key, label = "humaneval", "HumanEval"
    category = BenchmarkCategory.CODING
    description = "Function-completion problems from docstring specifications."
    pool = samples.HUMANEVAL


class MBPPAdapter(_CodeAdapter):
    key, label = "mbpp", "MBPP"
    category = BenchmarkCategory.CODING
    description = "Mostly Basic Python Problems — short, self-contained tasks."
    pool = samples.MBPP


BUILTIN_ADAPTERS: list[type[BenchmarkAdapter]] = [
    MMLUAdapter,
    MMLUProAdapter,
    GSM8KAdapter,
    ARCAdapter,
    HellaSwagAdapter,
    TruthfulQAAdapter,
    WinograndeAdapter,
    BBHAdapter,
    HumanEvalAdapter,
    MBPPAdapter,
    LongContextAdapter,
    InstructionFollowingAdapter,
    SafetyAdapter,
]
