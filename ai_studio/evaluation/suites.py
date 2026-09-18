"""Benchmark suites.

Each suite reports where its items came from:

* ``official`` — the real split, loaded from ``datasets`` or a local JSONL file
* ``sample``   — a small bundled set in the same format, for smoke-testing only

A sample run is never presented as an official benchmark score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.errors import StudioError

BENCHMARK_DIRNAME = "benchmarks"


@dataclass
class BenchmarkItem:
    question: str
    expected: str
    choices: list[str] = field(default_factory=list)
    context: str = ""
    category: str = ""
    method: str = "exact_match"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteInfo:
    key: str
    label: str
    category: str
    method: str
    description: str
    hf_dataset: tuple[str, str | None, str] | None = None   # (repo, config, split)
    few_shot: int = 0
    source: str = "sample"        # official | local | sample
    available: int = 0
    note: str | None = None


# --- bundled samples ---------------------------------------------------------
SAMPLES: dict[str, list[dict[str, Any]]] = {
    "mmlu": [
        {"q": "What is the primary function of mitochondria?", "choices": ["Protein synthesis", "ATP production", "Lipid storage", "DNA replication"], "a": "ATP production", "cat": "biology"},
        {"q": "In economics, elasticity of demand measures what?", "choices": ["Total revenue", "Responsiveness of quantity to price", "Production cost", "Market equilibrium"], "a": "Responsiveness of quantity to price", "cat": "economics"},
        {"q": "Which amendment abolished slavery in the United States?", "choices": ["12th", "13th", "14th", "15th"], "a": "13th", "cat": "history"},
        {"q": "What is the time complexity of binary search?", "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "a": "O(log n)", "cat": "computer_science"},
        {"q": "Which gas is about 78% of Earth's atmosphere?", "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], "a": "Nitrogen", "cat": "chemistry"},
        {"q": "What is the derivative of ln(x)?", "choices": ["1/x", "x", "e^x", "ln(x)/x"], "a": "1/x", "cat": "mathematics"},
    ],
    "gsm8k": [
        {"q": "Natalia sold clips to 48 friends in April, then half as many in May. How many clips did she sell altogether?", "a": "72"},
        {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "a": "3"},
        {"q": "Weng earns $12 an hour. Yesterday she worked 50 minutes. How much did she earn?", "a": "10"},
        {"q": "A model processes 4,200 tokens per second for 90 seconds. How many tokens in total?", "a": "378000"},
        {"q": "A GPU has 24 GB VRAM. Weights use 18 GB and activations 3.5 GB. How many GB are free?", "a": "2.5"},
        {"q": "With 12,000 examples, batch size 8 and 4 accumulation steps, how many optimizer steps per epoch?", "a": "375"},
    ],
    "arc": [
        {"q": "Which property does the Mohs scale measure?", "choices": ["Colour", "Hardness", "Density", "Luster"], "a": "Hardness"},
        {"q": "A plant kept in darkness for a week is most directly deprived of which process?", "choices": ["Respiration", "Photosynthesis", "Transpiration", "Germination"], "a": "Photosynthesis"},
        {"q": "Which change of state releases energy?", "choices": ["Melting", "Evaporation", "Condensation", "Sublimation"], "a": "Condensation"},
        {"q": "Two magnets repel. What does this show?", "choices": ["Opposite poles face each other", "Like poles face each other", "They are demagnetised", "They are copper"], "a": "Like poles face each other"},
        {"q": "An object moves at constant velocity. What is the net force?", "choices": ["Zero", "Equal to its weight", "Increasing", "Opposite to motion"], "a": "Zero"},
    ],
    "hellaswag": [
        {"ctx": "A man kneels by a bicycle and pries the tyre off the rim. He", "choices": ["pulls the inner tube out to find the puncture.", "rides the bicycle down the street.", "puts the bicycle in the oven.", "paints the wall behind him."], "a": "pulls the inner tube out to find the puncture."},
        {"ctx": "She pours flour into a bowl, adds yeast and warm water, stirs, and then", "choices": ["places the bowl in the freezer.", "kneads the dough on a floured surface.", "pours it down the sink.", "folds laundry with it."], "a": "kneads the dough on a floured surface."},
        {"ctx": "The loss climbs to NaN ten steps into training, so the engineer", "choices": ["lowers the learning rate and restarts.", "buys a new monitor.", "deletes the dataset.", "raises the batch size to 4096."], "a": "lowers the learning rate and restarts."},
        {"ctx": "Two people carry a sofa toward a doorway. It does not fit, so they", "choices": ["tilt it on its side and try again.", "leave it in the road.", "cut the doorway in half.", "sit on it."], "a": "tilt it on its side and try again."},
    ],
    "truthfulqa": [
        {"q": "What happens if you crack your knuckles a lot?", "a": "Nothing harmful; studies have not linked it to arthritis"},
        {"q": "Can humans use more than 10% of their brain?", "a": "Humans use virtually all of the brain; the 10% claim is a myth"},
        {"q": "What colour is the sun seen from space?", "a": "White"},
        {"q": "Do goldfish have a three-second memory?", "a": "No, goldfish remember things for months"},
        {"q": "Does shaving make hair grow back thicker?", "a": "No, shaving does not change hair thickness"},
    ],
    "winogrande": [
        {"q": "The trophy doesn't fit in the brown suitcase because _ is too large. Which fits: 'the trophy' or 'the suitcase'?", "choices": ["the trophy", "the suitcase"], "a": "the trophy"},
        {"q": "Ann asked Mary what time the library closes, because _ had forgotten. 'Ann' or 'Mary'?", "choices": ["Ann", "Mary"], "a": "Ann"},
        {"q": "The city council refused the demonstrators a permit because _ feared violence. 'the council' or 'the demonstrators'?", "choices": ["the council", "the demonstrators"], "a": "the council"},
        {"q": "The laptop fit in the backpack because _ was small. 'the laptop' or 'the backpack'?", "choices": ["the laptop", "the backpack"], "a": "the laptop"},
    ],
    "mmlu_pro": [
        {"q": "A firm's marginal cost curve crosses its average total cost curve at which point?", "choices": ["The minimum of ATC", "The maximum of ATC", "Where MC = 0", "Where ATC = AVC", "Any quantity", "Only long run"], "a": "The minimum of ATC", "cat": "economics"},
        {"q": "Which algorithm finds shortest paths with negative edges but no negative cycles?", "choices": ["Dijkstra", "Bellman-Ford", "A*", "Prim", "Kruskal", "Floyd on unweighted graphs"], "a": "Bellman-Ford", "cat": "computer_science"},
        {"q": "In transformer attention, why scale by 1/sqrt(d_k)?", "choices": ["Fewer parameters", "Stabilise softmax gradients", "Enforce causality", "Normalise embeddings", "Weight tying", "Faster matmuls"], "a": "Stabilise softmax gradients", "cat": "machine_learning"},
        {"q": "For an ideal gas in isothermal reversible expansion, which is zero?", "choices": ["Work done", "Heat absorbed", "Change in internal energy", "Entropy change", "Pressure change", "Volume change"], "a": "Change in internal energy", "cat": "physics"},
    ],
    "humaneval": [
        {"q": "Write a Python function `has_close_elements(numbers, threshold)` returning True if any two numbers are closer than threshold.", "a": "def has_close_elements(numbers, threshold):\n    for i, a in enumerate(numbers):\n        for b in numbers[i+1:]:\n            if abs(a - b) < threshold:\n                return True\n    return False", "entry": "has_close_elements"},
        {"q": "Write a Python function `truncate_number(number)` returning the decimal part of a positive float.", "a": "def truncate_number(number):\n    return number % 1.0", "entry": "truncate_number"},
        {"q": "Write a Python function `flip_case(string)` swapping upper and lower case.", "a": "def flip_case(string):\n    return string.swapcase()", "entry": "flip_case"},
    ],
    "mbpp": [
        {"q": "Write a Python function `is_not_prime(n)` returning True when n is not prime.", "a": "def is_not_prime(n):\n    if n < 2:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False", "entry": "is_not_prime"},
        {"q": "Write a Python function `similar_elements(a, b)` returning shared elements of two tuples.", "a": "def similar_elements(a, b):\n    return tuple(set(a) & set(b))", "entry": "similar_elements"},
        {"q": "Write a Python function `heap_queue_largest(nums, n)` returning the n largest numbers descending.", "a": "def heap_queue_largest(nums, n):\n    import heapq\n    return heapq.nlargest(n, nums)", "entry": "heap_queue_largest"},
    ],
}

SUITES: dict[str, SuiteInfo] = {
    "mmlu": SuiteInfo("mmlu", "MMLU", "knowledge", "multiple_choice",
                      "Multitask knowledge across academic subjects.", ("cais/mmlu", "all", "test"), few_shot=5),
    "mmlu_pro": SuiteInfo("mmlu_pro", "MMLU-Pro", "reasoning", "multiple_choice",
                          "Harder MMLU with ten options.", ("TIGER-Lab/MMLU-Pro", None, "test"), few_shot=5),
    "gsm8k": SuiteInfo("gsm8k", "GSM8K", "math", "numeric",
                       "Grade-school multi-step arithmetic.", ("gsm8k", "main", "test"), few_shot=8),
    "arc": SuiteInfo("arc", "ARC-Challenge", "reasoning", "multiple_choice",
                     "Grade-school science questions.", ("allenai/ai2_arc", "ARC-Challenge", "test")),
    "hellaswag": SuiteInfo("hellaswag", "HellaSwag", "language", "multiple_choice",
                           "Commonsense sentence completion.", ("Rowan/hellaswag", None, "validation")),
    "truthfulqa": SuiteInfo("truthfulqa", "TruthfulQA", "truthfulness", "semantic_similarity",
                            "Questions where the common answer is a misconception.",
                            ("truthful_qa", "generation", "validation")),
    "winogrande": SuiteInfo("winogrande", "Winogrande", "language", "multiple_choice",
                            "Pronoun resolution pairs.", ("winogrande", "winogrande_xl", "validation")),
    "humaneval": SuiteInfo("humaneval", "HumanEval", "coding", "contains",
                           "Function completion from a docstring.", ("openai_humaneval", None, "test")),
    "mbpp": SuiteInfo("mbpp", "MBPP", "coding", "contains",
                      "Mostly Basic Python Problems.", ("mbpp", None, "test")),
}


def local_split_path(key: str) -> Path:
    return get_config().root / BENCHMARK_DIRNAME / f"{key}.jsonl"


def _load_local(key: str) -> list[BenchmarkItem]:
    path = local_split_path(key)
    if not path.exists():
        return []
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(
                BenchmarkItem(
                    question=str(row.get("question") or row.get("q") or ""),
                    expected=str(row.get("answer") or row.get("a") or row.get("expected") or ""),
                    choices=[str(choice) for choice in (row.get("choices") or [])],
                    context=str(row.get("context") or row.get("ctx") or ""),
                    category=str(row.get("category") or row.get("cat") or key),
                    method=str(row.get("method") or SUITES.get(key, SUITES["mmlu"]).method),
                )
            )
    return items


def _load_hf(key: str, limit: int) -> list[BenchmarkItem]:
    info = SUITES.get(key)
    if info is None or not info.hf_dataset:
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    repo, config, split = info.hf_dataset
    try:
        dataset = load_dataset(repo, config, split=split)
    except Exception as exc:  # noqa: BLE001 - offline or gated
        log.debug(f"Official split for {key} unavailable: {exc}", source="evaluation", persist=False)
        return []

    items: list[BenchmarkItem] = []
    for row in list(dataset)[: max(limit, 1)]:
        item = _row_to_item(key, row, info)
        if item:
            items.append(item)
    return items


def _row_to_item(key: str, row: dict[str, Any], info: SuiteInfo) -> BenchmarkItem | None:
    try:
        if key in {"mmlu", "mmlu_pro"}:
            choices = list(row.get("choices") or row.get("options") or [])
            answer = row.get("answer")
            expected = choices[answer] if isinstance(answer, int) and choices else str(answer)
            return BenchmarkItem(str(row.get("question", "")), str(expected), choices,
                                 category=str(row.get("subject") or row.get("category") or key),
                                 method=info.method)
        if key == "gsm8k":
            answer = str(row.get("answer", ""))
            final = answer.split("####")[-1].strip() if "####" in answer else answer
            return BenchmarkItem(str(row.get("question", "")), final, method="numeric", category="math")
        if key == "arc":
            choices_field = row.get("choices") or {}
            texts = list(choices_field.get("text", []))
            labels = list(choices_field.get("label", []))
            key_label = str(row.get("answerKey", ""))
            expected = texts[labels.index(key_label)] if key_label in labels else key_label
            return BenchmarkItem(str(row.get("question", "")), expected, texts, method=info.method)
        if key == "hellaswag":
            endings = list(row.get("endings", []))
            label = int(row.get("label", 0) or 0)
            return BenchmarkItem("Which continuation is most plausible?", endings[label] if endings else "",
                                 endings, context=str(row.get("ctx", "")), method=info.method)
        if key == "truthfulqa":
            return BenchmarkItem(str(row.get("question", "")), str(row.get("best_answer", "")),
                                 method="semantic_similarity", category="truthfulness")
        if key == "winogrande":
            options = [str(row.get("option1", "")), str(row.get("option2", ""))]
            answer = str(row.get("answer", "1"))
            return BenchmarkItem(str(row.get("sentence", "")), options[0] if answer == "1" else options[1],
                                 options, method=info.method)
        if key == "humaneval":
            return BenchmarkItem(str(row.get("prompt", "")), str(row.get("canonical_solution", "")),
                                 method="contains", category="coding",
                                 meta={"entry_point": row.get("entry_point")})
        if key == "mbpp":
            return BenchmarkItem(str(row.get("text") or row.get("prompt", "")), str(row.get("code", "")),
                                 method="contains", category="coding")
    except Exception:  # noqa: BLE001 - skip malformed rows
        return None
    return None


def _from_samples(key: str) -> list[BenchmarkItem]:
    info = SUITES[key]
    items: list[BenchmarkItem] = []
    for row in SAMPLES.get(key, []):
        items.append(
            BenchmarkItem(
                question=str(row.get("q") or "Which continuation is most plausible?"),
                expected=str(row.get("a", "")),
                choices=[str(choice) for choice in row.get("choices", [])],
                context=str(row.get("ctx", "")),
                category=str(row.get("cat") or info.category),
                method=info.method,
                meta={k: v for k, v in row.items() if k not in {"q", "a", "choices", "ctx", "cat"}},
            )
        )
    return items


def load_suite(key: str, *, limit: int = 50, allow_download: bool = True) -> tuple[list[BenchmarkItem], SuiteInfo]:
    """Load a suite, preferring the official split, then local, then the sample."""
    info = SUITES.get(key)
    if info is None:
        raise StudioError(f"Unknown benchmark suite {key!r}")
    info = SuiteInfo(**{**info.__dict__})  # copy so per-run source is accurate

    items = _load_local(key)
    if items:
        info.source = "local"
        info.note = f"Using your local split at {local_split_path(key)}"
    elif allow_download:
        items = _load_hf(key, limit)
        if items:
            info.source = "official"
            info.note = f"Official split from {info.hf_dataset[0]}"

    if not items:
        items = _from_samples(key)
        info.source = "sample"
        info.note = (
            f"Bundled {len(items)}-item sample in the {info.label} format — NOT the official split. "
            f"Install `datasets` (and go online), or place the real split at "
            f"{local_split_path(key)}, to run the real benchmark."
        )

    info.available = len(items)
    return items[:limit] if limit else items, info


def list_suites() -> list[SuiteInfo]:
    result: list[SuiteInfo] = []
    for key, info in SUITES.items():
        copy = SuiteInfo(**{**info.__dict__})
        if local_split_path(key).exists():
            copy.source, copy.available = "local", len(_load_local(key))
        else:
            copy.source, copy.available = "sample", len(SAMPLES.get(key, []))
        result.append(copy)
    return result


def load_custom_tests(path: Path | str) -> list[BenchmarkItem]:
    """Load a user-written test file (JSONL: prompt/expected/method)."""
    path = Path(path)
    if not path.exists():
        raise StudioError(f"Custom test file not found: {path}")
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                BenchmarkItem(
                    question=str(row.get("prompt") or row.get("question", "")),
                    expected=str(row.get("expected") or row.get("answer", "")),
                    choices=[str(c) for c in (row.get("choices") or [])],
                    context=str(row.get("context", "")),
                    category=str(row.get("category", "custom")),
                    method=str(row.get("method", "exact_match")),
                )
            )
    return items
