"""Scoring methods.

Each evaluator returns a score in [0, 1] plus an explanation, so a benchmark
table can always show *why* an answer was marked right or wrong.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

EVALUATION_METHODS = (
    "exact_match",
    "contains",
    "regex",
    "multiple_choice",
    "numeric",
    "semantic_similarity",
    "llm_judge",
)

ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCTUATION = str.maketrans("", "", string.punctuation)


@dataclass
class Score:
    score: float
    correct: bool
    method: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "correct": self.correct, "method": self.method, "detail": self.detail}


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^(the\s+)?answer\s*(is)?\s*[:\-]?\s*", "", text)
    text = text.translate(PUNCTUATION)
    text = ARTICLES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(response: str, expected: str, **_: Any) -> Score:
    ok = normalize(response) == normalize(expected)
    return Score(1.0 if ok else 0.0, ok, "exact_match")


def contains(response: str, expected: str, **_: Any) -> Score:
    ok = normalize(expected) in normalize(response)
    return Score(1.0 if ok else 0.0, ok, "contains")


def regex_match(response: str, expected: str, *, pattern: str | None = None, **_: Any) -> Score:
    source = pattern or expected
    try:
        ok = bool(re.search(source, response or "", re.IGNORECASE | re.MULTILINE))
    except re.error as exc:
        return Score(0.0, False, "regex", f"Invalid pattern: {exc}")
    return Score(1.0 if ok else 0.0, ok, "regex", f"pattern: {source}")


def multiple_choice(response: str, expected: str, *, choices: list[str] | None = None, **_: Any) -> Score:
    """Accept an option letter, the option text, or the letter in a sentence."""
    choices = choices or []
    response = (response or "").strip()
    picked: str | None = None

    if choices:
        letters = [chr(65 + index) for index in range(len(choices))]
        match = re.search(r"\b([A-Z])\b[\).:]?", response[:160])
        if match and match.group(1) in letters:
            picked = choices[letters.index(match.group(1))]
        if picked is None:
            lowered = response.lower()
            for choice in choices:
                if choice and choice.lower() in lowered:
                    picked = choice
                    break

    expected_text = expected
    if choices and len(expected) == 1 and expected.upper() in [chr(65 + i) for i in range(len(choices))]:
        expected_text = choices[ord(expected.upper()) - 65]

    if picked is None:
        ok = normalize(response) == normalize(expected_text)
        return Score(1.0 if ok else 0.0, ok, "multiple_choice", "no option detected in the response")
    ok = normalize(picked) == normalize(expected_text)
    return Score(1.0 if ok else 0.0, ok, "multiple_choice", f"selected: {picked}")


def numeric_match(response: str, expected: str, *, tolerance: float = 1e-6, **_: Any) -> Score:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", (response or "").replace("$", ""))
    target = re.findall(r"-?\d[\d,]*\.?\d*", (expected or "").replace("$", ""))
    if not numbers or not target:
        return Score(0.0, False, "numeric", "no number found")
    try:
        got = float(numbers[-1].replace(",", "").rstrip("."))
        want = float(target[-1].replace(",", "").rstrip("."))
    except ValueError:
        return Score(0.0, False, "numeric", "unparseable number")
    ok = abs(got - want) <= max(tolerance, abs(want) * 1e-6)
    return Score(1.0 if ok else 0.0, ok, "numeric", f"got {got}, expected {want}")


def semantic_similarity(
    response: str, expected: str, *, threshold: float = 0.75, embedding_model: str | None = None, **_: Any
) -> Score:
    """Cosine similarity between sentence embeddings (needs sentence-transformers)."""
    from ai_studio.rag.embeddings import DEFAULT_MODEL, get_embedding_model

    model = get_embedding_model(embedding_model or DEFAULT_MODEL)
    vectors = model.encode([response or "", expected or ""])
    similarity = float(vectors[0] @ vectors[1])
    return Score(
        max(0.0, min(1.0, similarity)),
        similarity >= threshold,
        "semantic_similarity",
        f"cosine {similarity:.3f} (threshold {threshold})",
    )


def llm_judge(
    response: str,
    expected: str,
    *,
    question: str = "",
    judge_model_id: str | None = None,
    **_: Any,
) -> Score:
    """Ask a local model to grade. Only runs when a judge model is configured."""
    if not judge_model_id:
        return Score(0.0, False, "llm_judge", "No judge model configured — item not graded.")

    from ai_studio.inference.generator import GenerationSettings, generate

    prompt = (
        "You are grading an answer. Reply with only CORRECT or INCORRECT.\n\n"
        f"Question: {question}\nReference answer: {expected}\nCandidate answer: {response}\n\nVerdict:"
    )
    try:
        verdict, _ = generate(
            prompt,
            model_id=judge_model_id,
            settings=GenerationSettings(temperature=0.0, max_new_tokens=8),
        )
    except Exception as exc:  # noqa: BLE001
        return Score(0.0, False, "llm_judge", f"Judge failed: {exc}")
    ok = "correct" in verdict.lower() and "incorrect" not in verdict.lower()
    return Score(1.0 if ok else 0.0, ok, "llm_judge", f"judge said: {verdict.strip()[:60]}")


EVALUATORS = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex_match,
    "multiple_choice": multiple_choice,
    "numeric": numeric_match,
    "semantic_similarity": semantic_similarity,
    "llm_judge": llm_judge,
}


def evaluate(method: str, response: str, expected: str, **kwargs: Any) -> Score:
    evaluator = EVALUATORS.get(method)
    if evaluator is None:
        return Score(0.0, False, method, f"Unknown evaluation method {method!r}")
    try:
        return evaluator(response, expected, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a scoring failure is reported, not fatal
        return Score(0.0, False, method, f"Evaluator error: {exc}")
