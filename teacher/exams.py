"""Sitting a model down in front of a real benchmark.

The suites, their items and the grading all come from ``ai_studio.evaluation``
— this module is what stands between a folder-shaped Teacher model and that
machinery, and what makes the result honest to read.

Three things it insists on:

* **Where the questions came from.** A suite falls back to a handful of bundled
  example items when the real split cannot be loaded. Six questions is a check
  that the plumbing works, not a benchmark score, and it is labelled as such.
* **What chance looks like.** A four-choice question is 25% for a model that
  has learned nothing. Printing 25% without that number beside it invites
  exactly the wrong conclusion.
* **Whether the result means anything.** With twenty items, a score has to
  clear roughly two standard deviations of the chance rate before it is
  evidence of anything at all. Below that this says so.
"""

from __future__ import annotations

import math
import time

from ai_studio.core.errors import StudioError
from ai_studio.evaluation.benchmark_runner import build_prompt
from ai_studio.evaluation.evaluators import evaluate
from ai_studio.evaluation.suites import BenchmarkItem, list_suites, load_suite
from teacher import lessons
from teacher.workspace import Model, TeacherError

#: How much to let it write. A letter needs a few tokens; a worked answer or a
#: function needs room. Generating 256 tokens per item for a multiple-choice
#: question is most of the runtime for none of the benefit.
ROOM = {
    "multiple_choice": 8,
    "numeric": 96,
    "exact_match": 32,
    "contains": 160,
    "semantic_similarity": 96,
}


def catalogue() -> list[dict]:
    """Every suite, with where its items would come from right now."""
    return [
        {
            "suite": info.key,
            "label": info.label,
            "category": info.category,
            "method": info.method,
            "description": info.description,
            "source": info.source,
            "available": info.available,
        }
        for info in list_suites()
    ]


def chance_rate(items: list[BenchmarkItem]) -> float | None:
    """What a model that knows nothing scores, or None when there is no such number."""
    spreads = [len(item.choices) for item in items if item.choices]
    if not spreads or len(spreads) != len(items):
        return None
    return sum(1 / count for count in spreads) / len(spreads)


def beats_chance(correct: int, total: int, chance: float | None) -> bool | None:
    """Is this score distinguishable from guessing?

    A binomial standard deviation, two of them. Not a p-value — just the line
    below which a number is noise and should not be read as a result.
    """
    if chance is None or not total:
        return None
    expected = chance * total
    spread = math.sqrt(total * chance * (1 - chance)) or 1e-9
    return correct > expected + 2 * spread


def sit(
    model: Model,
    suite: str,
    *,
    limit: int = 20,
    few_shot: int | None = None,
    allow_download: bool = True,
    on_item=None,
    record: bool = True,
) -> dict:
    """Run one suite against one model and report what happened.

    Nothing here is estimated. Every item is generated and graded, and the
    count of what was right is the count of what was right. The result goes
    into the model's record, tied to the weights it was taken on, so a model
    card can show it — and can leave it out once those weights are gone.
    """
    try:
        items, info = load_suite(suite, limit=limit + (few_shot or 0), allow_download=allow_download)
    except StudioError as exc:
        raise TeacherError(exc.display()) from exc
    if not items:
        raise TeacherError(f"The {suite} suite has no items to run.")

    shots_wanted = info.few_shot if few_shot is None else int(few_shot)
    shots = items[:shots_wanted] if shots_wanted else []
    questions = items[len(shots):][:limit]
    if not questions:
        raise TeacherError(
            f"After {len(shots)} worked examples there were no questions left. "
            f"Raise --items, or lower --shots."
        )

    started = time.time()
    results: list[dict] = []
    correct = 0

    for index, item in enumerate(questions, start=1):
        prompt = build_prompt(item, shots)
        method = item.method or info.method
        reply, _stats = lessons.talk(
            model, prompt,
            max_new_tokens=ROOM.get(method, 64),
            temperature=0.0,          # greedy, so the run repeats exactly
            top_k=0, top_p=1.0, seed=0,
        )
        score = evaluate(method, reply, item.expected, choices=item.choices)
        correct += bool(score.correct)
        results.append({
            "question": item.question[:160],
            "expected": item.expected[:80],
            "answered": reply.strip()[:80],
            "correct": bool(score.correct),
            "category": item.category,
        })
        if on_item:
            on_item(index, len(questions), bool(score.correct))

    chance = chance_rate(questions)
    accuracy = correct / len(questions)
    outcome = {
        "model": model.name,
        "suite": info.key,
        "label": info.label,
        "method": info.method,
        "items": len(questions),
        "shots": len(shots),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "chance": round(chance, 4) if chance is not None else None,
        "beats_chance": beats_chance(correct, len(questions), chance),
        "source": info.source,
        "official": info.source in ("official", "local"),
        "note": info.note,
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    if record:
        keep(model, outcome)
    return outcome


def keep(model: Model, outcome: dict) -> dict:
    """Write a result into the model's record — the score, not every answer."""
    entry = {key: value for key, value in outcome.items() if key not in ("results", "model")}
    entry["at"] = time.time()
    entry["weights"] = model.weights_stamp()
    model.note(exams=[*model.history().get("exams", []), entry])
    return entry


def verdict(outcome: dict) -> str:
    """One sentence a person can act on, rather than a number to misread."""
    if not outcome["official"]:
        return ("This ran on the bundled example items, not the real split — it "
                "shows the plumbing works and nothing more.")
    if outcome["beats_chance"] is False:
        return ("Indistinguishable from guessing. A model this size is expected "
                "to score at chance on this; it is not a fault.")
    if outcome["beats_chance"] is None:
        return "There is no chance rate to compare against, so read the answers."
    return "Better than guessing, by enough items to mean something."
