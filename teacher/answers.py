"""Teaching a model to answer, rather than to carry on.

A model trained on plain text learns to continue it. Give it the first half of
a sentence and you get the second half — which is what a language model is, and
not what most people mean when they ask a question.

Instruction tuning is the difference. It trains on *pairs* — a question and its
answer, an instruction and what to do — and takes the loss **only on the
answer**. The model is never rewarded for reproducing the question; it is
rewarded for what follows it. That masking is what turns a text continuer into
something that replies.

The templates here are AI Studio's, not new ones, because the template used at
training has to be the one used at inference. A model taught with
``### Question:`` and then asked something bare will produce nonsense, so the
style is written into the model's history when it is taught and read back when
it is asked.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai_studio.data.dataset_builder import record_to_text
from ai_studio.training.data import SequenceDataset, _prompt_length
from teacher.workspace import Model, TeacherError

#: The shapes a pair can arrive in, and what each is called underneath.
SHAPES = {
    "instruction": ("instruction", "output"),
    "qa": ("question", "answer"),
    "chat": ("messages", None),
}

#: What to say to a model taught in each style, so the prompt matches training.
OPENERS = {
    "instruction": "### Instruction:\n{prompt}\n\n### Response:\n",
    "qa": "### Question:\n{prompt}\n\n### Answer:\n",
    "chat": "<|user|>\n{prompt}\n<|assistant|>\n",
}

ALIASES = {
    "instruction": ("instruction", "prompt", "task"),
    "input": ("input", "context_input"),
    "output": ("output", "response", "completion"),
    "question": ("question", "q"),
    "answer": ("answer", "a"),
    "context": ("context", "ctx"),
}


def _pick(row: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise(row: Any) -> dict | None:
    """Turn one row of a file into a record in a shape the templates know."""
    if not isinstance(row, dict):
        return None

    messages = row.get("messages") or row.get("conversation")
    if isinstance(messages, list) and messages:
        kept = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
            if isinstance(m, dict) and str(m.get("content", "")).strip()
        ]
        return {"messages": kept} if len(kept) >= 2 else None

    instruction = _pick(row, ALIASES["instruction"])
    output = _pick(row, ALIASES["output"])
    if instruction and output:
        return {"instruction": instruction, "input": _pick(row, ALIASES["input"]),
                "output": output}

    question = _pick(row, ALIASES["question"])
    answer = _pick(row, ALIASES["answer"])
    if question and answer:
        return {"question": question, "answer": answer, "context": _pick(row, ALIASES["context"])}
    return None


def read_pairs(paths: list[str]) -> tuple[list[dict], str, list[tuple[str, str]]]:
    """Read question-and-answer pairs from files. Returns (records, style, skipped)."""
    records: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for entry in paths:
        root = Path(entry).expanduser()
        if not root.exists():
            skipped.append((entry, "no such file or folder"))
            continue
        files = sorted(p for p in (root.rglob("*") if root.is_dir() else [root])
                       if p.is_file() and p.suffix.lower() in (".jsonl", ".ndjson", ".json", ".csv", ".tsv"))
        if not files:
            skipped.append((entry, "no .jsonl, .json or .csv inside"))
            continue

        for path in files:
            try:
                rows = _rows_of(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a lesson
                skipped.append((str(path), str(exc)))
                continue
            usable = [record for record in (_normalise(row) for row in rows) if record]
            if not usable:
                skipped.append((str(path), "no question-and-answer pairs found in it"))
                continue
            records.extend(usable)

    if not records:
        raise TeacherError(
            "No question-and-answer pairs were found.\n"
            "  Each row needs a question and its answer. Any of these work:\n"
            '    {"instruction": "...", "output": "..."}\n'
            '    {"question": "...", "answer": "..."}\n'
            '    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}\n'
            "  as JSONL, JSON or CSV with those column names."
            + ("\n  Looked at: " + "; ".join(f"{p} — {why}" for p, why in skipped[:4])
               if skipped else "")
        )
    return records, style_of(records), skipped


def _rows_of(path: Path) -> list[Any]:
    if path.suffix.lower() in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))

    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "train", "items", "conversations"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def style_of(records: list[dict]) -> str:
    """Which template these records are written for."""
    if any("messages" in record for record in records):
        return "chat"
    if any("question" in record for record in records):
        return "qa"
    return "instruction"


def summarise(records: list[dict], style: str) -> str:
    characters = sum(len(record_to_text(record, style)) for record in records)
    return f"{len(records):,} pair(s), {characters:,} characters, {style} style"


def build_dataset(records: list[dict], tokenizer, *, style: str, max_length: int):
    """Tokenize the pairs, masking each prompt so only the answer is learned."""
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id or 0

    sequences: list[list[int]] = []
    prompts: list[int] = []
    for record in records:
        text = record_to_text(record, style)
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"][:max_length]
        if len(ids) < 2:
            continue
        sequences.append(ids)
        prompts.append(_prompt_length(record, style, tokenizer, ""))

    if not sequences:
        raise TeacherError("Every pair was empty once tokenized.")
    return SequenceDataset(sequences, pad, max_length, prompts)


def opener(model: Model, prompt: str) -> str:
    """Wrap a question the way this model was taught to receive one.

    A model taught with "### Question:" and then asked something bare will
    answer as if continuing a document, which looks like the training failed
    when it did not.
    """
    style = model.history().get("answer_style")
    if not style or style not in OPENERS:
        return prompt
    return OPENERS[style].format(prompt=prompt)
