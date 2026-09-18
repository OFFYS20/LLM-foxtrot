"""Dataset builder: documents → training datasets.

Produces JSONL splits on disk plus a metadata record. Splits are assigned by
*document*, not by row, so overlapping chunks from one book cannot leak across
train/validation/test.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import ValidationError
from ai_studio.core.paths import dir_size, remove_path, slugify, text_sha256
from ai_studio.data.chunking import Chunk, chunk_by_characters, chunk_by_tokens, deduplicate

DATASET_MODES = (
    "raw_lm",
    "continued_pretraining",
    "instruction",
    "qa",
    "chat",
    "retrieval",
)

MODE_LABELS = {
    "raw_lm": "Raw language modelling",
    "continued_pretraining": "Continued pretraining",
    "instruction": "Instruction tuning",
    "qa": "Question answering",
    "chat": "Conversation / chat",
    "retrieval": "Knowledge retrieval",
}

SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class BuildOptions:
    mode: str = "raw_lm"
    block_size: int = 1024
    overlap: int = 128
    min_chunk_chars: int = 64
    max_chunk_chars: int = 40_000
    train_split: float = 0.90
    validation_split: float = 0.05
    test_split: float = 0.05
    shuffle: bool = True
    seed: int = 42
    deduplicate: bool = True
    near_duplicate_shingle: int = 0
    tokenizer_id: str | None = None
    system_prompt: str = ""
    group_by_document: bool = True

    def validate(self) -> None:
        if self.mode not in DATASET_MODES:
            raise ValidationError(f"Unknown dataset mode {self.mode!r}; expected one of {DATASET_MODES}")
        total = self.train_split + self.validation_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(f"Splits must sum to 1.0 (got {total:.3f})")
        for name, value in (
            ("train_split", self.train_split),
            ("validation_split", self.validation_split),
            ("test_split", self.test_split),
        ):
            if value < 0 or value > 1:
                raise ValidationError(f"{name} must be between 0 and 1")
        if self.block_size < 8:
            raise ValidationError("block_size must be at least 8")
        if self.overlap >= self.block_size:
            raise ValidationError("overlap must be smaller than block_size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildReport:
    documents_used: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    duplicates_removed: int = 0
    invalid_records: int = 0
    rows: dict[str, int] = field(default_factory=dict)
    tokens: int = 0
    token_method: str = "estimated"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"{self.rows.get('train', 0):,} train",
            f"{self.rows.get('validation', 0):,} validation",
            f"{self.rows.get('test', 0):,} test",
            f"~{self.tokens:,} tokens ({self.token_method})",
        ]
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed:,} duplicates removed")
        if self.invalid_records:
            parts.append(f"{self.invalid_records:,} invalid rows skipped")
        return " · ".join(parts)


# --------------------------------------------------------------- validation
def validate_record(record: Any, mode: str) -> str | None:
    """Return an error message, or None when the record is usable."""
    if not isinstance(record, dict):
        return f"expected an object, got {type(record).__name__}"

    if mode in {"raw_lm", "continued_pretraining", "retrieval"}:
        text = record.get("text")
        if not isinstance(text, str):
            return "missing string field 'text'"
        if not text.strip():
            return "'text' is empty"
        return None

    if mode == "instruction":
        for key in ("instruction", "output"):
            value = record.get(key)
            if not isinstance(value, str):
                return f"missing string field '{key}'"
            if not value.strip():
                return f"'{key}' is empty"
        if "input" in record and not isinstance(record["input"], str):
            return "'input' must be a string"
        return None

    if mode == "qa":
        question = record.get("question") or record.get("instruction")
        answer = record.get("answer") or record.get("output")
        if not isinstance(question, str) or not question.strip():
            return "missing 'question'"
        if not isinstance(answer, str) or not answer.strip():
            return "missing 'answer'"
        return None

    if mode == "chat":
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return "'messages' must be a non-empty array"
        roles = set()
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                return f"messages[{position}] is not an object"
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                return f"messages[{position}].role is invalid ({role!r})"
            if not isinstance(content, str) or not content.strip():
                return f"messages[{position}].content is empty"
            roles.add(role)
        if "assistant" not in roles:
            return "conversation has no assistant turn"
        return None

    return None


def validate_records(records: Iterable[Any], mode: str, *, limit: int = 200) -> tuple[int, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    invalid = 0
    for index, record in enumerate(records):
        problem = validate_record(record, mode)
        if problem:
            invalid += 1
            if len(issues) < limit:
                issues.append({"row": index, "error": problem})
    return invalid, issues


# ------------------------------------------------------------------ builders
def _records_from_chunks(chunks: list[Chunk], options: BuildOptions) -> list[dict[str, Any]]:
    mode = options.mode
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        if mode in {"raw_lm", "continued_pretraining"}:
            records.append({"text": text})
        elif mode == "retrieval":
            records.append(
                {
                    "text": text,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.index,
                    "tokens": chunk.token_count,
                }
            )
        elif mode == "instruction":
            # Without a generator model the honest conversion is a completion
            # task: continue the passage. Synthetic Q/A comes from the
            # "generate dataset" flow, which is labelled as synthetic.
            split_at = max(len(text) // 3, 80)
            records.append(
                {
                    "instruction": "Continue the following passage.",
                    "input": text[:split_at].strip(),
                    "output": text[split_at:].strip(),
                }
            )
        elif mode == "qa":
            split_at = max(len(text) // 3, 80)
            records.append(
                {
                    "question": "What does this passage say?",
                    "context": text[:split_at].strip(),
                    "answer": text[split_at:].strip(),
                }
            )
        elif mode == "chat":
            split_at = max(len(text) // 3, 80)
            messages = []
            if options.system_prompt:
                messages.append({"role": "system", "content": options.system_prompt})
            messages.append({"role": "user", "content": text[:split_at].strip()})
            messages.append({"role": "assistant", "content": text[split_at:].strip()})
            records.append({"messages": messages})
    return [record for record in records if validate_record(record, mode) is None]


def build_from_documents(
    name: str,
    document_ids: list[str],
    options: BuildOptions | None = None,
    *,
    description: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Chunk the selected documents and write train/validation/test JSONL."""
    options = options or BuildOptions()
    options.validate()
    if not document_ids:
        raise ValidationError("Select at least one document.")

    from ai_studio.data.ingestion import document_text

    db = get_db()
    tokenizer = None
    token_method = "estimated"
    if options.tokenizer_id:
        from ai_studio.models.tokenizer_manager import load_tokenizer

        tokenizer = load_tokenizer(options.tokenizer_id)
        token_method = "tokenizer"

    report = BuildReport(token_method=token_method)
    grouped: dict[str, list[Chunk]] = {}

    for position, document_id in enumerate(document_ids):
        if progress:
            progress((position + 1) / max(1, len(document_ids)) * 0.6, desc=f"Chunking {position + 1}/{len(document_ids)}")
        try:
            text = document_text(document_id)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the build
            report.documents_skipped += 1
            report.warnings.append(f"{document_id}: {exc}")
            continue
        if not text.strip():
            report.documents_skipped += 1
            continue

        if tokenizer is not None:
            chunks = chunk_by_tokens(
                text,
                tokenizer,
                block_size=options.block_size,
                overlap=options.overlap,
                document_id=document_id,
                min_tokens=max(8, options.block_size // 16),
            )
        else:
            chunks = chunk_by_characters(
                text,
                chunk_size=options.block_size * 4,  # ≈4 chars/token
                overlap=options.overlap * 4,
                document_id=document_id,
                min_chars=options.min_chunk_chars,
            )
        chunks = [c for c in chunks if len(c.text) <= options.max_chunk_chars or True]
        if chunks:
            grouped[document_id] = chunks
            report.documents_used += 1

    all_chunks = [chunk for chunks in grouped.values() for chunk in chunks]
    if not all_chunks:
        raise ValidationError(
            "No usable chunks were produced.",
            hint="The selected documents may be empty, or the block size may exceed their length.",
        )

    if options.deduplicate:
        kept, removed = deduplicate(all_chunks, similarity_shingle=options.near_duplicate_shingle)
        report.duplicates_removed = removed
        surviving = {id(chunk) for chunk in kept}
        grouped = {
            document_id: [c for c in chunks if id(c) in surviving]
            for document_id, chunks in grouped.items()
        }
        grouped = {k: v for k, v in grouped.items() if v}
        all_chunks = kept

    report.chunks_created = len(all_chunks)
    report.tokens = sum(chunk.token_count for chunk in all_chunks)

    if progress:
        progress(0.7, desc="Splitting")
    splits = _split_groups(grouped, options, report)
    split_records = {
        split: _records_from_chunks(chunks, options) for split, chunks in splits.items()
    }

    if options.shuffle:
        rng = random.Random(options.seed)
        for records in split_records.values():
            rng.shuffle(records)

    return _persist(
        name=name,
        mode=options.mode,
        split_records=split_records,
        options=options,
        report=report,
        description=description,
        source_document_ids=list(grouped.keys()),
        progress=progress,
    )


def build_from_records(
    name: str,
    records: list[Any],
    options: BuildOptions | None = None,
    *,
    description: str | None = None,
    is_synthetic: bool = False,
    source_document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a dataset from already-structured records (imported or generated)."""
    options = options or BuildOptions(mode="instruction")
    options.validate()

    report = BuildReport(token_method="estimated")
    valid: list[dict[str, Any]] = []
    for record in records:
        problem = validate_record(record, options.mode)
        if problem:
            report.invalid_records += 1
            if len(report.warnings) < 50:
                report.warnings.append(f"row {len(valid) + report.invalid_records}: {problem}")
            continue
        valid.append(record)

    if not valid:
        raise ValidationError(
            f"No valid {options.mode} records found.",
            hint=f"First problems: {'; '.join(report.warnings[:3])}" if report.warnings else None,
        )

    if options.deduplicate:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for record in valid:
            fingerprint = text_sha256(json.dumps(record, sort_keys=True, default=str))
            if fingerprint in seen:
                report.duplicates_removed += 1
                continue
            seen.add(fingerprint)
            unique.append(record)
        valid = unique

    rng = random.Random(options.seed)
    if options.shuffle:
        rng.shuffle(valid)

    report.chunks_created = len(valid)
    report.tokens = sum(len(json.dumps(record, default=str)) // 4 for record in valid)

    total = len(valid)
    train_end, val_end = _split_points(total, options)
    split_records = {
        "train": valid[:train_end],
        "validation": valid[train_end:val_end],
        "test": valid[val_end:],
    }
    _warn_thin_splits(report, total, options, train_end, val_end, "record")

    return _persist(
        name=name,
        mode=options.mode,
        split_records=split_records,
        options=options,
        report=report,
        description=description,
        source_document_ids=source_document_ids or [],
        is_synthetic=is_synthetic,
    )


def _split_points(total: int, options: BuildOptions) -> tuple[int, int]:
    """Return ``(train_end, validation_end)`` index boundaries for ``total`` rows.

    The three slices never overlap: a row assigned to train can never also
    appear in test. Train always keeps at least one row, and validation/test
    each get one only when their proportion is non-zero and rows remain.
    """
    if total <= 0:
        return 0, 0
    n_validation = round(total * options.validation_split)
    n_test = round(total * options.test_split)
    if options.validation_split > 0:
        n_validation = max(1, n_validation)
    if options.test_split > 0:
        n_test = max(1, n_test)
    while total - n_validation - n_test < 1:
        if n_test > 0:
            n_test -= 1
        elif n_validation > 0:
            n_validation -= 1
        else:
            break
    train_end = total - n_validation - n_test
    return train_end, train_end + n_validation


def _warn_thin_splits(report: BuildReport | None, total: int, options: BuildOptions,
                      train_end: int, val_end: int, unit: str) -> None:
    """Record when the realized split cannot match what was asked for."""
    if report is None:
        return
    missing = []
    if options.validation_split > 0 and val_end == train_end:
        missing.append("validation")
    if options.test_split > 0 and val_end == total:
        missing.append("test")
    if missing:
        report.warnings.append(
            f"Only {total} {unit}(s) available, so the "
            f"{' and '.join(missing)} split is empty. Add more data for a held-out set."
        )


def _split_groups(
    grouped: dict[str, list[Chunk]], options: BuildOptions, report: BuildReport | None = None
) -> dict[str, list[Chunk]]:
    """Assign whole documents to splits so overlapping chunks cannot leak.

    With very few documents an exact proportional split is impossible; the
    realized proportions are reported rather than silently applied.
    """
    splits: dict[str, list[Chunk]] = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(options.seed)
    document_ids = list(grouped)
    rng.shuffle(document_ids)
    count = len(document_ids)

    if not options.group_by_document or count < 3:
        # Not enough documents to hold out whole ones — split by row and say so,
        # because with overlap > 0 adjacent chunks share text across the boundary.
        chunks = [chunk for document_id in document_ids for chunk in grouped[document_id]]
        total = len(chunks)
        train_end, val_end = _split_points(total, options)
        splits["train"] = chunks[:train_end]
        splits["validation"] = chunks[train_end:val_end]
        splits["test"] = chunks[val_end:]
        _warn_thin_splits(report, total, options, train_end, val_end, "chunk")
        if report is not None and options.overlap > 0 and count < 3:
            report.warnings.append(
                f"Only {count} document(s): split row-wise rather than by document, so "
                f"overlapping chunks may appear on both sides of a split boundary. "
                f"Import more documents for a clean held-out set."
            )
        return splits

    want_validation = options.validation_split > 0
    want_test = options.test_split > 0
    n_validation = max(1 if want_validation else 0, round(count * options.validation_split))
    n_test = max(1 if want_test else 0, round(count * options.test_split))
    # Train always keeps at least one document.
    while count - n_validation - n_test < 1:
        if n_test > 0:
            n_test -= 1
        elif n_validation > 0:
            n_validation -= 1
        else:
            break

    test_ids = document_ids[:n_test]
    validation_ids = document_ids[n_test : n_test + n_validation]
    train_ids = document_ids[n_test + n_validation :]

    for document_id in train_ids:
        splits["train"].extend(grouped[document_id])
    for document_id in validation_ids:
        splits["validation"].extend(grouped[document_id])
    for document_id in test_ids:
        splits["test"].extend(grouped[document_id])

    if report is not None:
        total = sum(len(chunks) for chunks in splits.values()) or 1
        realized = splits["train"] and len(splits["train"]) / total or 0.0
        if abs(realized - options.train_split) > 0.1:
            report.warnings.append(
                f"Requested {options.train_split:.0%}/{options.validation_split:.0%}/"
                f"{options.test_split:.0%} but only {count} documents are available, so whole-document "
                f"held-out sets give {realized:.0%} train "
                f"({len(train_ids)}/{len(validation_ids)}/{len(test_ids)} documents)."
            )
    return splits


def _persist(
    *,
    name: str,
    mode: str,
    split_records: dict[str, list[dict[str, Any]]],
    options: BuildOptions,
    report: BuildReport,
    description: str | None,
    source_document_ids: list[str],
    is_synthetic: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    db = get_db()
    if db.get("datasets", name, key="name"):
        raise ValidationError(f"A dataset named {name!r} already exists.")

    directory = get_config().datasets_dir / slugify(name)
    if directory.exists():
        raise ValidationError(f"Dataset directory {directory.name!r} already exists.")
    directory.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.85, desc="Writing JSONL")

    hasher_parts: list[str] = []
    for split, records in split_records.items():
        path = directory / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                handle.write(line + "\n")
                if len(hasher_parts) < 2000:
                    hasher_parts.append(line)
        report.rows[split] = len(records)

    content_hash = text_sha256("\n".join(hasher_parts))
    meta = {
        "name": name,
        "mode": mode,
        "options": options.to_dict(),
        "report": report.to_dict(),
        "content_hash": content_hash,
        "created_at": time.time(),
        "source_document_ids": source_document_ids,
        "is_synthetic": is_synthetic,
    }
    (directory / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    record = {
        "id": new_id("ds"),
        "name": name,
        "mode": mode,
        "description": description,
        "path": str(directory),
        "tokenizer_id": options.tokenizer_id,
        "rows": sum(report.rows.values()),
        "train_rows": report.rows.get("train", 0),
        "validation_rows": report.rows.get("validation", 0),
        "test_rows": report.rows.get("test", 0),
        "token_count": report.tokens,
        "token_method": report.token_method,
        "is_synthetic": int(is_synthetic),
        "content_hash": content_hash,
        "source_document_ids": source_document_ids,
        "config": options.to_dict(),
        "stats": {**report.to_dict(), "size_bytes": dir_size(directory)},
        "notes": None,
        "created_at": time.time(),
    }
    db.insert("datasets", record)
    log.info(f"Built dataset '{name}' ({mode}): {report.summary()}", source="dataset",
             context={"dataset_id": record["id"]})
    return db.require("datasets", record["id"])


# -------------------------------------------------------------------- access
def load_split(dataset_id: str, split: str = "train", *, limit: int | None = None) -> list[dict[str, Any]]:
    record = get_db().require("datasets", dataset_id)
    path = Path(record["path"]) / f"{split}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def preview_dataset(dataset_id: str, split: str = "train", limit: int = 10) -> dict[str, Any]:
    record = get_db().require("datasets", dataset_id)
    rows = load_split(dataset_id, split, limit=limit)
    return {"dataset": record, "split": split, "rows": rows, "count": len(rows)}


def record_to_text(record: dict[str, Any], mode: str, *, system_prompt: str = "") -> str:
    """Flatten one record into the string the model actually trains on."""
    if mode in {"raw_lm", "continued_pretraining", "retrieval"}:
        return str(record.get("text", ""))
    if mode == "instruction":
        instruction = record.get("instruction", "")
        user_input = record.get("input", "")
        output = record.get("output", "")
        prompt = f"### Instruction:\n{instruction}\n"
        if user_input:
            prompt += f"\n### Input:\n{user_input}\n"
        return f"{prompt}\n### Response:\n{output}"
    if mode == "qa":
        question = record.get("question") or record.get("instruction", "")
        context = record.get("context", "")
        answer = record.get("answer") or record.get("output", "")
        block = f"### Question:\n{question}\n"
        if context:
            block += f"\n### Context:\n{context}\n"
        return f"{block}\n### Answer:\n{answer}"
    if mode == "chat":
        messages = record.get("messages", [])
        lines = []
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            lines.append(f"<|system|>\n{system_prompt}")
        for message in messages:
            lines.append(f"<|{message.get('role')}|>\n{message.get('content')}")
        return "\n".join(lines)
    return json.dumps(record, ensure_ascii=False)


def list_datasets() -> list[dict[str, Any]]:
    return get_db().list("datasets", order_by="created_at DESC")


def delete_dataset(dataset_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("datasets", dataset_id)
    if remove_files and record.get("path"):
        remove_path(Path(record["path"]))
    db.delete("datasets", dataset_id)
    log.warning(f"Deleted dataset {record['name']}", source="dataset")
