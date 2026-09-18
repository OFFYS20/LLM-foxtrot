"""Turning a built dataset into tensors the trainer can consume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_studio.core import logging as log
from ai_studio.core.errors import ValidationError
from ai_studio.data.dataset_builder import load_split, record_to_text

IGNORE_INDEX = -100


@dataclass
class TokenizedStats:
    sequences: int
    tokens: int
    truncated: int
    packed: bool


class PackedLMDataset(Dataset):
    """Concatenate everything and cut fixed-length blocks (efficient for raw LM).

    ``labels`` equals ``input_ids``: both this project's transformer and the
    HuggingFace causal models shift internally, so position *t* predicts *t+1*.
    """

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) < block_size:
            raise ValidationError(
                f"Not enough tokens for a single {block_size}-token block "
                f"(have {len(token_ids)}).",
                hint="Lower max_sequence_length, or add more training data.",
            )
        usable = len(token_ids) // block_size * block_size
        self.block_size = block_size
        self.data = torch.tensor(token_ids[:usable], dtype=torch.long)
        self.length = usable // block_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        block = self.data[start : start + self.block_size].contiguous()
        return {
            "input_ids": block,
            "labels": block.clone(),
            "attention_mask": torch.ones_like(block),
        }


class SequenceDataset(Dataset):
    """One example per record, padded — used for instruction/chat datasets."""

    def __init__(
        self,
        sequences: list[list[int]],
        pad_token_id: int,
        max_length: int,
        prompt_lengths: list[int] | None = None,
    ) -> None:
        if not sequences:
            raise ValidationError("No sequences to train on after tokenization.")
        self.sequences = sequences
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.prompt_lengths = prompt_lengths or [0] * len(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.sequences[index][: self.max_length]
        prompt_length = min(self.prompt_lengths[index], len(ids))
        padding = self.max_length - len(ids)

        input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
        input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[: len(ids)] = 1

        labels = input_ids.clone()
        labels[len(ids) :] = IGNORE_INDEX          # never learn from padding
        if prompt_length:
            labels[:prompt_length] = IGNORE_INDEX  # only learn the response
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_torch_dataset(
    dataset_id: str,
    split: str,
    tokenizer: Any,
    *,
    max_length: int,
    mode: str,
    packing: bool = True,
    system_prompt: str = "",
    max_records: int | None = None,
) -> tuple[Dataset | None, TokenizedStats]:
    """Load a split and tokenize it into a torch Dataset."""
    records = load_split(dataset_id, split, limit=max_records)
    if not records:
        return None, TokenizedStats(0, 0, 0, packing)

    supervised = mode in {"instruction", "qa", "chat"}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    if supervised or not packing:
        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        truncated = 0
        for record in records:
            text = record_to_text(record, mode, system_prompt=system_prompt)
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            if len(ids) > max_length:
                truncated += 1
                ids = ids[:max_length]
            if len(ids) < 2:
                continue
            sequences.append(ids)
            prompt_lengths.append(_prompt_length(record, mode, tokenizer, system_prompt))
        if not sequences:
            return None, TokenizedStats(0, 0, truncated, False)
        dataset = SequenceDataset(sequences, pad_id, max_length, prompt_lengths)
        return dataset, TokenizedStats(
            sequences=len(sequences),
            tokens=sum(len(s) for s in sequences),
            truncated=truncated,
            packed=False,
        )

    # Packed causal LM: one long stream cut into blocks.
    stream: list[int] = []
    eos = tokenizer.eos_token_id
    for record in records:
        text = record_to_text(record, mode, system_prompt=system_prompt)
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        stream.extend(ids)
        if eos is not None and (not ids or ids[-1] != eos):
            stream.append(eos)

    if len(stream) < max_length:
        # Too little text to pack — fall back to padded sequences so the run
        # can still proceed rather than failing outright.
        log.warning(
            f"Only {len(stream)} tokens in '{split}' — falling back to padded sequences.",
            source="training",
        )
        return build_torch_dataset(
            dataset_id, split, tokenizer, max_length=max_length, mode=mode,
            packing=False, system_prompt=system_prompt, max_records=max_records,
        )

    dataset = PackedLMDataset(stream, max_length)
    return dataset, TokenizedStats(
        sequences=len(dataset), tokens=len(stream), truncated=0, packed=True
    )


def _prompt_length(record: dict[str, Any], mode: str, tokenizer: Any, system_prompt: str) -> int:
    """Token count of the prompt portion, so loss is only taken on the response."""
    try:
        if mode == "instruction":
            prompt = f"### Instruction:\n{record.get('instruction', '')}\n"
            if record.get("input"):
                prompt += f"\n### Input:\n{record['input']}\n"
            prompt += "\n### Response:\n"
        elif mode == "qa":
            question = record.get("question") or record.get("instruction", "")
            prompt = f"### Question:\n{question}\n"
            if record.get("context"):
                prompt += f"\n### Context:\n{record['context']}\n"
            prompt += "\n### Answer:\n"
        elif mode == "chat":
            messages = record.get("messages", [])
            lines = []
            if system_prompt and not any(m.get("role") == "system" for m in messages):
                lines.append(f"<|system|>\n{system_prompt}")
            for message in messages[:-1]:
                lines.append(f"<|{message.get('role')}|>\n{message.get('content')}")
            lines.append("<|assistant|>\n")
            prompt = "\n".join(lines)
        else:
            return 0
        return len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    except Exception:  # noqa: BLE001 - masking is an optimisation, never fatal
        return 0


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [key for key in batch[0] if not key.startswith("_")]
    return {key: torch.stack([item[key] for item in batch]) for key in keys}
