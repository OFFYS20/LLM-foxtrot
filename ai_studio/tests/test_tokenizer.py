"""Tokenizer training, saving, loading and round-tripping."""

from __future__ import annotations

import pytest

from ai_studio.core.errors import ValidationError
from ai_studio.models.tokenizer_manager import (
    TokenizerConfig,
    load_tokenizer,
    preview_tokenization,
    train_tokenizer,
)


def test_train_and_roundtrip_byte_level_bpe(sample_corpus):
    record = train_tokenizer(
        TokenizerConfig(name="t-roundtrip", kind="byte_level_bpe", vocab_size=400, min_frequency=2),
        texts=[sample_corpus],
    )
    assert record["vocab_size"] > 50
    outcome = preview_tokenization(record["id"], "the cat sat on the mat .")
    assert outcome["token_count"] > 0
    assert outcome["roundtrip_ok"], "byte-level BPE must decode losslessly"


@pytest.mark.parametrize("kind", ["byte_level_bpe", "bpe", "wordpiece", "unigram"])
def test_every_algorithm_trains(kind, sample_corpus):
    record = train_tokenizer(
        TokenizerConfig(name=f"t-{kind}", kind=kind, vocab_size=300, min_frequency=2),
        texts=[sample_corpus],
    )
    assert record["kind"] == kind
    assert record["vocab_size"] > 20


def test_special_tokens_are_applied(sample_corpus):
    record = train_tokenizer(
        TokenizerConfig(name="t-special", kind="byte_level_bpe", vocab_size=300),
        texts=[sample_corpus],
    )
    tokenizer = load_tokenizer(record["id"])
    ids = tokenizer("hello", add_special_tokens=True)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    assert tokens[0] == "<s>" and tokens[-1] == "</s>", "BOS/EOS must wrap the sequence"


def test_rejects_unknown_algorithm():
    with pytest.raises(ValidationError):
        TokenizerConfig(name="x", kind="not-a-real-algorithm").validate()


def test_rejects_tiny_corpus():
    with pytest.raises(ValidationError):
        train_tokenizer(TokenizerConfig(name="t-tiny", vocab_size=300), texts=["too short"])
