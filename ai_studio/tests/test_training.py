"""Training: preflight safety, tokenized datasets, and a real optimization loop.

The loop tests run actual PyTorch steps on a tiny model — if training were
simulated, the loss assertions here would not hold.
"""

from __future__ import annotations

import pytest
import torch

from ai_studio.core.errors import ValidationError
from ai_studio.models.transformer import TransformerConfig, TransformerLM
from ai_studio.training.config import TrainingConfig, preflight
from ai_studio.training.data import IGNORE_INDEX, PackedLMDataset, SequenceDataset, collate
from ai_studio.training.trainer import Trainer, TrainingControl, perplexity

VOCAB = 48


def tiny_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(
        TransformerConfig(vocab_size=VOCAB, hidden_size=32, num_layers=2, num_heads=4,
                          max_position_embeddings=64, dropout=0.0)
    )


def repeating_dataset(block_size: int = 16, blocks: int = 24) -> PackedLMDataset:
    """A perfectly learnable pattern: the model should drive the loss down fast."""
    pattern = [3, 4, 5, 6, 7, 8, 9, 10]
    return PackedLMDataset(pattern * (block_size * blocks // len(pattern) + 1), block_size)


# ------------------------------------------------------------------ preflight
def test_preflight_passes_for_a_small_cpu_run():
    result = preflight(TrainingConfig(batch_size=1, max_sequence_length=128), parameter_count=1_000_000)
    assert result.ok
    assert result.blockers == []
    assert result.estimated_mb > 0
    assert result.device in {"cpu", "cuda", "mps"}


def test_preflight_blocks_a_run_that_cannot_fit():
    result = preflight(
        TrainingConfig(batch_size=64, max_sequence_length=4096),
        parameter_count=70_000_000_000,
    )
    assert not result.ok
    assert result.blockers, "an impossible run must be refused, not started"
    assert result.suggestions, "a refusal must come with a way forward"


def test_preflight_refuses_quantized_training_without_a_gpu():
    result = preflight(TrainingConfig(method="qlora", quantization="4bit"), parameter_count=1_000_000)
    if result.device != "cuda":
        assert not result.ok
        assert any("GPU" in blocker for blocker in result.blockers)


def test_preflight_estimate_reacts_to_the_knobs_it_suggests():
    big = preflight(TrainingConfig(batch_size=16, max_sequence_length=2048), parameter_count=50_000_000)
    small = preflight(TrainingConfig(batch_size=1, max_sequence_length=256), parameter_count=50_000_000)
    assert small.estimated_mb < big.estimated_mb

    plain = preflight(TrainingConfig(optimizer="adamw"), parameter_count=50_000_000)
    thrifty = preflight(TrainingConfig(optimizer="adafactor"), parameter_count=50_000_000)
    assert thrifty.estimated_mb < plain.estimated_mb

    checkpointed = preflight(
        TrainingConfig(batch_size=16, max_sequence_length=2048, gradient_checkpointing=True),
        parameter_count=50_000_000,
    )
    assert checkpointed.estimated_mb < big.estimated_mb


def test_preflight_warns_about_lost_precision_on_cpu():
    result = preflight(TrainingConfig(precision="fp16"), parameter_count=1_000_000)
    if result.device == "cpu":
        assert any("FP16" in warning or "CPU" in warning for warning in result.warnings)


def test_lora_preflight_is_cheaper_than_full_finetuning():
    full = preflight(TrainingConfig(method="finetune"), parameter_count=1_000_000_000)
    lora = preflight(
        TrainingConfig(method="lora"), parameter_count=1_000_000_000, trainable_parameters=4_000_000
    )
    assert lora.estimated_mb < full.estimated_mb


# ------------------------------------------------------------- torch datasets
def test_packed_dataset_cuts_exact_blocks():
    dataset = PackedLMDataset(list(range(100)), block_size=16)
    assert len(dataset) == 6, "the trailing partial block is dropped"
    item = dataset[0]
    assert item["input_ids"].shape == (16,)
    assert torch.equal(item["labels"], item["input_ids"]), "causal LM shifts inside the model"
    assert item["attention_mask"].sum() == 16


def test_packed_dataset_refuses_a_corpus_smaller_than_one_block():
    with pytest.raises(ValidationError):
        PackedLMDataset(list(range(10)), block_size=64)


def test_supervised_dataset_masks_padding_and_prompt():
    dataset = SequenceDataset(
        sequences=[[1, 2, 3, 4, 5]], pad_token_id=0, max_length=8, prompt_lengths=[3]
    )
    item = dataset[0]
    assert item["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    assert item["attention_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert item["labels"].tolist() == [IGNORE_INDEX] * 3 + [4, 5] + [IGNORE_INDEX] * 3


def test_supervised_dataset_without_a_prompt_learns_everything():
    item = SequenceDataset([[1, 2, 3]], pad_token_id=0, max_length=4)[0]
    assert item["labels"].tolist() == [1, 2, 3, IGNORE_INDEX]


def test_empty_supervised_dataset_is_rejected():
    with pytest.raises(ValidationError):
        SequenceDataset([], pad_token_id=0, max_length=8)


def test_collate_stacks_a_batch():
    dataset = PackedLMDataset(list(range(64)), block_size=8)
    batch = collate([dataset[0], dataset[1], dataset[2]])
    assert batch["input_ids"].shape == (3, 8)
    assert batch["labels"].shape == (3, 8)


# ------------------------------------------------------------- the real loop
def test_training_actually_reduces_the_loss():
    config = TrainingConfig(
        method="scratch", max_steps=40, batch_size=4, learning_rate=3e-3,
        max_sequence_length=16, log_interval=1, eval_interval=0, checkpoint_interval=0,
        lr_scheduler="constant", precision="fp32",
    )
    config.validate()
    trainer = Trainer(model=tiny_model(1), train_dataset=repeating_dataset(), config=config)
    result = trainer.train()

    assert result.status == "completed"
    assert result.steps == 40
    assert result.tokens_processed > 0
    first, last = result.history[0]["loss"], result.history[-1]["loss"]
    assert last < first * 0.7, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_weights_change_during_training():
    model = tiny_model(2)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    config = TrainingConfig(max_steps=5, batch_size=2, max_sequence_length=16,
                            log_interval=1, eval_interval=0, checkpoint_interval=0)
    Trainer(model=model, train_dataset=repeating_dataset(), config=config).train()

    changed = [
        name for name, tensor in model.state_dict().items()
        if not torch.equal(tensor, before[name])
    ]
    assert changed, "no parameter moved — nothing was actually trained"


def test_stop_ends_the_run_cleanly_and_reports_it():
    control = TrainingControl()
    config = TrainingConfig(max_steps=1000, batch_size=2, max_sequence_length=16,
                            log_interval=1, eval_interval=0, checkpoint_interval=0)

    def on_metrics(metrics):
        if metrics.step >= 3:
            control.stop()

    trainer = Trainer(model=tiny_model(3), train_dataset=repeating_dataset(), config=config,
                      control=control, on_metrics=on_metrics)
    result = trainer.train()

    assert result.status == "stopped", "a stopped run must not be reported as completed"
    assert 0 < result.steps < 1000
    assert result.final_train_loss is not None


def test_evaluation_returns_a_real_held_out_loss():
    config = TrainingConfig(max_steps=10, batch_size=2, max_sequence_length=16,
                            log_interval=5, eval_interval=5, checkpoint_interval=0,
                            eval_max_batches=2)
    trainer = Trainer(
        model=tiny_model(4), train_dataset=repeating_dataset(), config=config,
        eval_dataset=repeating_dataset(blocks=4),
    )
    result = trainer.train()
    assert result.best_val_loss is not None and result.best_val_loss > 0


def test_checkpoint_callback_fires_on_the_configured_interval():
    calls: list[int] = []
    config = TrainingConfig(max_steps=9, batch_size=2, max_sequence_length=16,
                            log_interval=1, eval_interval=0, checkpoint_interval=3)
    Trainer(
        model=tiny_model(5), train_dataset=repeating_dataset(), config=config,
        on_checkpoint=lambda step, *_rest: calls.append(step),
    ).train()
    assert calls == [3, 6, 9]


def test_gradient_accumulation_keeps_the_effective_batch():
    config = TrainingConfig(max_steps=4, batch_size=2, gradient_accumulation_steps=3,
                            max_sequence_length=16, log_interval=1, eval_interval=0,
                            checkpoint_interval=0)
    result = Trainer(model=tiny_model(6), train_dataset=repeating_dataset(), config=config).train()
    assert result.steps == 4, "steps count optimizer updates, not micro-batches"
    assert config.effective_batch_size == 6


def test_warmup_raises_the_learning_rate_then_decays_it():
    config = TrainingConfig(max_steps=20, batch_size=2, learning_rate=1e-3, warmup_steps=5,
                            lr_scheduler="cosine", max_sequence_length=16, log_interval=1,
                            eval_interval=0, checkpoint_interval=0)
    result = Trainer(model=tiny_model(7), train_dataset=repeating_dataset(), config=config).train()
    rates = [entry["learning_rate"] for entry in result.history]
    assert rates[0] < rates[4], "learning rate must warm up"
    assert rates[-1] < max(rates), "and then decay"


def test_perplexity_matches_the_loss():
    import math

    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(50)) == pytest.approx(50, rel=1e-6)
    assert perplexity(None) is None
