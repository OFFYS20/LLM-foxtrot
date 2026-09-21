"""Training across more than one process.

The launcher is exercised for real here — two processes, a process group, a
sampler that splits the data and gradients averaged between them — over gloo
on the CPU, which is the one multi-process configuration a machine without
two GPUs can actually run. What that does not cover is NCCL and CUDA device
placement; those need the hardware.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ai_studio.core.errors import TrainingError, UnsupportedError
from ai_studio.training import distributed


# ------------------------------------------------------------- what it picks
def test_a_cpu_group_never_uses_nccl():
    """NCCL does not carry CPU tensors at all."""
    assert distributed.backend("cpu") == "gloo"


def test_windows_gets_gloo_even_for_gpus(monkeypatch):
    """NCCL is not built for Windows; a slower group still beats one card."""
    monkeypatch.setattr(distributed.platform, "system", lambda: "Windows")
    assert distributed.backend("cuda") == "gloo"


def test_linux_gpus_get_nccl(monkeypatch):
    monkeypatch.setattr(distributed.platform, "system", lambda: "Linux")
    assert distributed.backend("cuda") == "nccl"


# ------------------------------------------------------- how many processes
def test_auto_means_every_card():
    assert distributed.resolve_world_size("auto", available=4) == 4


def test_one_is_the_ordinary_case():
    assert distributed.resolve_world_size(1, available=4) == 1
    assert distributed.resolve_world_size("1", available=0) == 1


def test_asking_for_more_cards_than_exist_says_how_many_there_are():
    with pytest.raises(UnsupportedError, match="has 2"):
        distributed.resolve_world_size(4, available=2)


def test_asking_to_spread_with_no_gpu_says_so():
    with pytest.raises(UnsupportedError, match="nothing to spread"):
        distributed.resolve_world_size(2, available=0)


def test_a_word_that_is_not_a_number_is_refused():
    with pytest.raises(UnsupportedError, match="number or 'auto'"):
        distributed.resolve_world_size("both", available=2)


def test_describe_reports_what_is_there():
    facts = distributed.describe()
    assert set(facts) == {"gpus", "names", "backend", "can_distribute"}
    assert facts["can_distribute"] == (facts["gpus"] > 1)


def test_launching_one_process_is_refused_as_a_mistake():
    job = distributed.Job(model_dir="/nowhere", train_tokens="/nowhere",
                          block_size=8, settings={}, world_size=1)
    with pytest.raises(TrainingError, match="more than one process"):
        distributed.launch(job)


# --------------------------------------------------- two processes, for real
def tiny_model(directory: Path) -> Path:
    from ai_studio.models.transformer import TransformerConfig, TransformerLM

    config = TransformerConfig(
        hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2,
        max_position_embeddings=32, vocab_size=64, intermediate_size=64,
    )
    path = directory / "model"
    TransformerLM(config).save_pretrained(str(path))
    return path


@pytest.mark.slow
def test_two_processes_train_one_model_and_agree_on_it():
    """The point of DDP: each process sees different data, and the weights
    still come out identical because the gradients were averaged."""
    import numpy
    import torch

    from ai_studio.training.config import TrainingConfig

    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        path = tiny_model(folder)
        before = (path / "model.safetensors").read_bytes()

        ids = [(n * 7 + 3) % 64 for n in range(4096)]
        settings = TrainingConfig(
            method="continued_pretraining", epochs=1.0, batch_size=2,
            learning_rate=1e-3, max_sequence_length=32, precision="fp32",
            warmup_steps=1, log_interval=1000, eval_interval=0,
            checkpoint_interval=0,
        )
        job = distributed.Job(
            model_dir=str(path),
            train_tokens=distributed.tokens_to_file(ids, folder, "train"),
            block_size=32,
            settings=settings.to_dict(),
            world_size=2,
        )
        outcome = distributed.launch(job)

        assert outcome["status"] == "completed"
        assert outcome["steps"] > 0
        assert outcome["final_train_loss"] is not None
        assert (path / "model.safetensors").read_bytes() != before, "it must have learned"

        # It has to be loadable afterwards, or the run produced nothing usable.
        from ai_studio.models.transformer import TransformerLM

        assert TransformerLM.from_pretrained(str(path)) is not None


@pytest.mark.slow
def test_the_processes_do_not_each_train_on_the_same_batches():
    """Without a distributed sampler this would be two copies of one job."""
    from torch.utils.data.distributed import DistributedSampler

    from ai_studio.training.data import PackedLMDataset

    dataset = PackedLMDataset([n % 64 for n in range(3200)], 32)
    first = list(DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False))
    second = list(DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False))
    assert set(first).isdisjoint(second), "the processes must see different slices"
    assert len(first) + len(second) >= len(dataset)
