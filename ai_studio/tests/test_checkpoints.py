"""Checkpoints: real weights on disk, resumable state, atomicity and pruning.

Every assertion here is about files that actually exist — a checkpoint record
without weights behind it would fail these tests.
"""

from __future__ import annotations

import json

import pytest
import torch

from ai_studio.core.errors import CheckpointError
from ai_studio.models.transformer import TransformerConfig, TransformerLM
from ai_studio.training.checkpoint_manager import (
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint_weights,
    load_training_state,
    prune_checkpoints,
    restore_training_state,
    save_checkpoint,
    verify_checkpoint,
)

TRAINING_CONFIG = {"method": "scratch", "learning_rate": 0.001, "batch_size": 2}


def build_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(
        TransformerConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                          max_position_embeddings=32)
    )


def save(experiment_id: str, model, *, step: int = 10, optimizer=None, scheduler=None,
         val_loss: float | None = 1.5, is_best: bool = False, name: str | None = None):
    return save_checkpoint(
        experiment_id=experiment_id,
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        epoch=step / 10,
        train_loss=2.0,
        val_loss=val_loss,
        training_config=TRAINING_CONFIG,
        dataset_meta={"dataset_id": "ds-1", "rows": 100},
        is_best=is_best,
        name=name,
    )


def test_checkpoint_writes_real_weights_to_disk(tmp_path):
    from pathlib import Path

    model = build_model()
    record = save("exp-weights", model)
    directory = Path(record["path"])

    assert directory.is_dir()
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
    assert weights, "a checkpoint must contain an actual weight file"
    assert record["size_bytes"] > 0
    assert (directory / "config.json").exists()
    assert (directory / "checkpoint.json").exists()

    manifest = json.loads((directory / "checkpoint.json").read_text())
    assert manifest["step"] == 10
    assert manifest["training_config"] == TRAINING_CONFIG


def test_weights_reload_bit_identically():
    from pathlib import Path

    model = build_model(seed=1)
    record = save("exp-roundtrip", model)

    reloaded = TransformerLM.from_pretrained(Path(record["path"]))
    original_state = model.state_dict()
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, original_state[name]), f"{name} changed across save/load"


def test_reloaded_model_produces_identical_logits():
    model = build_model(seed=2).eval()
    record = save("exp-logits", model)
    reloaded = TransformerLM.from_pretrained(record["path"]).eval()

    ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        assert torch.allclose(model(ids)["logits"], reloaded(ids)["logits"], atol=1e-6)


def test_optimizer_and_scheduler_state_resume():
    model = build_model(seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Take real steps so the optimizer carries non-trivial moment estimates.
    for _ in range(3):
        loss = model(torch.randint(0, 64, (2, 8)), labels=torch.randint(0, 64, (2, 8)))["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    record = save("exp-resume", model, step=3, optimizer=optimizer, scheduler=scheduler)
    assert record["has_optimizer_state"] == 1

    fresh_model = build_model(seed=99)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1, gamma=0.5)

    state = load_training_state(record["path"])
    resumed = restore_training_state(state, optimizer=fresh_optimizer, scheduler=fresh_scheduler)

    assert resumed["step"] == 3
    assert resumed["training_config"] == TRAINING_CONFIG
    assert resumed["dataset_meta"]["dataset_id"] == "ds-1"
    assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert fresh_optimizer.state_dict()["state"], "optimizer moments were not restored"


def test_resuming_puts_the_checkpoints_weights_back():
    """The state alone would carry on from step N with untrained weights."""
    trained = build_model(seed=3)
    record = save("exp-weights", trained, step=5)
    fresh = build_model(seed=99)
    assert not torch.equal(fresh.embed_tokens.weight, trained.embed_tokens.weight)

    assert load_checkpoint_weights(record["path"], fresh) == "full"
    for (name, ours), theirs in zip(fresh.state_dict().items(), trained.state_dict().values()):
        assert torch.equal(ours, theirs), f"{name} was not restored"


def test_weights_that_do_not_fit_are_refused_not_half_loaded():
    record = save("exp-misfit", build_model(seed=3), step=5)
    other = TransformerLM(TransformerConfig(vocab_size=64, hidden_size=32, num_layers=3,
                                            num_heads=4, max_position_embeddings=32))
    with pytest.raises(CheckpointError):
        load_checkpoint_weights(record["path"], other)


def test_resume_is_refused_when_state_is_missing(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(CheckpointError):
        load_training_state(tmp_path / "bare")


def test_mismatched_optimizer_state_is_reported_not_silently_ignored():
    model = build_model(seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randint(0, 64, (2, 8)), labels=torch.randint(0, 64, (2, 8)))["loss"].backward()
    optimizer.step()
    record = save("exp-mismatch", model, optimizer=optimizer)

    other = TransformerLM(TransformerConfig(vocab_size=64, hidden_size=64, num_layers=1, num_heads=4))
    wrong_optimizer = torch.optim.AdamW(other.parameters(), lr=1e-3)
    with pytest.raises(CheckpointError):
        restore_training_state(load_training_state(record["path"]), optimizer=wrong_optimizer)


def test_verify_reports_a_complete_checkpoint():
    record = save("exp-verify", build_model(), optimizer=None)
    report = verify_checkpoint(record["path"])
    assert report["exists"] and report["has_weights"] and report["has_config"]
    assert report["resumable"] is True
    assert report["problems"] == []


def test_verify_flags_a_missing_directory(tmp_path):
    report = verify_checkpoint(tmp_path / "gone")
    assert report["exists"] is False
    assert report["resumable"] is False
    assert report["problems"]


def test_verify_flags_a_checkpoint_without_weights(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{}")
    report = verify_checkpoint(broken)
    assert report["has_weights"] is False
    assert report["resumable"] is False
    assert any("weight" in problem.lower() for problem in report["problems"])


def test_failed_save_leaves_no_partial_directory(tmp_path):
    from ai_studio.core.config import get_config

    class Unsavable:
        """Fails during the weight-writing step."""

        def state_dict(self):
            raise RuntimeError("disk exploded")

    before = set((get_config().checkpoints_dir / "exp-atomic").glob("*")) if (
        get_config().checkpoints_dir / "exp-atomic"
    ).exists() else set()

    with pytest.raises(CheckpointError):
        save_checkpoint(
            experiment_id="exp-atomic", model=Unsavable(), tokenizer=None, optimizer=None,
            scheduler=None, step=1, epoch=0.1, train_loss=1.0, val_loss=None,
            training_config=TRAINING_CONFIG,
        )

    directory = get_config().checkpoints_dir / "exp-atomic"
    after = set(directory.glob("*")) if directory.exists() else set()
    assert after == before, "a failed save must not leave a half-written checkpoint behind"
    assert not list(directory.glob(".*tmp*")) if directory.exists() else True


def test_pruning_keeps_the_newest_and_the_best():
    from pathlib import Path

    model = build_model(seed=5)
    # The best checkpoint is the oldest, so keeping it has to be deliberate.
    best = save("exp-prune", model, step=1, name="step-1", val_loss=0.1, is_best=True)
    others = [save("exp-prune", model, step=step, name=f"step-{step}") for step in (2, 3, 4, 5)]
    stale_path = Path(others[0]["path"])

    removed = prune_checkpoints("exp-prune", keep_last=2, keep_best=True)
    assert removed == 2

    kept = {row["name"] for row in list_checkpoints("exp-prune")}
    assert kept == {"step-1", "step-4", "step-5"}
    assert Path(best["path"]).exists(), "the best checkpoint must survive pruning"
    assert not stale_path.exists(), "pruned checkpoints must be removed from disk too"


def test_pruning_without_keep_best_drops_the_best_too():
    from pathlib import Path

    model = build_model(seed=8)
    best = save("exp-prune-hard", model, step=1, name="old-best", val_loss=0.1, is_best=True)
    for step in (2, 3):
        save("exp-prune-hard", model, step=step, name=f"s{step}")

    assert prune_checkpoints("exp-prune-hard", keep_last=2, keep_best=False) == 1
    assert not Path(best["path"]).exists()


def test_marking_a_new_best_clears_the_previous_one():
    model = build_model(seed=6)
    first = save("exp-best", model, step=1, name="a", val_loss=1.0, is_best=True)
    second = save("exp-best", model, step=2, name="b", val_loss=0.2, is_best=True)

    rows = {row["id"]: row for row in list_checkpoints("exp-best")}
    assert rows[second["id"]]["is_best"] == 1
    assert rows[first["id"]]["is_best"] == 0, "only one checkpoint may be the best"


def test_delete_removes_record_and_files():
    from pathlib import Path

    record = save("exp-delete", build_model(seed=7))
    path = Path(record["path"])
    assert path.exists()

    delete_checkpoint(record["id"])
    assert not path.exists()
    assert record["id"] not in {row["id"] for row in list_checkpoints("exp-delete")}
