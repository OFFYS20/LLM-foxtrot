"""Training across more than one GPU.

One process per card, each holding a full copy of the model and training on a
different slice of every epoch. PyTorch averages the gradients across the
processes at each backward pass, so the result is one model trained on all the
data — not N models that have to be reconciled afterwards.

Why processes and not threads: Python's interpreter lock means threads would
take turns rather than run at once, which is the whole point. And why spawn
rather than asking people to run `torchrun`: this is a program people
double-click, not a cluster job.

**CPU and GPU together is not offered, because it would be slower.** Every
step ends with a gradient exchange, so a step takes as long as the slowest
process; a CPU is one to two orders of magnitude behind a GPU at this
arithmetic, so adding one makes the GPUs wait. Splitting a model's layers
across both is worse still — activations would cross the bus twice per step.
Mixing the two is a way to fit a model that does not fit, never a way to go
faster.
"""

from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_studio.core import logging as log
from ai_studio.core.errors import TrainingError, UnsupportedError


def gpu_count() -> int:
    """How many CUDA devices this machine has."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - a broken CUDA install is not a crash
        return 0


def backend(device_type: str = "cuda") -> str:
    """The collective library to talk over.

    NCCL is the fast one and the right answer for GPUs on Linux. It is not
    built for Windows, where gloo is the only option — slower, but it works,
    and a slower multi-GPU run still beats using one card. CPU tensors always
    go over gloo; NCCL does not carry them.
    """
    if device_type != "cuda" or platform.system() == "Windows":
        return "gloo"
    return "nccl"


def describe() -> dict[str, Any]:
    """What this machine can do, for reporting rather than deciding."""
    count = gpu_count()
    names: list[str] = []
    if count:
        import torch

        names = [torch.cuda.get_device_properties(i).name for i in range(count)]
    return {
        "gpus": count,
        "names": names,
        "backend": backend() if count > 1 else None,
        "can_distribute": count > 1,
    }


def resolve_world_size(requested: str | int, *, available: int | None = None) -> int:
    """Turn --gpus into a number of processes, or say why it cannot be."""
    have = gpu_count() if available is None else available
    if isinstance(requested, str):
        text = requested.strip().lower()
        if text in ("auto", ""):
            return max(1, have)
        if not text.isdigit():
            raise UnsupportedError(
                f"--gpus takes a number or 'auto', not {requested!r}.")
        requested = int(text)

    requested = int(requested)
    if requested <= 1:
        return 1
    if have == 0:
        raise UnsupportedError(
            "There is no CUDA GPU here, so there is nothing to spread across.",
            hint="Drop --gpus, or pass --gpus 1.",
        )
    if requested > have:
        raise UnsupportedError(
            f"Asked for {requested} GPUs; this machine has {have}.",
            hint=f"Use --gpus {have}, or --gpus auto.",
        )
    return requested


@dataclass
class Job:
    """Everything a worker process needs, and nothing that cannot be pickled.

    A live model cannot cross a process boundary, so what travels is where to
    load it from. The token ids travel as a file the workers memory-map rather
    than as a few million integers through a pipe.
    """

    model_dir: str
    train_tokens: str
    block_size: int
    settings: dict[str, Any]
    eval_tokens: str | None = None
    world_size: int = 1
    lora: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _worker(rank: int, job: Job, results) -> None:  # pragma: no cover - runs in a child
    """One process of the group. Rank 0 is the one that saves and reports."""
    import numpy
    import torch
    import torch.distributed as dist

    from ai_studio.training.config import TrainingConfig
    from ai_studio.training.data import PackedLMDataset
    from ai_studio.training.trainer import Trainer

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")

        dist.init_process_group(backend=backend(device.type),
                                rank=rank, world_size=job.world_size)
    except Exception as exc:  # noqa: BLE001 - report rather than hang the group
        results.put({"rank": rank, "error": f"could not join the group: {exc}"})
        return

    try:
        network = load_causal_lm(job.model_dir)
        lora_stats = None
        if job.lora:
            from ai_studio.training.lora_trainer import apply_lora

            settings = TrainingConfig.from_dict(job.settings)
            settings.lora.rank = int(job.lora.get("rank", 16))
            settings.lora.alpha = int(job.lora.get("alpha", settings.lora.rank * 2))
            network, lora_stats = apply_lora(network, settings)

        settings = TrainingConfig.from_dict(job.settings)
        train_ids = numpy.load(job.train_tokens, mmap_mode="r").tolist()
        eval_ids = numpy.load(job.eval_tokens, mmap_mode="r").tolist() if job.eval_tokens else []

        trainer = Trainer(
            model=network,
            train_dataset=PackedLMDataset(train_ids, job.block_size),
            eval_dataset=(PackedLMDataset(eval_ids, job.block_size)
                          if len(eval_ids) >= job.block_size else None),
            config=settings,
            device=device,
            rank=rank,
            world_size=job.world_size,
            on_log=(lambda message, level="info": results.put(
                {"rank": rank, "log": message, "level": level})) if rank == 0 else None,
        )
        # Every process takes part in an evaluation, because the held-out
        # loss is pooled across them. Before and after, on the same text: the
        # difference is what the lesson did.
        before = trainer.evaluate()
        outcome = trainer.train()
        after = trainer.evaluate() if outcome.status == "completed" else None

        if rank == 0:
            saved = trainer.model
            if lora_stats:
                saved = saved.merge_and_unload()
            saved.save_pretrained(job.model_dir)
            results.put({
                "rank": 0,
                "result": {
                    "status": outcome.status,
                    "steps": outcome.steps,
                    "final_train_loss": outcome.final_train_loss,
                    "best_val_loss": outcome.best_val_loss,
                    "held_out_before": before,
                    "held_out_after": after,
                    "duration_seconds": outcome.duration_seconds,
                    "error": outcome.error,
                    "lora": {k: v for k, v in (lora_stats or {}).items() if k != "network"},
                },
            })
    except Exception as exc:  # noqa: BLE001 - one worker's failure is the run's
        results.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - already torn down
            pass


def load_causal_lm(path: str):
    """Load a model from its folder, whichever kind it is.

    The worker cannot be handed a live model — it has to build its own — so
    this has to work from the directory alone.
    """
    import json

    from ai_studio.models.transformer import TransformerLM

    config_path = Path(path) / "config.json"
    try:
        model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Could not read {config_path}: {exc}") from exc

    if model_type == "ai_studio_transformer":
        return TransformerLM.from_pretrained(str(path))

    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)


def tokens_to_file(ids: list[int], directory: Path, name: str) -> str:
    """Write token ids where the workers can memory-map them."""
    import numpy

    path = Path(directory) / f"{name}.npy"
    numpy.save(path, numpy.asarray(ids, dtype=numpy.int32))
    return str(path)


def launch(job: Job, *, on_log=None) -> dict[str, Any]:
    """Run ``job`` across ``job.world_size`` processes and return rank 0's result.

    Blocks until every process has finished. A failure in any of them is
    raised here rather than left to look like a hang.
    """
    if job.world_size <= 1:
        raise TrainingError("launch() is for more than one process.")

    import torch.multiprocessing as multiprocessing

    context = multiprocessing.get_context("spawn")
    results = context.Queue()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))

    # Say what will really be used: a CPU group is gloo whatever the platform.
    talking = backend("cuda" if gpu_count() else "cpu")
    log.info(f"Starting {job.world_size} training processes over {talking}",
             source="training")
    spawned = multiprocessing.spawn(
        _worker, args=(job, results), nprocs=job.world_size, join=False)

    outcome: dict[str, Any] | None = None
    failures: list[str] = []

    def drain() -> None:
        while not results.empty():
            message = results.get()
            if "log" in message and on_log:
                on_log(message["log"], message.get("level", "info"))
            elif "error" in message:
                failures.append(f"process {message['rank']}: {message['error']}")
            elif "result" in message:
                nonlocal outcome
                outcome = message["result"]

    try:
        while not spawned.join(timeout=0.2):
            drain()
        drain()
    except Exception as exc:  # noqa: BLE001 - a child dying before it could report
        drain()
        if failures:
            raise TrainingError(
                "Distributed training failed:\n  " + "\n  ".join(failures)) from exc
        raise TrainingError(
            f"A training process stopped before it could say why ({exc}).",
            hint="Starting extra processes re-imports the program that launched them. "
                 "If you are calling this from a script of your own, its top level has "
                 "to be guarded with `if __name__ == \"__main__\":`.",
        ) from exc

    if failures:
        raise TrainingError("Distributed training failed:\n  " + "\n  ".join(failures))
    if outcome is None:
        raise TrainingError("The training processes finished without reporting a result.")
    return outcome


def _free_port() -> int:
    """A port the group can rendezvous on."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
