"""Building a model and teaching it.

The model, the training loop and the checkpoint writer are AI Studio's — tested
there, reused here rather than reimplemented. What this module adds is the short
path: material in, a model that has learned from it out.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from ai_studio.models.transformer import (
    TransformerConfig,
    TransformerLM,
    design_config,
    format_parameter_count,
    parse_parameter_count,
)
from ai_studio.training.config import TrainingConfig, fit_batch_size, preflight
from ai_studio.training.data import PackedLMDataset
from ai_studio.training.trainer import Trainer, perplexity

from teacher.material import Material
from teacher.workspace import Model, TeacherError

#: Common sizes, offered as a shortcut. They are nothing more than numbers —
#: --size takes any count you name, and every one of them, these included,
#: goes through the same search. A name that quietly meant "near enough" would
#: be the one case where the number you wrote is not the number you get.
SIZES = {
    "1m": (1_000_000, "learns grammar in minutes on a laptop CPU"),
    "10m": (10_000_000, "a few MB of text and some patience; still fine on a CPU"),
    "100m": (100_000_000, "wants a GPU and a library's worth of text"),
    "200m": (200_000_000, "a GPU with 8GB or so, and a lot of text"),
    "500m": (500_000_000, "a GPU with room to spare and a great deal of text"),
    "1b": (1_000_000_000, "a serious GPU (24GB+) and gigabytes of text"),
}

#: What --size used to be called. Kept so older commands and scripts still run.
SIZE_ALIASES = {
    "tiny": "1m",
    "small": "10m",
    "medium": "100m",
    "large": "500m",
    "huge": "1b",
}


def resolve_size(name: str) -> int:
    """How many parameters were asked for.

    A count written any ordinary way — 70K, 4m, 1.5B, 250000 — or one of the
    shortcut names. Either way the answer is a number, because either way the
    same search runs against it.
    """
    key = str(name).strip().lower()
    key = SIZE_ALIASES.get(key, key)
    if key in SIZES:
        return SIZES[key][0]
    try:
        return parse_parameter_count(key)
    except ValueError as exc:
        raise TeacherError(
            f"{exc} There are shortcuts too: {', '.join(SIZES)}"
            f" (older names {', '.join(SIZE_ALIASES)} still work)."
        ) from exc


#: Weights alone, at four bytes each, against the memory this machine has. A
#: model has to be built before it can be trained, and building it is where an
#: impossible number stops being an abstraction.
def check_it_can_exist(target: int) -> None:
    from ai_studio.hardware.monitor import monitor

    weights_gb = target * 4 / 1024 ** 3
    snapshot = monitor.snapshot()
    available_gb = snapshot.ram_available_gb or 0
    if not available_gb or weights_gb <= available_gb * 0.6:
        return

    raise TeacherError(
        f"{format_parameter_count(target)} parameters cannot be built here.\n"
        f"  The weights alone would be {weights_gb:,.1f} GB at four bytes each, and "
        f"this machine has {available_gb:.1f} GB free.\n"
        f"  Training needs roughly four times the weights again, for gradients and "
        f"the optimizer.\n"
        f"  The largest that would fit here is around "
        f"{format_parameter_count(int(available_gb * 0.6 * 1024 ** 3 / 4))}."
    )


#: Pretrained starting points that are realistic to fine-tune at home. Anything
#: on the Hub works with --base, these are just the ones worth suggesting.
BASES = {
    "small": ("HuggingFaceTB/SmolLM2-135M", "135M — writes real English, fine-tunes on a CPU"),
    "gpt2": ("openai-community/gpt2", "124M — the classic; older, still capable"),
    "medium": ("HuggingFaceTB/SmolLM2-360M", "360M — better prose, wants a GPU"),
    "tiny-test": ("sshleifer/tiny-gpt2", "0.1M — a stub for checking the plumbing, not for use"),
}

SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
MIN_CHARACTERS = 2_000


# --------------------------------------------------------------- tokenizer
def train_tokenizer(text: str, vocab_size: int) -> Tokenizer:
    """A byte-level BPE tokenizer, which round-trips any text losslessly."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)

    # Attached after training, so the special-token ids are the real ones.
    bos = tokenizer.token_to_id("<s>")
    eos = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", bos), ("</s>", eos)],
    )
    return tokenizer


def load_tokenizer(model: Model):
    """Load the model's own tokenizer — an adopted one brings its own specials."""
    if model.kind() != "studio":
        from transformers import AutoTokenizer

        fast = AutoTokenizer.from_pretrained(str(model.path))
        if fast.pad_token is None:
            fast.pad_token = fast.eos_token
        return fast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(model.tokenizer_path),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )


# ------------------------------------------------------------------ create
def create(model: Model, material: Material, size: str, *, context: int = 0) -> dict:
    """Build a tokenizer and a fresh, untrained model sized to the material."""
    size = resolve_size(size)
    if material.characters < MIN_CHARACTERS:
        raise TeacherError(
            f"Only {material.characters:,} characters of material — too little to learn anything. "
            f"Give it at least {MIN_CHARACTERS:,}; a few hundred KB is a sensible start."
        )

    target = size
    check_it_can_exist(target)

    # Embeddings are vocabulary times width. Past about a quarter of the budget
    # there is nothing left for layers, and a model that is all embedding table
    # learns nothing. A vocabulary larger than the text can support is wasted
    # too, so the corpus caps it as well.
    ceiling = max(256, min(32000, target // 256))
    affordable = max(256, min(ceiling, material.characters // 40))
    tokenizer = train_tokenizer(material.text, affordable)
    actual_vocab = tokenizer.get_vocab_size()

    overrides = {"vocab_size": actual_vocab}
    if context:
        overrides["max_position_embeddings"] = context
    # Designed against the vocabulary this corpus actually produced. A preset
    # sized for a 32,000-token vocabulary misses badly once the corpus caps it.
    config: TransformerConfig = design_config(target, **overrides)
    problems = config.validate()
    if problems:
        raise TeacherError("This architecture will not build: " + "; ".join(problems))

    model.path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(model.tokenizer_path))

    network = TransformerLM(config)
    network.save_pretrained(str(model.path))

    built = config.parameter_count()["total"]
    return {
        "size": format_parameter_count(target),
        "asked_for": target,
        # Widths move in steps, so a number you asked for usually lands a little
        # either side. Reporting what was asked for would be a small lie that
        # compounds every time someone repeats it.
        "parameters": built,
        "vocab_size": actual_vocab,
        "context": config.max_position_embeddings,
        "layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "heads": config.num_heads,
    }


# ------------------------------------------------------------------ adopt
def adopt(model: Model, base: str, *, trust_remote_code: bool = False) -> dict:
    """Download a pretrained model and make it the starting point for this one.

    Starting from a model that already knows a language beats starting from
    noise: the lessons then teach it your material rather than English itself.
    """
    repo = BASES.get(base, (base, ""))[0]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - transformers is a hard dependency
        raise TeacherError("transformers is not installed.") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        network = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=trust_remote_code, dtype=torch.float32
        )
    except Exception as exc:  # noqa: BLE001 - network, auth and bad names all land here
        raise TeacherError(
            f"Could not fetch '{repo}': {exc}\n"
            f"Check the name on huggingface.co, and that you are online."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.path.mkdir(parents=True, exist_ok=True)
    network.save_pretrained(str(model.path))
    tokenizer.save_pretrained(str(model.path))

    if not model.tokenizer_path.exists():
        raise TeacherError(
            f"'{repo}' ships a tokenizer Teacher cannot save as tokenizer.json. "
            f"Pick another base model."
        )

    parameters = sum(p.numel() for p in network.parameters())
    context = int(getattr(network.config, "max_position_embeddings", 0)
                  or getattr(network.config, "n_positions", 0) or 1024)
    model.note(base_repo=repo, adopted_at=time.time())
    return {
        "base": repo,
        "parameters": parameters,
        "vocab_size": int(getattr(network.config, "vocab_size", 0)),
        "context": context,
        "model_type": getattr(network.config, "model_type", "unknown"),
    }


def load_network(model: Model):
    """Load a model's weights, whichever kind it is."""
    if model.kind() == "studio":
        return TransformerLM.from_pretrained(str(model.path))
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(model.path), dtype=torch.float32)


def context_length(model: Model, network) -> int:
    config = getattr(network, "config", None)
    for field in ("max_position_embeddings", "n_positions"):
        value = getattr(config, field, None)
        if value:
            return int(value)
    return int(model.architecture().get("max_position_embeddings") or 1024)


# ------------------------------------------------------------------- teach
def teach(
    model: Model,
    material: Material,
    *,
    epochs: float = 3.0,
    batch_size: int | str = 8,
    learning_rate: float = 3e-4,
    keep_checkpoints: int = 2,
    lora: bool = False,
    lora_rank: int = 16,
    gpus: str | int = 1,
    device: str = "auto",
    answers: list[str] | None = None,
    against: "Material | None" = None,
    resume_from: int = 0,
    record: bool = True,
    on_log=None,
) -> dict:
    """Run a real training pass and save what it learned.

    Ordinarily that is over ``material`` — plain text, which teaches the model
    to continue. With ``answers`` it is over question-and-answer pairs instead,
    which teaches it to reply.
    """
    if not answers and material.characters < MIN_CHARACTERS:
        raise TeacherError(
            f"Only {material.characters:,} characters — too little for a lesson. "
            f"Give it at least {MIN_CHARACTERS:,}."
        )

    tokenizer = load_tokenizer(model)
    network = load_network(model)
    # A pretrained model's context can be far longer than a lesson needs; cap it
    # so one training block does not demand more material than there is.
    block = min(context_length(model, network), 512)

    style = None
    measured_on = "the held-out pairs"
    train_ids: list[int] = []
    eval_ids: list[int] = []

    if answers:
        # Pairs, not prose: the loss is taken on the answer only, which is what
        # turns a text continuer into something that replies.
        from teacher import answers as pairs

        records, style, ignored = pairs.read_pairs(list(answers))
        if on_log:
            on_log(f"[PREP] {pairs.summarise(records, style)}")
            for path, why in ignored[:3]:
                on_log(f"[WARN] skipped {path} — {why}")
        if len(records) < 8:
            raise TeacherError(
                f"Only {len(records)} pair(s) — too few to learn a habit of answering. "
                f"A few hundred is a sensible start."
            )

        held = max(1, int(len(records) * 0.05))
        train_set = pairs.build_dataset(records[:-held], tokenizer, style=style, max_length=block)
        eval_set = pairs.build_dataset(records[-held:], tokenizer, style=style, max_length=block)
    else:
        # verbose=False silences "Token indices sequence length is longer than
        # the specified maximum" — true of the corpus, irrelevant here:
        # PackedLMDataset cuts it into `block`-sized windows and the model never
        # sees it whole.
        ids = tokenizer(material.text, add_special_tokens=False, verbose=False)["input_ids"]
        if len(ids) < block * 2:
            raise TeacherError(
                f"The material is only {len(ids):,} tokens, and one training block is "
                f"{block}. Add more material, or make a model with a shorter context."
            )

        if against is not None and against.characters:
            # Measured against writing from somewhere else entirely. The tail of
            # the same corpus flatters a model that memorised it; a separate
            # file does not, because there is nothing there to have memorised.
            measured_on = "a separate file"
            train_ids = ids
            eval_ids = tokenizer(
                against.text, add_special_tokens=False, verbose=False)["input_ids"]
            if len(eval_ids) < block:
                raise TeacherError(
                    f"The text to measure against is only {len(eval_ids):,} tokens, and "
                    f"one block is {block}. Give it more, or drop --eval-from."
                )
        else:
            # Hold out the tail so the reported loss is on text never trained on.
            measured_on = "the tail of the same text"
            split = max(block, int(len(ids) * 0.05))
            train_ids, eval_ids = ids[:-split], ids[-split:]

        train_set = PackedLMDataset(train_ids, block)
        eval_set = PackedLMDataset(eval_ids, block) if len(eval_ids) >= block else None

    arch = model.architecture()
    # Depth, vocabulary and MLP width decide the memory a step needs, and none
    # of them can be read off a parameter count.
    shape = {
        "hidden_size": arch.get("hidden_size"),
        "num_layers": arch.get("num_hidden_layers") or arch.get("num_layers"),
        "num_heads": arch.get("num_attention_heads") or arch.get("num_heads"),
        "intermediate_size": arch.get("intermediate_size"),
        "vocab_size": arch.get("vocab_size"),
    }

    def configure(size: int) -> TrainingConfig:
        return TrainingConfig(
            method="lora" if lora else "continued_pretraining",
            epochs=float(epochs),
            batch_size=int(size),
            learning_rate=float(learning_rate),
            max_sequence_length=block,
            # "auto" takes what the card can do. Holding a modern GPU at FP32
            # halves its throughput and doubles its activation memory for
            # nothing — most of the difference between a card at 50% and a card
            # that is working.
            precision="auto",
            lr_scheduler="cosine",
            warmup_steps=max(5, int(len(train_set) * epochs / size * 0.03)),
            log_interval=max(1, len(train_set) // size // 10),
            eval_interval=max(10, len(train_set) // size // 4) if eval_set else 0,
            # About ten times a lesson. Nothing was saved within a round before
            # this, so a crash three hours in lost all three hours.
            checkpoint_interval=max(25, int(len(train_set) * epochs / size) // 10),
            keep_last_checkpoints=keep_checkpoints,
        )

    automatic = str(batch_size).strip().lower() == "auto"
    settings = configure(1 if automatic else int(batch_size))

    chosen_device, world_size = _choose_hardware(device, gpus, on_log=on_log)
    settings.num_threads = 0 if chosen_device == "cuda" else settings.num_threads

    total_parameters = network.num_parameters()
    lora_stats = _attach_lora(model, network, settings, lora_rank) if lora else None
    if lora_stats:
        network = lora_stats.pop("network")

    estimate = dict(
        parameter_count=total_parameters,
        trainable_parameters=lora_stats["trainable_parameters"] if lora_stats else None,
        **shape,
    )
    if automatic:
        # With more than one process each sees its own slice, so the steps an
        # epoch has are counted per process, not over the whole dataset.
        chosen, why = fit_batch_size(
            settings, samples=len(train_set) // world_size, **estimate)
        settings = configure(chosen)
        if on_log:
            on_log(f"[PREP] batch {chosen} is the largest that fits — {why}")

    if world_size > 1 and on_log:
        # Worth saying out loud: the gradient step is over every process's
        # batch at once, which is not the number that was typed.
        on_log(f"[PREP] effective batch {settings.batch_size * world_size} "
               f"({settings.batch_size} per GPU across {world_size})")
    settings.validate()

    check = preflight(settings, **estimate)
    if not check.ok:
        raise TeacherError(
            "This lesson would not fit in memory:\n  "
            + "\n  ".join(check.blockers)
            + ("\n\nTry:\n  " + "\n  ".join(check.suggestions) if check.suggestions else "")
        )

    started = time.time()

    if world_size > 1 and style:
        # The worker processes rebuild their datasets from a file of token ids,
        # which pairs are not. Rather than train the wrong thing quietly, say so.
        if on_log:
            on_log("[WARN] answer lessons run on one GPU for now; ignoring --gpus")
        world_size = 1

    if world_size > 1:
        # The workers load the model themselves and rank 0 writes it back, so
        # the state has to be put aside before they start rather than after.
        _snapshot_previous(model, keep_checkpoints)
        result = _teach_across_gpus(
            model, settings, train_ids, eval_ids, block,
            world_size=world_size,
            lora={"rank": lora_rank} if lora else None,
            on_log=on_log,
        )
        if result.status == "failed":
            raise TeacherError(f"The lesson failed: {result.error}")
        lora_stats = result.lora or lora_stats
    else:
        from teacher import interrupted

        plan = {
            "epochs": float(epochs),
            "batch_size": settings.batch_size,
            "learning_rate": float(learning_rate),
            "block": block,
            "style": style,
            "sources": list(answers) if answers else list(material.sources),
            "characters": material.characters,
            "fingerprint": interrupted.fingerprint(material.text or "".join(
                str(path) for path in (answers or []))),
            "lora": bool(lora),
            "lora_rank": int(lora_rank),
        }
        held: dict = {}

        def keep_progress(step, loss, val_loss, is_best):
            worker = held.get("trainer")
            if worker is not None:
                interrupted.write(
                    model, worker.model, worker.optimizer, worker.scheduler,
                    step=step, plan={**plan, "total_steps": held.get("total", 0)},
                )

        trainer = Trainer(
            model=network,
            train_dataset=train_set,
            eval_dataset=eval_set,
            config=settings,
            device=chosen_device,
            on_log=on_log,
            on_checkpoint=keep_progress,
            start_step=int(resume_from or 0),
        )
        held["trainer"] = trainer
        held["total"] = int(len(train_set) * epochs / max(1, settings.batch_size))
        result = trainer.train()
        interrupted.clear(model)
        if result.status == "failed":
            raise TeacherError(f"The lesson failed: {result.error}")

        # An adapter on its own would leave the folder unable to load, and
        # would quietly break rollback, branch, pack and Bench. Fold it in
        # instead: what LoRA buys here is a cheaper lesson, not a different
        # kind of model.
        if lora_stats:
            try:
                network = network.merge_and_unload()
            except Exception as exc:  # noqa: BLE001
                raise TeacherError(
                    f"The lesson ran, but its adapter could not be merged back in: {exc}\n"
                    f"The weights on disk are unchanged."
                ) from exc

        # Keep the previous state until the new one is safely written.
        _snapshot_previous(model, keep_checkpoints)
        network.save_pretrained(str(model.path))

    if style:
        # Read back when the model is asked something, so the prompt matches
        # the template it was taught with.
        model.note(answer_style=style)

    lesson = {
        "at": started,
        "style": style or "text",
        "measured_on": measured_on,
        "sources": list(answers) if answers else material.sources,
        "characters": material.characters,
        "pairs": len(train_set) if style else None,
        "tokens": len(train_ids) if not style else sum(
            len(sequence) for sequence in train_set.sequences),
        "epochs": float(epochs),
        "steps": result.steps,
        "final_loss": result.final_train_loss,
        "held_out_loss": result.best_val_loss,
        "perplexity": perplexity(result.best_val_loss or result.final_train_loss),
        "seconds": round(result.duration_seconds, 1),
        "device": f"{check.device} x{world_size}" if world_size > 1 else check.device,
        "gpus": world_size,
        "status": result.status,
    }
    if lora_stats:
        lesson["lora"] = {key: value for key, value in lora_stats.items() if key != "network"}
    if model.kind() != "studio" and model.history().get("base_loss") is None:
        # The first measurement is this model's own starting point.
        first = lesson["held_out_loss"] or lesson["final_loss"]
        if first is not None:
            model.note(base_loss=float(first))

    if record:
        model.record(lesson)
    return lesson


class _Outcome:
    """What came back from the worker processes, shaped like a TrainingResult."""

    def __init__(self, payload: dict) -> None:
        self.status = payload.get("status", "completed")
        self.steps = payload.get("steps", 0)
        self.final_train_loss = payload.get("final_train_loss")
        self.best_val_loss = payload.get("best_val_loss")
        self.duration_seconds = payload.get("duration_seconds", 0.0)
        self.error = payload.get("error")
        self.lora = payload.get("lora") or None


def _choose_hardware(device: str, gpus: str | int, *, on_log=None) -> tuple[str, int]:
    """Which device to train on, and across how many processes."""
    from ai_studio.core.errors import StudioError
    from ai_studio.training import distributed

    wanted = str(device).strip().lower()
    if wanted not in ("auto", "cpu", "cuda", "gpu"):
        raise TeacherError(f"Unknown device {device!r}. Use auto, cpu or cuda.")

    if wanted == "cpu":
        chosen = "cpu"
    elif wanted in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise TeacherError(
                "--device cuda was asked for, but no CUDA GPU is visible here.\n"
                "  Check that a GPU driver and a CUDA build of PyTorch are installed, "
                "or drop the flag to use the CPU."
            )
        chosen = "cuda"
    else:
        chosen = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        world_size = distributed.resolve_world_size(gpus)
    except StudioError as exc:
        raise TeacherError(exc.display()) from exc

    if chosen == "cpu" and world_size > 1:
        # Spreading a CPU run over processes fights for the same cores and adds
        # a gradient exchange to every step. It is slower, not faster.
        if on_log:
            on_log("[PREP] no GPU to spread across — training on the CPU in one process")
        world_size = 1
    if world_size > 1 and on_log:
        names = distributed.describe()["names"]
        on_log(f"[PREP] {world_size} GPUs over {distributed.backend('cuda')}: "
               f"{', '.join(names[:world_size])}")
    return chosen, world_size


def _teach_across_gpus(model, settings, train_ids, eval_ids, block, *,
                       world_size: int, lora: dict | None, on_log=None) -> _Outcome:
    """One lesson, one process per GPU, gradients averaged at every step."""
    import tempfile

    from ai_studio.core.errors import StudioError
    from ai_studio.training import distributed

    with tempfile.TemporaryDirectory(prefix="teacher-ddp-") as scratch:
        folder = Path(scratch)
        job = distributed.Job(
            model_dir=str(model.path),
            train_tokens=distributed.tokens_to_file(train_ids, folder, "train"),
            eval_tokens=(distributed.tokens_to_file(eval_ids, folder, "eval")
                         if len(eval_ids) >= block else None),
            block_size=block,
            settings=settings.to_dict(),
            world_size=world_size,
            lora=lora,
        )
        try:
            return _Outcome(distributed.launch(job, on_log=on_log))
        except StudioError as exc:
            raise TeacherError(exc.display()) from exc


def _attach_lora(model: Model, network, settings, rank: int) -> dict:
    """Wrap the model in LoRA adapters, or say plainly why that cannot happen."""
    if model.kind() == "studio":
        raise TeacherError(
            "LoRA trains a small adapter on top of a model that already knows a "
            "language. This one was built here, starting from noise — there is "
            "nothing to adapt yet, so train all of it instead (drop --lora)."
        )

    settings.lora.rank = int(rank)
    settings.lora.alpha = int(rank) * 2
    settings.lora.validate()

    from ai_studio.core.errors import StudioError
    from ai_studio.training.lora_trainer import apply_lora

    try:
        wrapped, stats = apply_lora(network, settings)
    except StudioError as exc:
        raise TeacherError(exc.display()) from exc
    stats["network"] = wrapped
    return stats


def _snapshot_previous(model: Model, keep: int) -> None:
    """Copy the current weights aside before they are overwritten."""
    if keep <= 0 or not (model.path / "model.safetensors").exists():
        return
    model.checkpoints.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = model.checkpoints / f"before-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        source = model.path / name
        if source.exists():
            shutil.copy2(source, target / name)

    existing = sorted(p for p in model.checkpoints.iterdir() if p.is_dir())
    for stale in existing[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


# ------------------------------------------------------------- teach until
#: How far to go. "best" means keep going until it stops improving.
TARGETS = {
    "words": 0.28,
    "sentences": 0.15,
    "best": 0.0,
}

#: Below this much relative improvement a round has bought nothing worth having.
PLATEAU = 0.01


def teach_until(
    model: Model,
    material: Material,
    *,
    target: str = "sentences",
    epochs_per_round: float = 3.0,
    max_rounds: int = 20,
    max_minutes: float = 0.0,
    on_round=None,
    **lesson_options,
) -> dict:
    """Keep teaching until the model reaches ``target`` or stops improving.

    Every round saves the model, so stopping early — by limit, by plateau or by
    Ctrl-C — always leaves the last completed round on disk.
    """
    if target not in TARGETS:
        raise TeacherError(f"Unknown target '{target}'. Choose one of: {', '.join(TARGETS)}")

    ceiling = TARGETS[target]
    started = time.time()
    deadline = started + max_minutes * 60 if max_minutes else None

    history: list[float] = []
    rounds: list[dict] = []
    reason = "reached the limit of rounds"

    # A model already past the target needs no lesson at all.
    previous = model.history().get("lessons", [])
    if previous and previous[-1].get("held_out_loss") is not None:
        _label, _note, share = judge(model, previous[-1]["held_out_loss"])
        if ceiling and share < ceiling:
            return {
                "rounds": 0,
                "reason": "it is already past that stage",
                "target": target,
                "first_loss": previous[-1]["held_out_loss"],
                "held_out_loss": previous[-1]["held_out_loss"],
                "epochs": 0.0,
                "seconds": 0.0,
                "lessons": [],
            }

    for index in range(1, max_rounds + 1):
        lesson = teach(
            model, material,
            epochs=epochs_per_round,
            record=False,
            **lesson_options,
        )
        rounds.append(lesson)
        loss = lesson["held_out_loss"] or lesson["final_loss"]
        label, _note, share = judge(model, loss)

        if on_round:
            on_round(index, lesson, label, share)

        if loss is None:
            reason = "there is no held-out text to measure against"
            break

        if ceiling and share < ceiling:
            reason = f"it reached '{label}'"
            break

        if history:
            gain = (history[-1] - loss) / max(history[-1], 1e-9)
            if gain < PLATEAU:
                reason = "it stopped improving"
                break
        history.append(loss)

        if deadline and time.time() >= deadline:
            reason = f"it ran for {max_minutes:g} minute(s)"
            break

    final = rounds[-1] if rounds else {}
    summary = {
        "at": started,
        "sources": material.sources,
        "characters": material.characters,
        "tokens": final.get("tokens", 0),
        "epochs": epochs_per_round * len(rounds),
        "rounds": len(rounds),
        "steps": sum(r.get("steps", 0) for r in rounds),
        "final_loss": final.get("final_loss"),
        "held_out_loss": final.get("held_out_loss"),
        "first_loss": rounds[0].get("held_out_loss") if rounds else None,
        "perplexity": final.get("perplexity"),
        "seconds": round(time.time() - started, 1),
        "device": final.get("device", "cpu"),
        "status": final.get("status", "completed"),
        "target": target,
        "reason": reason,
        "lessons": [r.get("held_out_loss") for r in rounds],
    }
    model.record(summary)
    return summary


# -------------------------------------------------------------------- talk
def talk(model: Model, prompt: str, *, max_new_tokens: int = 120, temperature: float = 0.8,
         top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1,
         seed: int | None = None) -> tuple[str, dict]:
    """Generate a continuation, mirroring what the Bench page does in the browser."""
    tokenizer = load_tokenizer(model)
    network = load_network(model).eval()

    from teacher import answers as pairs

    # A model taught with "### Question:" and then asked something bare answers
    # as if continuing a document, which looks like the training failed.
    prompt = pairs.opener(model, prompt)

    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids = [bos] + ids

    if seed is not None:
        torch.manual_seed(seed)

    started = time.time()
    with torch.no_grad():
        if model.kind() == "studio":
            output = network.generate(
                torch.tensor([ids]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output = network.generate(
                torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                top_k=top_k or None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    produced = output[0][len(ids):].tolist()
    elapsed = max(time.time() - started, 1e-6)
    text = tokenizer.decode(produced, skip_special_tokens=True)
    return text, {
        "prompt_tokens": len(ids),
        "generated": len(produced),
        "tokens_per_second": len(produced) / elapsed,
        "seconds": elapsed,
    }


# ------------------------------------------------------------------ compare
def compare(models: list[Model], prompt: str, **sampling) -> dict:
    """Run one prompt through several models under identical sampling.

    The seed is fixed for every model, so a difference in what comes back is a
    difference in the models and not in the dice.
    """
    if len(models) < 2:
        raise TeacherError("Name at least two models to compare.")
    sampling.setdefault("seed", 0)

    entries = []
    for model in models:
        arch = model.architecture()
        taught = model.history().get("lessons", [])
        last = taught[-1] if taught else {}
        loss = last.get("held_out_loss")
        label, note, share = judge(model, loss) if taught else ("never taught", "", 1.0)

        text, stats = talk(model, prompt, **sampling)
        entries.append({
            "model": model.name,
            "kind": model.kind(),
            "base": model.base_repo(),
            "branched_from": model.history().get("branched_from"),
            "vocab_size": arch.get("vocab_size"),
            "lessons": len(taught),
            "taught_characters": model.taught_characters(),
            "stage": label,
            "advice": note,
            "share": round(share, 4),
            "held_out_loss": loss,
            "reply": text,
            "tokens_per_second": round(stats["tokens_per_second"], 1),
        })

    same_vocabulary, comparable = comparability(entries)
    return {
        "prompt": prompt,
        "seed": sampling["seed"],
        "models": entries,
        "same_vocabulary": same_vocabulary,
        "losses_comparable": comparable,
    }


def comparability(entries: list[dict]) -> tuple[bool, bool]:
    """Whether these models' losses mean the same thing.

    A loss is an average over a vocabulary. Two models that carve text up
    differently are not being scored on the same scale at all, and putting
    their numbers side by side would invite exactly the wrong conclusion.
    """
    same_vocabulary = len({entry.get("vocab_size") for entry in entries}) == 1
    measured = all(entry.get("held_out_loss") is not None for entry in entries)
    return same_vocabulary, same_vocabulary and measured


# ------------------------------------------------------------------ verdict
# An untrained model's loss sits at ln(vocabulary): it is guessing uniformly.
# Progress is best read as a fraction of that baseline, because the baseline
# moves with the vocabulary size.
STAGES = (
    (0.70, "barely started",
     "It is still close to guessing. Expect noise, and broken characters where it "
     "picks half of a multi-byte letter."),
    (0.55, "learning the alphabet",
     "Fragments of real words, in no particular order."),
    (0.28, "learning words",
     "Recognisable words, shaky sentences."),
    (0.15, "learning sentences",
     "Sentences in the shape of your material."),
    (0.00, "has the shape of your text",
     "About as far as a model this size can go on this material."),
)


#: A pretrained model already knows a language, so the question is not whether it
#: has learned to write but whether it has taken on *your* material. Its baseline
#: is the loss of its own first lesson, not a uniform guess.
FITTING = (
    (0.95, "barely moved",
     "It still writes like its base model. Teach it more, or give it more material."),
    (0.75, "picking up your material",
     "Your material is starting to show through its own voice."),
    (0.55, "adapting well",
     "It writes in the register of your material."),
    (0.00, "closely fitted",
     "Fitted to your material. Going further risks memorising it rather than "
     "learning from it."),
)


def verdict(
    held_out_loss: float | None,
    vocab_size: int,
    *,
    baseline: float | None = None,
) -> tuple[str, str, float]:
    """Describe how far along a model is, as a share of where it started.

    ``baseline`` is the loss it began from: ``ln(vocabulary)`` for a model built
    from noise, or its own first measured loss for an adopted one.
    """
    if held_out_loss is None:
        return "unmeasured", "No held-out text, so there is nothing to judge it by.", 1.0

    stages = STAGES
    if baseline is None:
        baseline = math.log(max(vocab_size, 2))
    else:
        stages = FITTING

    share = max(held_out_loss, 0.0) / max(baseline, 1e-9)
    for threshold, label, note in stages:
        if share >= threshold:
            return label, note, share
    return stages[-1][1], stages[-1][2], share


def baseline_for(model: Model) -> float | None:
    """A pretrained model's starting loss, once its first lesson has measured it."""
    if model.kind() == "studio":
        return None
    return model.history().get("base_loss")


def judge(model: Model, held_out_loss: float | None) -> tuple[str, str, float]:
    """The verdict for this model, using whichever baseline suits its kind."""
    return verdict(held_out_loss, model_vocab(model), baseline=baseline_for(model))


def model_vocab(model: Model) -> int:
    return int(model.architecture().get("vocab_size") or 2)


def describe_sizes() -> str:
    lines = [f"  {key:<7} {format_parameter_count(target):>5}  {blurb}"
             for key, (target, blurb) in SIZES.items()]
    lines.append("")
    lines.append("  or any count you name: 70K, 4M, 51M, 1.5B, 250000")
    return "\n".join(lines)
