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

from ai_studio.models.transformer import SIZE_PRESETS, TransformerConfig, TransformerLM, preset_config
from ai_studio.training.config import TrainingConfig, fit_batch_size, preflight
from ai_studio.training.data import PackedLMDataset
from ai_studio.training.trainer import Trainer, perplexity

from teacher.material import Material
from teacher.workspace import Model, TeacherError

SIZES = {
    "tiny": ("nano-1m", "learns grammar in minutes on a laptop CPU"),
    "small": ("tiny-10m", "a few MB of text and some patience; still fine on a CPU"),
    "medium": ("base-100m", "wants a GPU and a library's worth of text"),
    "large": ("large-500m", "a GPU with room to spare and a lot of text"),
    "huge": ("huge-1b", "a serious GPU (24GB+) and gigabytes of text"),
}

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
    if size not in SIZES:
        raise TeacherError(f"Unknown size '{size}'. Choose one of: {', '.join(SIZES)}")
    if material.characters < MIN_CHARACTERS:
        raise TeacherError(
            f"Only {material.characters:,} characters of material — too little to learn anything. "
            f"Give it at least {MIN_CHARACTERS:,}; a few hundred KB is a sensible start."
        )

    preset_name = SIZES[size][0]
    preset = SIZE_PRESETS[preset_name]

    # A vocabulary larger than the text can support wastes most of the model's
    # parameters on embeddings it never learns, so cap it by corpus size.
    affordable = max(256, min(preset["vocab_size"], material.characters // 40))
    tokenizer = train_tokenizer(material.text, affordable)
    actual_vocab = tokenizer.get_vocab_size()

    overrides = {"vocab_size": actual_vocab}
    if context:
        overrides["max_position_embeddings"] = context
    config: TransformerConfig = preset_config(preset_name, **overrides)
    problems = config.validate()
    if problems:
        raise TeacherError("This architecture will not build: " + "; ".join(problems))

    model.path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(model.tokenizer_path))

    network = TransformerLM(config)
    network.save_pretrained(str(model.path))

    return {
        "size": size,
        "preset": preset_name,
        "vocab_size": actual_vocab,
        "parameters": config.parameter_count()["total"],
        "context": config.max_position_embeddings,
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
    record: bool = True,
    on_log=None,
) -> dict:
    """Run a real training pass over ``material`` and save what it learned."""
    if material.characters < MIN_CHARACTERS:
        raise TeacherError(
            f"Only {material.characters:,} characters — too little for a lesson. "
            f"Give it at least {MIN_CHARACTERS:,}."
        )

    tokenizer = load_tokenizer(model)
    network = load_network(model)
    # A pretrained model's context can be far longer than a lesson needs; cap it
    # so one training block does not demand more material than there is.
    block = min(context_length(model, network), 512)

    # verbose=False silences "Token indices sequence length is longer than the
    # specified maximum" — true of the corpus, irrelevant here: PackedLMDataset
    # cuts it into `block`-sized windows and the model never sees it whole.
    ids = tokenizer(material.text, add_special_tokens=False, verbose=False)["input_ids"]
    if len(ids) < block * 2:
        raise TeacherError(
            f"The material is only {len(ids):,} tokens, and one training block is {block}. "
            f"Add more material, or make a model with a shorter context."
        )

    # Hold out the tail so the reported loss is measured on text never trained on.
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
            checkpoint_interval=0,
            keep_last_checkpoints=keep_checkpoints,
        )

    automatic = str(batch_size).strip().lower() == "auto"
    settings = configure(1 if automatic else int(batch_size))

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
        chosen, why = fit_batch_size(settings, samples=len(train_set), **estimate)
        settings = configure(chosen)
        if on_log:
            on_log(f"[PREP] batch {chosen} is the largest that fits — {why}")
    settings.validate()

    check = preflight(settings, **estimate)
    if not check.ok:
        raise TeacherError(
            "This lesson would not fit in memory:\n  "
            + "\n  ".join(check.blockers)
            + ("\n\nTry:\n  " + "\n  ".join(check.suggestions) if check.suggestions else "")
        )

    trainer = Trainer(
        model=network,
        train_dataset=train_set,
        eval_dataset=eval_set,
        config=settings,
        device="cuda" if torch.cuda.is_available() else "cpu",
        on_log=on_log,
    )

    started = time.time()
    result = trainer.train()
    if result.status == "failed":
        raise TeacherError(f"The lesson failed: {result.error}")

    # An adapter on its own would leave the folder unable to load, and would
    # quietly break rollback, branch, pack and Bench. Fold it in instead: what
    # LoRA buys here is a cheaper lesson, not a different kind of model.
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

    lesson = {
        "at": started,
        "sources": material.sources,
        "characters": material.characters,
        "tokens": len(train_ids),
        "epochs": float(epochs),
        "steps": result.steps,
        "final_loss": result.final_train_loss,
        "held_out_loss": result.best_val_loss,
        "perplexity": perplexity(result.best_val_loss or result.final_train_loss),
        "seconds": round(result.duration_seconds, 1),
        "device": check.device,
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
    lines = []
    for key, (preset, blurb) in SIZES.items():
        params = preset_config(preset).parameter_count()["total"]
        size = f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        lines.append(f"  {key:<7} {size:>5}  {blurb}")
    return "\n".join(lines)
