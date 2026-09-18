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
from ai_studio.training.config import TrainingConfig, preflight
from ai_studio.training.data import PackedLMDataset
from ai_studio.training.trainer import Trainer, perplexity

from teacher.material import Material
from teacher.workspace import Model, TeacherError

# Each tier is ten times the one below it.
SIZES = {
    "tiny": ("nano-1m", "learns grammar in minutes on a laptop CPU"),
    "small": ("tiny-10m", "a few MB of text and some patience; still fine on a CPU"),
    "medium": ("base-100m", "wants a GPU and a library's worth of text"),
    "large": ("huge-1b", "a serious GPU (24GB+) and gigabytes of text"),
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


def load_tokenizer(model: Model) -> PreTrainedTokenizerFast:
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(model.tokenizer_path),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    return fast


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


# ------------------------------------------------------------------- teach
def teach(
    model: Model,
    material: Material,
    *,
    epochs: float = 3.0,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    keep_checkpoints: int = 2,
    on_log=None,
) -> dict:
    """Run a real training pass over ``material`` and save what it learned."""
    if material.characters < MIN_CHARACTERS:
        raise TeacherError(
            f"Only {material.characters:,} characters — too little for a lesson. "
            f"Give it at least {MIN_CHARACTERS:,}."
        )

    tokenizer = load_tokenizer(model)
    network = TransformerLM.from_pretrained(str(model.path))
    config = network.config
    block = config.max_position_embeddings

    ids = tokenizer(material.text, add_special_tokens=False)["input_ids"]
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

    settings = TrainingConfig(
        method="continued_pretraining",
        epochs=float(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        max_sequence_length=block,
        precision="fp32",
        lr_scheduler="cosine",
        warmup_steps=max(5, int(len(train_set) * epochs / batch_size * 0.03)),
        log_interval=max(1, len(train_set) // batch_size // 10),
        eval_interval=max(10, len(train_set) // batch_size // 4) if eval_set else 0,
        checkpoint_interval=0,
        keep_last_checkpoints=keep_checkpoints,
    )
    settings.validate()

    check = preflight(settings, parameter_count=network.num_parameters())
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
    model.record(lesson)
    return lesson


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


# -------------------------------------------------------------------- talk
def talk(model: Model, prompt: str, *, max_new_tokens: int = 120, temperature: float = 0.8,
         top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1,
         seed: int | None = None) -> tuple[str, dict]:
    """Generate a continuation, mirroring what the Bench page does in the browser."""
    tokenizer = load_tokenizer(model)
    network = TransformerLM.from_pretrained(str(model.path)).eval()

    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids = [bos] + ids

    if seed is not None:
        torch.manual_seed(seed)

    started = time.time()
    with torch.no_grad():
        output = network.generate(
            torch.tensor([ids]),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
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


def verdict(held_out_loss: float | None, vocab_size: int) -> tuple[str, str, float]:
    """Describe how far along a model is, as a share of the untrained baseline."""
    baseline = math.log(max(vocab_size, 2))
    if held_out_loss is None:
        return "unmeasured", "No held-out text, so there is nothing to judge it by.", 1.0
    share = max(held_out_loss, 0.0) / baseline
    for threshold, label, note in STAGES:
        if share >= threshold:
            return label, note, share
    return STAGES[-1][1], STAGES[-1][2], share


def model_vocab(model: Model) -> int:
    return int(model.architecture().get("vocab_size") or 2)


def describe_sizes() -> str:
    lines = []
    for key, (preset, blurb) in SIZES.items():
        params = preset_config(preset).parameter_count()["total"]
        size = f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        lines.append(f"  {key:<7} {size:>5}  {blurb}")
    return "\n".join(lines)
