"""Tokenizer training and management.

Trains real tokenizers with the HuggingFace ``tokenizers`` library and saves
them in a directory that ``transformers`` can load directly, so a tokenizer
trained here works with both the from-scratch models and HF models.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ai_studio.core import logging as log
from ai_studio.core.config import get_config
from ai_studio.core.database import get_db, new_id
from ai_studio.core.errors import DependencyMissingError, TokenizerError, ValidationError
from ai_studio.core.paths import dir_size, remove_path, slugify

TOKENIZER_KINDS = ("bpe", "byte_level_bpe", "wordpiece", "unigram")

DEFAULT_SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}


@dataclass
class TokenizerConfig:
    name: str
    kind: str = "byte_level_bpe"
    vocab_size: int = 8000
    min_frequency: int = 2
    lowercase: bool = False
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    extra_special_tokens: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in TOKENIZER_KINDS:
            raise ValidationError(f"Unknown tokenizer kind {self.kind!r}; expected one of {TOKENIZER_KINDS}")
        if not (50 <= self.vocab_size <= 500_000):
            raise ValidationError("vocab_size must be between 50 and 500,000")
        if self.min_frequency < 1:
            raise ValidationError("min_frequency must be at least 1")
        if not self.name.strip():
            raise ValidationError("Tokenizer name is required")

    @property
    def special_tokens(self) -> list[str]:
        ordered = [self.unk_token, self.bos_token, self.eos_token, self.pad_token]
        seen: list[str] = []
        for token in [*ordered, *self.extra_special_tokens]:
            if token and token not in seen:
                seen.append(token)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401

        return tokenizers
    except ImportError as exc:
        raise DependencyMissingError("tokenizers", "Tokenizer training") from exc


def _build_tokenizer(config: TokenizerConfig):
    """Construct an untrained tokenizer + matching trainer."""
    _require_tokenizers()
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    specials = config.special_tokens

    if config.kind == "byte_level_bpe":
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
    elif config.kind == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=specials,
            show_progress=False,
        )
    elif config.kind == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token=config.unk_token, max_input_chars_per_word=100))
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=specials,
            show_progress=False,
        )
    else:  # unigram — the SentencePiece-style option
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=config.vocab_size,
            special_tokens=specials,
            unk_token=config.unk_token,
            show_progress=False,
        )

    normalizer_steps = [normalizers.NFKC()]
    if config.lowercase:
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps)

    return tokenizer, trainer


def _attach_post_processor(tokenizer, config: TokenizerConfig) -> None:
    """Wrap sequences in BOS/EOS so causal LM training sees document bounds.

    Must run *after* training: the template needs the real vocabulary ids, and
    before training every special token would resolve to id 0.
    """
    from tokenizers.processors import TemplateProcessing

    bos_id = tokenizer.token_to_id(config.bos_token)
    eos_id = tokenizer.token_to_id(config.eos_token)
    if bos_id is None or eos_id is None:
        log.warning(
            "BOS/EOS not present in the trained vocabulary — skipping the post-processor.",
            source="tokenizer",
        )
        return
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        pair=f"{config.bos_token} $A {config.eos_token} {config.bos_token} $B {config.eos_token}",
        special_tokens=[(config.bos_token, bos_id), (config.eos_token, eos_id)],
    )


def train_tokenizer(
    config: TokenizerConfig,
    texts: Iterable[str] | None = None,
    *,
    document_ids: list[str] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Train a tokenizer on the given texts and/or Data Library documents."""
    config.validate()
    _require_tokenizers()

    corpus: list[str] = [text for text in (texts or []) if text and text.strip()]
    used_documents: list[str] = []

    if document_ids:
        from ai_studio.data.ingestion import document_text

        for document_id in document_ids:
            try:
                content = document_text(document_id)
            except Exception as exc:  # noqa: BLE001 - skip unreadable, keep training
                log.warning(f"Tokenizer: skipping document {document_id}: {exc}", source="tokenizer")
                continue
            if content.strip():
                corpus.append(content)
                used_documents.append(document_id)

    if not corpus:
        raise ValidationError(
            "No text to train on.",
            hint="Select at least one document in the Data Library, or paste text.",
        )

    total_chars = sum(len(text) for text in corpus)
    if total_chars < 500:
        raise ValidationError(
            f"Only {total_chars} characters of training text — far too little for a tokenizer.",
            hint="Import more documents; a few hundred KB is a sensible minimum.",
        )

    tokenizer, trainer = _build_tokenizer(config)
    if progress:
        progress(0.2, desc=f"Training {config.kind} tokenizer on {total_chars:,} characters")

    started = time.perf_counter()
    try:
        tokenizer.train_from_iterator(_iter_chunks(corpus), trainer=trainer, length=len(corpus))
    except Exception as exc:  # noqa: BLE001
        raise TokenizerError(f"Tokenizer training failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    _attach_post_processor(tokenizer, config)

    config_dir = get_config().tokenizers_dir / slugify(config.name)
    if config_dir.exists():
        raise ValidationError(
            f"A tokenizer directory named {config_dir.name!r} already exists.",
            hint="Pick a different name or delete the existing tokenizer.",
        )
    config_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(config_dir / "tokenizer.json"))
    _write_hf_config(config_dir, config, tokenizer.get_vocab_size())

    if progress:
        progress(0.9, desc="Saving tokenizer")

    record = {
        "id": new_id("tok"),
        "name": config.name,
        "kind": config.kind,
        "vocab_size": tokenizer.get_vocab_size(),
        "path": str(config_dir),
        "special_tokens": {
            "unk_token": config.unk_token,
            "bos_token": config.bos_token,
            "eos_token": config.eos_token,
            "pad_token": config.pad_token,
            "extra": config.extra_special_tokens,
        },
        "trained_on": used_documents,
        "stats": {
            "documents": len(used_documents),
            "texts": len(corpus),
            "characters": total_chars,
            "training_seconds": round(elapsed, 2),
            "requested_vocab_size": config.vocab_size,
            "min_frequency": config.min_frequency,
        },
        "source": "trained",
        "created_at": time.time(),
    }
    get_db().insert("tokenizers", record)
    log.info(
        f"Trained tokenizer '{config.name}' ({config.kind}, vocab {record['vocab_size']:,}) "
        f"on {total_chars:,} chars in {elapsed:.1f}s",
        source="tokenizer",
        context={"tokenizer_id": record["id"]},
    )
    return record


def _iter_chunks(corpus: list[str], chunk_chars: int = 200_000) -> Iterator[str]:
    """Feed the trainer in bounded pieces so memory stays flat on big corpora."""
    for text in corpus:
        if len(text) <= chunk_chars:
            yield text
            continue
        for start in range(0, len(text), chunk_chars):
            yield text[start : start + chunk_chars]


def _write_hf_config(directory: Path, config: TokenizerConfig, vocab_size: int) -> None:
    """Write the files ``transformers`` expects next to tokenizer.json."""
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 1_000_000,
                "unk_token": config.unk_token,
                "bos_token": config.bos_token,
                "eos_token": config.eos_token,
                "pad_token": config.pad_token,
                "clean_up_tokenization_spaces": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": config.unk_token,
                "bos_token": config.bos_token,
                "eos_token": config.eos_token,
                "pad_token": config.pad_token,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "ai_studio_tokenizer.json").write_text(
        json.dumps({**config.to_dict(), "actual_vocab_size": vocab_size}, indent=2),
        encoding="utf-8",
    )


def load_tokenizer(tokenizer_id_or_path: str):
    """Load a tokenizer as a ``PreTrainedTokenizerFast``."""
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError as exc:
        raise DependencyMissingError("transformers", "Loading tokenizers") from exc

    path = tokenizer_id_or_path
    record = get_db().get("tokenizers", tokenizer_id_or_path)
    if record:
        path = record["path"]

    directory = Path(path)
    if not directory.exists():
        # Not a local path — try it as a Hugging Face repo id.
        try:
            return AutoTokenizer.from_pretrained(str(path))
        except Exception as exc:  # noqa: BLE001
            raise TokenizerError(
                f"Tokenizer {path!r} was not found locally or on Hugging Face: {exc}"
            ) from exc

    tokenizer_file = directory / "tokenizer.json"
    specials_file = directory / "special_tokens_map.json"
    if tokenizer_file.exists():
        # Build directly from tokenizer.json: AutoTokenizer would first resolve the
        # model config, which warns noisily for this project's own architecture.
        specials = DEFAULT_SPECIAL_TOKENS
        if specials_file.exists():
            try:
                specials = {
                    key: value
                    for key, value in json.loads(specials_file.read_text()).items()
                    if isinstance(value, str)
                }
            except Exception:  # noqa: BLE001
                specials = DEFAULT_SPECIAL_TOKENS
        try:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **specials)
        except Exception as exc:  # noqa: BLE001
            raise TokenizerError(f"Could not load the tokenizer in {directory}: {exc}") from exc
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(directory))
        except Exception as exc:  # noqa: BLE001
            raise TokenizerError(f"No usable tokenizer in {directory}: {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def inspect_tokenizer(tokenizer_id: str, *, limit: int = 200) -> dict[str, Any]:
    """Vocabulary sample and configuration for the Tokenizer screen."""
    record = get_db().require("tokenizers", tokenizer_id)
    tokenizer = load_tokenizer(tokenizer_id)
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    return {
        "record": record,
        "vocab_size": len(vocab),
        "first_tokens": ordered[:limit],
        "last_tokens": ordered[-limit:],
        "special_tokens": {
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token,
            "unk": tokenizer.unk_token,
        },
    }


def preview_tokenization(tokenizer_id: str, text: str) -> dict[str, Any]:
    """Encode/decode a sample so you can see exactly how text is split."""
    if not text:
        raise ValidationError("Enter some text to tokenize.")
    tokenizer = load_tokenizer(tokenizer_id)
    encoding = tokenizer(text, add_special_tokens=True)
    ids = encoding["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "token_count": len(ids),
        "character_count": len(text),
        "chars_per_token": round(len(text) / max(1, len(ids)), 2),
        "ids": ids,
        "tokens": pieces,
        "decoded": decoded,
        "roundtrip_ok": tokenizer.decode(ids, skip_special_tokens=True).strip() == text.strip(),
    }


def list_tokenizers() -> list[dict[str, Any]]:
    return get_db().list("tokenizers", order_by="created_at DESC")


def register_pretrained_tokenizer(repo_id: str, name: str | None = None) -> dict[str, Any]:
    """Download and register a tokenizer from a Hugging Face repository."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DependencyMissingError("transformers", "Hugging Face tokenizers") from exc

    display = name or repo_id.split("/")[-1]
    directory = get_config().tokenizers_dir / slugify(display)
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(directory))
    except Exception as exc:  # noqa: BLE001
        raise TokenizerError(f"Could not fetch tokenizer {repo_id!r}: {exc}") from exc

    record = {
        "id": new_id("tok"),
        "name": display,
        "kind": "pretrained",
        "vocab_size": len(tokenizer.get_vocab()),
        "path": str(directory),
        "special_tokens": {
            "unk_token": tokenizer.unk_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
        },
        "trained_on": [],
        "stats": {"repo_id": repo_id, "size_bytes": dir_size(directory)},
        "source": "huggingface",
        "created_at": time.time(),
    }
    get_db().insert("tokenizers", record)
    log.info(f"Registered tokenizer {repo_id}", source="tokenizer")
    return record


def delete_tokenizer(tokenizer_id: str, *, remove_files: bool = True) -> None:
    db = get_db()
    record = db.require("tokenizers", tokenizer_id)
    if remove_files and record.get("path"):
        remove_path(Path(record["path"]))
    db.delete("tokenizers", tokenizer_id)
    log.warning(f"Deleted tokenizer {record['name']}", source="tokenizer")
