"""Configuration.

Values come from ``config.yaml`` (searched next to the package, then the CWD),
with environment overrides prefixed ``AISTUDIO_``. Everything the application
writes lives under ``storage.root``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = "config.yaml"

#: Relative storage paths resolve against the directory holding config.yaml,
#: falling back to the package directory. Set at import time by `load_config`.
_BASE_DIR: Path = PACKAGE_ROOT


@dataclass
class StorageConfig:
    root: str = "./storage"
    models: str = "./storage/models"
    datasets: str = "./storage/datasets"
    checkpoints: str = "./storage/checkpoints"
    documents: str = "./storage/documents"
    tokenizers: str = "./storage/tokenizers"
    indexes: str = "./storage/indexes"
    exports: str = "./storage/exports"
    database: str = "./storage/ai_studio.db"


@dataclass
class TrainingDefaults:
    default_precision: str = "bf16"
    checkpoint_interval: int = 500
    eval_interval: int = 250
    log_interval: int = 10
    keep_last_checkpoints: int = 3
    seed: int = 42
    max_sequence_length: int = 512
    gradient_checkpointing: bool = False
    num_workers: int = 0


@dataclass
class DataDefaults:
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_chars: int = 64
    max_chunk_chars: int = 40_000
    train_split: float = 0.90
    validation_split: float = 0.05
    test_split: float = 0.05
    max_upload_mb: int = 512


@dataclass
class RagDefaults:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.0


@dataclass
class InferenceDefaults:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    context_strategy: str = "trim_oldest"  # trim_oldest | summarize | new_context | retrieve


@dataclass
class UIConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    title: str = "AI Studio"
    refresh_seconds: float = 2.0


@dataclass
class AppConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    data: DataDefaults = field(default_factory=DataDefaults)
    rag: RagDefaults = field(default_factory=RagDefaults)
    inference: InferenceDefaults = field(default_factory=InferenceDefaults)
    ui: UIConfig = field(default_factory=UIConfig)

    # ------------------------------------------------------------- resolved
    @property
    def root(self) -> Path:
        return _abs(self.storage.root)

    @property
    def models_dir(self) -> Path:
        return _abs(self.storage.models)

    @property
    def datasets_dir(self) -> Path:
        return _abs(self.storage.datasets)

    @property
    def checkpoints_dir(self) -> Path:
        return _abs(self.storage.checkpoints)

    @property
    def documents_dir(self) -> Path:
        return _abs(self.storage.documents)

    @property
    def tokenizers_dir(self) -> Path:
        return _abs(self.storage.tokenizers)

    @property
    def indexes_dir(self) -> Path:
        return _abs(self.storage.indexes)

    @property
    def exports_dir(self) -> Path:
        return _abs(self.storage.exports)

    @property
    def database_path(self) -> Path:
        return _abs(self.storage.database)

    @property
    def originals_dir(self) -> Path:
        """Imported files, byte-for-byte as supplied."""
        return self.documents_dir / "original"

    @property
    def cleaned_dir(self) -> Path:
        """Preprocessed text — kept separately so originals are never altered."""
        return self.documents_dir / "cleaned"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def ensure_directories(self) -> None:
        for path in (
            self.root,
            self.models_dir,
            self.datasets_dir,
            self.checkpoints_dir,
            self.documents_dir,
            self.originals_dir,
            self.cleaned_dir,
            self.tokenizers_dir,
            self.indexes_dir,
            self.exports_dir,
            self.logs_dir,
            self.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _abs(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_BASE_DIR / path).resolve()


def _merge(instance: Any, data: dict[str, Any]) -> Any:
    """Overlay a plain dict onto a (possibly nested) dataclass instance."""
    if not is_dataclass(instance) or not isinstance(data, dict):
        return instance
    known = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif value is not None:
            try:
                setattr(instance, key, type(current)(value) if current is not None else value)
            except (TypeError, ValueError):
                setattr(instance, key, value)
    return instance


def _apply_env(config: AppConfig) -> AppConfig:
    """``AISTUDIO_<SECTION>_<FIELD>`` overrides any file value."""
    for section_field in fields(config):
        section = getattr(config, section_field.name)
        if not is_dataclass(section):
            continue
        for item in fields(section):
            env_key = f"AISTUDIO_{section_field.name.upper()}_{item.name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            current = getattr(section, item.name)
            try:
                if isinstance(current, bool):
                    value: Any = raw.strip().lower() in {"1", "true", "yes", "on"}
                elif isinstance(current, int):
                    value = int(raw)
                elif isinstance(current, float):
                    value = float(raw)
                else:
                    value = raw
            except ValueError:
                value = raw
            setattr(section, item.name, value)
    return config


def find_config_file() -> Path | None:
    explicit = os.environ.get("AISTUDIO_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for candidate in (
        Path.cwd() / CONFIG_FILENAME,
        PACKAGE_ROOT / CONFIG_FILENAME,
        PACKAGE_ROOT.parent / CONFIG_FILENAME,
    ):
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path | None = None) -> AppConfig:
    global _BASE_DIR
    config = AppConfig()
    source = path or find_config_file()
    if source is not None:
        _BASE_DIR = source.parent
    if source and source.exists():
        try:
            import yaml

            raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            _merge(config, raw)
        except ImportError:
            pass  # PyYAML missing → defaults, which are already valid
        except Exception as exc:  # noqa: BLE001 - a broken config must not block startup
            print(f"[config] ignoring {source}: {exc}")
    _apply_env(config)
    config.ensure_directories()
    return config


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()
