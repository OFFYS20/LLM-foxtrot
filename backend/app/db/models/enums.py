"""Shared vocabulary for the catalog, training, and evaluation domains."""

from __future__ import annotations

from enum import StrEnum


class ModelStatus(StrEnum):
    READY = "ready"
    LOADED = "loaded"
    TRAINING = "training"
    STOPPED = "stopped"
    IMPORTING = "importing"
    ERROR = "error"


class ModelFormat(StrEnum):
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    HF_REPO = "hf_repo"


class ModelSource(StrEnum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM_PATH = "custom_path"
    CHECKPOINT = "checkpoint"
    DEMO = "demo"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class DatasetFormat(StrEnum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TXT = "txt"
    PARQUET = "parquet"
    HF_DATASET = "hf_dataset"


class DatasetTemplate(StrEnum):
    INSTRUCTION = "instruction"
    CHAT = "chat"
    PLAIN_TEXT = "plain_text"
    RAW = "raw"


class DatasetStatus(StrEnum):
    READY = "ready"
    IMPORTING = "importing"
    INVALID = "invalid"
    ERROR = "error"


class TrainingMethod(StrEnum):
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"
    QLORA = "qlora"
    CONTINUED_PRETRAINING = "continued_pretraining"
    INSTRUCTION_TUNING = "instruction_tuning"


class Optimizer(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAFACTOR = "adafactor"
    SGD = "sgd"
    LION = "lion"


class LRScheduler(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_RESTARTS = "cosine_with_restarts"
    CONSTANT = "constant"
    CONSTANT_WARMUP = "constant_with_warmup"
    POLYNOMIAL = "polynomial"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RunProvenance(StrEnum):
    """Where a number came from — surfaced in the UI so nothing looks official
    that was not actually measured."""

    MEASURED = "measured"  # produced by a real engine on this machine
    SIMULATED = "simulated"  # demo mode / synthetic data
    IMPORTED = "imported"  # user-supplied result file


class BenchmarkCategory(StrEnum):
    REASONING = "reasoning"
    MATH = "math"
    CODING = "coding"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    LONG_CONTEXT = "long_context"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    SAFETY = "safety"
    CUSTOM = "custom"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
