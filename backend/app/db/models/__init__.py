"""ORM model registry — import this module to register every mapper."""

from app.db.models.catalog import Dataset, Model, Project
from app.db.models.chat import Conversation, Message, PlaygroundComparison
from app.db.models.enums import (
    BenchmarkCategory,
    DatasetFormat,
    DatasetStatus,
    DatasetTemplate,
    JobStatus,
    LogLevel,
    LRScheduler,
    ModelFormat,
    ModelSource,
    ModelStatus,
    Optimizer,
    Precision,
    RunProvenance,
    TrainingMethod,
)
from app.db.models.evaluation import BenchmarkItem, BenchmarkRun, ModelComparison
from app.db.models.system import HardwareSample, LogEntry, Setting
from app.db.models.training import Checkpoint, Experiment, TrainingJob, TrainingMetric

__all__ = [
    "BenchmarkCategory",
    "BenchmarkItem",
    "BenchmarkRun",
    "Checkpoint",
    "Conversation",
    "Dataset",
    "DatasetFormat",
    "DatasetStatus",
    "DatasetTemplate",
    "Experiment",
    "HardwareSample",
    "JobStatus",
    "LRScheduler",
    "LogEntry",
    "LogLevel",
    "Message",
    "Model",
    "ModelComparison",
    "ModelFormat",
    "ModelSource",
    "ModelStatus",
    "Optimizer",
    "PlaygroundComparison",
    "Precision",
    "Project",
    "RunProvenance",
    "Setting",
    "TrainingJob",
    "TrainingMethod",
    "TrainingMetric",
]
