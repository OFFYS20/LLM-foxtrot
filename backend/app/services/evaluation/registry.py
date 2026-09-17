"""Benchmark suite registry."""

from __future__ import annotations

from typing import Any

from app.core.errors import NotFoundError, ValidationError
from app.schemas.benchmarks import BenchmarkSuiteInfo
from app.services.evaluation.base import BenchmarkAdapter
from app.services.evaluation.custom import CustomDatasetAdapter
from app.services.evaluation.suites import BUILTIN_ADAPTERS


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._classes = {cls.key: cls for cls in BUILTIN_ADAPTERS}

    def register(self, adapter_cls: type[BenchmarkAdapter]) -> None:
        """Third-party suites plug in here without touching the runner."""
        self._classes[adapter_cls.key] = adapter_cls

    def keys(self) -> list[str]:
        return [*self._classes.keys(), "custom"]

    def get(self, key: str, *, dataset: Any | None = None) -> BenchmarkAdapter:
        if key == "custom":
            if dataset is None:
                raise ValidationError("The custom suite requires a dataset_id")
            return CustomDatasetAdapter(dataset)
        cls = self._classes.get(key)
        if cls is None:
            raise NotFoundError(f"Unknown benchmark suite {key!r}")
        return cls()

    def list_info(self) -> list[BenchmarkSuiteInfo]:
        infos = [cls().info() for cls in self._classes.values()]
        infos.append(
            BenchmarkSuiteInfo(
                key="custom",
                label="Custom dataset",
                category="custom",
                description="Evaluate against any imported dataset with question/answer columns.",
                metric="exact_match",
                default_shots=0,
                available_items=0,
                data_source="user_dataset",
                official=False,
                notes="Pick a dataset when configuring the run.",
            )
        )
        return infos


registry = BenchmarkRegistry()
