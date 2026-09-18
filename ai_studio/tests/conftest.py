"""Test fixtures — every test runs against a throwaway storage root."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_ROOT = Path(tempfile.mkdtemp(prefix="ai-studio-tests-"))
for key in ("ROOT", "MODELS", "DATASETS", "CHECKPOINTS", "DOCUMENTS", "TOKENIZERS", "INDEXES", "EXPORTS"):
    os.environ[f"AISTUDIO_STORAGE_{key}"] = str(TMP_ROOT / key.lower())
os.environ["AISTUDIO_STORAGE_DATABASE"] = str(TMP_ROOT / "test.db")

from ai_studio.core.config import get_config  # noqa: E402
from ai_studio.core.database import get_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _storage() -> Path:
    config = get_config()
    config.ensure_directories()
    get_db()
    return TMP_ROOT


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_corpus() -> str:
    subjects = ["the cat", "the dog", "a bird", "the fox"]
    verbs = ["sat on", "jumped over", "ran past", "looked at"]
    objects = ["the mat", "the fence", "the river", "the moon"]
    return " ".join(
        f"{s} {v} {o} ." for _ in range(30) for s in subjects for v in verbs for o in objects
    )
