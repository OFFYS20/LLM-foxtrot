"""Test fixtures — each test module gets an isolated database and data root."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TMP_ROOT = Path(tempfile.mkdtemp(prefix="foxtrot-tests-"))
os.environ.setdefault("FOXTROT_DATABASE_URL", f"sqlite:///{TMP_ROOT / 'test.db'}")
os.environ.setdefault("FOXTROT_DATA_DIR", str(TMP_ROOT / "data"))
os.environ.setdefault("FOXTROT_SEED_DEMO_DATA", "false")
os.environ.setdefault("FOXTROT_DEMO_MODE", "true")
os.environ.setdefault("FOXTROT_HARDWARE_PROVIDER", "simulated")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def data_root() -> Path:
    return TMP_ROOT / "data"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def demo_model(client):
    from app.services.demo.seed import demo_data_present, seed_demo_data

    if not demo_data_present():
        seed_demo_data()
    models = client.get("/api/models").json()["items"]
    return next(m for m in models if m["name"] == "foxtrot-7b-base")


@pytest.fixture
def demo_dataset(client, demo_model):
    datasets = client.get("/api/datasets").json()["items"]
    return datasets[0]
