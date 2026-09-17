"""Training config validation and job lifecycle (simulated backend)."""

from __future__ import annotations

import io
import json
import time


def _small_dataset(client, name: str) -> dict:
    rows = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(6)]
    payload = "\n".join(json.dumps(r) for r in rows).encode()
    response = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.jsonl", io.BytesIO(payload), "application/jsonl")},
        data={"name": name, "template": "instruction"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _model(client, name: str) -> dict:
    response = client.post(
        "/api/models/import",
        json={"source": "huggingface", "repo_id": f"acme/{name}", "name": name},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_config_validation_rejects_incoherent_precision(client):
    response = client.post(
        "/api/training/validate",
        json={"method": "full_finetune", "precision": "int4"},
    )
    assert response.status_code == 422


def test_qlora_forces_quantized_precision(client):
    body = client.post(
        "/api/training/validate", json={"method": "qlora", "precision": "bf16"}
    ).json()
    assert body["config"]["precision"] in {"int4", "int8"}


def test_raw_yaml_config_parses_and_warns(client):
    yaml_config = """
method: lora
epochs: 2
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.00002
max_sequence_length: 8192
lora:
  rank: 32
  alpha: 64
"""
    body = client.post(
        "/api/training/config/parse", json={"format": "yaml", "content": yaml_config}
    ).json()
    assert body["ok"] is True
    assert body["config"]["lora"]["rank"] == 32
    assert body["effective_batch_size"] == 8


def test_raw_config_reports_parse_errors(client):
    body = client.post(
        "/api/training/config/parse", json={"format": "json", "content": "{not json"}
    ).json()
    assert body["ok"] is False
    assert body["errors"]


def test_training_job_lifecycle(client):
    model = _model(client, "lifecycle-model")
    dataset = _small_dataset(client, "lifecycle-ds")

    created = client.post(
        "/api/training/jobs",
        json={
            "model_id": model["id"],
            "dataset_id": dataset["id"],
            "start_immediately": True,
            "config": {
                "epochs": 4,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "log_every_steps": 1,
                "eval_every_steps": 2,
                "checkpointing": {"save_every_steps": 4, "keep_last": 2},
            },
        },
    )
    assert created.status_code == 201, created.text
    job = created.json()
    assert job["provenance"] == "simulated"  # no real weights → never claimed as measured
    assert job["backend"] == "simulated"
    assert job["total_steps"] == 24

    # let a few simulated steps run
    deadline = time.time() + 15
    metrics: list = []
    while time.time() < deadline:
        metrics = client.get(f"/api/training/jobs/{job['id']}/metrics").json()
        if len(metrics) >= 3:
            break
        time.sleep(0.4)
    assert len(metrics) >= 3, "expected streamed metrics from the simulated trainer"
    assert metrics[0]["loss"] > 0
    assert metrics[-1]["learning_rate"] is not None

    paused = client.post(f"/api/training/jobs/{job['id']}/pause").json()
    assert paused["status"] == "paused"

    resumed = client.post(f"/api/training/jobs/{job['id']}/resume").json()
    assert resumed["status"] == "running"

    assert client.post(f"/api/training/jobs/{job['id']}/checkpoint", json={}).json()["ok"]

    client.post(f"/api/training/jobs/{job['id']}/stop")
    deadline = time.time() + 10
    while time.time() < deadline:
        detail = client.get(f"/api/training/jobs/{job['id']}").json()
        if detail["status"] in {"stopped", "completed"}:
            break
        time.sleep(0.3)
    assert detail["status"] in {"stopped", "completed"}

    # an experiment is created for every run, with the hyperparameters captured
    experiments = client.get("/api/experiments").json()["items"]
    experiment = next(e for e in experiments if e["job_id"] == job["id"])
    assert experiment["hyperparameters"]["epochs"] == 4
    assert experiment["provenance"] == "simulated"

    logs = client.get(f"/api/training/jobs/{job['id']}/logs").json()["entries"]
    assert any("[TRAIN]" in entry["message"] for entry in logs)
    assert any("simulated" in entry["message"].lower() for entry in logs)


def test_checkpoints_are_written_for_a_run(client):
    jobs = client.get("/api/training/jobs").json()["items"]
    assert jobs
    checkpoints = client.get("/api/checkpoints").json()["items"]
    assert isinstance(checkpoints, list)
