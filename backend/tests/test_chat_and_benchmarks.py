"""Chat streaming, playground comparison and benchmark runs (demo engine)."""

from __future__ import annotations

import json
import time


def _model(client, name: str) -> dict:
    response = client.post(
        "/api/models/import",
        json={"source": "huggingface", "repo_id": f"acme/{name}", "name": name},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_non_streaming_completion_reports_usage(client):
    model = _model(client, "chat-model")
    body = client.post(
        "/api/chat/completions",
        json={
            "model_id": model["id"],
            "messages": [{"role": "user", "content": "Explain gradient accumulation."}],
            "stream": False,
            "persist": False,
            "params": {"max_tokens": 64},
        },
    ).json()

    assert body["content"]
    usage = body["usage"]
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["time_to_first_token_ms"] > 0
    assert usage["provenance"] == "simulated"  # demo engine must never claim otherwise
    assert "simulated" in body["content"].lower()


def test_streaming_completion_emits_tokens_then_usage(client):
    model = _model(client, "stream-model")
    frames = []
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={
            "model_id": model["id"],
            "messages": [{"role": "user", "content": "Write a haiku about VRAM."}],
            "stream": True,
            "persist": False,
            "params": {"max_tokens": 40},
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            frames.append(json.loads(payload))

    assert any(f["type"] == "token" for f in frames)
    usage_frame = next(f for f in frames if f["type"] == "usage")
    assert usage_frame["usage"]["tokens_per_sec"] > 0
    assert usage_frame["usage"]["finish_reason"] in {"stop", "length"}


def test_conversation_persistence_and_message_editing(client):
    model = _model(client, "conv-model")
    conversation = client.post(
        "/api/conversations",
        json={"title": "Test chat", "model_id": model["id"], "system_prompt": "Be brief."},
    ).json()

    client.post(
        "/api/chat/completions",
        json={
            "model_id": model["id"],
            "conversation_id": conversation["id"],
            "messages": [{"role": "user", "content": "Hello there"}],
            "stream": False,
            "persist": True,
            "params": {"max_tokens": 32},
        },
    )

    detail = client.get(f"/api/conversations/{conversation['id']}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assistant = detail["messages"][1]
    assert assistant["tokens_per_sec"] > 0
    assert assistant["finish_reason"]

    edited = client.patch(
        f"/api/messages/{detail['messages'][0]['id']}", json={"content": "Edited prompt"}
    ).json()
    assert edited["content"] == "Edited prompt"

    deleted = client.delete(f"/api/messages/{assistant['id']}").json()
    assert deleted["ok"] is True
    assert len(client.get(f"/api/conversations/{conversation['id']}").json()["messages"]) == 1


def test_playground_compares_models_side_by_side(client):
    a = _model(client, "pg-a")
    b = _model(client, "pg-b")
    body = client.post(
        "/api/playground/run",
        json={
            "model_ids": [a["id"], b["id"]],
            "prompt": "Summarise LoRA in one sentence.",
            "params": {"max_tokens": 48},
        },
    ).json()

    assert len(body["entries"]) == 2
    for entry in body["entries"]:
        assert entry["latency_ms"] > 0
        assert entry["tokens_per_sec"] > 0
        assert entry["provenance"] == "simulated"

    vote = client.post(
        f"/api/playground/comparisons/{body['id']}/vote", json={"winner_model_id": a["id"]}
    ).json()
    assert vote["winner_model_id"] == a["id"]
    assert vote["votes"][a["id"]] == 1

    history = client.get("/api/playground/comparisons").json()
    assert history["total"] >= 1


def test_playground_rejects_more_than_four_models(client):
    ids = [_model(client, f"pg-limit-{i}")["id"] for i in range(5)]
    response = client.post("/api/playground/run", json={"model_ids": ids, "prompt": "hi"})
    assert response.status_code == 422


def test_benchmark_run_produces_graded_items(client):
    model = _model(client, "bench-model")
    run = client.post(
        "/api/benchmarks/run",
        json={
            "model_id": model["id"],
            "suite": "gsm8k",
            "config": {"num_examples": 3, "max_tokens": 48, "temperature": 0.0},
        },
    ).json()
    assert run["status"] == "queued"
    assert run["config"]["official_split"] is False  # bundled sample, never called official

    deadline = time.time() + 60
    detail = {}
    while time.time() < deadline:
        detail = client.get(f"/api/benchmarks/results/{run['id']}").json()
        if detail["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)

    assert detail["status"] == "completed", detail.get("error")
    assert detail["completed_items"] == 3
    assert len(detail["items"]) == 3
    assert detail["overall_score"] is not None
    assert detail["avg_latency_ms"] > 0
    for item in detail["items"]:
        assert item["prompt"]
        assert item["expected"]

    incorrect = client.get(f"/api/benchmarks/results/{run['id']}?only_incorrect=true").json()[
        "items"
    ]
    assert all(item["correct"] is False for item in incorrect)


def test_code_suites_do_not_report_execution_pass_at_1(client):
    suites = {s["key"]: s for s in client.get("/api/benchmarks/suites").json()}
    assert suites["humaneval"]["requires_execution"] is True
    notes = suites["humaneval"]["notes"].lower()
    assert "pass@1 is disabled" in notes
    assert "never runs model-generated code" in notes


def test_model_comparison_exposes_individual_metrics(client):
    a = _model(client, "cmp-a")
    b = _model(client, "cmp-b")
    body = client.post("/api/evaluations/compare", json={"model_ids": [a["id"], b["id"]]}).json()
    assert len(body["rows"]) == 2
    row = body["rows"][0]
    for field in (
        "parameters",
        "size_bytes",
        "context_length",
        "benchmark_score",
        "validation_loss",
        "inference_tokens_per_sec",
        "vram_usage_mb",
    ):
        assert field in row
    assert "measured_metrics" in row
