"""Dataset import, validation and preview."""

from __future__ import annotations

import io
import json


def _jsonl(rows) -> bytes:
    return "\n".join(json.dumps(row) for row in rows).encode()


def test_upload_validate_and_preview(client):
    payload = _jsonl(
        [
            {"instruction": "Explain LoRA", "input": "", "output": "Low-rank adapters."},
            {"instruction": "Explain QLoRA", "input": "", "output": "Quantized base + LoRA."},
            {"instruction": "", "output": "missing instruction"},
        ]
    )
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("sft.jsonl", io.BytesIO(payload), "application/jsonl")},
        data={"name": "unit-sft", "template": "instruction"},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["rows"] == 3
    assert dataset["template"] == "instruction"
    assert dataset["validation_report"]["invalid_rows"] == 1
    assert dataset["status"] == "invalid"

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?limit=2").json()
    assert len(preview["rows"]) == 2
    assert "instruction" in preview["columns"]

    report = client.post(
        f"/api/datasets/{dataset['id']}/validate", json={"template": "instruction"}
    ).json()
    assert report["valid_rows"] == 2
    assert any("instruction" in issue["message"] for issue in report["issues"])


def test_upload_rejects_unsupported_suffix(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("weights.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")},
        data={"name": "bad-upload"},
    )
    assert response.status_code == 422


def test_upload_rejects_path_traversal_filename(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("../../evil.jsonl", io.BytesIO(b'{"text": "x"}'), "application/jsonl")},
        data={"name": "evil"},
    )
    # The filename is sanitised to `evil.jsonl`; nothing may be written outside the root.
    assert response.status_code in {201, 409, 422}
    if response.status_code == 201:
        assert "/data/datasets/" in response.json()["local_path"].replace("\\", "/")


def test_templates_expose_required_schemas(client):
    templates = {t["key"]: t for t in client.get("/api/datasets/templates").json()}
    assert set(templates) >= {"instruction", "chat", "plain_text"}
    assert templates["instruction"]["schema_example"] == {
        "instruction": "",
        "input": "",
        "output": "",
    }
    assert templates["chat"]["schema_example"]["messages"][0]["role"] == "system"
    assert templates["plain_text"]["schema_example"] == {"text": ""}
