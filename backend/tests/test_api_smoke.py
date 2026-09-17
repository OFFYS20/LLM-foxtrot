"""End-to-end API smoke tests over the demo providers."""

from __future__ import annotations


def test_health_reports_compute_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["compute_mode"] in {"cuda", "rocm", "mps", "cpu"}


def test_system_info_lists_capabilities(client):
    body = client.get("/api/system").json()
    names = {c["name"] for c in body["capabilities"]}
    assert "gpu" in names
    assert any(name.startswith("inference.") for name in names)


def test_root_aliases_match_api_prefix(client):
    assert client.get("/models").status_code == client.get("/api/models").status_code


def test_models_crud_roundtrip(client):
    created = client.post(
        "/api/models/import",
        json={"source": "huggingface", "repo_id": "acme/tiny-llm", "precision": "fp16"},
    )
    assert created.status_code == 201, created.text
    model = created.json()
    assert model["repo_id"] == "acme/tiny-llm"
    assert model["is_demo"] is False

    clone = client.post(f"/api/models/{model['id']}/clone", json={"name": "tiny-llm-clone"}).json()
    assert clone["parent_model_id"] == model["id"]

    assert client.delete(f"/api/models/{clone['id']}").json()["ok"] is True
    assert client.delete(f"/api/models/{model['id']}").json()["ok"] is True


def test_model_import_rejects_bad_repo_id(client):
    response = client.post(
        "/api/models/import", json={"source": "huggingface", "repo_id": "../../etc/passwd"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"validation_error", "request_validation_error"}


def test_model_import_rejects_path_traversal(client):
    response = client.post(
        "/api/models/import", json={"source": "local", "path": "../../../etc", "name": "evil"}
    )
    assert response.status_code in {400, 422}


def test_benchmark_suites_do_not_claim_official_results(client):
    suites = client.get("/api/benchmarks/suites").json()
    assert suites
    for suite in suites:
        if suite["data_source"] == "bundled_sample":
            assert suite["official"] is False
            assert "not the official" in (suite["notes"] or "").lower()


def test_hardware_snapshot_declares_provenance(client):
    body = client.get("/api/hardware").json()
    assert body["provenance"] in {"measured", "simulated"}
    assert "compute_mode" in body
