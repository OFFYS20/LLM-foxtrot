"""Project CRUD — the container models, datasets and runs are organised under."""

from __future__ import annotations


def test_create_read_update_delete(client):
    created = client.post(
        "/api/projects",
        json={"name": "Instruction tuning", "description": "SFT experiments", "tags": ["sft"]},
    )
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["name"] == "Instruction tuning"
    assert project["is_demo"] is False
    assert project["experiment_count"] == 0
    assert project["running_job_count"] == 0

    listed = client.get("/api/projects").json()
    assert project["id"] in {row["id"] for row in listed["items"]}

    fetched = client.get(f"/api/projects/{project['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["description"] == "SFT experiments"

    updated = client.patch(
        f"/api/projects/{project['id']}", json={"description": "Renamed", "tags": ["sft", "lora"]}
    ).json()
    assert updated["description"] == "Renamed"
    assert updated["tags"] == ["sft", "lora"]
    assert updated["name"] == "Instruction tuning", "an unset field must not be cleared"

    assert client.delete(f"/api/projects/{project['id']}").json()["ok"] is True
    assert client.get(f"/api/projects/{project['id']}").status_code == 404


def test_duplicate_names_are_rejected(client):
    first = client.post("/api/projects", json={"name": "Unique name"})
    assert first.status_code == 201
    second = client.post("/api/projects", json={"name": "Unique name"})
    assert second.status_code == 409, second.text
    client.delete(f"/api/projects/{first.json()['id']}")


def test_blank_names_are_rejected(client):
    assert client.post("/api/projects", json={"name": "   "}).status_code == 422


def test_unknown_references_are_rejected(client):
    response = client.post(
        "/api/projects", json={"name": "Bad reference", "base_model_id": "mdl-does-not-exist"}
    )
    assert response.status_code == 422, response.text
    assert "does not exist" in response.text


def test_detail_resolves_names_and_counts(client, demo_model, demo_dataset):
    project = client.post(
        "/api/projects",
        json={
            "name": "Counted project",
            "base_model_id": demo_model["id"],
            "default_dataset_id": demo_dataset["id"],
        },
    ).json()

    assert project["base_model_name"]
    assert project["default_dataset_name"] == demo_dataset["name"]
    assert project["checkpoint_count"] == 0
    assert project["last_activity_at"] is None

    client.delete(f"/api/projects/{project['id']}")


def test_search_filters_the_listing(client):
    a = client.post("/api/projects", json={"name": "Retrieval work"}).json()
    b = client.post("/api/projects", json={"name": "Quantization work"}).json()

    found = client.get("/api/projects", params={"search": "retrieval"}).json()["items"]
    names = {row["name"] for row in found}
    assert "Retrieval work" in names
    assert "Quantization work" not in names

    client.delete(f"/api/projects/{a['id']}")
    client.delete(f"/api/projects/{b['id']}")


def test_deleting_a_project_detaches_rather_than_destroys_runs(client, demo_model, demo_dataset):
    project = client.post("/api/projects", json={"name": "Detach me"}).json()

    job = client.post(
        "/api/training/jobs",
        json={
            "name": "run-in-project",
            "project_id": project["id"],
            "model_id": demo_model["id"],
            "dataset_id": demo_dataset["id"],
            "start_immediately": False,
        },
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["id"]

    assert client.delete(f"/api/projects/{project['id']}").json()["ok"] is True

    survivor = client.get(f"/api/training/jobs/{job_id}")
    assert survivor.status_code == 200, "deleting a project must not delete its runs"
    assert survivor.json()["project_id"] is None

    client.delete(f"/api/training/jobs/{job_id}")
