"""teacher serve: an OpenAI-shaped API over a real model.

Every request here runs real generation on a tiny model. The replies are
noise — it is untrained — but the shapes, the token counts, the stopping and
the repeatability are what a client depends on, and those are exact.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-serve-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import generation, lessons, serve, workspace  # noqa: E402
from teacher.material import gather  # noqa: E402

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def model():
    text = " ".join(f"the {a} saw the {b} by the {c} ." for a in ("cat", "dog", "fox")
                    for b in ("hill", "sea", "moon") for c in ("road", "wall", "tree")) * 30
    made = workspace.get("served", must_exist=False)
    lessons.create(made, gather([], raw_text=text), "tiny", context=64)
    return made


@pytest.fixture(scope="module")
def client(model):
    return TestClient(serve.build_app([model.name]))


def complete(client, **body):
    body = {"model": "served", "prompt": "the cat", "max_tokens": 8, "seed": 1, **body}
    return client.post("/v1/completions", json=body)


def test_the_models_being_served_are_listed(client):
    listed = client.get("/v1/models").json()
    assert listed["object"] == "list" and [m["id"] for m in listed["data"]] == ["served"]


def test_a_completion_has_the_shape_clients_expect(client):
    reply = complete(client).json()
    assert reply["object"] == "text_completion" and reply["model"] == "served"
    choice = reply["choices"][0]
    assert isinstance(choice["text"], str) and choice["finish_reason"] in ("stop", "length")
    usage = reply["usage"]
    assert 1 <= usage["completion_tokens"] <= 8
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_the_same_seed_gives_the_same_reply(client):
    first = complete(client, seed=7, temperature=1.0).json()["choices"][0]["text"]
    again = complete(client, seed=7, temperature=1.0).json()["choices"][0]["text"]
    assert first == again


def test_a_chat_reply_comes_back_as_an_assistant_message(client):
    reply = client.post("/v1/chat/completions", json={
        "model": "served", "max_tokens": 6, "seed": 1,
        "messages": [{"role": "system", "content": "be brief"},
                     {"role": "user", "content": "the dog"}],
    }).json()
    assert reply["object"] == "chat.completion"
    assert reply["choices"][0]["message"]["role"] == "assistant"


def events(response) -> list:
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", "a stream ends by saying so"
    return [json.loads(line[6:]) for line in lines[:-1]]


def test_a_stream_adds_up_to_the_reply_it_streams(client):
    whole = complete(client, max_tokens=12, temperature=0).json()["choices"][0]["text"]
    streamed = events(complete(client, max_tokens=12, temperature=0, stream=True))
    assert "".join(chunk["choices"][0]["text"] for chunk in streamed) == whole
    assert streamed[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_a_chat_stream_opens_with_the_role(client):
    streamed = events(client.post("/v1/chat/completions", json={
        "model": "served", "max_tokens": 5, "seed": 1, "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "the fox"}],
    }))
    assert streamed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert streamed[-1]["usage"]["completion_tokens"] >= 1, "usage comes last when asked for"


def test_generation_stops_where_it_is_told_to(client):
    whole = complete(client, max_tokens=16, temperature=0).json()["choices"][0]["text"]
    if len(whole.strip()) < 6:
        pytest.skip("the untrained model said too little to cut")
    stop = whole[3:6]
    cut = complete(client, max_tokens=16, temperature=0, stop=[stop]).json()["choices"][0]
    assert cut["text"] == whole[:whole.index(stop)]
    assert cut["finish_reason"] == "stop"


def test_an_unknown_model_is_a_clear_404(client):
    reply = complete(client, model="gpt-4")
    assert reply.status_code == 404
    assert reply.json()["error"]["code"] == "model_not_found"
    assert "served" in reply.json()["error"]["message"], "it says what is being served"


def test_a_prompt_longer_than_the_context_is_refused(client):
    reply = complete(client, prompt="the cat sat " * 200)
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "context_length_exceeded"


def test_bad_requests_are_explained(client):
    assert complete(client, temperature=9).status_code == 400
    assert complete(client, n=3).status_code == 400
    assert client.post("/v1/completions", content=b"{not json").status_code == 400
    ended_by_assistant = client.post("/v1/chat/completions", json={
        "model": "served", "messages": [{"role": "assistant", "content": "hi"}]})
    assert ended_by_assistant.status_code == 400


def test_a_key_keeps_out_whoever_does_not_have_it(model):
    guarded = TestClient(serve.build_app([model.name], api_key="sesame"))
    assert guarded.get("/v1/models").status_code == 401
    assert guarded.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert guarded.get("/v1/models", headers={"Authorization": "Bearer sesame"}).status_code == 200
    assert "models" not in guarded.get("/health").json(), "nor does health give the names away"


def test_new_weights_on_disk_are_picked_up_without_a_restart(model):
    served = serve.Models([model.name])
    before = served.get(model.name)
    assert served.get(model.name) is before, "unchanged weights are not read twice"
    weights = model.path / "model.safetensors"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert served.get(model.name) is not before


# -------------------------------------------------------------- templates
def test_a_model_taught_questions_is_asked_in_its_template():
    prompt = serve.chat_prompt(
        [{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."},
         {"role": "user", "content": "How?"}], "qa")
    assert prompt == ("### Question:\nWhy?\n\n### Answer:\nBecause.\n\n"
                      "### Question:\nHow?\n\n### Answer:\n")


def test_a_text_model_is_given_a_transcript_to_continue():
    prompt = serve.chat_prompt([{"role": "user", "content": "Hello"}], None)
    assert prompt.endswith("User: Hello\nAssistant:")
    assert "\nUser:" in serve.plain_stops(None)


def test_text_that_might_begin_a_stop_string_is_held_back():
    assert generation._held_back("the end of ###", ("### Question:",)) == 3
    assert generation._held_back("nothing here", ("### Question:",)) == 0
    assert generation._cut("an answer\n### Question: more", ("\n### Question:",)) == ("an answer", True)
