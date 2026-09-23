"""A local API that speaks OpenAI's dialect, so other programs can use a model.

    python -m teacher serve seabot

Anything that talks to OpenAI's API can then be pointed at
``http://127.0.0.1:8008/v1`` instead: editors, chat front-ends, scripts using
the ``openai`` package. The model answers from this machine; nothing is sent
anywhere.

What it offers, and no more:

* ``GET  /v1/models``            — the models being served
* ``POST /v1/completions``       — continue a prompt
* ``POST /v1/chat/completions``  — reply to a conversation
* ``GET  /health``               — is it up

Both generation routes stream with ``"stream": true``.

What it is careful about:

* **Where it listens.** 127.0.0.1 unless told otherwise, so only this machine
  can reach it. Listening wider is allowed, and says so, and is what
  ``--api-key`` is for.
* **One reply at a time.** A model generating two replies at once would need
  twice the memory and give each half the speed; requests wait their turn.
* **Current weights.** A lesson that rewrites the weights while the server is
  running is picked up by the next request, not after a restart.
* **The template.** A model taught to answer in a particular form is asked in
  that form, because asked any other way it answers as if continuing a
  document.
"""

# No "from __future__ import annotations" here: FastAPI reads the handlers'
# annotations at runtime to tell a Request from a query parameter, and a
# string annotation naming a class imported inside a function reads as neither.

import hmac
import json
import threading
import time
import uuid

from teacher import generation, workspace
from teacher.answers import OPENERS
from teacher.workspace import TeacherError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8008

#: The longest reply a request may ask for; a client asking for more gets this.
MOST_TOKENS = 2048
#: The longest prompt accepted, in characters, before any tokenizing is done.
MOST_CHARACTERS = 200_000

#: How each template marks the start of the next turn. Generation stops there,
#: so a reply does not run on into a question it wrote for itself.
TURN_MARKS = {
    "qa": ("\n### Question:",),
    "instruction": ("\n### Instruction:",),
    "chat": ("<|user|>", "<|system|>"),
}


# ----------------------------------------------------------- conversations
def chat_prompt(messages: list[dict], style: str | None) -> str:
    """A conversation, laid out the way this model was taught to read one.

    A model taught questions and answers is shown earlier turns as earlier
    questions and answers. One taught only text has no template, so the turns
    are written out as a plain transcript for it to continue.
    """
    system = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if not turns or turns[-1]["role"] != "user":
        raise TeacherError("A conversation has to end with something from the user.")

    if style == "chat":
        lines = [f"<|system|>\n{system}"] if system else []
        lines += [f"<|{m['role']}|>\n{m['content']}" for m in turns]
        return "\n".join(lines) + "\n<|assistant|>\n"

    if style in OPENERS:
        opener = OPENERS[style]
        parts = [system + "\n\n"] if system else []
        for message in turns:
            if message["role"] == "user":
                parts.append(opener.format(prompt=message["content"]))
            else:
                parts.append(message["content"].strip() + "\n\n")
        return "".join(parts)

    lines = [system, ""] if system else []
    for message in turns:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines) + "\nAssistant:"


def plain_stops(style: str | None) -> tuple[str, ...]:
    """Where a reply to a chat request ends, for a model with no template."""
    return TURN_MARKS.get(style, ("\nUser:",))


# ----------------------------------------------------------------- request
def _messages(body: dict) -> list[dict]:
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'messages' must be a non-empty list.")
    messages = []
    for item in raw:
        if not isinstance(item, dict) or item.get("role") not in ("system", "user", "assistant"):
            raise ValueError("Each message needs a role of system, user or assistant.")
        content = item.get("content")
        if isinstance(content, list):
            # The newer shape: a list of parts. Text parts are all a text model can read.
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
        if not isinstance(content, str):
            raise ValueError("Each message needs text content.")
        messages.append({"role": item["role"], "content": content})
    return messages


def _sampling(body: dict, stops: tuple[str, ...]) -> generation.Sampling:
    def number(name, default, low, high, kind=float):
        value = body.get(name, default)
        if value is None:
            value = default
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number.") from None
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be between {low} and {high}.")
        return value

    if body.get("n", 1) not in (1, None):
        raise ValueError("Only one choice per request ('n': 1) is supported.")
    stop = body.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    if not isinstance(stop, (list, tuple)) or not all(isinstance(s, str) for s in stop):
        raise ValueError("'stop' must be a string or a list of strings.")
    if len(stop) > 4:
        raise ValueError("At most 4 stop sequences.")

    # Newer clients send max_completion_tokens; older ones max_tokens.
    wanted = body.get("max_completion_tokens")
    if wanted is None:
        wanted = body.get("max_tokens")
    try:
        wanted = 256 if wanted is None else int(wanted)
    except (TypeError, ValueError):
        raise ValueError("'max_tokens' must be a whole number.") from None
    if wanted < 1:
        raise ValueError("'max_tokens' must be at least 1.")
    seed = body.get("seed")
    return generation.Sampling(
        max_new_tokens=wanted,
        temperature=number("temperature", 0.8, 0.0, 2.0),
        top_p=number("top_p", 0.95, 0.0, 1.0),
        top_k=40,
        repetition_penalty=number("repetition_penalty", 1.1, 1.0, 2.0),
        seed=int(seed) if isinstance(seed, (int, float)) else None,
        stop=tuple(stop) + tuple(stops),
    )


# --------------------------------------------------------------------- app
class Models:
    """The models being served, each loaded once and reloaded when retaught."""

    def __init__(self, names: list[str]) -> None:
        self.names = [workspace.get(name).name for name in names]
        self.held: dict[str, generation.Loaded] = {}
        self.lock = threading.Lock()
        self.started = int(time.time())

    def get(self, name: str | None) -> generation.Loaded:
        if not name and len(self.names) == 1:
            name = self.names[0]
        if name not in self.names:
            raise LookupError(
                f"No model called {name!r} is being served. "
                f"Served: {', '.join(self.names)}."
            )
        model = workspace.get(name)
        with self.lock:
            held = self.held.get(name)
            if held is None or held.stamp != model.weights_stamp():
                self.held[name] = generation.load(model, keep=False)
            return self.held[name]


def build_app(names: list[str], *, api_key: str | None = None):
    """The FastAPI application serving ``names``."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - installed with Gradio
        raise TeacherError(
            "Serving needs FastAPI, which comes with Gradio: pip install gradio"
        ) from exc

    models = Models(names)
    for name in models.names:
        models.get(name)  # load now: a broken model should fail at start, not at first use
    # One reply at a time, across every model: two at once would each run at
    # half speed in twice the memory.
    turn = threading.Lock()

    app = FastAPI(title="Teacher", docs_url=None, redoc_url=None, openapi_url=None)

    def error(status: int, message: str, kind: str = "invalid_request_error",
              code: str | None = None) -> JSONResponse:
        return JSONResponse(status_code=status, content={
            "error": {"message": message, "type": kind, "param": None, "code": code}})

    @app.middleware("http")
    async def check_key(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            given = request.headers.get("authorization", "")
            expected = f"Bearer {api_key}"
            if not hmac.compare_digest(given.encode(), expected.encode()):
                return error(401, "Missing or wrong API key. Send: Authorization: Bearer <key>",
                             "authentication_error", "invalid_api_key")
        return await call_next(request)

    @app.get("/health")
    def health():
        # Behind a key, even the names of the models are the key's to tell.
        return {"status": "ok"} if api_key else {"status": "ok", "models": models.names}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": name, "object": "model", "created": models.started, "owned_by": "teacher"}
            for name in models.names
        ]}

    async def read(request: Request) -> dict:
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"The request body is not valid JSON: {exc.msg}.") from None
        if not isinstance(body, dict):
            raise ValueError("The request body must be a JSON object.")
        return body

    async def respond(body: dict, prompt: str, loaded, sampling, *, chat: bool):
        """Run one generation, whole or streamed, in the shape the route promises."""
        if len(prompt) > MOST_CHARACTERS:
            return error(400, f"The prompt is {len(prompt):,} characters; the most accepted "
                              f"is {MOST_CHARACTERS:,}.")
        ids = generation.encode(loaded, prompt)
        sampling.max_new_tokens = min(sampling.max_new_tokens, MOST_TOKENS)
        if len(ids) >= loaded.context:
            return error(400, f"This model's context is {loaded.context:,} tokens; the prompt "
                              f"is {len(ids):,}.", code="context_length_exceeded")

        reply_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        name = loaded.model.name
        kind = "chat.completion" if chat else "text_completion"

        def usage(result):
            return {"prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}

        if not body.get("stream"):
            def whole():
                with turn:
                    return generation.generate(loaded, ids, sampling)

            # Off the event loop, so a long reply does not stall every other request.
            result = await run_in_threadpool(whole)
            choice = ({"index": 0, "message": {"role": "assistant", "content": result["text"]},
                       "finish_reason": result["finish_reason"]} if chat else
                      {"index": 0, "text": result["text"], "logprobs": None,
                       "finish_reason": result["finish_reason"]})
            return {"id": reply_id, "object": kind, "created": created, "model": name,
                    "choices": [choice], "usage": usage(result)}

        return StreamingResponse(
            _stream(loaded, ids, sampling, turn, chat=chat, reply_id=reply_id,
                    created=created, name=name, usage=usage,
                    with_usage=bool((body.get("stream_options") or {}).get("include_usage"))),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await read(request)
            loaded = await run_in_threadpool(models.get, body.get("model"))
            prompt = body.get("prompt", "")
            if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
                prompt = prompt[0]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("'prompt' must be a non-empty string.")
            sampling = _sampling(body, ())
        except LookupError as exc:
            return error(404, str(exc), code="model_not_found")
        except (ValueError, TeacherError) as exc:
            return error(400, str(exc))
        return await respond(body, prompt, loaded, sampling, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await read(request)
            loaded = await run_in_threadpool(models.get, body.get("model"))
            style = loaded.model.history().get("answer_style")
            prompt = chat_prompt(_messages(body), style)
            sampling = _sampling(body, plain_stops(style))
        except LookupError as exc:
            return error(404, str(exc), code="model_not_found")
        except (ValueError, TeacherError) as exc:
            return error(400, str(exc))
        return await respond(body, prompt, loaded, sampling, chat=True)

    return app


async def _stream(loaded, ids, sampling, turn, *, chat, reply_id, created, name, usage,
                  with_usage):
    """Server-sent events, one per piece of text, as OpenAI's API sends them.

    Generation runs in its own thread and hands text over through a queue; a
    client that goes away sets the flag that ends the generation early.
    """
    import queue

    from starlette.concurrency import run_in_threadpool

    pieces: queue.Queue = queue.Queue()
    cancelled = threading.Event()
    outcome: dict = {}

    def work():
        try:
            with turn:
                outcome.update(generation.generate(
                    loaded, ids, sampling, on_text=pieces.put, cancelled=cancelled))
        except Exception as exc:  # noqa: BLE001 - reported to the client, not raised into the void
            outcome["error"] = str(exc)
        finally:
            pieces.put(None)

    kind = "chat.completion.chunk" if chat else "text_completion"

    def event(choice: dict, **extra) -> str:
        payload = {"id": reply_id, "object": kind, "created": created, "model": name,
                   "choices": [choice] if choice else [], **extra}
        return f"data: {json.dumps(payload)}\n\n"

    def piece(text: str | None, finish: str | None = None, first: bool = False) -> dict:
        if chat:
            delta = {"role": "assistant", "content": text or ""} if first else (
                {"content": text} if text else {})
            return {"index": 0, "delta": delta, "finish_reason": finish}
        return {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish}

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        if chat:
            yield event(piece("", first=True))
        while True:
            text = await run_in_threadpool(pieces.get)
            if text is None:
                break
            yield event(piece(text))
        if "error" in outcome:
            yield f"data: {json.dumps({'error': {'message': outcome['error'], 'type': 'server_error'}})}\n\n"
        else:
            yield event(piece(None, outcome.get("finish_reason", "stop")))
            if with_usage:
                yield event({}, usage=usage(outcome))
        yield "data: [DONE]\n\n"
    finally:
        # Reached when the client disconnects, too: stop generating for no one.
        cancelled.set()


def run(names: list[str], *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
        api_key: str | None = None, on_ready=None) -> None:
    """Load the models and serve them until stopped."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - installed with Gradio
        raise TeacherError("Serving needs uvicorn, which comes with Gradio: pip install gradio") from exc
    app = build_app(names, api_key=api_key)
    if on_ready:
        on_ready()
    uvicorn.run(app, host=host, port=port, log_level="warning")
