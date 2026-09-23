"""Teacher's window.

Everything the command line does, without the command line: bring in text, make
a model, teach it, watch how it is doing, and talk to it. The same functions
underneath, so a model made here is the same model made there.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from ai_studio.core import gradio_compat as compat
from teacher import card as cards
from teacher import exams, lessons, workspace
from teacher.material import gather
from teacher.workspace import TeacherError

#: What a lesson teaches: text to continue, or pairs to answer.
TEXT = "Text — it learns to continue it"
PAIRS = "Question-and-answer pairs — it learns to reply"

CSS = """
:root { --t-accent: #a85c32; }
.teacher-title { font-size: 21px; font-weight: 700; margin: 0; }
.teacher-sub { color: #6b7883; font-size: 13px; margin: 2px 0 0; }
.stage { font-size: 15px; font-weight: 600; }
.mono textarea, .mono input { font-family: ui-monospace, "IBM Plex Mono", monospace !important; }
footer { display: none !important; }
"""

#: One background lesson at a time, so two clicks cannot fight over the weights.
_lock = threading.Lock()


# ------------------------------------------------------------------ helpers
def hardware() -> dict:
    """What this machine has to train on."""
    from ai_studio.training import distributed

    return distributed.describe()


def hardware_summary() -> str:
    facts = hardware()
    if not facts["gpus"]:
        return "no GPU found — training on the CPU"
    if facts["gpus"] == 1:
        return f"one GPU: {facts['names'][0]}"
    return f"{facts['gpus']} GPUs, gradients averaged between them"


def gpu_picker():
    """How many GPUs to train on — a slider only when there is a choice to make.

    Gradio 6 refuses a slider whose ends are the same number, which is what one
    GPU (or none) gives; that refusal stopped the whole window opening.
    """
    count = hardware()["gpus"]
    label = f"GPUs to use ({hardware_summary()})"
    if count > 1:
        return gr.Slider(1, count, value=1, step=1, label=label)
    return gr.Number(value=1, precision=0, label=label, interactive=False)


def model_names() -> list[str]:
    return [model.name for model in workspace.every()]


def fmt(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:,.0f}"


def friendly(exc: Exception) -> str:
    """A message someone can act on, not a traceback."""
    if isinstance(exc, TeacherError):
        return f"### Stopped\n\n{exc}"
    return f"### Something went wrong\n\n```\n{type(exc).__name__}: {exc}\n```"


def collect(paths, pasted: str, folder: str) -> "gather":
    sources: list[str] = []
    if paths:
        sources.extend(getattr(item, "name", item) for item in paths)
    if folder and folder.strip():
        sources.append(folder.strip())
    return gather(sources, raw_text=pasted or "")


def paths_from(files, folder: str) -> list[str]:
    """The files and the folder given, as paths — pairs are read from files, not pasted."""
    sources = [getattr(item, "name", item) for item in (files or [])]
    if folder and folder.strip():
        sources.append(folder.strip())
    return sources


def parse_batch(value) -> int | str:
    """The batch box: a whole number, or auto."""
    text = str(value if value is not None else "").strip().lower()
    if text == "auto":
        return "auto"
    try:
        number = float(text or 8)
    except ValueError:
        raise TeacherError(f"Batch size '{value}' is not a number. Use a whole number, "
                           f"or auto.") from None
    if number < 1 or number != int(number):
        raise TeacherError("Batch size is a whole number, 1 or more — or auto.")
    return int(number)


def parse_rate(value):
    """The rate box: empty for the measured default, or a number."""
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "default"):
        return None
    try:
        rate = float(text)
    except ValueError:
        raise TeacherError(f"Learning rate '{value}' is not a number. Leave it empty for "
                           f"the measured default, or type something like 5e-5.") from None
    if not 0 < rate < 1:
        raise TeacherError(f"{rate:g} is not a learning rate — they sit between about "
                           f"1e-6 and 1e-2.")
    return rate


def lesson_report(model, lesson: dict) -> str:
    """What one lesson did, before and after, in words someone can act on."""
    label, note, _share = lessons.judge(model, lesson["held_out_loss"])
    before, after = lesson.get("held_out_before"), lesson["held_out_loss"]
    blocks = [f"### {model.name}: {label}", f"_{note}_"]
    if after is not None:
        change = f"**{before:.4f} → {after:.4f}**" if before is not None else f"**{after:.4f}**"
        blocks.append(f"Held-out loss {change} (perplexity {lesson['perplexity']:.1f}), "
                      f"measured on {lesson.get('measured_on') or 'held-out text'}.")
    if before is not None and after is not None and after > before:
        blocks.append("⚠️ **This lesson made it worse** on text it did not train on. "
                      "Roll it back on the **Keep** tab, then try fewer epochs or a lower "
                      "learning rate.")
    best = lesson.get("best_held_out")
    if best is not None and after is not None and best < after - 0.01 and not (
            before is not None and after > before):
        blocks.append(f"_It was lowest part-way through ({best:.4f}) and rose after — "
                      f"fewer epochs would have stopped there._")
    rate = lesson.get("learning_rate")
    blocks.append(
        f"{lesson['epochs']:g} epoch(s) · {lesson['steps']} steps · batch "
        f"{lesson.get('batch_size', '?')} · rate {f'{rate:.0e}' if rate else '?'}"
        f"{' (' + lesson['rate_from'] + ')' if lesson.get('rate_from') else ''} · "
        f"{lesson['seconds']:.0f}s on {lesson['device']}")
    if lesson.get("lora"):
        detail = lesson["lora"]
        blocks.append(f"LoRA adapter, rank {detail['rank']}: {fmt(detail['trainable_parameters'])} "
                      f"of {fmt(detail['total_parameters'])} weights trained, then merged in.")
    return "\n\n".join(blocks)


def describe(name: str) -> str:
    """The card shown for whichever model is selected."""
    if not name:
        return "_No model selected._"
    try:
        model = workspace.get(name)
    except TeacherError as exc:
        return str(exc)

    arch = model.architecture()
    history = model.history()
    taught = history.get("lessons", [])
    kind = model.kind()

    origin = (f"Built here · **{fmt(_params(arch))}** parameters"
              if kind == "studio"
              else f"Based on **{model.base_repo() or 'a pretrained model'}**")
    layers = arch.get("num_hidden_layers") or arch.get("num_layers", "?")
    context = arch.get("max_position_embeddings") or arch.get("n_positions", "?")

    # Each entry becomes its own paragraph, so nothing runs together.
    blocks = [
        f"### {model.name}",
        f"{origin} · vocabulary {arch.get('vocab_size', 0):,}",
        f"{layers} layers · context {context} tokens · "
        f"{model.size_bytes() / 1024 ** 2:.0f} MB on disk",
    ]

    if history.get("branched_from"):
        blocks.append(f"Branched from **{history['branched_from']}** "
                      f"at `{history['branched_at']}`")

    states = model.saved_states()
    if states:
        blocks.append(f"{len(states)} saved state(s) to go back to, newest "
                      f"`{states[-1]['stamp']}`")

    if not taught:
        blocks.append("**Never taught anything** — it will produce noise until you "
                      "give it a lesson." if not history.get("branched_from") else
                      "No lessons *since the branch* — its weights carry everything "
                      "the original had learned by then.")
        return "\n\n".join(blocks)

    from teacher import interrupted

    waiting = interrupted.waiting(model)
    if waiting:
        blocks.append(f"⏸ **An unfinished lesson is waiting** — {interrupted.describe(waiting)}. "
                      f"Resume it on the **Teach** tab.")

    last = taught[-1]
    loss = last.get("held_out_loss") or last.get("final_loss")
    label, note, _share = lessons.judge(model, loss)
    blocks.append(f"<span class='stage'>{label}</span>")
    blocks.append(f"_{note}_")
    if last.get("held_out_before") is not None and last.get("held_out_loss") is not None:
        worse = last["held_out_loss"] > last["held_out_before"]
        blocks.append(f"Last lesson: held-out {last['held_out_before']:.4f} → "
                      f"{last['held_out_loss']:.4f}" + (" — **it got worse**" if worse else ""))
    blocks.append(
        f"{len(taught)} lesson(s) · {fmt(model.taught_characters())} characters taught · "
        f"held-out loss **{loss:.4f}**" if loss else f"{len(taught)} lesson(s)"
    )

    trend = [entry.get("held_out_loss") for entry in taught if entry.get("held_out_loss")]
    if len(trend) > 1:
        rows = ["| lesson | held-out loss |", "|---|---|"]
        rows += [f"| {index} | {value:.4f} |"
                 for index, value in list(enumerate(trend, start=1))[-8:]]
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _params(arch: dict) -> int | None:
    if arch.get("model_type") != "ai_studio_transformer":
        return None
    try:
        from ai_studio.models.transformer import TransformerConfig

        known = {k: v for k, v in arch.items() if k in TransformerConfig.__dataclass_fields__}
        return TransformerConfig(**known).parameter_count()["total"]
    except Exception:  # noqa: BLE001
        return None


def refresh_everything(selected: str | None = None):
    names = model_names()
    pick = selected if selected in names else (names[0] if names else None)
    update = gr.update(choices=names, value=pick)
    # The same list, without a selection, for the compare box.
    return update, gr.update(choices=names), update, describe(pick or "")


# -------------------------------------------------------------------- build
def do_create(name, mode, size, custom_size, base, custom_base, files, pasted, folder,
              progress=gr.Progress()):
    try:
        if not name or not name.strip():
            raise TeacherError("Give the model a name.")
        model = workspace.get(name, must_exist=False)
        if model.exists():
            raise TeacherError(f"'{model.name}' already exists. Teach it instead, or pick a new name.")

        if mode == "Start from a pretrained model":
            chosen = (custom_base or "").strip() or lessons.BASES[base][0]
            progress(0.2, desc=f"Downloading {chosen}")
            built = lessons.adopt(model, chosen)
            report = (f"### Adopted {built['base']}\n\n"
                      f"**{fmt(built['parameters'])}** parameters · vocabulary "
                      f"{built['vocab_size']:,} · context {built['context']}\n\n"
                      f"It already writes the language it was trained on. Teach it your "
                      f"material next — go to **Teach**.")
        else:
            material = collect(files, pasted, folder)
            if not material.characters:
                raise TeacherError("Add some text first — the tokenizer is built from it.")
            progress(0.3, desc="Training a tokenizer and building the model")
            wanted = (custom_size or "").strip() or size
            built = lessons.create(model, material, wanted)

            count = f"**{fmt(built['parameters'])}** parameters"
            if built.get("asked_for"):
                drift = built["parameters"] / built["asked_for"] - 1
                count += (f" — you asked for {fmt(built['asked_for'])}, "
                          f"{drift * 100:+.1f}%")
            report = (f"### Built {model.name}\n\n"
                      f"{count}\n\n"
                      f"{built['hidden_size']} wide · {built['layers']} layers · "
                      f"{built['heads']} heads · vocabulary {built['vocab_size']:,} · "
                      f"context {built['context']}\n\n"
                      f"Built from {material.summary()}. It knows nothing yet — "
                      f"go to **Teach**.")
    except Exception as exc:  # noqa: BLE001 - one failure must not kill the window
        return (friendly(exc), *refresh_everything())
    return (report, *refresh_everything(model.name))


# -------------------------------------------------------------------- teach
def do_teach(name, files, pasted, folder, epochs, batch, rate, auto, target,
             max_rounds, gpus=1, kind=TEXT, lora=False, lora_rank=16, device="auto",
             keep=2, against="", progress=gr.Progress()):
    if not _lock.acquire(blocking=False):
        return "### Busy\n\nA lesson is already running. Wait for it to finish.", gr.update()
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)

        answers = None
        if kind == PAIRS:
            # Pairs are files: .jsonl, .json or .csv, read as questions and answers.
            answers = paths_from(files, folder)
            if not answers:
                raise TeacherError("Add a file of question-and-answer pairs — .jsonl, .json "
                                   "or .csv — or a folder of them.")
            material = gather([])
        else:
            material = collect(files, pasted, folder)
            if not material.characters:
                raise TeacherError("Add some text to teach it from.")

        measured = None
        if against and str(against).strip():
            measured = gather([str(against).strip()], dedupe=False)
            if not measured.characters:
                raise TeacherError(f"Nothing readable in {against}, so there would be "
                                   f"nothing to measure against.")

        options = dict(
            batch_size=parse_batch(batch), learning_rate=parse_rate(rate),
            gpus=int(gpus or 1), lora=bool(lora), lora_rank=int(lora_rank or 16),
            device=device or "auto", keep_checkpoints=int(keep if keep is not None else 2),
            answers=answers, against=measured,
        )
        rounds: list[str] = []

        if auto:
            def on_round(index, lesson, label, share):
                loss = lesson["held_out_loss"] or lesson["final_loss"]
                rounds.append(f"| {index} | {loss:.4f} | {label} |")
                progress(min(0.95, index / max_rounds), desc=f"round {index} — {label}")

            summary = lessons.teach_until(
                model, material, target=target, epochs_per_round=float(epochs),
                max_rounds=int(max_rounds), on_round=on_round, **options,
            )
            if not summary["rounds"]:
                return f"### Nothing to do\n\n{summary['reason'].capitalize()}.", describe(name)

            table = "\n".join(["| round | held-out loss | stage |", "|---|---|---|", *rounds])
            label, note, _share = lessons.judge(model, summary["held_out_loss"])
            start = summary.get("held_out_before")
            change = (f"Held-out loss **{start:.4f} → {summary['held_out_loss']:.4f}**. "
                      if start is not None and summary["held_out_loss"] is not None else "")
            body = (f"### {model.name}: {label}\n\n_{note}_\n\n{table}\n\n{change}"
                    f"**{summary['rounds']} rounds**, {summary.get('rounds_kept', summary['rounds'])} "
                    f"kept ({summary['epochs']:g} epochs in these weights), "
                    f"{summary['seconds']:.0f}s — stopped because {summary['reason']}.")
        else:
            progress(0.2, desc="Teaching")
            lesson = lessons.teach(model, material, epochs=float(epochs), **options)
            body = lesson_report(model, lesson)
    except Exception as exc:  # noqa: BLE001
        return friendly(exc), describe(name or "")
    finally:
        _lock.release()
    return body, describe(name)


def resume_state(name: str | None):
    """Show the resume button only when there is something to resume."""
    from teacher import interrupted

    try:
        waiting = interrupted.waiting(workspace.get(name)) if name else None
    except TeacherError:
        waiting = None
    if not waiting:
        return gr.update(value="", visible=False), gr.update(visible=False)
    return (gr.update(value=f"⏸ **Unfinished lesson** — {interrupted.describe(waiting)}.",
                      visible=True), gr.update(visible=True))


def do_resume(name, progress=gr.Progress()):
    if not _lock.acquire(blocking=False):
        return "### Busy\n\nA lesson is already running. Wait for it to finish.", gr.update()
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        progress(0.2, desc="Carrying on from the last save")
        lesson = lessons.resume(model)
        body = "Resumed from where it was interrupted.\n\n" + lesson_report(model, lesson)
    except Exception as exc:  # noqa: BLE001
        return friendly(exc), describe(name or "")
    finally:
        _lock.release()
    return body, describe(name)


# --------------------------------------------------------------------- test
def suite_choices() -> list[tuple[str, str]]:
    try:
        return [(f"{row['label']} — {row['category']}", row["suite"]) for row in exams.catalogue()]
    except Exception:  # noqa: BLE001 - a broken suite list must not stop the window opening
        return [("MMLU", "mmlu")]


def do_test(name, suite, items, shots, offline):
    """Sit the model down in front of a benchmark, item by item."""
    if not name:
        yield "Choose a model first."
        return
    done: dict = {}
    progress: list[str] = []

    def run():
        try:
            done["outcome"] = exams.sit(
                workspace.get(name), suite, limit=int(items or 20),
                few_shot=int(shots) if shots not in (None, "") else None,
                allow_download=not offline,
                on_item=lambda i, total, right: progress.append(
                    f"{i}/{total} {'right' if right else 'wrong'}"),
            )
        except Exception as exc:  # noqa: BLE001
            done["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        if progress:
            yield f"Asking… {progress[-1]}"
        time.sleep(0.5)
    worker.join()
    if "error" in done:
        yield friendly(done["error"])
        return

    outcome = done["outcome"]
    lines = [
        f"### {outcome['label']}: {outcome['correct']}/{outcome['items']} "
        f"({outcome['accuracy'] * 100:.0f}%)",
    ]
    if outcome["chance"] is not None:
        lines.append(f"Guessing would score **{outcome['chance'] * 100:.0f}%**.")
    lines.append(exams.verdict(outcome))
    if not outcome["official"]:
        lines.append(f"_{outcome['note']}_")
    lines.append(f"{outcome['shots']} worked example(s) in the prompt · greedy, so it repeats "
                 f"exactly · {outcome['seconds']:.0f}s · kept in the model's record")
    def cell(text: str, width: int) -> str:
        # A pipe inside a cell would end it early.
        return " ".join(text.split())[:width].replace("|", "\\|")

    rows = ["| | question | expected | answered |", "|---|---|---|---|"]
    rows += [f"| {'✓' if r['correct'] else '✗'} | {cell(r['question'], 70)} | "
             f"{cell(r['expected'], 20)} | {cell(r['answered'], 20)} |"
             for r in outcome["results"][:10]]
    lines.append("\n".join(rows))
    yield "\n\n".join(lines)


# ------------------------------------------------------------------ compare
def do_compare(names, prompt, tokens, temperature):
    try:
        chosen = [workspace.get(name) for name in (names or [])]
        if len(chosen) < 2:
            raise TeacherError("Pick at least two models to compare.")
        if not (prompt or "").strip():
            raise TeacherError("Write a prompt for them all to continue.")
        result = lessons.compare(chosen, prompt, max_new_tokens=int(tokens),
                                 temperature=float(temperature))
    except Exception as exc:  # noqa: BLE001
        return friendly(exc)

    blocks = [f"Same prompt, same seed ({result['seed']}) — a difference is the models, "
              f"not the dice."]
    for entry in result["models"]:
        loss = entry["held_out_loss"]
        quoted = "\n".join("> " + line for line in (entry["reply"].strip() or "(nothing)").splitlines())
        blocks.append(f"**{entry['model']}** · {entry['stage']}"
                      + (f" · held-out {loss:.4f}" if loss is not None else "")
                      + "\n\n" + quoted)
    if result["losses_comparable"]:
        blocks.append("Their held-out losses were measured on the same text with the same "
                      "vocabulary, so the lower one is doing better.")
    else:
        blocks.append("_Their held-out losses cannot be compared: "
                      + ("they split text into different vocabularies" if not result["same_vocabulary"]
                         else "they were measured on different text")
                      + ". Read the replies instead._")
    return "\n\n".join(blocks)


# --------------------------------------------------------------- the card
def do_card(name):
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        known = cards.facts(model)
        written = cards.write(model, known=known)
    except Exception as exc:  # noqa: BLE001
        return friendly(exc), ""
    return (f"### Model card written\n\n`{written}`\n\n{len(known['lessons'])} lesson(s) in "
            f"these weights · {len(known['sources'])} source(s) · {len(known['exams'])} "
            f"benchmark result(s). No licence is chosen for the model — that is yours "
            f"to decide.", cards.render(known))


# -------------------------------------------------------------------- GGUF
def do_export(name, precision, destination, converter):
    from teacher import export

    if not name:
        yield "Choose a model first."
        return
    model_name = name
    lines: list[str] = []
    done: dict = {}

    def run():
        try:
            model = workspace.get(model_name)
            target = (Path(destination).expanduser() if (destination or "").strip()
                      else Path.home() / f"{model.name}-{precision}.gguf")
            done["written"] = export.to_gguf(
                model, target, precision=precision, converter=(converter or "").strip() or None,
                on_log=lambda message, level="info": lines.append(message))
            done["card"] = cards.beside_export(model, done["written"])
        except Exception as exc:  # noqa: BLE001
            done["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        if lines:
            yield "Converting…\n\n```\n" + "\n".join(lines[-8:]) + "\n```"
        time.sleep(0.5)
    worker.join()
    if "error" in done:
        yield friendly(done["error"])
        return

    written = done["written"]
    _entry, card_path, why_not = done["card"]
    advice = "\n".join(export.advice(written))
    yield (f"### Exported\n\n`{written['path']}` — {written['bytes'] / 1024 ** 2:,.1f} MB\n\n"
           + (f"Model card beside it: `{card_path}`\n\n" if card_path
              else f"No model card beside it — {why_not}\n\n")
           + f"```\n{advice}\n```")


# -------------------------------------------------------------------- serve
#: The API server started from the window, if one is running.
_serving: dict = {}


def do_serve(name, port, key):
    """Start the OpenAI-style API for one model, on this machine only."""
    if _serving.get("server"):
        return (f"Already serving **{_serving['name']}** at "
                f"`http://127.0.0.1:{_serving['port']}/v1`. Stop it first.")
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        import uvicorn

        from teacher import serve

        app = serve.build_app([name], api_key=(key or "").strip() or None)
        server = uvicorn.Server(uvicorn.Config(app, host=serve.DEFAULT_HOST, port=int(port),
                                               log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            if server.started or not thread.is_alive():
                break
            time.sleep(0.1)
        if not server.started:
            server.should_exit = True
            raise TeacherError(f"Could not listen on port {int(port)} — is something else "
                               f"using it? Try another port.")
    except Exception as exc:  # noqa: BLE001
        return friendly(exc)
    _serving.update(server=server, thread=thread, name=name, port=int(port))
    return (f"### Serving {name}\n\n`http://127.0.0.1:{int(port)}/v1`\n\n"
            f"Point any OpenAI client there, with **{name}** as the model"
            + (" and your key as the API key." if (key or "").strip() else ".")
            + " Only this machine can reach it.")


def do_stop_serving():
    server = _serving.pop("server", None)
    if server is None:
        return "Nothing is being served."
    server.should_exit = True
    _serving.pop("thread").join(timeout=10)
    name = _serving.pop("name", "")
    _serving.pop("port", None)
    return f"Stopped serving {name}."


# ----------------------------------------------------------------- the web
def do_web(query, count, addresses):
    """Search the web, keep the pages as text, and point Make/Teach at them.

    A generator, so the window shows each address as it is read rather than
    sitting blank for a minute.
    """
    from teacher import websearch

    query = (query or "").strip()
    urls = [line.strip() for line in (addresses or "").splitlines() if line.strip()]
    if not query and not urls:
        yield "Type something to search for, or paste an address.", gr.update()
        return

    progress: list[str] = []
    done: dict = {}

    def run():
        try:
            done["harvest"] = websearch.harvest(
                query, urls=urls, limit=int(count),
                on_step=lambda i, total, url: progress.append(f"{i}/{total} — {url}"),
            )
        except Exception as exc:  # noqa: BLE001 - a failed search must not kill the window
            done["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    shown = 0
    while worker.is_alive():
        if len(progress) > shown:
            shown = len(progress)
            yield "Reading:\n\n" + "\n".join(f"- `{line}`" for line in progress), gr.update()
        time.sleep(0.3)
    worker.join()

    if "error" in done:
        yield friendly(done["error"]), gr.update()
        return

    collected = done["harvest"]
    try:
        folder = websearch.folder_for(query or urls[0], workspace.root())
        websearch.save(collected, folder)
    except Exception as exc:  # noqa: BLE001
        yield friendly(exc), gr.update()
        return

    lines = [f"**Kept {collected.summary()}** in `{folder}`", ""]
    lines += [f"- [{page.title or page.url}]({page.url}) — {page.characters:,} characters"
              for page in collected.pages]
    if collected.skipped:
        lines += ["", "Could not read:"]
        lines += [f"- `{url}` — {why}" for url, why in collected.skipped[:5]]
    lines += [
        "",
        "_Every file records the address it came from and the date. A page you "
        "found is someone else's writing, under someone else's terms._",
        "",
        "The **folder** box above now points at it — press *Teach*.",
    ]
    yield "\n".join(lines), str(folder)


# --------------------------------------------------------------------- chat
def do_chat(message, chat_history, name, tokens, temperature, top_p, top_k):
    chat_history = chat_history or []
    if not name:
        chat_history.append({"role": "assistant", "content": "Choose a model on the left first."})
        return chat_history, ""
    if not message or not message.strip():
        return chat_history, ""

    chat_history.append({"role": "user", "content": message})
    try:
        model = workspace.get(name)
        text, stats = lessons.talk(
            model, message, max_new_tokens=int(tokens), temperature=float(temperature),
            top_p=float(top_p), top_k=int(top_k),
        )
        reply = text.strip() or "_(it produced nothing)_"
        reply += f"\n\n<sub>{stats['generated']} tokens · {stats['tokens_per_second']:.0f}/s</sub>"
    except Exception as exc:  # noqa: BLE001
        reply = friendly(exc)
    chat_history.append({"role": "assistant", "content": reply})
    return chat_history, ""


# --------------------------------------------------------------------- keep
def do_pack(name, destination):
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        target = Path(destination).expanduser() if destination.strip() else (model.path / "for-bench")
        copied = model.pack(target)
        return (f"### Copied\n\n{len(copied)} file(s) to `{target}`\n\n"
                f"Open `web_chat/index.html` and drop that folder in.")
    except Exception as exc:  # noqa: BLE001
        return friendly(exc)


def do_branch(name, new_name, stamp):
    """Carry on from a saved state in a new model, leaving the original alone."""
    if not name:
        return "Pick a model first.", *refresh_everything(name)
    if not (new_name or "").strip():
        return "Give the copy a name.", *refresh_everything(name)
    try:
        source = workspace.get(name)
        made = workspace.branch(source, new_name, at=(stamp or "").strip() or None)
    except Exception as exc:  # noqa: BLE001
        return friendly(exc), *refresh_everything(name)

    where = f"`{stamp}`" if stamp else "its current weights"
    return (
        f"Branched **{source.name}** ({where}) into **{made.name}**.\n\n"
        f"`{made.path}`\n\n"
        f"**{source.name} is untouched.** Select {made.name} above and teach it "
        f"as hard as you like — whatever happens, the original is still there.",
        *refresh_everything(made.name),
    )


def do_rollback(name, stamp=""):
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        restored = model.rollback(to=(stamp or "").strip() or None)
        return (f"### Rolled back\n\n{model.name} restored from `{restored}`. "
                f"Its lesson history is unchanged — it still lists what was taught.")
    except Exception as exc:  # noqa: BLE001
        return friendly(exc)


def state_choices(name: str | None):
    """The saved states of the selected model, newest first, for a dropdown."""
    if not name:
        return gr.update(choices=[], value=None)
    try:
        states = workspace.get(name).saved_states()
    except TeacherError:
        return gr.update(choices=[], value=None)
    labels = [
        f"{state['stamp']}  ({time.strftime('%H:%M', time.localtime(state['at']))})"
        for state in reversed(states)
    ]
    return gr.update(choices=labels, value=labels[0] if labels else None)


def chosen_stamp(label: str | None) -> str:
    """The stamp out of a dropdown label like "20260919-172927  (17:29)"."""
    return (label or "").split()[0] if label else ''


def do_forget(name, confirm):
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        if not confirm:
            raise TeacherError("Tick the box to confirm — this cannot be undone.")
        model = workspace.get(name)
        import shutil

        shutil.rmtree(model.path)
        return (f"### Deleted\n\n{model.name} and everything it learned is gone.",
                *refresh_everything())
    except Exception as exc:  # noqa: BLE001
        return (friendly(exc), *refresh_everything(name))


# -------------------------------------------------------------------- build
def build() -> gr.Blocks:
    names = model_names()
    first = names[0] if names else None

    with compat.blocks(title="Teacher", css=CSS, theme=gr.themes.Soft()) as app:
        gr.HTML("<p class='teacher-title'>Teacher</p>"
                "<p class='teacher-sub'>Make a language model, teach it your writing, "
                "and talk to it. Everything runs on this machine.</p>")

        with gr.Row():
            with gr.Column(scale=1):
                picker = gr.Dropdown(names, value=first, label="Model", interactive=True)
                refresh = gr.Button("↻ Refresh", size="sm")
                card = gr.Markdown(describe(first or ""))
            with gr.Column(scale=2):
                with gr.Tabs():
                    # ---------------------------------------------- make
                    with gr.Tab("Make"):
                        gr.Markdown(
                            "Start from a **pretrained model** if you want it to write proper "
                            "sentences quickly. Start **from scratch** if you want to watch a "
                            "model learn a language from nothing."
                        )
                        new_name = gr.Textbox(label="Name", placeholder="bookbot")
                        mode = gr.Radio(
                            ["Start from a pretrained model", "Build from scratch"],
                            value="Start from a pretrained model", label="How",
                        )
                        with gr.Group() as base_group:
                            base = gr.Dropdown(
                                [(f"{key} — {note}", key) for key, (_repo, note) in lessons.BASES.items()],
                                value="small", label="Pretrained model",
                            )
                            custom_base = gr.Textbox(
                                label="…or any Hugging Face name",
                                placeholder="HuggingFaceTB/SmolLM2-135M",
                            )
                        with gr.Group(visible=False) as scratch_group:
                            size = gr.Radio(
                                [(f"{key} — {note}", key) for key, (_target, note) in lessons.SIZES.items()],
                                value=next(iter(lessons.SIZES)), label="Size",
                            )
                            custom_size = gr.Textbox(
                                label="…or type how many parameters you want",
                                placeholder="70K, 5M, 51M, 1.5B — anything you like",
                            )
                            gr.Markdown(
                                "_Widths move in steps, so a number lands near rather than on "
                                "it; you will be told what was actually built. The text below "
                                "also builds the tokenizer._"
                            )
                        make_button = gr.Button("Make it", variant="primary")
                        make_out = gr.Markdown()

                    # --------------------------------------------- teach
                    with gr.Tab("Teach"):
                        gr.Markdown("Give it material, then let it learn. Every round is saved.")
                        resume_note = gr.Markdown(visible=False)
                        resume_button = gr.Button("Resume the unfinished lesson", visible=False)
                        kind = gr.Radio([TEXT, PAIRS], value=TEXT, label="What it learns from")
                        auto = gr.Checkbox(
                            value=True, label="Keep going until it is done (recommended)")
                        with gr.Row():
                            target = gr.Dropdown(
                                list(lessons.TARGETS), value="best", label="Until")
                            max_rounds = gr.Number(value=20, precision=0, label="Most rounds")
                        with gr.Row():
                            epochs = gr.Number(value=3, label="Epochs (per round)")
                            batch = gr.Textbox(value="8", label="Batch size",
                                               info="a number, or auto to fill the hardware")
                            rate = gr.Textbox(value="", label="Learning rate",
                                              placeholder="default for this kind of model",
                                              info="empty for the measured default — 5e-5 "
                                                   "pretrained, 1e-3 LoRA, 3e-3 from scratch")
                        gpus = gpu_picker()
                        with gr.Accordion("More", open=False):
                            with gr.Row():
                                lora = gr.Checkbox(
                                    value=False, label="LoRA — train a small adapter, not every "
                                                       "weight (pretrained models only)")
                                lora_rank = gr.Number(value=16, precision=0, label="LoRA rank")
                            with gr.Row():
                                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto",
                                                     label="Train on")
                                keep = gr.Number(value=2, precision=0,
                                                 label="Saved states to keep")
                            against = gr.Textbox(
                                label="Measure against a separate file or folder",
                                placeholder="leave empty to hold out the end of the material",
                                info="the tail of the same text flatters a model that "
                                     "memorised it; separate text does not")
                        teach_button = gr.Button("Teach", variant="primary")
                        teach_out = gr.Markdown()

                    # ---------------------------------------------- chat
                    with gr.Tab("Chat"):
                        chat = compat.chatbot(height=380, label=None)
                        with gr.Row():
                            message = gr.Textbox(
                                placeholder="Write something for it to continue…",
                                show_label=False, scale=5, elem_classes="mono",
                            )
                            send = gr.Button("Send", variant="primary", scale=1)
                        with gr.Accordion("Sampling", open=False):
                            with gr.Row():
                                tokens = gr.Slider(16, 512, value=120, step=8, label="Length")
                                temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature")
                            with gr.Row():
                                top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top-p")
                                top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k")
                        clear_chat = gr.Button("Clear", size="sm")
                        with gr.Accordion("Compare models", open=False):
                            gr.Markdown("One prompt through several models at the same seed, "
                                        "so a difference is the models and not the dice.")
                            compare_names = gr.Dropdown(names, multiselect=True,
                                                        label="Models")
                            compare_prompt = gr.Textbox(label="Prompt", lines=2)
                            compare_button = gr.Button("Compare")
                            compare_out = gr.Markdown()

                    # ---------------------------------------------- test
                    with gr.Tab("Test"):
                        gr.Markdown(
                            "Real benchmark questions, graded, with the score **guessing** "
                            "would get beside it. A small model scoring at chance on a "
                            "knowledge test is expected — that is not a failed lesson."
                        )
                        with gr.Row():
                            suite = gr.Dropdown(suite_choices(), value="arc", label="Benchmark")
                            items = gr.Number(value=20, precision=0, label="Questions")
                            shots = gr.Number(value=None, precision=0,
                                              label="Worked examples (empty: the suite's own)")
                        offline = gr.Checkbox(
                            value=False,
                            label="Don't download the official questions — use what is here")
                        test_button = gr.Button("Run it", variant="primary")
                        test_out = gr.Markdown()

                    # ---------------------------------------------- keep
                    with gr.Tab("Keep"):
                        gr.Markdown("### Use it in the browser page")
                        gr.Markdown(
                            "Copies the three files the Bench page needs. Only for models built "
                            "from scratch — a pretrained one runs here, not there."
                        )
                        destination = gr.Textbox(
                            label="Copy to", placeholder="leave empty to put it beside the model")
                        pack_button = gr.Button("Copy files")

                        gr.Markdown("### Model card")
                        gr.Markdown(
                            "A README for the model, written from its own record: what it "
                            "started from, what it was taught and from where, how it scores, "
                            "and the base model's licence."
                        )
                        card_button = gr.Button("Write the model card")
                        with gr.Accordion("The card", open=False):
                            card_view = gr.Markdown()

                        gr.Markdown("### Export to GGUF")
                        gr.Markdown(
                            "One file for llama.cpp, Ollama and LM Studio, with its model "
                            "card beside it. Needs llama.cpp's converter; pretrained models only."
                        )
                        with gr.Row():
                            precision = gr.Dropdown(["f16", "bf16", "f32", "q8_0"], value="f16",
                                                    label="Precision")
                            export_to = gr.Textbox(label="Write to",
                                                   placeholder="leave empty for your home folder")
                        converter = gr.Textbox(label="llama.cpp's convert_hf_to_gguf.py, "
                                                     "if Teacher cannot find it")
                        export_button = gr.Button("Export")

                        gr.Markdown("### Serve it to other programs")
                        gr.Markdown(
                            "An API in OpenAI's format on this machine, so chat front-ends, "
                            "editors and scripts can use the model."
                        )
                        with gr.Row():
                            serve_port = gr.Number(value=8008, precision=0, label="Port")
                            serve_key = gr.Textbox(label="Key (optional)", type="password")
                        with gr.Row():
                            serve_button = gr.Button("Start serving")
                            stop_button = gr.Button("Stop", size="sm")

                        gr.Markdown("### Saved states")
                        gr.Markdown(
                            "Teacher copies the weights aside before every lesson. Go back "
                            "to one, or branch it into a new model and carry on training "
                            "*that* — which leaves this one exactly as it is."
                        )
                        states = gr.Dropdown(label="Saved state", choices=[], interactive=True)
                        with gr.Row():
                            rollback_button = gr.Button("Roll this model back to it")
                            refresh_states = gr.Button("Refresh", size="sm")
                        branch_name = gr.Textbox(
                            label="…or branch it into a new model called",
                            placeholder="mymodel-v2")
                        branch_button = gr.Button("Branch")

                        gr.Markdown("### Delete")
                        confirm = gr.Checkbox(label="Yes, delete this model and everything it learned")
                        forget_button = gr.Button("Delete", variant="stop")
                        keep_out = gr.Markdown()

        # ------------------------------------------------- shared material
        with gr.Accordion("Text — used by Make and Teach", open=True):
            gr.Markdown(
                "Any of these, or all three together. Files can be `.txt`, `.md`, `.pdf`, "
                "`.docx`, `.epub`, `.html`, `.csv` or `.json`."
            )
            with gr.Row():
                files = gr.File(label="Files", file_count="multiple", type="filepath")
                with gr.Column():
                    folder = gr.Textbox(
                        label="…or a folder on this machine",
                        placeholder=str(Path.home() / "mytext"),
                    )
                    pasted = gr.Textbox(label="…or paste text", lines=4)
            check = gr.Button("Check what this adds up to", size="sm")
            material_out = gr.Markdown()

            with gr.Accordion("…or fetch it from the web", open=False):
                gr.Markdown(
                    "Searches the web, reads what it finds and saves each page as a text "
                    "file with the address it came from. Nothing is overwritten and "
                    "nothing is thrown away — the folder box above is pointed at the "
                    "result so *Make* and *Teach* can use it."
                )
                with gr.Row():
                    web_query = gr.Textbox(
                        label="Search for", scale=3,
                        placeholder="victorian lighthouse keepers",
                    )
                    web_count = gr.Slider(1, 15, value=5, step=1, label="Pages to read")
                web_urls = gr.Textbox(
                    label="…or paste addresses, one per line", lines=2,
                    placeholder="https://en.wikipedia.org/wiki/Lighthouse_keeper",
                )
                web_button = gr.Button("Search and keep", size="sm")
                web_out = gr.Markdown()

        # ------------------------------------------------------- wiring
        def toggle(chosen):
            pretrained = chosen == "Start from a pretrained model"
            return gr.update(visible=pretrained), gr.update(visible=not pretrained)

        mode.change(toggle, mode, [base_group, scratch_group])

        def do_check(files_in, pasted_in, folder_in):
            try:
                material = collect(files_in, pasted_in, folder_in)
            except Exception as exc:  # noqa: BLE001
                return friendly(exc)
            if not material.characters:
                return "Nothing readable found yet."
            report = f"**{material.summary()}**"
            if material.characters < lessons.MIN_CHARACTERS:
                report += (f"\n\nThat is under {lessons.MIN_CHARACTERS:,} characters — too little "
                           f"to learn from. Add more.")
            if material.skipped:
                report += "\n\nSkipped:\n" + "\n".join(
                    f"- `{path}` — {why}" for path, why in material.skipped[:5])
            return report

        # The dropdown is built once at launch; without this a model made later
        # would be missing from it after a page reload.
        app.load(refresh_everything, picker, [picker, compare_names, picker, card])

        check.click(do_check, [files, pasted, folder], material_out)
        web_button.click(do_web, [web_query, web_count, web_urls], [web_out, folder])
        picker.change(describe, picker, card)
        refresh.click(refresh_everything, picker, [picker, compare_names, picker, card])

        make_button.click(
            do_create,
            [new_name, mode, size, custom_size, base, custom_base, files, pasted, folder],
            [make_out, picker, compare_names, picker, card],
        )
        teach_button.click(
            do_teach,
            [picker, files, pasted, folder, epochs, batch, rate, auto, target, max_rounds,
             gpus, kind, lora, lora_rank, device, keep, against],
            [teach_out, card],
        ).then(resume_state, picker, [resume_note, resume_button])
        app.load(resume_state, picker, [resume_note, resume_button])
        picker.change(resume_state, picker, [resume_note, resume_button])
        resume_button.click(do_resume, picker, [teach_out, card]).then(
            resume_state, picker, [resume_note, resume_button])

        test_button.click(do_test, [picker, suite, items, shots, offline], test_out)
        compare_button.click(do_compare, [compare_names, compare_prompt, tokens, temperature],
                             compare_out)
        card_button.click(do_card, picker, [keep_out, card_view])
        export_button.click(do_export, [picker, precision, export_to, converter], keep_out)
        serve_button.click(do_serve, [picker, serve_port, serve_key], keep_out)
        stop_button.click(do_stop_serving, None, keep_out)

        send.click(do_chat, [message, chat, picker, tokens, temperature, top_p, top_k],
                   [chat, message])
        message.submit(do_chat, [message, chat, picker, tokens, temperature, top_p, top_k],
                       [chat, message])
        clear_chat.click(lambda: [], outputs=chat)

        pack_button.click(do_pack, [picker, destination], keep_out)

        # The saved-state list is per model, and grows with every lesson.
        app.load(state_choices, picker, states)
        picker.change(state_choices, picker, states)
        refresh_states.click(state_choices, picker, states)
        rollback_button.click(
            lambda name, label: do_rollback(name, chosen_stamp(label)),
            [picker, states], keep_out,
        ).then(state_choices, picker, states)
        branch_button.click(
            lambda name, new, label: do_branch(name, new, chosen_stamp(label)),
            [picker, branch_name, states],
            [keep_out, picker, compare_names, picker, card],
        ).then(state_choices, picker, states)
        forget_button.click(do_forget, [picker, confirm],
                            [keep_out, picker, compare_names, picker, card])

    return app


def launch(host: str = "127.0.0.1", port: int = 7861, share: bool = False,
           open_browser: bool = True) -> None:
    compat.launch(
        build().queue(default_concurrency_limit=2),
        server_name=host, server_port=port, share=share,
        inbrowser=open_browser, quiet=False, show_api=False,
    )
