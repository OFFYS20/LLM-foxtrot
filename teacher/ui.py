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

from teacher import lessons, workspace
from teacher.material import gather
from teacher.workspace import TeacherError

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

    if not taught:
        blocks.append("**Never taught anything** — it will produce noise until you "
                      "give it a lesson.")
        return "\n\n".join(blocks)

    last = taught[-1]
    loss = last.get("held_out_loss") or last.get("final_loss")
    label, note, _share = lessons.judge(model, loss)
    blocks.append(f"<span class='stage'>{label}</span>")
    blocks.append(f"_{note}_")
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
    return update, update, update, describe(pick or "")


# -------------------------------------------------------------------- build
def do_create(name, mode, size, base, custom_base, files, pasted, folder, progress=gr.Progress()):
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
            built = lessons.create(model, material, size)
            report = (f"### Built {model.name}\n\n"
                      f"**{fmt(built['parameters'])}** parameters · vocabulary "
                      f"{built['vocab_size']:,} · context {built['context']}\n\n"
                      f"Built from {material.summary()}. It knows nothing yet — "
                      f"go to **Teach**.")
    except Exception as exc:  # noqa: BLE001 - one failure must not kill the window
        return (friendly(exc), *refresh_everything())
    return (report, *refresh_everything(model.name))


# -------------------------------------------------------------------- teach
def do_teach(name, files, pasted, folder, epochs, batch, rate, auto, target,
             max_rounds, progress=gr.Progress()):
    if not _lock.acquire(blocking=False):
        return "### Busy\n\nA lesson is already running. Wait for it to finish.", gr.update()
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        material = collect(files, pasted, folder)
        if not material.characters:
            raise TeacherError("Add some text to teach it from.")

        rounds: list[str] = []

        if auto:
            def on_round(index, lesson, label, share):
                loss = lesson["held_out_loss"] or lesson["final_loss"]
                rounds.append(f"| {index} | {loss:.4f} | {label} |")
                progress(min(0.95, index / max_rounds), desc=f"round {index} — {label}")

            summary = lessons.teach_until(
                model, material, target=target, epochs_per_round=float(epochs),
                max_rounds=int(max_rounds), batch_size=int(batch), learning_rate=float(rate),
                on_round=on_round,
            )
            if not summary["rounds"]:
                return f"### Nothing to do\n\n{summary['reason'].capitalize()}.", describe(name)

            table = "\n".join(["| round | held-out loss | stage |", "|---|---|---|", *rounds])
            label, note, _share = lessons.judge(model, summary["held_out_loss"])
            body = (f"### {model.name}: {label}\n\n_{note}_\n\n{table}\n\n"
                    f"**{summary['rounds']} rounds** ({summary['epochs']:g} epochs) in "
                    f"{summary['seconds']:.0f}s — stopped because {summary['reason']}.")
        else:
            progress(0.2, desc="Teaching")
            lesson = lessons.teach(
                model, material, epochs=float(epochs),
                batch_size=int(batch), learning_rate=float(rate),
            )
            label, note, _share = lessons.judge(model, lesson["held_out_loss"])
            body = (f"### {model.name}: {label}\n\n_{note}_\n\n"
                    f"Held-out loss **{lesson['held_out_loss']:.4f}** "
                    f"(perplexity {lesson['perplexity']:.1f}) after "
                    f"{lesson['epochs']:g} epoch(s) in {lesson['seconds']:.0f}s on "
                    f"{lesson['device']}.")
    except Exception as exc:  # noqa: BLE001
        return friendly(exc), describe(name or "")
    finally:
        _lock.release()
    return body, describe(name)


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


def do_rollback(name):
    try:
        if not name:
            raise TeacherError("Choose a model first.")
        model = workspace.get(name)
        restored = model.rollback()
        return (f"### Rolled back\n\n{model.name} restored from `{restored}`. "
                f"Its lesson history is unchanged — it still lists what was taught.")
    except Exception as exc:  # noqa: BLE001
        return friendly(exc)


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

    with gr.Blocks(title="Teacher", css=CSS, theme=gr.themes.Soft()) as app:
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
                                [(f"{key} — {note}", key) for key, (_p, note) in lessons.SIZES.items()],
                                value="tiny", label="Size",
                            )
                            gr.Markdown("_From scratch, the text below also builds the tokenizer._")
                        make_button = gr.Button("Make it", variant="primary")
                        make_out = gr.Markdown()

                    # --------------------------------------------- teach
                    with gr.Tab("Teach"):
                        gr.Markdown("Give it material, then let it learn. Every round is saved.")
                        auto = gr.Checkbox(
                            value=True, label="Keep going until it is done (recommended)")
                        with gr.Row():
                            target = gr.Dropdown(
                                list(lessons.TARGETS), value="best", label="Until")
                            max_rounds = gr.Number(value=20, precision=0, label="Most rounds")
                        with gr.Row():
                            epochs = gr.Number(value=3, label="Epochs (per round)")
                            batch = gr.Number(value=8, precision=0, label="Batch size")
                            rate = gr.Number(value=3e-4, label="Learning rate")
                        teach_button = gr.Button("Teach", variant="primary")
                        teach_out = gr.Markdown()

                    # ---------------------------------------------- chat
                    with gr.Tab("Chat"):
                        chat = gr.Chatbot(type="messages", height=380, label=None)
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

                        gr.Markdown("### Undo the last lesson")
                        gr.Markdown("Teacher saves the weights before every lesson.")
                        rollback_button = gr.Button("Roll back")

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
        app.load(refresh_everything, picker, [picker, picker, picker, card])

        check.click(do_check, [files, pasted, folder], material_out)
        picker.change(describe, picker, card)
        refresh.click(refresh_everything, picker, [picker, picker, picker, card])

        make_button.click(
            do_create,
            [new_name, mode, size, base, custom_base, files, pasted, folder],
            [make_out, picker, picker, picker, card],
        )
        teach_button.click(
            do_teach,
            [picker, files, pasted, folder, epochs, batch, rate, auto, target, max_rounds],
            [teach_out, card],
        )

        send.click(do_chat, [message, chat, picker, tokens, temperature, top_p, top_k],
                   [chat, message])
        message.submit(do_chat, [message, chat, picker, tokens, temperature, top_p, top_k],
                       [chat, message])
        clear_chat.click(lambda: [], outputs=chat)

        pack_button.click(do_pack, [picker, destination], keep_out)
        rollback_button.click(do_rollback, picker, keep_out)
        forget_button.click(do_forget, [picker, confirm],
                            [keep_out, picker, picker, picker, card])

    return app


def launch(host: str = "127.0.0.1", port: int = 7861, share: bool = False,
           open_browser: bool = True) -> None:
    build().queue(default_concurrency_limit=2).launch(
        server_name=host, server_port=port, share=share,
        inbrowser=open_browser, quiet=False, show_api=False,
    )
