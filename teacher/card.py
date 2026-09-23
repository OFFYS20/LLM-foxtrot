"""A model card: what a model is, what it was taught, and on whose terms.

A card is what someone reads before they trust a model they did not make. It is
written from the model's own record and nothing else — every number in it was
measured by a lesson or a benchmark, and anything the record does not know is
said to be unknown rather than filled in.

Three things it is careful about:

* **Which lessons are in these weights.** A rolled-back lesson is still in the
  record, because it happened. It is not in the weights, so the card lists it
  apart, as undone.
* **Which scores describe these weights.** A benchmark taken before the last
  lesson describes a model that no longer exists; it is kept out of the table.
* **Licences.** Teacher's own licence covers Teacher. It says nothing about a
  base model's weights or about the material, so the card reports the base
  model's declared licence, lists where the material came from, and leaves the
  choice of a licence for this model to the person who made it.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path

from teacher import workspace
from teacher.workspace import Model, TeacherError

#: How a card says it was written by Teacher — and so may be rewritten by it.
MARK = "<!-- Written by Teacher"

#: A web page saved by `teacher web` starts with its title, address and date.
HEADERS = ("Source:", "Retrieved:")


# ------------------------------------------------------------------ facts
def parameter_count(model: Model) -> int | None:
    """Counted from the weights file's header — nothing is loaded."""
    try:
        from safetensors import safe_open

        total = 0
        with safe_open(str(model.path / "model.safetensors"), "pt") as handle:
            for key in handle.keys():
                total += math.prod(handle.get_slice(key).get_shape())
        return total
    except Exception:  # noqa: BLE001 - an unreadable file means "not known"
        return None


def shape(model: Model) -> dict:
    """The architecture, under one set of names whichever kind the model is."""
    arch = model.architecture()

    def first(*names):
        return next((arch[name] for name in names if arch.get(name)), None)

    return {
        "type": arch.get("model_type") or "unknown",
        "layers": first("num_layers", "num_hidden_layers", "n_layer"),
        "width": first("hidden_size", "n_embd"),
        "heads": first("num_heads", "num_attention_heads", "n_head"),
        "vocabulary": first("vocab_size"),
        "context": first("max_position_embeddings", "n_positions"),
    }


def describe_source(source: str) -> dict:
    """Where one piece of material came from, as far as its file says."""
    if source.startswith("("):
        return {"kind": "typed", "label": source.strip("()")}
    path = Path(source)
    entry = {"kind": "file", "name": path.name}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = [handle.readline().strip() for _ in range(3)]
    except OSError:
        entry["missing"] = True
        return entry
    # Exactly the layout `teacher web` writes — title, address, date — and
    # nothing looser, so a file that merely mentions a source is not taken
    # for a web page.
    title, address, date = head
    if address.startswith(HEADERS[0]) and date.startswith(HEADERS[1]):
        entry.update(kind="web", title=title or path.stem,
                     url=address[len(HEADERS[0]):].strip(),
                     retrieved=date[len(HEADERS[1]):].strip() or None)
    return entry


def facts(model: Model, *, look_up_licence: bool = True) -> dict:
    """Everything the card says, as data. ``teacher card --json`` prints this."""
    from teacher import lessons

    history = model.history()
    stamp = model.weights_stamp()
    carried = workspace.lineage(history, stamp)

    base = history.get("base_repo")
    licence = history.get("base_license")
    if base and not licence and look_up_licence:
        licence = lessons.licence_of(base)
        if licence:
            model.note(base_license=licence)

    sources: list[dict] = []
    seen: set[str] = set()
    for lesson in carried["in_effect"]:
        for source in lesson.get("sources") or []:
            if source not in seen:
                seen.add(source)
                sources.append(describe_source(str(source)))

    exams = history.get("exams", [])
    taught = carried["in_effect"]
    last = taught[-1] if taught else {}
    stage = None
    if last.get("held_out_loss") is not None:
        label, note, _share = lessons.judge(model, last["held_out_loss"])
        stage = {"label": label, "note": note}

    return {
        "name": model.name,
        "kind": model.kind(),
        "made": history.get("made") or ("adopted" if base else "from scratch"),
        "created_at": history.get("adopted_at") or history.get("created_at"),
        "base_repo": base,
        "base_license": licence,
        "branched_from": history.get("branched_from"),
        "branched_at": history.get("branched_at"),
        "branched_on": history.get("branched_on"),
        "parameters": parameter_count(model),
        "shape": shape(model),
        "answer_style": history.get("answer_style"),
        "lessons": taught,
        "undone": carried["undone"],
        "certain": carried["certain"],
        "rollbacks": history.get("rollbacks", []),
        "characters": sum(int(lesson.get("characters") or 0) for lesson in taught),
        "sources": sources,
        "stage": stage,
        "exams": [exam for exam in exams if exam.get("weights") == stamp],
        "older_exams": [exam for exam in exams if exam.get("weights") != stamp],
        "exports": [item for item in history.get("exports", []) if item.get("weights") == stamp],
    }


# ----------------------------------------------------------------- render
def _date(at) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(at)) if at else "unknown"


def _short(number: int | None) -> str:
    from ai_studio.models.transformer import format_parameter_count

    return format_parameter_count(number) if number else "unknown"


def _count(number: int | None) -> str:
    return f"{number:,} ({_short(number)})" if number else "unknown"


def _number(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "—"


def _loss(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _front_matter(f: dict, export: dict | None) -> list[str]:
    lines = ["---"]
    if f["base_repo"]:
        lines.append(f"base_model: {f['base_repo']}")
    if f["kind"] == "pretrained" and not export:
        # Beside a GGUF file the card describes that file, which Transformers
        # does not load; the gguf tag is what the Hub goes by there.
        lines.append("library_name: transformers")
    lines += ["pipeline_tag: text-generation", "tags:", "- teacher", "- text-generation"]
    if export:
        lines.append("- gguf")
    lines += [
        "# license: not set. Teacher does not choose a licence for a model it did not",
        "# write the material for — see the Licence section before sharing this.",
        "---",
    ]
    return lines


def _a(phrase: str) -> str:
    """"a" or "an", as the number is said aloud: an 8M, an 11M, a 135M."""
    spoken_vowel = phrase[:1] == "8" or re.match(r"1[18](?!\d)", phrase)
    return ("an " if spoken_vowel else "a ") + phrase


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _summary(f: dict) -> str:
    size = f"{_short(f['parameters'])}-parameter language model" if f["parameters"] \
        else "language model"
    taught = len(f["lessons"])
    amount = (f"{f['characters']:,} characters of material over {taught} "
              f"lesson{'s' if taught != 1 else ''}")
    if f["made"] == "adopted":
        base = f"[{f['base_repo']}](https://huggingface.co/{f['base_repo']})"
        if not taught:
            return f"{_cap(_a(size))}: {base}, adopted with Teacher and not taught anything yet."
        return f"{_cap(_a(size))}, fine-tuned from {base} with Teacher on {amount}."
    if not taught:
        return (f"{_cap(_a(size))} built from scratch with Teacher and not taught "
                f"anything yet — its weights are still random.")
    return f"{_cap(_a(size))} built from scratch with Teacher and taught {amount}."


def render(f: dict, *, export: dict | None = None) -> str:
    """The card as Markdown, in the layout the Hugging Face Hub expects."""
    shape_ = f["shape"]
    out = _front_matter(f, export)
    out += [
        f"{MARK} from the model's own record by `teacher card {f['name']}`. "
        f"Every number below was measured; nothing is estimated. -->",
        "",
        f"# {f['name']}",
        "",
        _summary(f),
        "",
        "| | |",
        "|---|---|",
        f"| Parameters | {_count(f['parameters'])} |",
        f"| Architecture | {shape_['type']} — {shape_['layers'] or '?'} layers, "
        f"{shape_['width'] or '?'} wide, {shape_['heads'] or '?'} heads |",
        f"| Vocabulary | {shape_['vocabulary']:,} tokens |" if shape_["vocabulary"]
        else "| Vocabulary | unknown |",
        f"| Context | {shape_['context']:,} tokens |" if shape_["context"]
        else "| Context | unknown |",
    ]
    if f["made"] == "adopted":
        out.append(f"| Started from | {f['base_repo']} (adopted {_date(f['created_at'])}) |")
    else:
        out.append(f"| Made | from scratch, {_date(f['created_at'])} |")
    if f["branched_from"]:
        out.append(f"| Branched from | {f['branched_from']} at {f['branched_at']}, "
                   f"{_date(f['branched_on'])} |")
    if f["lessons"]:
        last = f["lessons"][-1]
        out.append(f"| Held-out loss | {_loss(last.get('held_out_loss'))} "
                   f"(on {last.get('measured_on') or 'held-out text'}) |")
    if f["stage"]:
        out.append(f"| Where it stands | {f['stage']['label']} |")
    style = f["answer_style"]
    out.append(f"| Answers questions | {'yes — taught in the ' + style + ' style' if style else 'no — it continues text'} |")
    out.append("")

    out += _how_to_use(f, export)
    out += _lessons(f)
    out += _sources(f)
    out += _exams(f)
    if export:
        out += _this_file(export)
    out += _licence(f)
    out += _limits(f)
    return "\n".join(out).rstrip() + "\n"


def _how_to_use(f: dict, export: dict | None) -> list[str]:
    from teacher.answers import OPENERS

    if export:
        out = [
            "## Using it", "",
            "```",
            f'llama-cli -m {export["file"]} -p "your prompt"',
            f"printf 'FROM ./{export['file']}\\n' > Modelfile && "
            f"ollama create {f['name']} -f Modelfile",
            "```", "",
            "LM Studio: put the file in its models folder and pick it from the list.", "",
        ]
        if f["answer_style"] in OPENERS:
            template = OPENERS[f["answer_style"]].format(prompt="{your question}")
            out += ["It was taught to answer in this form, and answers best when asked "
                    "in it:", "", "```", template.rstrip("\n"), "```", ""]
        return out

    out = ["## Using it", "", "```", f'python -m teacher ask {f["name"]} "your prompt"',
           f"python -m teacher serve {f['name']}      # an OpenAI-compatible API", "```", ""]
    if f["answer_style"] in OPENERS:
        template = OPENERS[f["answer_style"]].format(prompt="{your question}")
        out += ["It was taught to answer in this form, and answers best when asked in it "
                "(Teacher does this for you):", "", "```", template.rstrip("\n"), "```", ""]
    if f["kind"] == "pretrained":
        out += [
            "Outside Teacher, with 🤗 Transformers:", "", "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            'tokenizer = AutoTokenizer.from_pretrained("path/to/this/folder")',
            'model = AutoModelForCausalLM.from_pretrained("path/to/this/folder")',
            "```", "",
        ]
    else:
        out += [
            "It uses Teacher's own architecture, so 🤗 Transformers cannot load it on its "
            "own. Run it with Teacher, with AI Studio, or in a browser with Bench "
            f"(`python -m teacher pack {f['name']}`).", "",
        ]
    return out


def _lessons(f: dict) -> list[str]:
    out = ["## What it was taught", ""]
    if not f["lessons"]:
        out += ["Nothing yet — these are the weights it was made with.", ""]
    else:
        out += [
            "| # | Date | Kind | Material | Epochs | Steps | Rate | Held-out loss | Measured on | Device |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for number, lesson in enumerate(f["lessons"], start=1):
            kind = "answers" if lesson.get("style") not in (None, "text") else "text"
            if lesson.get("lora"):
                kind += " (LoRA)"
            if lesson.get("rounds_in_effect"):
                kind += (f", {lesson['rounds_in_effect']} of its "
                         f"{lesson.get('rounds_kept', lesson.get('rounds'))} rounds")
            if lesson.get("inherited"):
                kind += ", before the branch"
            amount = (f"{lesson['pairs']:,} pairs" if lesson.get("pairs")
                      else f"{int(lesson.get('characters') or 0):,} chars")
            rate = lesson.get("learning_rate")
            out.append(
                f"| {number} | {_date(lesson.get('at'))} | {kind} | {amount} | "
                f"{_number(lesson.get('epochs'))} | {_number(lesson.get('steps'))} | "
                f"{f'{rate:.0e}' if rate else '—'} | {_loss(lesson.get('held_out_loss'))} | "
                f"{lesson.get('measured_on') or '—'} | {lesson.get('device') or '—'} |"
            )
        out.append("")
        out.append("The held-out loss is measured on text the lesson did not train on. "
                   "Lower is better; it only compares fairly between lessons measured "
                   "on the same kind of text.")
        out.append("")
    if f["undone"]:
        out += [
            f"{len(f['undone'])} more lesson(s) are in the record but not in these weights — "
            "undone by a rollback, or taught to the model this was branched from after "
            "the branch was taken:", "",
        ]
        for lesson in f["undone"]:
            out.append(f"- {_date(lesson.get('at'))}: {int(lesson.get('characters') or 0):,} "
                       f"characters, held-out loss {_loss(lesson.get('held_out_loss'))}")
        out.append("")
    if not f["certain"] and f["lessons"]:
        out += [
            "Some of these lessons were recorded before Teacher noted which weights each "
            "lesson made, so a rollback from that time cannot be told apart from a lesson "
            "that stuck. They are listed as taught.", "",
        ]
    return out


def _sources(f: dict) -> list[str]:
    out = ["## Where the material came from", ""]
    if not f["sources"]:
        return out + ["No material is recorded for the lessons in these weights.", ""]
    for source in f["sources"]:
        if source["kind"] == "web":
            out.append(f"- [{source['title']}]({source['url']}) — web page, retrieved "
                       f"{source.get('retrieved') or 'on an unrecorded date'}")
        elif source["kind"] == "typed":
            out.append(f"- {source['label']}")
        else:
            gone = " (no longer on disk)" if source.get("missing") else ""
            out.append(f"- `{source['name']}` — a local file{gone}")
    out += ["", "Local files are named, not located: their paths stay on the machine "
            "that trained the model.", ""]
    return out


def _exams(f: dict) -> list[str]:
    from teacher.exams import verdict

    out = ["## How it scores", ""]
    if not f["exams"]:
        out.append(f"Not benchmarked on these weights. Run: `python -m teacher test {f['name']} "
                   "arc`")
    else:
        out += [
            "| Benchmark | Items | Right | Score | Guessing scores | What it means |",
            "|---|---|---|---|---|---|",
        ]
        for exam in f["exams"]:
            chance = exam.get("chance")
            out.append(
                f"| {exam.get('label') or exam.get('suite')} | {exam.get('items')} | "
                f"{exam.get('correct')} | {exam.get('accuracy', 0) * 100:.0f}% | "
                f"{f'{chance * 100:.0f}%' if chance is not None else '—'} | "
                f"{verdict(exam)} |"
            )
        out += ["", "Greedy decoding, so each run repeats exactly. With this few items a "
                "score has to clear about two standard deviations of the guessing rate "
                "before it means anything, and the last column says whether it did."]
    if f["older_exams"]:
        out += ["", f"{len(f['older_exams'])} earlier result(s) were taken on weights this "
                "model no longer has, and are left out."]
    return out + [""]


def _this_file(export: dict) -> list[str]:
    return [
        "## This file", "",
        "| | |", "|---|---|",
        f"| File | `{export['file']}` |",
        f"| Format | GGUF, {export['precision']} |",
        f"| Size | {export['bytes']:,} bytes |",
        f"| SHA-256 | `{export['sha256']}` |",
        "", "Converted by llama.cpp's own `convert_hf_to_gguf.py`.", "",
    ]


def _licence(f: dict) -> list[str]:
    out = ["## Licence", ""]
    out.append("Teacher, the program that made this card, is Apache 2.0. That licence covers "
               "the program. It does not cover these weights, and it does not cover the "
               "material they were taught.")
    out.append("")
    if f["base_repo"]:
        if f["base_license"]:
            out.append(f"The base model, {f['base_repo']}, is released by its authors under "
                       f"**{f['base_license']}** (from its model card). These weights are "
                       "derived from it, so its terms come with them.")
        else:
            out.append(f"Teacher could not find the licence of the base model, {f['base_repo']}. "
                       f"Read https://huggingface.co/{f['base_repo']} before sharing this.")
        out.append("")
    if any(source["kind"] == "web" for source in f["sources"]):
        out.append("Web pages keep their authors' terms; each one is listed above with its "
                   "address.")
        out.append("")
    out.append("No licence has been chosen for this model. The person who made it decides "
               "that, knowing what the material was.")
    return out + [""]


def _limits(f: dict) -> list[str]:
    size = _a(f"{_short(f['parameters'])}-parameter model") if f["parameters"] else "a model"
    return [
        "## Limitations", "",
        f"- {_cap(size)} knows what was in its material"
        + (" and in its base model" if f["base_repo"] else "")
        + ", and will state things that are false as fluently as things that are true.",
        "- It can repeat passages of its material word for word. Do not teach it anything "
        "you would not want repeated.",
        "- It has not been evaluated for bias, safety or factual accuracy beyond the "
        "scores above.",
        "",
    ]


# ------------------------------------------------------------------ write
def written_by_teacher(path: Path) -> bool:
    try:
        return MARK in path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return False


def write(model: Model, destination: Path | None = None, *, export: dict | None = None,
          known: dict | None = None, look_up_licence: bool = True) -> Path:
    """Write the card, by default as README.md inside the model's folder.

    A file already there that Teacher did not write is left alone: someone's
    own README is not Teacher's to replace.
    """
    target = Path(destination) if destination else model.path / "README.md"
    if target.exists() and not written_by_teacher(target):
        raise TeacherError(
            f"{target} already exists and was not written by Teacher, so it was left "
            f"alone.\n  Write the card somewhere else:  teacher card {model.name} --out CARD.md"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render(known or facts(model, look_up_licence=look_up_licence), export=export)
    target.write_text(text, encoding="utf-8")
    return target


def beside_export(model: Model, written: dict) -> tuple[dict, Path | None, str | None]:
    """Record an export and write its card next to it: seabot.gguf, seabot.md.

    Returns the export as recorded, where the card went, and why it did not go
    anywhere if it did not — a card is worth having, but never worth failing
    an export that has already succeeded.
    """
    path = Path(written["path"])
    entry = {
        "at": time.time(),
        "file": path.name,
        "precision": written.get("precision"),
        "bytes": written.get("bytes") or path.stat().st_size,
        "sha256": sha256(path),
        "weights": model.weights_stamp(),
    }
    model.note(exports=[*model.history().get("exports", []), entry])
    try:
        return entry, write(model, path.with_suffix(".md"), export=entry), None
    except TeacherError as exc:
        return entry, None, str(exc).splitlines()[0]
    except OSError as exc:
        return entry, None, f"the card could not be written: {exc}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
