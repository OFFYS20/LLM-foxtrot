"""Teacher's command line.

    teacher new mymodel --from ./books
    teacher teach mymodel --from ./more-books
    teacher ask mymodel "once upon a"
    teacher show mymodel
    teacher pack mymodel -o ./for-bench
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from teacher import __version__, lessons, workspace
from teacher.material import gather
from teacher.workspace import TeacherError

DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def style(text: str, code: str) -> str:
    return f"{code}{text}{OFF}" if sys.stdout.isatty() else text


#: When true, prose goes nowhere and every command ends in one JSON object.
_JSON = {"on": False, "emitted": False}


def say(message: str = "") -> None:
    if _JSON["on"]:
        return
    print(message, flush=True)


def emit(**payload) -> int:
    """The machine-readable result of a command. Prints only in --json mode."""
    if not _JSON["on"] or _JSON["emitted"]:
        return 0
    import json as _json

    _JSON["emitted"] = True
    print(_json.dumps({"ok": True, **payload}, default=str, indent=2), flush=True)
    return 0


def fmt_count(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:,.0f}"


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def collect(args) -> "gather":
    sources = list(args.source or [])
    if getattr(args, "web", "") and args.web.strip():
        sources.append(str(fetch_material(args.web, getattr(args, "web_results", 5))[0]))
        # So the "teach it again" advice below names the folder that was gathered
        # rather than "<your text>".
        args.source = sources
    material = gather(sources, raw_text=args.text or "", clean=not args.raw)
    if material.partial:
        say(style(f"  read only in part — a row file has a ceiling on how much is read:", DIM))
        for path, dropped in material.partial[:5]:
            say(style(f"    {path} — {dropped:,} row(s) left out", DIM))
        say(style("    split the file into smaller ones to use all of it", DIM))
    if material.skipped:
        say(style(f"  skipped {len(material.skipped)} item(s):", DIM))
        for path, why in material.skipped[:5]:
            say(style(f"    {path} — {why}", DIM))
        if len(material.skipped) > 5:
            say(style(f"    …and {len(material.skipped) - 5} more", DIM))
    if not material.characters:
        if getattr(args, "answers", None):
            # The pairs are the material for this lesson; --from is optional.
            return material
        raise TeacherError("Nothing readable was found. Point --from at a file or folder of text.")
    say(f"  material: {material.summary()}")
    return material


def fetch_material(query: str, results: int, urls: list[str] | None = None,
                   destination: Path | None = None) -> tuple[Path, "object"]:
    """Search the web, read what it finds, and keep it as text files.

    Returns the folder it was written to and the harvest itself.
    """
    from teacher import websearch

    if query and query.strip():
        say(style(f"  searching the web for {query!r}", DIM))
    else:
        say(style(f"  reading {len(urls or [])} address(es)", DIM))

    def on_step(index, total, url):
        say(style(f"    {index}/{total}  {url[:88]}", DIM))

    collected = websearch.harvest(query, urls=urls, limit=results, on_step=on_step)
    folder = Path(destination) if destination else websearch.folder_for(
        query or (urls or [""])[0], workspace.root())
    websearch.save(collected, folder)
    say(style(f"  kept {collected.summary()} in {folder}", DIM))
    for url, why in collected.skipped[:5]:
        say(style(f"    skipped {url} — {why}", DIM))
    say(style("  each file records the address it came from — web pages carry "
              "their own terms", DIM))
    return folder, collected


# ------------------------------------------------------------------ commands
def cmd_new(args) -> int:
    model = workspace.get(args.name, must_exist=False)
    if model.path.exists() and model.exists() and not args.replace:
        raise TeacherError(f"'{model.name}' already exists. Teach it more, or pass --replace.")

    say(f"{style('Reading material', BOLD)}")
    material = collect(args)

    if args.base:
        say(f"\n{style('Adopting ' + args.base, BOLD)}")
        say(style("  downloading — this happens once", DIM))
        built = lessons.adopt(model, args.base, trust_remote_code=args.trust_remote_code)
        say(f"  base:       {built['base']}")
        say(f"  parameters: {fmt_count(built['parameters'])}")
        say(f"  vocabulary: {built['vocab_size']:,} tokens")
        say(f"  context:    {built['context']} tokens")
        say(style("  it already writes the language it was trained on; your lessons "
                  "teach it your material", DIM))
    else:
        say(f"\n{style('Building ' + model.name, BOLD)}")
        built = lessons.create(model, material, args.size, context=args.context)
        drift = built["parameters"] / built["asked_for"] - 1
        say(f"  asked for:  {fmt_count(built['asked_for'])} parameters")
        say(f"  built:      {fmt_count(built['parameters'])}  ({drift * 100:+.1f}%)")
        say(style("  widths move in steps, so a number lands near rather than on", DIM))
        say(f"  shape:      {built['hidden_size']} wide, {built['layers']} layers, "
            f"{built['heads']} heads")
        say(f"  vocabulary: {built['vocab_size']:,} tokens")
        say(f"  context:    {built['context']} tokens")
    say(style(f"  stored in {model.path}", DIM))

    if args.no_teach:
        say("\nMade but not taught — it will produce noise until you run: "
            f"teacher teach {model.name} --from <material>")
        return emit(command="new", model=model.name, taught=False, **built)

    say(f"\n{style('First lesson', BOLD)}")
    return run_lesson(model, material, args)


def cmd_teach(args) -> int:
    model = workspace.get(args.name)
    say(f"{style('Reading answers' if args.answers else 'Reading material', BOLD)}")
    material = collect(args)
    say(f"\n{style('Teaching ' + model.name, BOLD)}")
    return run_lesson(model, material, args)


def run_lesson(model, material, args) -> int:
    if getattr(args, "until", None):
        return run_until(model, material, args)

    last = {"line": ""}

    def on_log(message: str, level: str = "info") -> None:
        if "[TRAIN]" in message or "[PREP" in message or "[WARN" in message:
            text = message.split("] ", 1)[-1]
            if text != last["line"]:
                say(style("  " + text, DIM))
                last["line"] = text

    lesson = lessons.teach(
        model, material,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.rate,
        keep_checkpoints=getattr(args, "keep", 2),
        lora=getattr(args, "lora", False),
        lora_rank=getattr(args, "lora_rank", 16),
        gpus=getattr(args, "gpus", 1),
        device=getattr(args, "device", "auto"),
        answers=getattr(args, "answers", None),
        on_log=on_log,
    )

    say("")
    say(f"  learned from:  {fmt_count(lesson['tokens'])} tokens over {lesson['epochs']:g} epoch(s)")
    say(f"  loss:          {lesson['final_loss']:.4f}" if lesson["final_loss"] is not None else "  loss: —")
    if lesson["held_out_loss"] is not None:
        say(f"  on held-out:   {lesson['held_out_loss']:.4f}  (perplexity {lesson['perplexity']:.1f})")
    say(f"  took:          {lesson['seconds']:.1f}s on {lesson['device']}")
    if lesson.get("lora"):
        detail = lesson["lora"]
        say(f"  adapter:       rank {detail['rank']} on {', '.join(detail['target_modules'])}")
        say(f"                 {fmt_count(detail['trainable_parameters'])} of "
            f"{fmt_count(detail['total_parameters'])} weights trained "
            f"({detail['trainable_percent']:.2f}%), then merged in")

    report_progress(model, lesson, args)
    label, _note, share = lessons.judge(model, lesson["held_out_loss"])
    return emit(command="teach", model=model.name, stage=label, share=round(share, 4),
                held_out_loss=lesson["held_out_loss"], epochs=lesson["epochs"],
                seconds=lesson["seconds"], device=lesson["device"])


def run_until(model, material, args) -> int:
    """Teach in rounds until the model gets where you asked, or stops improving."""
    say(style(f"  teaching until '{args.until}' — {args.epochs:g} epoch(s) per round, "
              f"at most {args.max_rounds}", DIM))
    say(style("  every round is saved, so Ctrl-C is safe\n", DIM))

    def on_round(index, lesson, label, share):
        loss = lesson["held_out_loss"] or lesson["final_loss"]
        bar = "#" * max(1, int((1 - min(share, 1.0)) * 24))
        say(f"  round {index:>2}  held-out {loss:>7.4f}  {bar:<24} {label}")

    summary = lessons.teach_until(
        model, material,
        target=args.until,
        epochs_per_round=args.epochs,
        max_rounds=args.max_rounds,
        max_minutes=args.max_minutes,
        batch_size=args.batch,
        learning_rate=args.rate,
        keep_checkpoints=getattr(args, "keep", 2),
        lora=getattr(args, "lora", False),
        lora_rank=getattr(args, "lora_rank", 16),
        gpus=getattr(args, "gpus", 1),
        device=getattr(args, "device", "auto"),
        answers=getattr(args, "answers", None),
        on_round=on_round,
    )

    if not summary["rounds"]:
        say(f"\n  Nothing to do — {summary['reason']}.")
        return emit(command="teach", model=model.name, rounds=0,
                    reason=summary["reason"], held_out_loss=summary["held_out_loss"])

    first, last_loss = summary["first_loss"], summary["held_out_loss"]
    say("")
    say(f"  rounds:        {summary['rounds']} ({summary['epochs']:g} epochs total)")
    if first is not None and last_loss is not None:
        say(f"  held-out loss: {first:.4f} -> {last_loss:.4f}")
    say(f"  took:          {fmt_duration(summary['seconds'])} on {summary['device']}")
    say(f"  stopped:       {summary['reason']}")

    label, note, share = lessons.judge(model, last_loss)
    say(f"\n  {style(model.name + ': ' + label, BOLD)}")
    say(style(f"  {note}", DIM))

    if summary["reason"] == "it stopped improving" and share >= 0.28:
        say(style("\n  More epochs will not help from here — it needs more material.", DIM))
    else:
        say(f"\n  Try it:  python -m teacher ask {model.name} \"...\"")
    return emit(command="teach", model=model.name, stage=label, share=round(share, 4),
                rounds=summary["rounds"], epochs=summary["epochs"],
                held_out_loss=summary["held_out_loss"], first_loss=summary["first_loss"],
                reason=summary["reason"], seconds=summary["seconds"])


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def report_progress(model, lesson, args) -> None:
    """Say how far along the model is, and what to do next — not just "ready"."""
    label, note, share = lessons.judge(model, lesson["held_out_loss"])
    say(f"\n  {style(model.name + ': ' + label, BOLD)}")
    say(style(f"  {note}", DIM))

    source = " ".join(f'--from "{path}"' for path in (args.source or [])) or "--from <your text>"
    if share >= 0.55:
        more = max(10, int(args.epochs * 4))
        say(f"\n  Not done yet — teach it again, longer:")
        say(f"    python -m teacher teach {model.name} {source} --epochs {more}")
    elif share >= 0.15:
        more = max(10, int(args.epochs * 2))
        say(f"\n  Worth another lesson:")
        say(f"    python -m teacher teach {model.name} {source} --epochs {more}")
        say(style("  Stop when the held-out number stops falling between lessons.", DIM))
    else:
        say(f"\n  Try it:  python -m teacher ask {model.name} \"...\"")
        say(style("  If it still disappoints, it needs more material rather than more "
                  "epochs.", DIM))


def cmd_ask(args) -> int:
    model = workspace.get(args.name)
    prompt = " ".join(args.prompt).strip()

    if not prompt:
        say(style(f"Talking to {model.name}. Blank line or Ctrl-C to stop.", DIM))
        try:
            while True:
                line = input("\nyou  ").strip()
                if not line:
                    return 0
                respond(model, line, args)
        except (EOFError, KeyboardInterrupt):
            say("")
            return 0

    return respond(model, prompt, args)


def respond(model, prompt: str, args) -> None:
    text, stats = lessons.talk(
        model, prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    say(f"\n{style(prompt, DIM)}{text}")
    say(style(
        f"\n  {stats['generated']} tokens at {stats['tokens_per_second']:.0f}/s"
        f" (prompt {stats['prompt_tokens']})", DIM))
    return emit(command="ask", model=model.name, prompt=prompt, reply=text, **stats)


def cmd_web(args) -> int:
    """Look something up on the web and keep the pages as material."""
    from teacher import websearch

    query = " ".join(args.query or []).strip()
    if not query and not args.url:
        raise TeacherError("Give something to search for, or --url ADDRESS.")

    if args.list:
        if not query:
            raise TeacherError("--list shows what a search found; give it something to search for.")
        found = websearch.search(query, limit=args.results)
        say(f"{style(str(len(found)) + ' result(s) for ' + repr(query), BOLD)}\n")
        for index, result in enumerate(found, start=1):
            say(f"  {index}. {result.title}")
            say(style(f"     {result.url}", DIM))
            if result.snippet:
                say(style(f"     {result.snippet[:160]}", DIM))
        say(style("\n  Nothing was downloaded. Drop --list to read and keep them.", DIM))
        return emit(command="web", query=query, fetched=False,
                    results=[r.to_dict() for r in found])

    say(f"{style('Gathering from the web', BOLD)}")
    destination = Path(args.out).expanduser() if args.out else None
    folder, collected = fetch_material(query, args.results, urls=list(args.url or []),
                                       destination=destination)
    say(f"\n  {len(collected.pages)} page(s), {collected.characters:,} characters")
    say(style(f"\n  Teach from it:  python -m teacher teach NAME --from \"{folder}\"", DIM))
    return emit(command="web", query=query, fetched=True, folder=str(folder),
                characters=collected.characters,
                pages=[page.to_dict() for page in collected.pages],
                skipped=[{"url": url, "why": why} for url, why in collected.skipped])


def cmd_list(args) -> int:
    models = workspace.every()
    if not models:
        say(f"No models yet in {workspace.root()}.")
        say("Make one with:  teacher new mymodel --from ./some-text")
        return emit(command="list", home=str(workspace.root()), models=[])

    say(f"{'NAME':<22}{'PARAMS':>9}{'TAUGHT':>12}{'LESSONS':>9}{'SIZE':>9}")
    for model in models:
        arch = model.architecture()
        params = arch.get("parameters")
        if params is None:
            from ai_studio.models.transformer import TransformerConfig
            try:
                params = TransformerConfig(**{
                    k: v for k, v in arch.items()
                    if k in TransformerConfig.__dataclass_fields__
                }).parameter_count()["total"]
            except Exception:  # noqa: BLE001
                params = None
        history = model.history()
        say(f"{model.name:<22}{fmt_count(params):>9}"
            f"{fmt_count(model.taught_characters()) + 'c':>12}"
            f"{len(history.get('lessons', [])):>9}{fmt_bytes(model.size_bytes()):>9}")
    say(style(f"\nin {workspace.root()}", DIM))
    return emit(command="list", home=str(workspace.root()), models=[
        {"name": m.name, "kind": m.kind(), "base": m.base_repo(),
         "lessons": len(m.history().get("lessons", [])),
         "taught_characters": m.taught_characters()}
        for m in models
    ])


GUIDE = """\
You are driving Teacher, a program on my computer that trains small language
models from my own text. Run the commands for me and tell me what happened in
plain language — I do not want to read terminal output.

WHERE THINGS ARE
  Run every command from:  {repo}
  Models are stored in:    {home}
  Python to use:           {python}

HOW TO TALK TO IT
  Add --json to any command and you get one JSON object back instead of prose:
  {{"ok": true, ...}} on success, {{"ok": false, "error": "..."}} on failure.
  Read that, not the human text. Never invent a result you did not run.

THE COMMANDS
  {python} -m teacher --json list
      What models exist already.

  {python} -m teacher --json bases
      Pretrained models worth starting from.

  {python} -m teacher --json new NAME --base small --from PATH
      Make a model starting from a pretrained one (best output, needs a
      download). Drop --base to build from scratch instead, which also takes
      --size takes any parameter count I name — 70K, 4M, 51M, 1.5B, 250000.
      1m|10m|100m|200m|500m|1b are shortcuts for common numbers, nothing more.
      The reply carries "asked_for" and "parameters"; quote "parameters", the
      one it built. Anything past 10M wants a GPU; do not choose one on a
      laptop. A number this machine cannot hold is refused with the arithmetic.

  {python} -m teacher --json teach NAME --from PATH --until best
      Teach it until it stops improving. Add --epochs N to set the size of one
      round, --max-rounds N to cap it, --max-minutes N to put a clock on it.
      Returns stage, held_out_loss, rounds and why it stopped.

  {python} -m teacher --json ask NAME "some words"
      Get a continuation. Returns the reply and how fast it ran.

  {python} -m teacher --json show NAME
      Architecture, every lesson, and how far along it is.

  {python} -m teacher --json rollback NAME [--to STAMP]
      Undo the last lesson if it made the model worse, or go back to a named
      saved state. Two are kept by default; --keep N on a lesson keeps more.

  {python} -m teacher --json compare A B --prompt "some words"
      The same prompt through two or more models at the same seed. Returns each
      one's reply, stage and held-out loss, and whether those losses can be
      compared at all — they cannot across different vocabularies.

  {python} -m teacher --json teach NAME --answers PATH --epochs 12
      Teach it to answer rather than continue, from question-and-answer pairs:
      .jsonl, .json or .csv with question/answer, instruction/output or
      messages. The loss is taken on the answer only. Afterwards `ask` wraps
      my question in the template it was taught with, so do not add a prefix
      of your own.

  {python} -m teacher --json export NAME --precision f16
      Convert to GGUF for llama.cpp, Ollama or LM Studio. Needs llama.cpp's
      converter cloned; if it is missing, or it refuses the model, nothing is
      written and the error says why — tell me that rather than calling it
      exported. Only works for models started from a pretrained base.

  {python} -m teacher --json test NAME arc --items 20
      Run a real benchmark suite. Returns accuracy, chance, beats_chance,
      source and official. Read all of them: official:false means it ran on
      bundled example items rather than the real split and is not a score, and
      beats_chance:false means the result is within noise of guessing. A model
      this size scoring at chance on a knowledge benchmark is expected and not
      a fault — say so rather than letting me think training failed. Always
      quote the chance rate beside the accuracy. --list shows every suite.

  {python} -m teacher --json checkpoints NAME
      The saved states this model can go back to or branch from, newest last.

  {python} -m teacher --json branch NAME NEW-NAME [--at STAMP]
      Copy the model, or one of its saved states, into a new model. The
      original is not touched, so this is how to try a different training run
      without risking what already works.

  {python} -m teacher --json web "SOMETHING" --results 5
      Search the web, read what it finds, and save each page as a text file in
      a new folder. Returns that folder, and a list of the pages with their
      addresses. Add --url ADDRESS to read a page you already know, with or
      without a search; add --list to see what a search found without
      downloading anything; add -o DIR to choose where it goes.
      Then teach from it with --from FOLDER.

      Add --gpus auto to use every GPU in the machine: one process per card,
      gradients averaged every step. The batch is per GPU, so the real step is
      batch x gpus. --device auto|cpu|cuda picks where to train. Do not
      suggest using the CPU and the GPU together; it is slower, not faster.

      Add --batch auto to fill the hardware: it picks the largest batch that
      fits in memory and still leaves enough optimizer steps to learn from.
      Worth using whenever a GPU is sitting at half load.

      Add --lora to teach or new to train a small adapter instead of every
      weight: far less memory, so a bigger base fits. Pretrained models only —
      it is refused on one built from scratch, with a reason.

      new and teach also take --web "SOMETHING" to do both in one step:
      {python} -m teacher --json new NAME --base small --web "SOMETHING" --until best

      Pages run about 10,000-34,000 characters each, so five pages is roughly
      50,000-150,000. That is enough to fine-tune a pretrained model and far
      too little to build one from scratch, so always pair --web with --base
      unless I have asked you to gather dozens of pages.

  PATH is a file or a folder. It reads .txt .md .pdf .docx .epub .html .csv
  .json and walks folders. --text "..." works instead of --from for short text.

WHAT THE RESULTS MEAN
  "stage" is how far along the model is. For one built from scratch:
  barely started -> learning the alphabet -> learning words -> learning
  sentences -> has the shape of your text. For one started from a pretrained
  model: barely moved -> picking up your material -> adapting well ->
  closely fitted.
  "held_out_loss" is measured on text it never trained on. Lower is better.
  Falling between lessons means it is still learning; flat means it has
  stopped, and more epochs will not help — it needs more material.

THE FULL VERSION
  docs/OPERATING.md in this repository covers all three programs — Teacher,
  the Bench browser page and AI Studio — with every command and flag. If you
  need something that is not here, read that.

RULES
  - Ask me where my text is before you start. Do not guess a path.
  - Only go to the web if I asked for it or agreed to it, and tell me which
    pages you took. Those pages are someone else's writing under someone
    else's terms; every saved file keeps its address for that reason. Five to
    ten pages is a normal run — do not pull hundreds.
  - With web material, use --base. A handful of pages cannot teach a model
    English from nothing; it will stop at "barely started" and say so.
  - Start from a pretrained model unless I say I want to watch one learn from
    nothing; the output is far better.
  - Use --until best rather than picking an epoch count.
  - If a command returns ok:false, tell me the error and what to do about it.
    Do not retry the same thing.
  - Training takes minutes to hours. Say so before starting a long one.
  - If a lesson raises the held-out loss, say so and offer to roll it back.
  - Before trying something that might not work — much more material, a very
    different setting — branch the model first and train the branch. Then
    neither of us has to undo anything, and `compare` says which won.
  - Do not read two models' losses against each other unless compare says
    they are comparable. It tells you.
  - Tell me the stage and whether it is still improving. Skip the numbers
    unless I ask.
"""


def cmd_guide(args) -> int:
    """Instructions a person can paste into another AI so it drives Teacher."""
    repo = Path(__file__).resolve().parent.parent
    python = sys.executable or "python"
    text = GUIDE.format(repo=repo, home=workspace.root(), python=python)
    if _JSON["on"]:
        return emit(command="guide", guide=text, repo=str(repo),
                    home=str(workspace.root()), python=python)
    say(style("Copy everything below into ChatGPT, Claude or any assistant that "
              "can run commands on this machine.\n", DIM))
    say("-" * 72)
    say(text)
    say("-" * 72)
    return 0


def cmd_ui(args) -> int:
    try:
        from teacher import ui
    except ImportError as exc:
        raise TeacherError(
            f"The window needs gradio: pip install gradio  ({exc})"
        ) from exc
    if _JSON["on"]:
        return emit(command="ui", url=f"http://{args.host}:{args.port}",
                    note="the window runs until stopped; start it without --json")
    say(f"Opening Teacher at http://{args.host}:{args.port}")
    say(style("Close this window or press Ctrl-C to stop it.\n", DIM))
    ui.launch(host=args.host, port=args.port, share=args.share,
              open_browser=not args.no_browser)
    return 0


def cmd_bases(args) -> int:
    say("Starting from a pretrained model teaches it your material instead of the")
    say("language itself. Use any Hugging Face name, or one of these shortcuts:\n")
    say(f"  {'SHORTCUT':<12}{'MODEL':<36}NOTES")
    for key, (repo, note) in lessons.BASES.items():
        say(f"  {key:<12}{repo:<36}{note}")
    say(style("\n  python -m teacher new mymodel --base small --from ./mytext", DIM))
    say(style("  Models built this way run in the Teacher UI, not the Bench web page.", DIM))
    return emit(command="bases", bases=[
        {"shortcut": key, "repo": repo, "note": note}
        for key, (repo, note) in lessons.BASES.items()
    ])


def cmd_show(args) -> int:
    model = workspace.get(args.name)
    arch = model.architecture()
    history = model.history()
    lessons_taught = history.get("lessons", [])

    say(style(model.name, BOLD))
    say(f"  path        {model.path}")
    say(f"  layers      {arch.get('num_layers')}   hidden {arch.get('hidden_size')}   "
        f"heads {arch.get('num_heads')}")
    say(f"  context     {arch.get('max_position_embeddings')} tokens")
    say(f"  vocabulary  {arch.get('vocab_size'):,} tokens")
    say(f"  on disk     {fmt_bytes(model.size_bytes())}")
    if history.get("branched_from"):
        say(f"  branched    from {history['branched_from']} at {history['branched_at']}")
    states = model.saved_states()
    if states:
        say(f"  saved       {len(states)} state(s), newest {states[-1]['stamp']}")

    if not lessons_taught:
        say("\n  Never taught anything — it will produce noise.")
        return emit(command="show", model=model.name, kind=model.kind(),
                    base=model.base_repo(), architecture=arch, lessons=0,
                    stage="never taught", path=str(model.path),
                    saved_states=[state["stamp"] for state in states],
                    branched_from=history.get("branched_from"),
                    branched_at=history.get("branched_at"))

    say(f"\n  {len(lessons_taught)} lesson(s), {fmt_count(model.taught_characters())} characters total")
    for index, lesson in enumerate(lessons_taught[-8:], start=max(1, len(lessons_taught) - 7)):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(lesson.get("at", 0)))
        loss = lesson.get("held_out_loss") or lesson.get("final_loss")
        sources = lesson.get("sources", [])
        first = Path(sources[0]).name if sources else "?"
        more = f" +{len(sources) - 1}" if len(sources) > 1 else ""
        say(f"    {index:>2}. {when}  loss {loss:.4f}   "
            f"{fmt_count(lesson.get('characters'))}c  {first}{more}"
            if loss is not None else f"    {index:>2}. {when}  {first}{more}")

    last = lessons_taught[-1]
    label, note, share = lessons.judge(model, last.get("held_out_loss"))
    return emit(command="show", model=model.name, kind=model.kind(), base=model.base_repo(),
                architecture=arch, lessons=len(lessons_taught), stage=label, advice=note,
                share=round(share, 4), held_out_loss=last.get("held_out_loss"),
                taught_characters=model.taught_characters(), path=str(model.path),
                saved_states=[state["stamp"] for state in states],
                branched_from=history.get("branched_from"),
                branched_at=history.get("branched_at"))


def cmd_pack(args) -> int:
    model = workspace.get(args.name)
    destination = Path(args.out).expanduser()
    copied = model.pack(destination)
    say(f"Copied {len(copied)} file(s) to {destination}:")
    for name in copied:
        say(f"  {name}  {fmt_bytes((destination / name).stat().st_size)}")
    say(style("\nOpen web_chat/index.html and drop that folder in to talk to it.", DIM))
    return emit(command="pack", model=model.name, destination=str(destination), files=copied)


def cmd_compare(args) -> int:
    """The same prompt through two or more models, under identical sampling."""
    models = [workspace.get(name) for name in args.names]
    prompt = " ".join(args.prompt).strip() or "The"

    result = lessons.compare(
        models, prompt,
        max_new_tokens=args.tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, seed=args.seed,
    )

    say(f"{style('Prompt', BOLD)}  {prompt!r}   "
        + style(f"(seed {result['seed']}, {args.tokens} tokens, temperature "
                f"{args.temperature})", DIM))

    for entry in result["models"]:
        say(f"\n{style(entry['model'], BOLD)}  —  {entry['stage']}")
        detail = [f"{entry['lessons']} lesson(s)"]
        if entry["held_out_loss"] is not None:
            detail.append(f"held-out {entry['held_out_loss']:.4f}")
        detail.append(f"vocabulary {entry['vocab_size']:,}")
        if entry["branched_from"]:
            detail.append(f"branched from {entry['branched_from']}")
        say(style("  " + " · ".join(detail), DIM))
        say(f"  {style(prompt, DIM)}{entry['reply']}")

    say("")
    if result["losses_comparable"]:
        best = min(result["models"], key=lambda entry: entry["held_out_loss"])
        say(f"  Lowest held-out loss: {style(best['model'], BOLD)} "
            f"({best['held_out_loss']:.4f})")
        say(style("  Same vocabulary, so the numbers line up — but they were each "
                  "measured on their own material.", DIM))
        say(style("  Only trust the comparison if you taught them the same text.", DIM))
    elif not result["same_vocabulary"]:
        say(style("  These use different vocabularies, so their losses are not "
                  "comparable — a loss is an average over the tokens a model has, "
                  "and these carve text up differently. Judge by reading.", DIM))
    else:
        say(style("  Not every model here has been measured on held-out text yet. "
                  "Judge by reading.", DIM))
    return emit(command="compare", **result)


def cmd_test(args) -> int:
    """Sit a model down in front of a real benchmark suite."""
    from teacher import exams

    if args.list:
        rows = exams.catalogue()
        say(f"  {'SUITE':<12}{'NAME':<16}{'MEASURES':<16}{'ITEMS HERE':>11}")
        for row in rows:
            source = "sample" if row["source"] == "sample" else row["source"]
            say(f"  {row['suite']:<12}{row['label']:<16}{row['category']:<16}"
                f"{str(row['available']) + ' ' + source:>11}")
        say(style("\n  'sample' means the bundled example items, not the real split.", DIM))
        say(style("  pip install datasets — then the official splits download on first use.", DIM))
        return emit(command="test", suites=rows)

    if not args.name:
        raise TeacherError("Name a model to test, or pass --list to see the suites.")
    model = workspace.get(args.name)
    say(f"{style(model.name + ' sitting ' + args.suite, BOLD)}")

    def on_item(index, total, was_right):
        say(style(f"  {index}/{total}  {'right' if was_right else 'wrong'}", DIM))

    outcome = exams.sit(
        model, args.suite, limit=args.items, few_shot=args.shots,
        allow_download=not args.offline, on_item=on_item,
    )

    say("")
    say(f"  {style(outcome['label'], BOLD)}  —  {outcome['correct']}/{outcome['items']} "
        f"({outcome['accuracy'] * 100:.0f}%)")
    if outcome["chance"] is not None:
        say(f"  guessing would score  {outcome['chance'] * 100:.0f}%")
    say(f"  {outcome['shots']} worked example(s) in the prompt, {outcome['seconds']:.0f}s")

    if not outcome["official"]:
        say(style(f"\n  {outcome['note']}", DIM))
    say(f"\n  {exams.verdict(outcome)}")
    if outcome["official"] and outcome["beats_chance"] is False:
        say(style("  Benchmarks like this measure knowledge a model of this size "
                  "was never going to hold.", DIM))
    return emit(command="test", **outcome)


def cmd_checkpoints(args) -> int:
    """The saved states a model can be rolled back to or branched from."""
    model = workspace.get(args.name)
    states = model.saved_states()
    if not states:
        say(f"{model.name} has no saved states yet.")
        say(style("  One is written before every lesson. Teach it something first.", DIM))
        return emit(command="checkpoints", model=model.name, states=[], keep=None)

    say(style(f"{model.name} — {len(states)} saved state(s)", BOLD))
    say(style("  written before each lesson; the newest is what rollback restores\n", DIM))
    say(f"  {'STAMP':<18}{'TAKEN':<20}{'SIZE':>9}")
    for state in states:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(state["at"]))
        say(f"  {state['stamp']:<18}{when:<20}{fmt_bytes(state['bytes']):>9}")

    newest = states[-1]["stamp"]
    say(style(f"\n  Go back:    python -m teacher rollback {model.name} --to {newest}", DIM))
    say(style(f"  Branch it:  python -m teacher branch {model.name} {model.name}-v2 "
              f"--at {newest}", DIM))
    say(style("\n  Only the last few are kept. Pass --keep N to a lesson to hold more.", DIM))
    return emit(command="checkpoints", model=model.name, states=states)


def cmd_branch(args) -> int:
    """Carry on from a saved state in a new model, leaving the original alone."""
    source = workspace.get(args.name)
    made = workspace.branch(source, args.new_name, at=args.at)
    from_what = args.at or "its current weights"

    say(f"Branched {source.name} ({from_what}) into {style(made.name, BOLD)}.")
    say(style(f"  {made.path}", DIM))
    say(style(f"  {source.name} is untouched — train {made.name} as hard as you like.", DIM))
    say(style(f"  Its lesson list starts empty; {source.name}'s record is kept inside it.", DIM))
    say(f"\n  Teach it:  python -m teacher teach {made.name} --from <material> --until best")
    return emit(command="branch", model=made.name, branched_from=source.name,
                branched_at=from_what, path=str(made.path))


def cmd_export(args) -> int:
    """Convert a model to GGUF, for llama.cpp, Ollama and LM Studio."""
    from teacher import export

    model = workspace.get(args.name)
    destination = Path(args.out).expanduser() if args.out else (
        Path.cwd() / f"{model.name}-{args.precision}.gguf")

    say(f"{style('Converting ' + model.name + ' to GGUF', BOLD)}")
    say(style(f"  {args.precision} — {export.TYPES[args.precision]}", DIM))
    say(style("  llama.cpp's own converter does this; it takes a few minutes", DIM))

    written = export.to_gguf(
        model, destination, precision=args.precision, converter=args.converter,
        on_log=lambda message, level="info": say(style("  " + message, DIM)),
    )

    say(f"\n  {written['path']}  {fmt_bytes(written['bytes'])}")
    say("")
    for line in export.advice(written):
        say(style("  " + line, DIM))
    return emit(command="export", **written)


def cmd_rollback(args) -> int:
    model = workspace.get(args.name)
    saved = model.earlier_states()
    restored = model.rollback(to=args.to)
    which = f"the state saved at {args.to}" if args.to else "the state saved before its last lesson"
    say(f"Rolled {model.name} back to {which} ({restored}).")
    say(style("  Its history still lists every lesson — that is a record of what happened.", DIM))
    if not args.to and len(saved) > 1:
        say(style(f"  Earlier ones are still there: python -m teacher checkpoints {model.name}", DIM))
    return emit(command="rollback", model=model.name, restored_from=restored,
                states_left=len(saved) - 1)


def cmd_forget(args) -> int:
    model = workspace.get(args.name)
    if not args.yes:
        raise TeacherError(
            f"This would delete {model.path} and everything {model.name} has learned. "
            f"Pass --yes if you mean it."
        )
    import shutil

    shutil.rmtree(model.path)
    say(f"Deleted {model.name}.")
    return emit(command="forget", model=model.name, deleted=True)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teacher",
        description="Teach a small language model from your own text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sizes:\n" + lessons.describe_sizes(),
    )
    parser.add_argument("--version", action="version", version=f"teacher {__version__}")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object instead of prose, for scripts and AI assistants")
    subs = parser.add_subparsers(dest="command", required=True)

    def add_material(sub):
        sub.add_argument("--from", dest="source", action="append", metavar="PATH",
                         help="a file or folder to learn from (repeatable)")
        sub.add_argument("--text", help="text to learn from, given directly")
        sub.add_argument("--raw", action="store_true",
                         help="skip cleaning and use the text exactly as found")
        sub.add_argument("--answers", action="append", metavar="PATH",
                         help="question-and-answer pairs to learn to reply from, instead "
                              "of plain text: .jsonl, .json or .csv with instruction/"
                              "output, question/answer, or messages columns (repeatable)")
        sub.add_argument("--web", metavar="QUERY", default="",
                         help="also search the web for this and learn from what it finds; "
                              "the pages are kept as text files, not thrown away")
        sub.add_argument("--web-results", dest="web_results", type=int, default=5,
                         help="how many pages --web reads (default 5)")

    def add_training(sub):
        sub.add_argument("--epochs", type=float, default=3.0,
                         help="passes over the material (default 3); one round when --until is used")
        sub.add_argument("--batch", default="8", metavar="N",
                         help="sequences per step (default 8). 'auto' picks the largest "
                              "that fits, which is usually what a half-idle GPU is missing")
        sub.add_argument("--rate", type=float, default=3e-4, help="learning rate (default 3e-4)")
        sub.add_argument("--until", choices=list(lessons.TARGETS), metavar="STAGE",
                         help="keep teaching until it reaches this stage, or stops improving: "
                              + ", ".join(lessons.TARGETS))
        sub.add_argument("--max-rounds", dest="max_rounds", type=int, default=20,
                         help="most rounds an --until run may take (default 20)")
        sub.add_argument("--max-minutes", dest="max_minutes", type=float, default=0.0,
                         help="stop an --until run after this long (default: no limit)")
        sub.add_argument("--gpus", default="1", metavar="N",
                         help="how many GPUs to spread the lesson across: a number, or "
                              "'auto' for all of them (default 1). Each one runs a full "
                              "copy and the gradients are averaged every step")
        sub.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                         help="where to train (default auto: the GPU if there is one)")
        sub.add_argument("--lora", action="store_true",
                         help="train a small adapter instead of every weight — far less "
                              "memory, so a bigger base fits. Only for pretrained models; "
                              "the adapter is folded back in when the lesson ends")
        sub.add_argument("--lora-rank", dest="lora_rank", type=int, default=16, metavar="N",
                         help="how much the adapter can change (default 16; 8 is thriftier, "
                              "64 learns more and costs more)")
        sub.add_argument("--keep", type=int, default=2, metavar="N",
                         help="how many saved states to keep behind this model (default 2); "
                              "each is a full copy of the weights")

    new = subs.add_parser("new", help="build a new model and teach it its first lesson")
    new.add_argument("name")
    new.add_argument("--size", default="1m", metavar="SIZE",
                     help="how big to make it from scratch — any number of parameters "
                          "(70K, 50M, 1.5B, 7000000), or a ready-made rung: "
                          + ", ".join(lessons.SIZES)
                          + " (default 1m; tiny/small/medium/large/huge still work)")
    new.add_argument("--base", metavar="MODEL",
                     help="start from a pretrained model instead of from noise — a shortcut ("
                          + ", ".join(lessons.BASES) + ") or any Hugging Face name")
    new.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                     help="allow a base model to run its own code (only for repos you trust)")
    new.add_argument("--context", type=int, default=0, help="context length in tokens")
    new.add_argument("--replace", action="store_true", help="overwrite a model of the same name")
    new.add_argument("--no-teach", action="store_true", help="build it but do not train yet")
    add_material(new)
    add_training(new)
    new.set_defaults(func=cmd_new)

    teach = subs.add_parser("teach", help="teach an existing model from more material")
    teach.add_argument("name")
    add_material(teach)
    add_training(teach)
    teach.set_defaults(func=cmd_teach)

    ask = subs.add_parser("ask", help="talk to a model")
    ask.add_argument("name")
    ask.add_argument("prompt", nargs="*", help="what to say; leave empty for a back-and-forth")
    ask.add_argument("--tokens", type=int, default=120, help="how much to generate (default 120)")
    ask.add_argument("--temperature", type=float, default=0.8)
    ask.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ask.add_argument("--top-k", dest="top_k", type=int, default=40)
    ask.add_argument("--seed", type=int, default=None, help="repeat an exact answer")
    ask.set_defaults(func=cmd_ask)

    listing = subs.add_parser("list", help="show every model you have")
    listing.set_defaults(func=cmd_list)

    web = subs.add_parser(
        "web", help="search the web and keep what it finds as material")
    web.add_argument("query", nargs="*", help="what to search for")
    web.add_argument("--results", type=int, default=5,
                     help="how many pages to read (default 5)")
    web.add_argument("--url", action="append", metavar="ADDRESS",
                     help="read this page too, or instead of searching (repeatable)")
    web.add_argument("-o", "--out", metavar="DIR",
                     help="where to keep the text files (default: a dated folder "
                          "under your models directory)")
    web.add_argument("--list", action="store_true",
                     help="only show what the search found; download nothing")
    web.set_defaults(func=cmd_web)

    bases = subs.add_parser("bases", help="pretrained models worth starting from")
    bases.set_defaults(func=cmd_bases)

    guide = subs.add_parser(
        "guide", help="print instructions to paste into ChatGPT or Claude so it can drive Teacher")
    guide.set_defaults(func=cmd_guide)

    ui = subs.add_parser("ui", help="open the window — everything here, without the terminal")
    ui.add_argument("--port", type=int, default=7861)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--share", action="store_true", help="make a public link")
    ui.add_argument("--no-browser", dest="no_browser", action="store_true")
    ui.set_defaults(func=cmd_ui)

    show = subs.add_parser("show", help="what one model is and what it has been taught")
    show.add_argument("name")
    show.set_defaults(func=cmd_show)

    pack = subs.add_parser("pack", help="copy the files the Bench web page needs")
    pack.add_argument("name")
    pack.add_argument("-o", "--out", required=True, metavar="DIR")
    pack.set_defaults(func=cmd_pack)

    compare = subs.add_parser(
        "compare", help="the same prompt through two or more models, side by side")
    compare.add_argument("names", nargs="+", metavar="NAME", help="two or more models")
    compare.add_argument("--prompt", dest="prompt", action="append", default=[],
                         help="what to say to each of them")
    compare.add_argument("--tokens", type=int, default=80,
                         help="how much each one generates (default 80)")
    compare.add_argument("--temperature", type=float, default=0.8)
    compare.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    compare.add_argument("--top-k", dest="top_k", type=int, default=40)
    compare.add_argument("--seed", type=int, default=0,
                         help="the same for every model, so the dice are not the difference")
    compare.set_defaults(func=cmd_compare)

    test = subs.add_parser(
        "test", help="run a real benchmark suite against a model")
    test.add_argument("name", nargs="?", help="the model to test")
    test.add_argument("suite", nargs="?", default="mmlu",
                      help="which suite (default mmlu); --list shows them all")
    test.add_argument("--items", type=int, default=20,
                      help="how many questions to ask (default 20)")
    test.add_argument("--shots", type=int, default=None,
                      help="worked examples in the prompt (default: the suite's own)")
    test.add_argument("--offline", action="store_true",
                      help="do not download the official split; use what is already here")
    test.add_argument("--list", action="store_true",
                      help="show every suite and where its items would come from")
    test.set_defaults(func=cmd_test)

    checkpoints = subs.add_parser(
        "checkpoints", help="the saved states a model can go back to or branch from")
    checkpoints.add_argument("name")
    checkpoints.set_defaults(func=cmd_checkpoints)

    branch = subs.add_parser(
        "branch", help="copy a model, or one of its saved states, into a new model")
    branch.add_argument("name", help="the model to branch from")
    branch.add_argument("new_name", metavar="new-name", help="what to call the copy")
    branch.add_argument("--at", metavar="STAMP",
                        help="branch from this saved state instead of the current weights "
                             "(see: teacher checkpoints NAME)")
    branch.set_defaults(func=cmd_branch)

    export = subs.add_parser(
        "export", help="convert a model to GGUF for llama.cpp, Ollama or LM Studio")
    export.add_argument("name")
    export.add_argument("-o", "--out", metavar="FILE",
                        help="where to write it (default: NAME-PRECISION.gguf here)")
    export.add_argument("--precision", default="f16",
                        choices=("f32", "f16", "bf16", "q8_0"),
                        help="how much of each weight to keep (default f16)")
    export.add_argument("--converter", metavar="PATH",
                        help="llama.cpp's convert_hf_to_gguf.py, if it is somewhere unusual")
    export.set_defaults(func=cmd_export)

    rollback = subs.add_parser(
        "rollback", help="undo the last lesson, restoring the weights saved before it")
    rollback.add_argument("name")
    rollback.add_argument("--to", metavar="STAMP",
                          help="restore this saved state instead of the newest")
    rollback.set_defaults(func=cmd_rollback)

    forget = subs.add_parser("forget", help="delete a model and everything it learned")
    forget.add_argument("name")
    forget.add_argument("--yes", action="store_true")
    forget.set_defaults(func=cmd_forget)

    return parser


def quiet_library_logging() -> None:
    """Teacher prints its own progress; the library's stdout log would double it."""
    import logging

    from ai_studio.core import logging as studio_logging

    studio_logging.configure()
    logging.getLogger("ai_studio").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _JSON["on"] = bool(getattr(args, "json", False))
    _JSON["emitted"] = False
    quiet_library_logging()
    try:
        code = args.func(args)
        # Anything reading stdout must find an object, whatever path ran.
        emit(command=args.command)
        return code
    except TeacherError as exc:
        if _JSON["on"] and not _JSON["emitted"]:
            import json as _json

            _JSON["emitted"] = True
            print(_json.dumps({"ok": False, "command": args.command,
                               "error": str(exc)}, indent=2), flush=True)
        else:
            say(f"\n{style('Stopped:', BOLD)} {exc}")
        return 1
    except KeyboardInterrupt:
        say("\nStopped. Nothing was overwritten.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
