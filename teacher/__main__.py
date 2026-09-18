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


def say(message: str = "") -> None:
    print(message, flush=True)


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
    material = gather(list(args.source or []), raw_text=args.text or "", clean=not args.raw)
    if material.skipped:
        say(style(f"  skipped {len(material.skipped)} item(s):", DIM))
        for path, why in material.skipped[:5]:
            say(style(f"    {path} — {why}", DIM))
        if len(material.skipped) > 5:
            say(style(f"    …and {len(material.skipped) - 5} more", DIM))
    if not material.characters:
        raise TeacherError("Nothing readable was found. Point --from at a file or folder of text.")
    say(f"  material: {material.summary()}")
    return material


# ------------------------------------------------------------------ commands
def cmd_new(args) -> int:
    model = workspace.get(args.name, must_exist=False)
    if model.path.exists() and model.exists() and not args.replace:
        raise TeacherError(f"'{model.name}' already exists. Teach it more, or pass --replace.")

    say(f"{style('Reading material', BOLD)}")
    material = collect(args)

    say(f"\n{style('Building ' + model.name, BOLD)}")
    built = lessons.create(model, material, args.size, context=args.context)
    say(f"  size:       {built['size']} ({built['preset']})")
    say(f"  parameters: {fmt_count(built['parameters'])}")
    say(f"  vocabulary: {built['vocab_size']:,} tokens")
    say(f"  context:    {built['context']} tokens")
    say(style(f"  stored in {model.path}", DIM))

    if args.no_teach:
        say("\nMade but not taught — it will produce noise until you run: "
            f"teacher teach {model.name} --from <material>")
        return 0

    say(f"\n{style('First lesson', BOLD)}")
    return run_lesson(model, material, args)


def cmd_teach(args) -> int:
    model = workspace.get(args.name)
    say(f"{style('Reading material', BOLD)}")
    material = collect(args)
    say(f"\n{style('Teaching ' + model.name, BOLD)}")
    return run_lesson(model, material, args)


def run_lesson(model, material, args) -> int:
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
        on_log=on_log,
    )

    say("")
    say(f"  learned from:  {fmt_count(lesson['tokens'])} tokens over {lesson['epochs']:g} epoch(s)")
    say(f"  loss:          {lesson['final_loss']:.4f}" if lesson["final_loss"] is not None else "  loss: —")
    if lesson["held_out_loss"] is not None:
        say(f"  on held-out:   {lesson['held_out_loss']:.4f}  (perplexity {lesson['perplexity']:.1f})")
    say(f"  took:          {lesson['seconds']:.1f}s on {lesson['device']}")

    report_progress(model, lesson, args)
    return 0


def report_progress(model, lesson, args) -> None:
    """Say how far along the model is, and what to do next — not just "ready"."""
    label, note, share = lessons.verdict(lesson["held_out_loss"], lessons.model_vocab(model))
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

    respond(model, prompt, args)
    return 0


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


def cmd_list(args) -> int:
    models = workspace.every()
    if not models:
        say(f"No models yet in {workspace.root()}.")
        say("Make one with:  teacher new mymodel --from ./some-text")
        return 0

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
    return 0


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

    if not lessons_taught:
        say("\n  Never taught anything — it will produce noise.")
        return 0

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
    return 0


def cmd_pack(args) -> int:
    model = workspace.get(args.name)
    destination = Path(args.out).expanduser()
    copied = model.pack(destination)
    say(f"Copied {len(copied)} file(s) to {destination}:")
    for name in copied:
        say(f"  {name}  {fmt_bytes((destination / name).stat().st_size)}")
    say(style("\nOpen web_chat/index.html and drop that folder in to talk to it.", DIM))
    return 0


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
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teacher",
        description="Teach a small language model from your own text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sizes:\n" + lessons.describe_sizes(),
    )
    parser.add_argument("--version", action="version", version=f"teacher {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    def add_material(sub):
        sub.add_argument("--from", dest="source", action="append", metavar="PATH",
                         help="a file or folder to learn from (repeatable)")
        sub.add_argument("--text", help="text to learn from, given directly")
        sub.add_argument("--raw", action="store_true",
                         help="skip cleaning and use the text exactly as found")

    def add_training(sub):
        sub.add_argument("--epochs", type=float, default=3.0, help="passes over the material (default 3)")
        sub.add_argument("--batch", type=int, default=8, help="sequences per step (default 8)")
        sub.add_argument("--rate", type=float, default=3e-4, help="learning rate (default 3e-4)")

    new = subs.add_parser("new", help="build a new model and teach it its first lesson")
    new.add_argument("name")
    new.add_argument("--size", default="tiny", choices=list(lessons.SIZES),
                     help="how big to make it (default tiny)")
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

    show = subs.add_parser("show", help="what one model is and what it has been taught")
    show.add_argument("name")
    show.set_defaults(func=cmd_show)

    pack = subs.add_parser("pack", help="copy the files the Bench web page needs")
    pack.add_argument("name")
    pack.add_argument("-o", "--out", required=True, metavar="DIR")
    pack.set_defaults(func=cmd_pack)

    forget = subs.add_parser("forget", help="delete a model and everything it learned")
    forget.add_argument("name")
    forget.add_argument("--yes", action="store_true")
    forget.set_defaults(func=cmd_forget)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TeacherError as exc:
        say(f"\n{style('Stopped:', BOLD)} {exc}")
        return 1
    except KeyboardInterrupt:
        say("\nStopped. Nothing was overwritten.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
