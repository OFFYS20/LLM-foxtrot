"""Taking a model out of Teacher and into the tools people already use.

llama.cpp, Ollama and LM Studio all read GGUF, and none of them read a folder
of safetensors. Converting is llama.cpp's own job — its ``convert_hf_to_gguf.py``
knows every architecture it supports, and reimplementing that here would mean
maintaining a second, worse copy of it.

So this finds the converter and runs it. **If it is not there, this says so and
stops.** It does not write a file named ``.gguf`` that llama.cpp cannot open,
and it does not report a quantisation that never happened — a converted model
that fails silently is worse than no converted model, because you find out
later and somewhere else.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from teacher.workspace import Model, TeacherError

#: Where llama.cpp's converter usually is, once someone has cloned it.
LIKELY = (
    "convert_hf_to_gguf.py",
    "llama.cpp/convert_hf_to_gguf.py",
    "../llama.cpp/convert_hf_to_gguf.py",
    "~/llama.cpp/convert_hf_to_gguf.py",
    "~/src/llama.cpp/convert_hf_to_gguf.py",
)

#: What llama.cpp's converter will write directly. Anything smaller than these
#: is a second step through its `llama-quantize`, which is not run here.
TYPES = {
    "f32": "every weight as it is, largest file",
    "f16": "half precision — the usual choice, and lossless enough",
    "bf16": "half precision with more range, for weights trained in bf16",
    "q8_0": "eight-bit, about half the size of f16",
}


def find_converter(hint: str | None = None) -> Path:
    """Locate llama.cpp's converter, or say exactly how to get it."""
    candidates = [hint] if hint else []
    candidates += [os.environ.get("LLAMA_CPP_CONVERT", "")]
    candidates += list(LIKELY)

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            path = path / "convert_hf_to_gguf.py"
        if path.is_file():
            return path.resolve()

    raise TeacherError(
        "llama.cpp's converter was not found, so nothing was written.\n"
        "  GGUF is llama.cpp's format and its converter is the thing that knows how to\n"
        "  write it. Writing a file here without it would produce something that only\n"
        "  looks like a model.\n\n"
        "  Get it with:\n"
        "    git clone https://github.com/ggerganov/llama.cpp\n"
        "    pip install -r llama.cpp/requirements.txt\n\n"
        "  Then point at it with --converter /path/to/convert_hf_to_gguf.py, or set\n"
        "  LLAMA_CPP_CONVERT. It is looked for beside this repository automatically."
    )


def to_gguf(
    model: Model,
    destination: Path,
    *,
    precision: str = "f16",
    converter: str | None = None,
    on_log=None,
) -> dict:
    """Convert a model to GGUF. Returns what was written, or raises saying why not."""
    if precision not in TYPES:
        raise TeacherError(
            f"Unknown precision {precision!r}. Choose one of: {', '.join(TYPES)}."
        )
    if not model.exists():
        raise TeacherError(f"{model.name} is missing {', '.join(model.missing())}.")
    if model.kind() == "studio":
        raise TeacherError(
            f"{model.name} was built here, and llama.cpp does not know this architecture.\n"
            f"  GGUF export works for models started from a pretrained base "
            f"(teacher new NAME --base small ...), whose architecture llama.cpp "
            f"already supports.\n"
            f"  A from-scratch model runs in Teacher, and in the Bench web page: "
            f"teacher pack {model.name} -o ./for-bench"
        )

    script = find_converter(converter)
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The converter writes beside the destination and the file is moved into
    # place only once it has finished. A converter that fails half-way leaves
    # nothing — and cannot spoil a good file from an earlier export.
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)

    command = [
        sys.executable, str(script), str(model.path),
        "--outfile", str(partial),
        "--outtype", precision,
    ]
    if on_log:
        on_log(f"[EXPORT] {' '.join(command[:2])} … --outtype {precision}")

    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        partial.unlink(missing_ok=True)
        raise TeacherError(f"Could not run the converter: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        partial.unlink(missing_ok=True)
        raise TeacherError("The converter ran for an hour without finishing.") from exc

    if finished.returncode != 0:
        partial.unlink(missing_ok=True)
        tail = (finished.stderr or finished.stdout or "").strip().splitlines()[-6:]
        raise TeacherError(
            "llama.cpp's converter refused this model:\n  "
            + "\n  ".join(tail or ["it gave no reason"])
            + "\n\n  Nothing was written."
        )
    if not partial.exists():
        raise TeacherError(
            f"The converter reported success but {destination} is not there.")
    partial.replace(destination)

    return {
        "model": model.name,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "precision": precision,
        "converter": str(script),
    }


def advice(written: dict) -> list[str]:
    """What to do with the file now it exists."""
    path = written["path"]
    return [
        f"llama.cpp:  llama-cli -m {path} -p \"your prompt\"",
        f"Ollama:     printf 'FROM {path}\\n' > Modelfile && "
        f"ollama create {written['model']} -f Modelfile",
        "LM Studio:  put it in your models folder and pick it from the list",
    ]
