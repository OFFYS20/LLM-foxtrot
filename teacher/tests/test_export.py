"""Converting a model to GGUF, and refusing to pretend when it cannot be done."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

TMP_HOME = Path(tempfile.mkdtemp(prefix="teacher-export-tests-"))
os.environ["TEACHER_HOME"] = str(TMP_HOME)

from teacher import export, workspace  # noqa: E402
from teacher.workspace import TeacherError  # noqa: E402


def make(name: str, model_type: str) -> workspace.Model:
    model = workspace.Model(name=name, path=TMP_HOME / name)
    model.path.mkdir(parents=True, exist_ok=True)
    (model.path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (model.path / "model.safetensors").write_bytes(b"weights")
    (model.path / "tokenizer.json").write_text("{}")
    return model


# --------------------------------------------------- finding the converter
def test_a_missing_converter_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_CONVERT", "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "LIKELY", ())

    with pytest.raises(TeacherError) as excinfo:
        export.find_converter()
    message = str(excinfo.value)
    assert "git clone" in message, "say how to get it"
    assert "nothing was written" in message.lower(), "and that nothing was produced"


def test_a_named_converter_is_used(tmp_path):
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter")
    assert export.find_converter(str(script)) == script.resolve()


def test_a_directory_is_looked_inside(tmp_path):
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter")
    assert export.find_converter(str(tmp_path)) == script.resolve()


def test_the_environment_variable_is_honoured(monkeypatch, tmp_path):
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter")
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script))
    monkeypatch.setattr(export, "LIKELY", ())
    assert export.find_converter() == script.resolve()


# ------------------------------------------------------- what it refuses
def test_a_from_scratch_model_is_refused_with_somewhere_else_to_go():
    """llama.cpp has never heard of this architecture, and a file it cannot
    open is worse than no file."""
    model = make("homemade", "ai_studio_transformer")
    with pytest.raises(TeacherError) as excinfo:
        export.to_gguf(model, TMP_HOME / "out.gguf")

    message = str(excinfo.value)
    assert "does not know this architecture" in message
    assert "teacher pack" in message, "point at the thing that does run it"
    assert not (TMP_HOME / "out.gguf").exists()


def test_an_unknown_precision_is_refused_before_anything_runs():
    model = make("adopted", "llama")
    with pytest.raises(TeacherError, match="Unknown precision"):
        export.to_gguf(model, TMP_HOME / "out.gguf", precision="q2_k")


def test_an_incomplete_model_is_refused():
    model = workspace.Model(name="half", path=TMP_HOME / "half")
    model.path.mkdir(parents=True, exist_ok=True)
    (model.path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(TeacherError, match="missing"):
        export.to_gguf(model, TMP_HOME / "out.gguf")


def test_a_converter_that_fails_is_reported_and_nothing_is_claimed(monkeypatch, tmp_path):
    """The failure mode this guards against: a .gguf that llama.cpp cannot open,
    found out later and somewhere else."""
    model = make("adopted2", "llama")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("import sys; sys.stderr.write('no tokenizer here\\n'); sys.exit(1)")

    with pytest.raises(TeacherError) as excinfo:
        export.to_gguf(model, tmp_path / "out.gguf", converter=str(script))
    assert "no tokenizer here" in str(excinfo.value), "llama.cpp's own reason, verbatim"
    assert "Nothing was written" in str(excinfo.value)
    assert not (tmp_path / "out.gguf").exists()


def test_a_converter_that_claims_success_but_writes_nothing_is_caught(monkeypatch, tmp_path):
    model = make("adopted3", "llama")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("import sys; sys.exit(0)")

    with pytest.raises(TeacherError, match="is not there"):
        export.to_gguf(model, tmp_path / "missing.gguf", converter=str(script))


def test_a_real_conversion_reports_what_it_wrote(tmp_path):
    model = make("adopted4", "llama")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text(
        "import sys\n"
        "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        "open(out, 'wb').write(b'GGUF' + b'\\x00' * 100)\n"
    )
    written = export.to_gguf(model, tmp_path / "ok.gguf", converter=str(script))

    assert written["bytes"] == 104
    assert written["precision"] == "f16"
    assert Path(written["path"]).read_bytes()[:4] == b"GGUF"


def test_every_precision_is_described():
    for name, description in export.TYPES.items():
        assert description, f"{name} has no explanation"


def test_the_advice_names_the_tools_people_actually_use():
    lines = " ".join(export.advice(
        {"path": "/tmp/x.gguf", "model": "bot", "bytes": 1, "precision": "f16"}))
    assert "llama-cli" in lines and "ollama create" in lines and "LM Studio" in lines
