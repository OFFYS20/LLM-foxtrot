"""Sandbox safety: model-generated code must never reach the operating system."""

from __future__ import annotations

import pytest

from ai_studio.core.errors import ValidationError, WebError
from ai_studio.tools.calculator import calculate
from ai_studio.tools.python_sandbox import run_python


# ------------------------------------------------------------------ escapes
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from subprocess import run",
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "exec('x = 1')",
        "eval('1 + 1')",
        "globals()",
        "getattr(int, 'mro')",
        "class Escape: pass",
        "lambda: 1",
        "try:\n    pass\nexcept Exception:\n    pass",
        "del x",
        "with open('f') as handle:\n    pass",
        "compile('1', '<s>', 'eval')",
        "x = 1\nx.__class__",
    ],
)
def test_escape_attempts_are_refused(code):
    with pytest.raises(ValidationError):
        run_python(code)


def test_oversized_snippets_are_refused():
    with pytest.raises(ValidationError):
        run_python("x = 1\n" * 2000)


def test_empty_input_is_refused():
    with pytest.raises(ValidationError):
        run_python("   ")


def test_syntax_errors_are_reported_not_raised_as_crashes():
    with pytest.raises(ValidationError) as excinfo:
        run_python("def (:")
    assert "Syntax error" in str(excinfo.value)


# ------------------------------------------------------------- allowed work
def test_arithmetic_and_printing_work():
    result = run_python("total = sum(range(10))\nprint(total)")
    assert result["stdout"].strip() == "45"
    assert result["error"] is None
    assert result["variables"]["total"] == "45"


def test_functions_and_loops_work():
    result = run_python(
        "def square(n):\n"
        "    return n * n\n"
        "print([square(i) for i in range(4)])\n"
    )
    assert result["stdout"].strip() == "[0, 1, 4, 9]"


def test_runtime_errors_are_captured_not_propagated():
    result = run_python("print(1 / 0)")
    assert result["error"] and "ZeroDivisionError" in result["error"]
    assert result["timed_out"] is False


def test_infinite_loops_time_out():
    result = run_python("while True:\n    pass")
    assert result["timed_out"] is True
    assert "Timed out" in result["error"]


# ------------------------------------------------------------- calculator
@pytest.mark.parametrize(
    "expression,expected",
    [("2 + 3 * 4", 14), ("(1 + 2) ** 3", 27), ("10 / 4", 2.5), ("-7 + 2", -5), ("17 % 5", 2)],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculate(expression)["value"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('id')", "open('/etc/passwd')", "[].__class__", "x = 1"],
)
def test_calculator_refuses_anything_that_is_not_arithmetic(expression):
    with pytest.raises(ValidationError):
        calculate(expression)


def test_calls_through_expressions_are_refused():
    # Only plain `name(...)` calls are allowed, so there is no way to reach a
    # method or a callable pulled out of a container.
    for code in ["[print][0]('x')", "{'p': print}['p']('x')", "(print if 1 else len)('x')"]:
        with pytest.raises(ValidationError):
            run_python(code)


def test_a_defined_function_cannot_smuggle_in_a_forbidden_call():
    with pytest.raises(ValidationError):
        run_python("def sneaky():\n    return eval('1')\nprint(sneaky())")


def test_a_timed_out_snippet_does_not_leave_a_thread_running():
    import threading

    before = threading.active_count()
    assert run_python("while True:\n    pass")["timed_out"] is True
    # Give the guard a moment to unwind the thread.
    deadline = __import__("time").monotonic() + 3
    while threading.active_count() > before and __import__("time").monotonic() < deadline:
        pass
    assert threading.active_count() == before, "a runaway snippet must be stopped, not abandoned"


# ------------------------------------------------------------- the registry
def test_every_built_in_tool_is_registered():
    from ai_studio.tools.registry import install_default_tools

    names = {tool.name for tool in install_default_tools().list()}
    assert {"calculator", "document_search", "python_sandbox",
            "web_search", "read_web_page"} <= names


def test_a_tool_that_needs_confirmation_does_not_run_without_it():
    """Declaring a tool dangerous and running it anyway would make the flag decoration."""
    from ai_studio.tools.registry import install_default_tools

    registry = install_default_tools()
    answer = registry.call("python_sandbox", {"code": "print(1)"})
    assert answer["ok"] is False
    assert answer["needs_confirmation"] is True


def test_a_confirmed_tool_runs():
    from ai_studio.tools.registry import install_default_tools

    registry = install_default_tools()
    answer = registry.call("python_sandbox", {"code": "print(6 * 7)"}, confirmed=True)
    assert answer["ok"] is True
    assert "42" in answer["result"]["stdout"]


def test_reaching_the_web_needs_confirmation():
    from ai_studio.tools.registry import install_default_tools

    registry = install_default_tools()
    for name in ("web_search", "read_web_page"):
        assert registry.get(name).requires_confirmation, f"{name} leaves this machine"


def test_a_failing_tool_is_data_not_a_crash(monkeypatch):
    from ai_studio.tools import web_search as module
    from ai_studio.tools.registry import install_default_tools

    def blocked(query, *, limit=5, engine="auto"):
        raise WebError("the site answered 429")

    monkeypatch.setattr(module.web, "search", blocked)
    answer = install_default_tools().call("web_search", {"query": "x"}, confirmed=True)
    assert answer["ok"] is False
    assert "429" in answer["error"]


# -------------------------------------------------------------- web_search
def test_web_search_hands_back_what_it_found(monkeypatch):
    from ai_studio.data.web import Result
    from ai_studio.tools import web_search as module

    monkeypatch.setattr(module.web, "search", lambda query, *, limit=5: [
        Result(title="One", url="https://a.test/1", snippet="first", engine="web"),
    ])
    answer = module.web_search("lighthouses", top_k=3)
    assert answer["count"] == 1
    assert answer["results"][0] == {"title": "One", "url": "https://a.test/1",
                                    "snippet": "first", "engine": "web"}


@pytest.mark.parametrize("query", ["", "   ", None])
def test_web_search_needs_something_to_search_for(query):
    from ai_studio.tools.web_search import web_search

    with pytest.raises(ValidationError):
        web_search(query)


def test_reading_a_page_says_when_it_was_cut_short(monkeypatch):
    from ai_studio.data.web import Page
    from ai_studio.tools import web_search as module

    monkeypatch.setattr(module.web, "fetch_page",
                        lambda url: Page(url=url, title="T", text="x" * 9_000))
    answer = module.read_web_page("https://a.test/", max_chars=1_000)
    assert answer["truncated"] is True
    assert len(answer["text"]) == 1_000
    assert answer["characters"] == 9_000, "the real length is reported, not the truncated one"


def test_reading_a_page_refuses_an_address_on_this_machine():
    from ai_studio.tools.web_search import read_web_page

    with pytest.raises(ValidationError):
        read_web_page("http://127.0.0.1:8000/secrets")
