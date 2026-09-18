"""A deliberately restricted Python evaluator.

This is *not* a general sandbox. It executes a tiny allow-listed subset with no
imports, no attribute access, no filesystem and no network, under a time limit.
Model-generated code never reaches the operating system.
"""

from __future__ import annotations

import ast
import io
import contextlib
import sys
import threading
import time
from typing import Any

from ai_studio.core.errors import ValidationError

MAX_SOURCE_CHARS = 2000
TIMEOUT_SECONDS = 2.0

FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Attribute,      # blocks ().__class__ escapes
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}


def _check(source: str) -> ast.Module:
    if len(source) > MAX_SOURCE_CHARS:
        raise ValidationError(f"Snippet is too long (max {MAX_SOURCE_CHARS} characters)")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValidationError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc

    # Functions the snippet defines itself may be called; nothing else beyond
    # the allow-list can be. Their bodies go through the same checks below.
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    callable_names = set(SAFE_BUILTINS) | defined

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValidationError(
                f"{type(node).__name__} is not allowed in the sandbox "
                "(no imports, attribute access, classes, or exception handling)."
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValidationError("Dunder names are not allowed in the sandbox")
        if isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
            raise ValidationError("Dunder names are not allowed in the sandbox")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in callable_names:
                raise ValidationError(f"Function {node.func.id!r} is not available in the sandbox")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValidationError("Only plain function calls are allowed in the sandbox")
    return tree


class _SandboxTimeout(Exception):
    """Raised inside the sandbox thread once the deadline passes."""


def run_python(code: str) -> dict[str, Any]:
    """Execute a restricted snippet and capture stdout."""
    if not code or not code.strip():
        raise ValidationError("Nothing to run")
    tree = _check(code)
    compiled = compile(tree, "<sandbox>", "exec")

    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    buffer = io.StringIO()
    error: list[str] = []
    timed_out: list[bool] = []
    deadline = time.monotonic() + TIMEOUT_SECONDS

    def _guard(frame: Any, event: str, arg: Any) -> Any:
        """Stop a runaway snippet instead of leaving a thread spinning forever."""
        if time.monotonic() > deadline:
            raise _SandboxTimeout
        return _guard

    def _run() -> None:
        sys.settrace(_guard)
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compiled, namespace)  # noqa: S102 - AST-restricted, no imports/attributes
        except _SandboxTimeout:
            timed_out.append(True)
        except Exception as exc:  # noqa: BLE001 - report, never propagate
            error.append(f"{type(exc).__name__}: {exc}")
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # A little slack so the in-thread guard fires first and the thread really ends.
    thread.join(timeout=TIMEOUT_SECONDS + 1.0)
    if timed_out or thread.is_alive():
        return {
            "stdout": buffer.getvalue(),
            "error": f"Timed out after {TIMEOUT_SECONDS}s",
            "variables": {},
            "timed_out": True,
        }

    return {
        "stdout": buffer.getvalue(),
        "error": error[0] if error else None,
        "variables": {
            key: repr(value)[:200]
            for key, value in namespace.items()
            if not key.startswith("__") and not callable(value)
        },
        "timed_out": False,
    }
