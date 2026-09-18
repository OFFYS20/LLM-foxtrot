"""A safe arithmetic evaluator — AST only, no eval of arbitrary code."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ai_studio.core.errors import ValidationError

OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

FUNCTIONS: dict[str, Any] = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "floor", "ceil", "fabs")
}
FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

MAX_EXPONENT = 1000


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValidationError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValidationError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > MAX_EXPONENT:
            raise ValidationError(f"Exponent too large (max {MAX_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValidationError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_evaluate(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ValidationError("Only whitelisted math functions may be called")
        return FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValidationError(f"Unknown name: {node.id}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(item) for item in node.elts]
    raise ValidationError(f"Unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> dict[str, Any]:
    if not expression or not expression.strip():
        raise ValidationError("Enter an expression")
    if len(expression) > 500:
        raise ValidationError("Expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"Could not parse the expression: {exc.msg}") from exc
    value = _evaluate(tree)
    return {"expression": expression, "value": value}
