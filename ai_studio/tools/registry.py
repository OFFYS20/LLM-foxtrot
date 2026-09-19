"""Tool registry.

Tools are explicit, allow-listed Python callables with a JSON-serialisable
schema. Nothing here gives a model shell access or unrestricted filesystem
access; each tool validates its own input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_studio.core import logging as log
from ai_studio.core.errors import NotFoundError, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_confirmation: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters},
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        log.debug(f"Registered tool '{tool.name}'", source="tools", persist=False)
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"Unknown tool {name!r}")
        return tool

    def list(self, *, enabled_only: bool = False) -> list[Tool]:
        tools = list(self._tools.values())
        return [tool for tool in tools if tool.enabled] if enabled_only else tools

    def schemas(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.list(enabled_only=enabled_only)]

    def call(
        self,
        name: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        tool = self.get(name)
        if not tool.enabled:
            raise ValidationError(f"Tool {name!r} is disabled.")
        if tool.requires_confirmation and not confirmed:
            # Declaring a tool dangerous and then running it anyway would make
            # the flag decoration. The caller has to say a person agreed.
            return {
                "ok": False,
                "tool": name,
                "needs_confirmation": True,
                "error": f"{name!r} runs only once someone confirms it.",
            }
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Tool arguments are not valid JSON: {exc}") from exc
        arguments = arguments or {}
        try:
            result = tool.handler(**arguments)
            return {"ok": True, "tool": name, "result": result}
        except ValidationError as exc:
            return {"ok": False, "tool": name, "error": exc.message}
        except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
            log.warning(f"Tool {name} failed: {exc}", source="tools")
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}


registry = ToolRegistry()


def install_default_tools() -> ToolRegistry:
    """Register the built-in tools (idempotent)."""
    from ai_studio.tools.calculator import calculate
    from ai_studio.tools.document_search import search_documents
    from ai_studio.tools.python_sandbox import run_python
    from ai_studio.tools.web_search import read_web_page, web_search

    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a arithmetic expression, e.g. '2 * (3 + 4) ** 2'.",
            handler=calculate,
            parameters={"expression": {"type": "string", "description": "The expression"}},
        )
    )
    registry.register(
        Tool(
            name="document_search",
            description="Search the Data Library and return matching excerpts.",
            handler=search_documents,
            parameters={
                "query": {"type": "string", "description": "What to look for"},
                "top_k": {"type": "integer", "description": "How many excerpts (default 4)"},
            },
        )
    )
    registry.register(
        Tool(
            name="python_sandbox",
            description=(
                "Run a small, restricted Python snippet and return its stdout. "
                "No imports, filesystem, or network access."
            ),
            handler=run_python,
            parameters={"code": {"type": "string", "description": "Python source"}},
            requires_confirmation=True,
        )
    )
    registry.register(
        Tool(
            name="web_search",
            description=(
                "Search the web and return titles, addresses and snippets. "
                "Leaves this machine: the query goes to a search engine."
            ),
            handler=web_search,
            parameters={
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "description": "How many results (default 5)"},
            },
            requires_confirmation=True,
        )
    )
    registry.register(
        Tool(
            name="read_web_page",
            description=(
                "Fetch one web page and return its readable text. "
                "http(s) only; addresses on this machine or its network are refused."
            ),
            handler=read_web_page,
            parameters={
                "url": {"type": "string", "description": "The page address"},
                "max_chars": {"type": "integer", "description": "How much text (default 4000)"},
            },
            requires_confirmation=True,
        )
    )
    return registry
