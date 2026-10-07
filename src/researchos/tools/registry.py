"""Typed tool registry.

Every tool declares a pydantic model for its arguments. The registry is the only path from a
model's tool call to code: arguments are parsed and validated before the handler runs, and any
failure is returned to the model as an error observation rather than raised.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from researchos.llm import ToolCall, ToolSpec


class ToolError(Exception):
    """An expected tool failure. The message is shown to the model, so it must not leak
    secrets or internal details."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]
    """Receives an instance of `args_model`; returns a JSON-serialisable observation."""
    terminal: bool = False
    """A successful call ends the agent run."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    content: str
    """JSON observation returned to the model."""
    terminal: bool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"unknown tool '{call.name}'; available: {', '.join(self._tools)}")
        try:
            args = tool.args_model.model_validate_json(call.arguments or "{}")
        except ValidationError as exc:
            return _error(f"invalid arguments for {call.name}: {_describe(exc)}")
        try:
            result = tool.handler(args)
        except ToolError as exc:
            return _error(str(exc))
        return ToolOutcome(ok=True, content=json.dumps(result), terminal=tool.terminal)


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(ok=False, content=json.dumps({"error": message}), terminal=False)


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '(arguments)'}: {err['msg']}"
        for err in exc.errors(include_url=False)
    )
