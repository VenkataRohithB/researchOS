"""Shared test helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from researchos.agent import ResearchAgent
from researchos.llm import LLMClient, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage
from researchos.project import Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.web import MockFetcher, MockSearch


def call(name: str, **arguments: Any) -> Message:
    return Message(
        role="assistant",
        tool_calls=(ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments)),),
    )


class ScriptedLLM:
    """Replays a fixed list of replies (or raises listed errors) and records what it was sent."""

    model = "scripted"

    def __init__(self, replies: Sequence[Message | LLMError]) -> None:
        self._replies = list(replies)
        self.requests: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.requests.append(list(messages))
        reply = self._replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        return LLMResponse(message=reply, usage=Usage(1_000, 100), model=self.model)

    def close(self) -> None:
        pass


def make_agent(
    tmp_path: Path, llm: LLMClient, *, max_steps: int = 20, max_cost_usd: float = 10.0
) -> tuple[ResearchAgent, Project]:
    project = Project.create(tmp_path, ResearchRequest(topic="test topic"))
    meter = UsageMeter(
        limits=Limits(max_steps=max_steps, max_cost_usd=max_cost_usd, max_wall_seconds=60),
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
        events_path=project.events_path,
        run_id="test-run",
    )
    tools = ToolRegistry(build_research_tools(project, MockSearch(), MockFetcher()))
    return ResearchAgent(llm=llm, tools=tools, meter=meter, project=project), project


def tool_results(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [json.loads(m.content or "") for m in messages if m.role == "tool"]
