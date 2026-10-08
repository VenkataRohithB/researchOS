"""Shared test helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from researchos.agent import ResearchAgent
from researchos.focused import FocusedModel
from researchos.llm import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage
from researchos.project import EvidenceInput, Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.user import NoUser, UserChannel
from researchos.web import FetchedPage, MockFetcher, MockSearch

SEED_URL = "https://seed.example/background"
SEED_QUOTE = "The seeded source states this background finding plainly"


def call(name: str, **arguments: Any) -> Message:
    return Message(
        role="assistant",
        tool_calls=(ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments)),),
    )


class ScriptedLLM:
    """Replays a fixed list of replies (or raises listed errors) and records what it was sent."""

    model = "scripted"

    def __init__(self, replies: Sequence[Message | BaseException]) -> None:
        self._replies = list(replies)
        self.requests: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.requests.append(list(messages))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return LLMResponse(message=reply, usage=Usage(1_000, 100), model=self.model)

    def close(self) -> None:
        pass


def make_agent(
    tmp_path: Path, llm: LLMClient, *, grounded: bool = True, **options: Any
) -> tuple[ResearchAgent, Project]:
    """`grounded` seeds one fetched source and a note citing it, so scripts may call
    finish_research directly."""
    project = Project.create(tmp_path, ResearchRequest(topic="test topic"))
    if grounded:
        seed = project.add_source(
            FetchedPage(url=SEED_URL, title="Seed source", text=f"Background. {SEED_QUOTE}.")
        )
        project.add_claim("seed finding", "fact", [EvidenceInput(seed.id, SEED_QUOTE, "supports")])
    return agent_for(project, llm, **options), project


def agent_for(
    project: Project,
    llm: LLMClient,
    *,
    max_steps: int = 20,
    max_cost_usd: float = 10.0,
    context_turns: int = 6,
    run_id: str = "test-run",
    reader: LLMClient | None = None,
    user: UserChannel | None = None,
) -> ResearchAgent:
    """`reader` answers the focused calls tools make (study_source); defaults to `llm`."""
    meter = UsageMeter(
        limits=Limits(max_steps=max_steps, max_cost_usd=max_cost_usd, max_wall_seconds=60),
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
        events_path=project.events_path,
        run_id=run_id,
    )
    tools = ToolRegistry(
        build_research_tools(
            project,
            MockSearch(),
            MockFetcher(),
            focused=FocusedModel(reader or llm, meter),
            user=user or NoUser(),
        )
    )
    return ResearchAgent(
        llm=llm,
        tools=tools,
        meter=meter,
        project=project,
        run_id=run_id,
        context_turns=context_turns,
    )


def tool_results(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [json.loads(m.content or "") for m in messages if m.role == "tool"]
