"""The research agent loop.

Each step the model sees the conversation so far and either calls tools or replies in text.
The harness executes the calls it chose, appends the observations and repeats until the model
calls a terminal tool, stalls, fails, or a budget limit is reached. The order of actions is
decided entirely by the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import resources
from pathlib import Path

from researchos.llm import (
    LLMAuthorizationError,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolSpec,
)
from researchos.project import Project, ResearchRequest
from researchos.tools import ToolRegistry
from researchos.usage import BudgetExceeded, UsageMeter, UsageSummary

logger = logging.getLogger(__name__)

NUDGE = "Continue the research by calling a tool. If the objective is met, call finish_research."


class RunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STALLED = "stalled"
    LLM_REFUSED = "llm_refused"
    LLM_FAILED = "llm_failed"


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    reason: str
    report_path: Path
    usage: UsageSummary


class ResearchAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        meter: UsageMeter,
        project: Project,
        max_idle_turns: int = 2,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._meter = meter
        self._project = project
        self._max_idle_turns = max_idle_turns

    def run(self) -> RunResult:
        messages = [
            Message(role="system", content=system_prompt()),
            Message(role="user", content=_describe_request(self._project.metadata.request)),
        ]
        specs = self._tools.specs()
        idle_turns = 0

        while True:
            try:
                step = self._meter.start_step()
            except BudgetExceeded as exc:
                return self._end(RunStatus.BUDGET_EXHAUSTED, str(exc))

            try:
                response = self._chat(messages, specs)
            except LLMAuthorizationError as exc:
                return self._end(RunStatus.LLM_REFUSED, str(exc))
            except LLMError as exc:
                return self._end(RunStatus.LLM_FAILED, str(exc))

            reply = response.message
            messages.append(reply)
            if reply.content:
                logger.info("step %d: model: %s", step, _clip(reply.content, 300))

            if not reply.tool_calls:
                idle_turns += 1
                if idle_turns > self._max_idle_turns:
                    return self._end(
                        RunStatus.STALLED, f"no tool call in {idle_turns} consecutive turns"
                    )
                messages.append(Message(role="user", content=NUDGE))
                continue
            idle_turns = 0

            finished = False
            for call in reply.tool_calls:
                logger.info("step %d: %s(%s)", step, call.name, _clip(call.arguments, 160))
                started = time.monotonic()
                outcome = self._tools.execute(call)
                self._meter.record_tool_call(
                    name=call.name, ok=outcome.ok, latency_seconds=time.monotonic() - started
                )
                if not outcome.ok:
                    logger.info("step %d: %s failed: %s", step, call.name, outcome.content)
                messages.append(Message(role="tool", content=outcome.content, tool_call_id=call.id))
                finished = finished or outcome.terminal
            if finished:
                return self._end(RunStatus.COMPLETED, "the agent finished the research")

    def _chat(self, messages: list[Message], specs: list[ToolSpec]) -> LLMResponse:
        started = time.monotonic()
        try:
            response = self._llm.chat(messages, specs)
        except LLMError as exc:
            self._meter.record_llm_failure(
                model=self._llm.model, error=str(exc), latency_seconds=time.monotonic() - started
            )
            raise
        self._meter.record_llm_call(
            model=response.model,
            usage=response.usage,
            latency_seconds=time.monotonic() - started,
        )
        return response

    def _end(self, status: RunStatus, reason: str) -> RunResult:
        if status is not RunStatus.COMPLETED:
            self._project.write_report(status=f"{status.value}: {reason}", summary=None)
        self._meter.record_run_end(status=status.value, reason=reason)
        logger.info("run ended: %s (%s)", status.value, reason)
        return RunResult(
            status=status,
            reason=reason,
            report_path=self._project.report_path,
            usage=self._meter.summary(),
        )


def system_prompt(today: datetime | None = None) -> str:
    template = resources.files("researchos.prompts").joinpath("system.md").read_text("utf-8")
    return template.replace("{today}", f"{today or datetime.now(UTC):%Y-%m-%d}")


def _describe_request(request: ResearchRequest) -> str:
    lines = [f"Topic: {request.topic}"]
    if request.goal:
        lines.append(f"Goal: {request.goal}")
    if request.knowledge_level:
        lines.append(f"My current knowledge: {request.knowledge_level}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
