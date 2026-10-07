"""The research agent loop.

Each step the model receives a fresh context built from the project's state (see
`researchos.context`) and either calls tools or replies in text. The harness executes the calls
it chose, records the turn, and persists state, then repeats until the model calls a terminal
tool, stalls, fails, or a budget limit is reached. The order of actions is decided entirely by
the model.

State is saved after every step, so a run that is interrupted (Ctrl-C, crash, budget) can be
resumed later from where it stopped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from researchos.context import Transcript, Turn, build_messages
from researchos.llm import LLMAuthorizationError, LLMClient, LLMError, LLMResponse, Message
from researchos.project import Project
from researchos.state import RunStatus
from researchos.tools import ToolRegistry
from researchos.usage import BudgetExceeded, UsageMeter, UsageSummary

logger = logging.getLogger(__name__)

NUDGE = "Continue the research by calling a tool. If the objective is met, call finish_research."


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
        run_id: str,
        context_turns: int = 6,
        max_idle_turns: int = 2,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._meter = meter
        self._project = project
        self._run_id = run_id
        self._context_turns = context_turns
        self._max_idle_turns = max_idle_turns
        self._transcript = Transcript(project.transcript_path)

    def run(self) -> RunResult:
        """Run until the research ends. Raises `ProjectLockedError` if another process is
        already running this project. On Ctrl-C the run is recorded as interrupted and
        `KeyboardInterrupt` propagates."""
        with self._project.lock():
            self._project.state.begin_run(self._run_id)
            self._project.save_state()
            try:
                status, reason = self._loop()
            except KeyboardInterrupt:
                self._end(RunStatus.INTERRUPTED, "interrupted by user")
                raise
            return self._end(status, reason)

    def _loop(self) -> tuple[RunStatus, str]:
        state = self._project.state
        idle_turns = 0

        while True:
            try:
                self._meter.start_step()
            except BudgetExceeded as exc:
                return RunStatus.BUDGET_EXHAUSTED, str(exc)
            state.step += 1
            step = state.step

            messages = build_messages(
                self._project,
                self._transcript,
                context_turns=self._context_turns,
                usage=self._meter.summary(),
                limits=self._meter.limits,
            )
            try:
                response = self._chat(messages)
            except LLMAuthorizationError as exc:
                return RunStatus.LLM_REFUSED, str(exc)
            except LLMError as exc:
                return RunStatus.LLM_FAILED, str(exc)

            reply = response.message
            turn: list[Message] = [reply]
            if reply.content:
                logger.info("step %d: model: %s", step, _clip(reply.content, 300))

            if not reply.tool_calls:
                idle_turns += 1
                if idle_turns > self._max_idle_turns:
                    self._record(step, turn)
                    return RunStatus.STALLED, f"no tool call in {idle_turns} consecutive turns"
                turn.append(Message(role="user", content=NUDGE))
                self._record(step, turn)
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
                turn.append(Message(role="tool", content=outcome.content, tool_call_id=call.id))
                finished = finished or outcome.terminal
            self._record(step, turn)
            if finished:
                return RunStatus.COMPLETED, "the agent finished the research"

    def _chat(self, messages: list[Message]) -> LLMResponse:
        started = time.monotonic()
        try:
            response = self._llm.chat(messages, self._tools.specs())
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

    def _record(self, step: int, messages: list[Message]) -> None:
        self._transcript.append(Turn(step=step, messages=tuple(messages)))
        self._project.state.runs[-1].usage = self._meter.summary()
        self._project.save_state()

    def _end(self, status: RunStatus, reason: str) -> RunResult:
        usage = self._meter.summary()
        self._project.state.end_run(status, reason, usage)
        self._project.save_state()
        if status is not RunStatus.COMPLETED:
            self._project.write_report(
                status=f"{status.value}: {reason}", summary=self._project.state.summary
            )
        self._meter.record_run_end(status=status.value, reason=reason)
        logger.info("run ended: %s (%s)", status.value, reason)
        return RunResult(
            status=status, reason=reason, report_path=self._project.report_path, usage=usage
        )


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
