"""Explicit research state, persisted after every agent step so research can be resumed.

The state holds the agent's *task* state (phase, agenda, runs, final summary). Research
knowledge (sources, notes) lives in the project's own files, and the step transcript in
`transcript.jsonl`; the state only refers to them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from researchos.usage import UsageSummary


class Phase(StrEnum):
    """The agent's self-reported stage. A label for humans and for resuming; it does not
    constrain which tools the agent may call."""

    RESEARCH_STARTED = "research_started"
    TOPIC_DECOMPOSED = "topic_decomposed"
    RESEARCHING = "researching"
    CROSS_CHECKING = "cross_checking"
    KNOWLEDGE_GRAPH_BUILDING = "knowledge_graph_building"
    CONTENT_GENERATION = "content_generation"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"
    PUBLISHED = "published"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STALLED = "stalled"
    LLM_REFUSED = "llm_refused"
    LLM_FAILED = "llm_failed"


AgendaStatus = Literal["open", "done", "dropped"]


class AgendaItem(BaseModel):
    id: str
    text: str
    status: AgendaStatus = "open"
    note: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    """Claims that cover this item; required for it to count as done."""


class RunRecord(BaseModel):
    run_id: str
    started_at: datetime
    ended_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    reason: str | None = None
    usage: UsageSummary | None = None


class ResearchState(BaseModel):
    phase: Phase = Phase.RESEARCH_STARTED
    phase_reason: str | None = None
    step: int = 0
    """Total agent steps across all runs."""
    agenda: list[AgendaItem] = Field(default_factory=list)
    summary: str | None = None
    """Synthesis from the most recent `finish_research`."""
    reviewed_claims: list[str] = Field(default_factory=list)
    """Weakly supported claims the agent has already been asked to cross-check."""
    runs: list[RunRecord] = Field(default_factory=list)

    @property
    def status(self) -> RunStatus | None:
        return self.runs[-1].status if self.runs else None

    def begin_run(self, run_id: str) -> RunRecord:
        """Start a run. A previous run still marked running was killed without cleanup."""
        if self.runs and self.runs[-1].status is RunStatus.RUNNING:
            self.runs[-1].status = RunStatus.INTERRUPTED
            self.runs[-1].reason = "process ended without finishing the run"
        run = RunRecord(run_id=run_id, started_at=datetime.now(UTC))
        self.runs.append(run)
        return run

    def end_run(self, status: RunStatus, reason: str, usage: UsageSummary) -> None:
        run = self.runs[-1]
        run.status, run.reason, run.usage = status, reason, usage
        run.ended_at = datetime.now(UTC)

    def add_agenda_items(self, texts: list[str]) -> list[AgendaItem]:
        start = len(self.agenda) + 1
        items = [AgendaItem(id=f"a{start + i}", text=t) for i, t in enumerate(texts)]
        self.agenda.extend(items)
        return items

    def set_agenda_status(
        self,
        item_id: str,
        status: AgendaStatus,
        note: str | None,
        claim_ids: list[str] | None = None,
    ) -> None:
        for item in self.agenda:
            if item.id == item_id:
                item.status, item.note = status, note
                item.claim_ids = list(claim_ids or []) if status == "done" else []
                return
        raise KeyError(item_id)
