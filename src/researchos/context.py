"""Builds what the model sees each step.

Instead of an ever-growing chat history, every step the model receives:

1. the system prompt,
2. a freshly rendered snapshot of the research state (request, user instructions, phase,
   agenda, sources, recent notes, budget), and
3. the last few complete turns from the transcript, for short-term continuity.

The prompt therefore stays roughly constant in size however long the research runs, and a
resumed run sees the same kind of context as an uninterrupted one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from researchos.llm import Message, ToolCall
from researchos.project import Project
from researchos.untrusted import untrusted
from researchos.usage import Limits, UsageSummary

MAX_NOTES_IN_CONTEXT = 30
MAX_SOURCES_IN_CONTEXT = 50
MAX_DIRECTIVES_IN_CONTEXT = 10


@dataclass(frozen=True)
class Turn:
    """One agent step: the model's reply followed by its tool results (or a nudge)."""

    step: int
    messages: tuple[Message, ...]


class Transcript:
    """Append-only log of turns, kept in memory and mirrored to a JSON Lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._turns: list[Turn] = []
        if path.exists():
            # A trailing partial line is a write cut short by a crash; that turn is lost.
            for line in path.read_text(encoding="utf-8").split("\n")[:-1]:
                if line.strip():
                    self._turns.append(_turn_from_json(json.loads(line)))

    def append(self, turn: Turn) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_turn_to_json(turn)) + "\n")
        self._turns.append(turn)

    def recent(self, count: int) -> list[Turn]:
        return self._turns[-count:] if count > 0 else []

    def __len__(self) -> int:
        return len(self._turns)


def build_messages(
    project: Project,
    transcript: Transcript,
    *,
    context_turns: int,
    usage: UsageSummary,
    limits: Limits,
) -> list[Message]:
    messages = [
        Message(role="system", content=system_prompt()),
        Message(role="user", content=render_snapshot(project, usage=usage, limits=limits)),
    ]
    for turn in transcript.recent(context_turns):
        messages.extend(turn.messages)
    return messages


def system_prompt(today: datetime | None = None) -> str:
    template = resources.files("researchos.prompts").joinpath("system.md").read_text("utf-8")
    return template.replace("{today}", f"{today or datetime.now(UTC):%Y-%m-%d}")


def render_snapshot(project: Project, *, usage: UsageSummary, limits: Limits) -> str:
    request = project.metadata.request
    state = project.state
    out = [f"Topic: {request.topic}"]
    if request.goal:
        out.append(f"Goal: {request.goal}")
    if request.knowledge_level:
        out.append(f"My current knowledge: {request.knowledge_level}")

    phase = f"{state.phase.value}" + (f" ({state.phase_reason})" if state.phase_reason else "")
    out += ["", "# Research state", f"Step {state.step}. Phase: {phase}."]
    if len(state.runs) > 1:
        out.append(
            f"This is run {len(state.runs)} of this project. Build on the existing state below; "
            "do not redo work that is already recorded."
        )

    directives = project.directives()[-MAX_DIRECTIVES_IN_CONTEXT:]
    if directives:
        out += ["", "## Instructions from the user (latest last; these override your plan)"]
        out += [f"- [{d.created_at:%Y-%m-%d %H:%M}] {d.text}" for d in directives]

    out += ["", "## Agenda"]
    if state.agenda:
        out += [
            f"- {item.id} [{item.status}] {item.text}" + (f" — {item.note}" if item.note else "")
            for item in state.agenda
        ]
    else:
        out.append("Empty. Use update_agenda to plan what needs to be learned.")

    if state.summary:
        out += ["", "## Summary from the previous run", state.summary]

    sources = project.sources()
    out += ["", f"## Sources fetched ({len(sources)})"]
    if sources:
        shown = sources[-MAX_SOURCES_IN_CONTEXT:]
        listing = "\n".join(f"{s.id}: {s.title} <{s.url}>" for s in shown)
        out.append(untrusted(listing, "source-titles"))
        if len(sources) > len(shown):
            out.append(f"({len(sources) - len(shown)} older sources not shown.)")
    else:
        out.append("None yet.")

    notes = project.notes()
    out += ["", f"## Notes ({len(notes)})"]
    if notes:
        if len(notes) > MAX_NOTES_IN_CONTEXT:
            out.append(
                f"Showing the latest {MAX_NOTES_IN_CONTEXT}; use read_notes to see older ones."
            )
        for note in notes[-MAX_NOTES_IN_CONTEXT:]:
            cites = f" [{', '.join(note.source_ids)}]" if note.source_ids else ""
            out.append(f"- {note.id}{cites}: {note.text}")
    else:
        out.append("None yet.")

    out += [
        "",
        "## Budget for this run",
        f"Step {usage.steps} of at most {limits.max_steps} · "
        f"cost ${usage.cost_usd:.4f}/${limits.max_cost_usd:.2f} · "
        f"time {usage.elapsed_seconds:.0f}s/{limits.max_wall_seconds:.0f}s",
    ]
    return "\n".join(out)


def _message_to_json(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
    }


def _message_from_json(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tuple(ToolCall(**c) for c in data.get("tool_calls") or ()),
        tool_call_id=data.get("tool_call_id"),
    )


def _turn_to_json(turn: Turn) -> dict[str, Any]:
    return {"step": turn.step, "messages": [_message_to_json(m) for m in turn.messages]}


def _turn_from_json(data: dict[str, Any]) -> Turn:
    return Turn(step=data["step"], messages=tuple(_message_from_json(m) for m in data["messages"]))
