"""Research tools available to the agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from researchos.project import Project, UnknownSourceError
from researchos.state import Phase
from researchos.tools.registry import Tool, ToolError
from researchos.untrusted import untrusted
from researchos.web import Fetcher, FetchError, SearchError, SearchProvider

FETCH_EXCERPT_CHARS = 4_000


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchWebArgs(_Args):
    query: str = Field(min_length=1, max_length=400, description="Search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class FetchSourceArgs(_Args):
    url: str = Field(min_length=1, max_length=2_000, description="http(s) URL to fetch.")


class ReadSourceArgs(_Args):
    source_id: str = Field(description="Id returned by fetch_source.")
    offset: int = Field(default=0, ge=0, description="Character offset to start reading at.")
    length: int = Field(default=4_000, ge=500, le=12_000)


class SaveNoteArgs(_Args):
    text: str = Field(min_length=1, max_length=4_000, description="The finding, in your words.")
    source_ids: list[str] = Field(
        default_factory=list, description="Ids of fetched sources that support this note."
    )


class ReadNotesArgs(_Args):
    offset: int = Field(default=0, ge=0, description="Index of the first note to return.")
    limit: int = Field(default=30, ge=1, le=100)


class AgendaUpdate(_Args):
    id: str = Field(description="Agenda item id, e.g. 'a3'.")
    status: Literal["open", "done", "dropped"]
    note: str | None = Field(default=None, max_length=500, description="Why, or what was found.")


class UpdateAgendaArgs(_Args):
    add: list[str] = Field(
        default_factory=list, max_length=20, description="New items: things still to learn or do."
    )
    update: list[AgendaUpdate] = Field(default_factory=list, max_length=50)


class SetPhaseArgs(_Args):
    phase: Phase
    reason: str = Field(min_length=1, max_length=300)


class FinishResearchArgs(_Args):
    summary: str = Field(
        min_length=1, max_length=8_000, description="Final synthesis of what was learned."
    )


def build_research_tools(project: Project, search: SearchProvider, fetcher: Fetcher) -> list[Tool]:
    def search_web(args: SearchWebArgs) -> dict[str, Any]:
        try:
            results = search.search(args.query, args.max_results)
        except SearchError as exc:
            raise ToolError(f"search failed: {exc}") from exc
        return {
            "results": [
                {"url": r.url, "content": untrusted(f"{r.title}\n{r.snippet}", r.url)}
                for r in results
            ]
        }

    def fetch_source(args: FetchSourceArgs) -> dict[str, Any]:
        try:
            page = fetcher.fetch(args.url)
        except FetchError as exc:
            raise ToolError(f"could not fetch {args.url}: {exc}") from exc
        source = project.add_source(page)
        return {
            "source_id": source.id,
            "url": source.url,
            "total_chars": source.chars,
            "excerpt": untrusted(
                f"Title: {source.title}\n\n{page.text[:FETCH_EXCERPT_CHARS]}", source.id
            ),
            "truncated": source.chars > FETCH_EXCERPT_CHARS,
        }

    def read_source(args: ReadSourceArgs) -> dict[str, Any]:
        try:
            text = project.read_source_text(args.source_id)
        except UnknownSourceError:
            raise ToolError(f"unknown source_id '{args.source_id}'") from None
        chunk = text[args.offset : args.offset + args.length]
        end = args.offset + len(chunk)
        return {
            "source_id": args.source_id,
            "offset": args.offset,
            "text": untrusted(chunk, args.source_id),
            "next_offset": end if end < len(text) else None,
        }

    def save_note(args: SaveNoteArgs) -> dict[str, Any]:
        try:
            note = project.add_note(args.text, args.source_ids)
        except UnknownSourceError as exc:
            raise ToolError(
                f"unknown source_id '{exc.args[0]}'; cite only ids returned by fetch_source"
            ) from None
        return {"note_id": note.id}

    def read_notes(args: ReadNotesArgs) -> dict[str, Any]:
        notes = project.notes()
        page = notes[args.offset : args.offset + args.limit]
        return {
            "total": len(notes),
            "notes": [{"id": n.id, "text": n.text, "source_ids": n.source_ids} for n in page],
        }

    def update_agenda(args: UpdateAgendaArgs) -> dict[str, Any]:
        known = {item.id for item in project.state.agenda}
        unknown = [u.id for u in args.update if u.id not in known]
        if unknown:
            raise ToolError(f"unknown agenda item(s): {', '.join(unknown)}")
        for update in args.update:
            project.state.set_agenda_status(update.id, update.status, update.note)
        added = project.state.add_agenda_items([text for text in args.add if text.strip()])
        return {"added": [{"id": item.id, "text": item.text} for item in added]}

    def set_phase(args: SetPhaseArgs) -> dict[str, Any]:
        project.state.phase, project.state.phase_reason = args.phase, args.reason
        return {"phase": args.phase.value}

    def finish_research(args: FinishResearchArgs) -> dict[str, Any]:
        project.state.summary = args.summary
        path = project.write_report(status="completed", summary=args.summary)
        return {"report": str(path.relative_to(project.root))}

    return [
        Tool(
            name="search_web",
            description="Search the web. Returns titles, URLs and snippets (untrusted).",
            args_model=SearchWebArgs,
            handler=search_web,
        ),
        Tool(
            name="fetch_source",
            description=(
                "Fetch a web page, store it as a source and return its id with an excerpt. "
                "Only fetched sources can be cited."
            ),
            args_model=FetchSourceArgs,
            handler=fetch_source,
        ),
        Tool(
            name="read_source",
            description="Read more of a previously fetched source, starting at a char offset.",
            args_model=ReadSourceArgs,
            handler=read_source,
        ),
        Tool(
            name="save_note",
            description="Record a research finding, citing the fetched sources that support it.",
            args_model=SaveNoteArgs,
            handler=save_note,
        ),
        Tool(
            name="read_notes",
            description="Page through all saved notes, oldest first.",
            args_model=ReadNotesArgs,
            handler=read_notes,
        ),
        Tool(
            name="update_agenda",
            description=(
                "Plan the research: add items still to learn or do, and mark items done or "
                "dropped as you go."
            ),
            args_model=UpdateAgendaArgs,
            handler=update_agenda,
        ),
        Tool(
            name="set_phase",
            description="Record which stage the research is in. A label; it restricts nothing.",
            args_model=SetPhaseArgs,
            handler=set_phase,
        ),
        Tool(
            name="finish_research",
            description="End the research when the objective is met. Writes the final report.",
            args_model=FinishResearchArgs,
            handler=finish_research,
            terminal=True,
        ),
    ]
