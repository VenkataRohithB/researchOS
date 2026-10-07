"""A research project: one directory holding everything the agent gathers and produces.

Layout::

    <workspace>/<project-id>/
        metadata/project.json   request that started the project
        metadata/state.json     task state: phase, agenda, runs (see `researchos.state`)
        metadata/transcript.jsonl  every agent step: model reply and tool results
        metadata/directives.jsonl  instructions from the user, added at any time
        metadata/events.jsonl   every LLM and tool call, with usage
        sources/sources.json    registry of fetched sources
        sources/snapshots/      extracted text of each source, as fetched
        notes/notes.jsonl       research notes, each citing sources by id
        report.md               human-readable output
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from researchos.state import ResearchState
from researchos.web import FetchedPage


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1)
    goal: str | None = None
    knowledge_level: str | None = None


class ProjectMetadata(BaseModel):
    id: str
    created_at: datetime
    request: ResearchRequest


class Source(BaseModel):
    id: str
    url: str
    title: str
    fetched_at: datetime
    content_sha256: str
    chars: int


class Note(BaseModel):
    id: str
    text: str
    source_ids: list[str]
    created_at: datetime


class Directive(BaseModel):
    text: str
    created_at: datetime


class UnknownSourceError(KeyError):
    pass


class ProjectNotFoundError(LookupError):
    pass


class ProjectLockedError(RuntimeError):
    pass


class Project:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = ProjectMetadata.model_validate_json(
            (root / "metadata" / "project.json").read_text(encoding="utf-8")
        )
        self._sources: dict[str, Source] = {
            s["id"]: Source.model_validate(s)
            for s in json.loads(self._sources_path.read_text(encoding="utf-8"))
        }
        self.state = ResearchState.model_validate_json(self._state_path.read_text(encoding="utf-8"))

    @classmethod
    def open(cls, workspace_dir: Path, ref: str) -> Project:
        """Open a project by id (a directory in the workspace) or by path."""
        for candidate in (workspace_dir / ref, Path(ref)):
            if (candidate / "metadata" / "project.json").is_file():
                return cls(candidate)
        raise ProjectNotFoundError(f"no research project '{ref}' in {workspace_dir}")

    @staticmethod
    def list_ids(workspace_dir: Path) -> list[str]:
        if not workspace_dir.is_dir():
            return []
        return sorted(
            p.name for p in workspace_dir.iterdir() if (p / "metadata" / "project.json").is_file()
        )

    @classmethod
    def create(cls, workspace_dir: Path, request: ResearchRequest) -> Project:
        created_at = datetime.now(UTC)
        base_id = f"{_slugify(request.topic)}-{created_at:%Y%m%d-%H%M%S}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        root, project_id = _claim_directory(workspace_dir, base_id)
        for sub in ("metadata", "sources/snapshots", "notes"):
            (root / sub).mkdir(parents=True)
        metadata = ProjectMetadata(id=project_id, created_at=created_at, request=request)
        _atomic_write(root / "metadata" / "project.json", metadata.model_dump_json(indent=2))
        _atomic_write(root / "metadata" / "state.json", ResearchState().model_dump_json(indent=2))
        _atomic_write(root / "sources" / "sources.json", "[]")
        (root / "notes" / "notes.jsonl").touch()
        return cls(root)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold an exclusive lock so only one process runs the agent on this project.
        The OS releases it if the process dies, so a crash never leaves a stale lock."""
        with (self.root / "metadata" / ".lock").open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ProjectLockedError(
                    f"project {self.metadata.id} is already being researched by another process"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def save_state(self) -> None:
        _atomic_write(self._state_path, self.state.model_dump_json(indent=2))

    @property
    def transcript_path(self) -> Path:
        return self.root / "metadata" / "transcript.jsonl"

    @property
    def _state_path(self) -> Path:
        return self.root / "metadata" / "state.json"

    @property
    def _directives_path(self) -> Path:
        return self.root / "metadata" / "directives.jsonl"

    @property
    def events_path(self) -> Path:
        return self.root / "metadata" / "events.jsonl"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    @property
    def _sources_path(self) -> Path:
        return self.root / "sources" / "sources.json"

    @property
    def _notes_path(self) -> Path:
        return self.root / "notes" / "notes.jsonl"

    # Sources -------------------------------------------------------------------------

    def add_source(self, page: FetchedPage) -> Source:
        source_id = "src-" + hashlib.sha256(page.url.encode()).hexdigest()[:10]
        source = Source(
            id=source_id,
            url=page.url,
            title=page.title,
            fetched_at=datetime.now(UTC),
            content_sha256=hashlib.sha256(page.text.encode()).hexdigest(),
            chars=len(page.text),
        )
        _atomic_write(self._snapshot_path(source_id), page.text)
        self._sources[source_id] = source
        _atomic_write(
            self._sources_path,
            json.dumps([s.model_dump(mode="json") for s in self._sources.values()], indent=2),
        )
        return source

    def sources(self) -> list[Source]:
        return list(self._sources.values())

    def get_source(self, source_id: str) -> Source:
        try:
            return self._sources[source_id]
        except KeyError:
            raise UnknownSourceError(source_id) from None

    def read_source_text(self, source_id: str) -> str:
        self.get_source(source_id)
        return self._snapshot_path(source_id).read_text(encoding="utf-8")

    def _snapshot_path(self, source_id: str) -> Path:
        return self.root / "sources" / "snapshots" / f"{source_id}.md"

    # Notes ---------------------------------------------------------------------------

    def add_note(self, text: str, source_ids: list[str]) -> Note:
        for source_id in source_ids:
            self.get_source(source_id)
        note = Note(
            id=f"note-{len(self.notes()) + 1:04d}",
            text=text,
            source_ids=source_ids,
            created_at=datetime.now(UTC),
        )
        with self._notes_path.open("a", encoding="utf-8") as f:
            f.write(note.model_dump_json() + "\n")
        return note

    def notes(self) -> list[Note]:
        lines = self._notes_path.read_text(encoding="utf-8").splitlines()
        return [Note.model_validate_json(line) for line in lines if line.strip()]

    # Directives ----------------------------------------------------------------------

    def add_directive(self, text: str) -> Directive:
        """Append a user instruction. Safe while a run is in progress: the file is
        append-only and the agent re-reads it every step."""
        directive = Directive(text=text, created_at=datetime.now(UTC))
        with self._directives_path.open("a", encoding="utf-8") as f:
            f.write(directive.model_dump_json() + "\n")
        return directive

    def directives(self) -> list[Directive]:
        if not self._directives_path.exists():
            return []
        # A line without its trailing newline is still being appended by another process.
        complete = self._directives_path.read_text(encoding="utf-8").split("\n")[:-1]
        return [Directive.model_validate_json(line) for line in complete if line.strip()]

    # Output --------------------------------------------------------------------------

    def write_report(self, *, status: str, summary: str | None) -> Path:
        request = self.metadata.request
        lines = [f"# {request.topic}", "", f"*Status: {status}*", ""]
        if summary:
            lines += ["## Summary", "", summary, ""]
        if self.state.agenda:
            lines += ["## Agenda", ""]
            lines += [f"- [{item.status}] {item.text}" for item in self.state.agenda]
            lines.append("")
        lines += ["## Notes", ""]
        notes = self.notes()
        lines += [_format_note(note) for note in notes] or ["*No notes were recorded.*"]
        lines += ["", "## Sources", ""]
        sources = self.sources()
        lines += [f"- `{s.id}` [{s.title}]({s.url})" for s in sources] or ["*None.*"]
        _atomic_write(self.report_path, "\n".join(lines) + "\n")
        return self.report_path


def _format_note(note: Note) -> str:
    cites = " ".join(f"`{sid}`" for sid in note.source_ids)
    return f"- {note.text}" + (f" — {cites}" if cites else "")


def _slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "research"


def _claim_directory(workspace_dir: Path, base_id: str) -> tuple[Path, str]:
    for attempt in range(1, 100):
        project_id = base_id if attempt == 1 else f"{base_id}-{attempt}"
        root = workspace_dir / project_id
        try:
            root.mkdir()
        except FileExistsError:
            continue
        return root, project_id
    raise RuntimeError(f"could not allocate a project directory for {base_id}")


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
