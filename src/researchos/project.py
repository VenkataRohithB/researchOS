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
        knowledge/claims.json   claims, each with quoted evidence from sources
        knowledge/concepts.json concepts and their relationships (the knowledge graph)
        report.md               human-readable output
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from researchos.evidence import (
    Assessment,
    Claim,
    ClaimKind,
    Evidence,
    QuoteError,
    SourceIdentity,
    SourceTier,
    Stance,
    assess,
    canonical_url,
    check_quote,
    default_tier,
    fingerprint,
    is_near_duplicate,
    site_of,
)
from researchos.knowledge import Concept, Level, concept_id
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
    """Canonical URL (see `researchos.evidence.canonical_url`)."""
    title: str
    fetched_at: datetime
    content_sha256: str
    chars: int
    published: str | None = None
    """Publication date as stated by the page, when it states one."""
    tier: SourceTier
    tier_rationale: str | None = None
    """Why the agent assigned the tier; None for the domain-based default."""
    duplicate_of: str | None = None
    """Id of an earlier source with essentially the same content."""

    @model_validator(mode="before")
    @classmethod
    def _default_tier(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tier" not in data and "url" in data:
            data = {**data, "tier": default_tier(str(data["url"]))}
        return data


@dataclass(frozen=True)
class EvidenceInput:
    source_id: str
    quote: str
    stance: Stance


class EvidenceError(ValueError):
    """Evidence was rejected; the message lists every problem and is shown to the model."""


class UnknownClaimError(KeyError):
    pass


class UnknownConceptError(KeyError):
    pass


@dataclass(frozen=True)
class ConceptChange:
    concept: Concept
    created: bool
    problems: list[str]
    """Requested changes that were refused, e.g. a parent link that would form a cycle."""


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
        self._fingerprints: dict[str, frozenset[int]] = {}
        if not self._claims_path.exists():
            self._migrate_notes()
        self._claims: dict[str, Claim] = {
            c["id"]: Claim.model_validate(c)
            for c in json.loads(self._claims_path.read_text(encoding="utf-8"))
        }
        self._concepts: dict[str, Concept] = (
            {
                c["id"]: Concept.model_validate(c)
                for c in json.loads(self._concepts_path.read_text(encoding="utf-8"))
            }
            if self._concepts_path.exists()
            else {}
        )

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
        for sub in ("metadata", "sources/snapshots", "knowledge"):
            (root / sub).mkdir(parents=True)
        metadata = ProjectMetadata(id=project_id, created_at=created_at, request=request)
        _atomic_write(root / "metadata" / "project.json", metadata.model_dump_json(indent=2))
        _atomic_write(root / "metadata" / "state.json", ResearchState().model_dump_json(indent=2))
        _atomic_write(root / "sources" / "sources.json", "[]")
        _atomic_write(root / "knowledge" / "claims.json", "[]")
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
    def _claims_path(self) -> Path:
        return self.root / "knowledge" / "claims.json"

    @property
    def _concepts_path(self) -> Path:
        return self.root / "knowledge" / "concepts.json"

    # Sources -------------------------------------------------------------------------

    def add_source(self, page: FetchedPage) -> Source:
        """Register a fetched page. Re-fetching the same canonical URL refreshes it; a page
        whose content duplicates an earlier source is recorded as a duplicate of it."""
        url = canonical_url(page.url)
        source_id = "src-" + hashlib.sha256(url.encode()).hexdigest()[:10]
        prints = fingerprint(page.text)
        duplicate_of = next(
            (
                other.id
                for other in self._sources.values()
                if other.id != source_id
                and other.duplicate_of is None
                and is_near_duplicate(prints, self._fingerprint(other.id))
            ),
            None,
        )
        previous = self._sources.get(source_id)
        source = Source(
            id=source_id,
            url=url,
            title=page.title,
            fetched_at=datetime.now(UTC),
            content_sha256=hashlib.sha256(page.text.encode()).hexdigest(),
            chars=len(page.text),
            published=page.published,
            tier=(
                previous.tier
                if previous and previous.tier_rationale
                else default_tier(url, self.metadata.request.topic)
            ),
            tier_rationale=previous.tier_rationale if previous else None,
            duplicate_of=duplicate_of,
        )
        _atomic_write(self._snapshot_path(source_id), page.text)
        self._fingerprints[source_id] = prints
        self._sources[source_id] = source
        self._save_sources()
        return source

    def set_source_tier(self, source_id: str, tier: SourceTier, rationale: str) -> Source:
        source = self.get_source(source_id).model_copy(
            update={"tier": tier, "tier_rationale": rationale}
        )
        self._sources[source_id] = source
        self._save_sources()
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

    def source_identities(self) -> dict[str, SourceIdentity]:
        return {
            s.id: SourceIdentity(site=site_of(s.url), original_id=s.duplicate_of or s.id)
            for s in self._sources.values()
        }

    def _fingerprint(self, source_id: str) -> frozenset[int]:
        if source_id not in self._fingerprints:
            self._fingerprints[source_id] = fingerprint(self.read_source_text(source_id))
        return self._fingerprints[source_id]

    def _snapshot_path(self, source_id: str) -> Path:
        return self.root / "sources" / "snapshots" / f"{source_id}.md"

    def _save_sources(self) -> None:
        _atomic_write(
            self._sources_path,
            json.dumps([s.model_dump(mode="json") for s in self._sources.values()], indent=2),
        )

    # Claims --------------------------------------------------------------------------

    def add_claim(
        self,
        text: str,
        kind: ClaimKind,
        evidence: Sequence[EvidenceInput],
        note: str | None = None,
    ) -> Claim:
        """Record a claim. All evidence is checked first; if any item is invalid nothing is
        recorded and `EvidenceError` lists every problem."""
        checked = self._check_evidence(evidence)
        now = datetime.now(UTC)
        claim = Claim(
            id=f"claim-{len(self._claims) + 1:04d}",
            text=text,
            kind=kind,
            evidence=checked,
            note=note,
            created_at=now,
            updated_at=now,
        )
        self._claims[claim.id] = claim
        self._save_claims()
        return claim

    def add_evidence(self, claim_id: str, evidence: Sequence[EvidenceInput]) -> Claim:
        claim = self.get_claim(claim_id)
        checked = self._check_evidence(evidence)
        existing = {(e.source_id, e.quote) for e in claim.evidence}
        new = [e for e in checked if (e.source_id, e.quote) not in existing]
        claim = claim.model_copy(
            update={"evidence": [*claim.evidence, *new], "updated_at": datetime.now(UTC)}
        )
        self._claims[claim_id] = claim
        self._save_claims()
        return claim

    def claims(self) -> list[Claim]:
        return list(self._claims.values())

    def get_claim(self, claim_id: str) -> Claim:
        try:
            return self._claims[claim_id]
        except KeyError:
            raise UnknownClaimError(claim_id) from None

    def assess(self, claim: Claim) -> Assessment:
        return assess(claim, self.source_identities())

    def _check_evidence(self, evidence: Sequence[EvidenceInput]) -> list[Evidence]:
        problems: list[str] = []
        texts: dict[str, str] = {}
        for i, item in enumerate(evidence, start=1):
            if item.source_id not in self._sources:
                problems.append(
                    f"evidence {i}: unknown source_id '{item.source_id}'; "
                    "cite only ids returned by fetch_source"
                )
                continue
            if item.source_id not in texts:
                texts[item.source_id] = self.read_source_text(item.source_id)
            try:
                check_quote(item.quote, texts[item.source_id])
            except QuoteError as exc:
                problems.append(f"evidence {i} ({item.source_id}): {exc}")
        if problems:
            raise EvidenceError("; ".join(problems))
        now = datetime.now(UTC)
        return [
            Evidence(source_id=e.source_id, quote=e.quote, stance=e.stance, added_at=now)
            for e in evidence
        ]

    def _save_claims(self) -> None:
        _atomic_write(
            self._claims_path,
            json.dumps([c.model_dump(mode="json") for c in self._claims.values()], indent=2),
        )

    def _migrate_notes(self) -> None:
        """Projects created before claims existed kept free-text notes without quotes. Carry
        them over as unsupported claims that remember which sources they cited."""
        self._claims_path.parent.mkdir(exist_ok=True)
        legacy = self.root / "notes" / "notes.jsonl"
        claims: list[Claim] = []
        if legacy.exists():
            for line in legacy.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                note = json.loads(line)
                cited = ", ".join(note.get("source_ids") or []) or "none"
                created = datetime.fromisoformat(note["created_at"])
                claims.append(
                    Claim(
                        id=f"claim-{len(claims) + 1:04d}",
                        text=note["text"],
                        kind="fact",
                        note=f"Migrated from {note['id']} (sources cited without quotes: {cited})",
                        created_at=created,
                        updated_at=created,
                    )
                )
        _atomic_write(
            self._claims_path, json.dumps([c.model_dump(mode="json") for c in claims], indent=2)
        )

    # Concepts ------------------------------------------------------------------------

    def concepts(self) -> list[Concept]:
        return list(self._concepts.values())

    def get_concept(self, ref: str) -> Concept:
        """Find a concept by id, title or alias."""
        found = self._resolve(ref)
        if found is None:
            raise UnknownConceptError(ref)
        return self._concepts[found]

    def merge_concept(
        self,
        title: str,
        *,
        summary: str | None = None,
        parent: str | None = None,
        prerequisites: Sequence[str] = (),
        related: Sequence[str] = (),
        aliases: Sequence[str] = (),
        claim_ids: Sequence[str] = (),
    ) -> ConceptChange:
        """Add what is new without overwriting what exists: fill empty fields and extend
        lists. Referenced concepts that do not exist yet are created as stubs."""
        existing = self._resolve(title)
        created = existing is None
        current = self._concepts[existing] if existing else self._new_concept(title)
        problems: list[str] = []
        update: dict[str, Any] = {
            "aliases": _union(current.aliases, [a for a in aliases if a != current.title]),
            "prerequisites": _union(current.prerequisites, self._refs(prerequisites, current.id)),
            "related": _union(current.related, self._refs(related, current.id)),
            "claim_ids": _union(current.claim_ids, [c for c in claim_ids if c in self._claims]),
        }
        if summary and not current.summary:
            update["summary"] = summary
        if parent and current.parent is None:
            update["parent"], problem = self._checked_parent(current.id, parent)
            problems += [problem] if problem else []
        concept = current.model_copy(update={**update, "updated_at": datetime.now(UTC)})
        self._concepts[concept.id] = concept
        self._save_concepts()
        return ConceptChange(concept=concept, created=created, problems=problems)

    def update_concept(
        self,
        ref: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        parent: str | None = None,
        prerequisites: Sequence[str] | None = None,
        related: Sequence[str] | None = None,
    ) -> ConceptChange:
        """Set fields explicitly (the agent correcting the graph). An empty `parent` makes the
        concept top-level."""
        current = self.get_concept(ref)
        problems: list[str] = []
        update: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if title:
            update["title"] = title
        if summary is not None:
            update["summary"] = summary
        if parent is not None:
            if parent == "":
                update["parent"] = None
            else:
                update["parent"], problem = self._checked_parent(current.id, parent)
                problems += [problem] if problem else []
        if prerequisites is not None:
            update["prerequisites"] = self._refs(prerequisites, current.id)
        if related is not None:
            update["related"] = self._refs(related, current.id)
        concept = current.model_copy(update=update)
        self._concepts[concept.id] = concept
        self._save_concepts()
        return ConceptChange(concept=concept, created=False, problems=problems)

    def set_explanations(self, ref: str, explanations: dict[Level, str]) -> Concept:
        current = self.get_concept(ref)
        concept = current.model_copy(
            update={
                "explanations": {**current.explanations, **explanations},
                "updated_at": datetime.now(UTC),
            }
        )
        self._concepts[concept.id] = concept
        self._save_concepts()
        return concept

    def children(self, ref: str) -> list[Concept]:
        parent = self.get_concept(ref).id
        return [c for c in self._concepts.values() if c.parent == parent]

    def _resolve(self, ref: str) -> str | None:
        if ref in self._concepts:
            return ref
        key = concept_id(ref)
        if key in self._concepts:
            return key
        for concept in self._concepts.values():
            if any(concept_id(alias) == key for alias in concept.aliases):
                return concept.id
        return None

    def _new_concept(self, title: str) -> Concept:
        now = datetime.now(UTC)
        return Concept(id=concept_id(title), title=title.strip(), created_at=now, updated_at=now)

    def _refs(self, titles: Sequence[str], own_id: str) -> list[str]:
        """Ids for referenced concepts, creating stubs for unknown ones."""
        ids: list[str] = []
        for title in titles:
            if not title.strip():
                continue
            ref = self._resolve(title)
            if ref is None:
                stub = self._new_concept(title)
                self._concepts[stub.id] = stub
                ref = stub.id
            if ref != own_id:
                ids.append(ref)
        return ids

    def _checked_parent(self, own_id: str, parent: str) -> tuple[str | None, str | None]:
        (parent_id,) = self._refs([parent], own_id) or (None,)
        if parent_id is None:
            return None, f"{own_id} cannot be its own parent"
        ancestor: str | None = parent_id
        while ancestor is not None:
            if ancestor == own_id:
                return None, f"making {parent_id} the parent of {own_id} would form a cycle"
            ancestor = self._concepts[ancestor].parent
        return parent_id, None

    def _save_concepts(self) -> None:
        _atomic_write(
            self._concepts_path,
            json.dumps([c.model_dump(mode="json") for c in self._concepts.values()], indent=2),
        )

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
        lines += ["## Claims", ""]
        claims = self.claims()
        lines += [self._format_claim(c) for c in claims] or ["*No claims were recorded.*"]
        lines += ["", "## Sources", ""]
        lines += [_format_source(s) for s in self.sources()] or ["*None.*"]
        _atomic_write(self.report_path, "\n".join(lines) + "\n")
        return self.report_path

    def _format_claim(self, claim: Claim) -> str:
        assessment = self.assess(claim)
        label = assessment.status.replace("_", " ")
        lines = [f"- **[{label}]** {claim.text} (`{claim.id}`, {claim.kind})"]
        lines += [f'  - {e.stance} `{e.source_id}`: "{e.quote}"' for e in claim.evidence]
        return "\n".join(lines)


def _format_source(source: Source) -> str:
    extra = f", duplicate of `{source.duplicate_of}`" if source.duplicate_of else ""
    return f"- `{source.id}` [{source.title}]({source.url}) ({source.tier}{extra})"


def _union(first: Sequence[str], second: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))


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
