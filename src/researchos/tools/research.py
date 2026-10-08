"""Research tools available to the agent."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from researchos.evidence import ClaimKind, ClaimStatus, SourceTier, Stance, words
from researchos.project import (
    EvidenceError,
    EvidenceInput,
    Project,
    Source,
    UnknownClaimError,
    UnknownSourceError,
)
from researchos.state import Phase
from researchos.tools.registry import Tool, ToolError
from researchos.untrusted import untrusted
from researchos.web import Fetcher, FetchError, SearchError, SearchProvider

FETCH_EXCERPT_CHARS = 4_000
PASSAGE_CHARS = 700


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


class SearchSourcesArgs(_Args):
    query: str = Field(min_length=1, max_length=300, description="Words to look for.")
    source_ids: list[str] = Field(
        default_factory=list, description="Limit to these sources; empty searches all of them."
    )
    max_passages: int = Field(default=5, ge=1, le=15)


class AssessSourceArgs(_Args):
    source_id: str
    tier: SourceTier = Field(
        description=(
            "official: the subject's own documentation or announcements; paper: peer-reviewed "
            "or preprint research; academic: university material; standard: standards body; "
            "government; reference: encyclopedias; news: journalism; community: forums, "
            "blogs, social media; other."
        )
    )
    rationale: str = Field(min_length=1, max_length=300)


class EvidenceArgs(_Args):
    source_id: str = Field(description="Id of a fetched source.")
    quote: str = Field(
        min_length=1,
        max_length=1_500,
        description="A passage copied verbatim from the source (5-120 words, no ellipses).",
    )
    stance: Stance = Field(
        default="supports", description="Whether the passage supports, contradicts or qualifies."
    )


class RecordClaimArgs(_Args):
    text: str = Field(min_length=1, max_length=1_000, description="One self-contained claim.")
    kind: ClaimKind = Field(
        default="fact",
        description=(
            "fact: stated by sources; interpretation: your reading of what sources say; "
            "inference: your conclusion drawn from several facts."
        ),
    )
    evidence: list[EvidenceArgs] = Field(default_factory=list, max_length=6)
    note: str | None = Field(
        default=None, max_length=500, description="Caveats, or how sources disagree."
    )


class AddEvidenceArgs(_Args):
    claim_id: str
    evidence: list[EvidenceArgs] = Field(min_length=1, max_length=6)


class GetClaimArgs(_Args):
    claim_id: str


class ListClaimsArgs(_Args):
    status: ClaimStatus | None = Field(default=None, description="Only claims with this status.")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=30, ge=1, le=100)


class AgendaUpdate(_Args):
    id: str = Field(description="Agenda item id, e.g. 'a3'.")
    status: Literal["open", "done", "dropped"]
    note: str | None = Field(default=None, max_length=500, description="Why, or what was found.")
    claim_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Required to mark an item done: the claims, backed by quotes, that cover it.",
    )


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
        result = {
            **_describe_source(source),
            "total_chars": source.chars,
            "excerpt": untrusted(
                f"Title: {source.title}\n\n{page.text[:FETCH_EXCERPT_CHARS]}", source.id
            ),
            "truncated": source.chars > FETCH_EXCERPT_CHARS,
        }
        if source.duplicate_of:
            result["warning"] = (
                f"this page has essentially the same content as {source.duplicate_of}; it does "
                "not count as an independent source"
            )
        return result

    def read_source(args: ReadSourceArgs) -> dict[str, Any]:
        text = _source_text(project, args.source_id)
        chunk = text[args.offset : args.offset + args.length]
        end = args.offset + len(chunk)
        return {
            "source_id": args.source_id,
            "offset": args.offset,
            "text": untrusted(chunk, args.source_id),
            "next_offset": end if end < len(text) else None,
        }

    def search_sources(args: SearchSourcesArgs) -> dict[str, Any]:
        ids = args.source_ids or [s.id for s in project.sources()]
        terms = set(words(args.query))
        if not terms:
            raise ToolError("query has no searchable words")
        scored: list[tuple[int, str, int, str]] = []
        for source_id in ids:
            text = _source_text(project, source_id)
            # Passages are blocks of consecutive non-blank lines.
            for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
                passage = match.group(0)
                hits = len(terms & set(words(passage)))
                if hits:
                    scored.append((hits, source_id, match.start(), passage[:PASSAGE_CHARS]))
        scored.sort(key=lambda item: -item[0])
        return {
            "passages": [
                {"source_id": sid, "offset": offset, "text": untrusted(passage, sid)}
                for _, sid, offset, passage in scored[: args.max_passages]
            ]
        }

    def assess_source(args: AssessSourceArgs) -> dict[str, Any]:
        try:
            source = project.set_source_tier(args.source_id, args.tier, args.rationale)
        except UnknownSourceError:
            raise ToolError(f"unknown source_id '{args.source_id}'") from None
        return _describe_source(source)

    def record_claim(args: RecordClaimArgs) -> dict[str, Any]:
        if args.kind == "fact" and not args.evidence:
            raise ToolError("a fact needs at least one quoted passage as evidence")
        try:
            claim = project.add_claim(args.text, args.kind, _evidence(args.evidence), args.note)
        except EvidenceError as exc:
            raise ToolError(f"claim not recorded: {exc}") from None
        return _describe_claim(project, claim.id)

    def add_evidence(args: AddEvidenceArgs) -> dict[str, Any]:
        try:
            project.add_evidence(args.claim_id, _evidence(args.evidence))
        except UnknownClaimError:
            raise ToolError(f"unknown claim_id '{args.claim_id}'") from None
        except EvidenceError as exc:
            raise ToolError(f"evidence not added: {exc}") from None
        return _describe_claim(project, args.claim_id)

    def get_claim(args: GetClaimArgs) -> dict[str, Any]:
        try:
            claim = project.get_claim(args.claim_id)
        except UnknownClaimError:
            raise ToolError(f"unknown claim_id '{args.claim_id}'") from None
        sources = {s.id: s for s in project.sources()}
        return {
            **_describe_claim(project, claim.id),
            "text": claim.text,
            "note": claim.note,
            "evidence": [
                {
                    **_describe_source(sources[e.source_id]),
                    "stance": e.stance,
                    "quote": untrusted(e.quote, e.source_id),
                }
                for e in claim.evidence
            ],
        }

    def list_claims(args: ListClaimsArgs) -> dict[str, Any]:
        claims = [
            c
            for c in project.claims()
            if args.status is None or project.assess(c).status == args.status
        ]
        page = claims[args.offset : args.offset + args.limit]
        return {
            "total": len(claims),
            "claims": [
                {**_describe_claim(project, c.id), "text": c.text, "kind": c.kind} for c in page
            ],
        }

    def update_agenda(args: UpdateAgendaArgs) -> dict[str, Any]:
        known = {item.id for item in project.state.agenda}
        unknown = [u.id for u in args.update if u.id not in known]
        if unknown:
            raise ToolError(f"unknown agenda item(s): {', '.join(unknown)}")
        problems = [p for u in args.update if u.status != "open" for p in _closing_problems(u)]
        if problems:
            raise ToolError("agenda not updated: " + "; ".join(problems))
        for update in args.update:
            project.state.set_agenda_status(update.id, update.status, update.note, update.claim_ids)
        added = project.state.add_agenda_items([text for text in args.add if text.strip()])
        return {"added": [{"id": item.id, "text": item.text} for item in added]}

    def set_phase(args: SetPhaseArgs) -> dict[str, Any]:
        project.state.phase, project.state.phase_reason = args.phase, args.reason
        return {"phase": args.phase.value}

    def _closing_problems(update: AgendaUpdate) -> list[str]:
        if update.status == "dropped":
            return [] if update.note else [f"{update.id}: give a reason in note to drop it"]
        if not update.claim_ids:
            return [f"{update.id}: list the claim_ids that cover it to mark it done"]
        problems = []
        for claim_id in update.claim_ids:
            try:
                assessment = project.assess(project.get_claim(claim_id))
            except UnknownClaimError:
                problems.append(f"{update.id}: unknown claim '{claim_id}'")
                continue
            if not assessment.supporting_sources:
                problems.append(f"{update.id}: {claim_id} has no supporting quote from a source")
        return problems

    def finish_research(args: FinishResearchArgs) -> dict[str, Any]:
        still_open = [item.id for item in project.state.agenda if item.status == "open"]
        if still_open:
            raise ToolError(
                f"cannot finish: agenda items still open: {', '.join(still_open)}. Research "
                "them and mark them done with the claims that cover them, or drop them with a "
                "reason."
            )
        weak = [
            claim_id
            for item in project.state.agenda
            for claim_id in item.claim_ids
            if claim_id not in project.state.reviewed_claims
            and project.assess(project.get_claim(claim_id)).status in ("single_source", "disputed")
        ]
        if weak:
            weak = list(dict.fromkeys(weak))
            project.state.reviewed_claims.extend(weak)
            raise ToolError(
                "review before finishing: the agenda relies on claims that are single-source or "
                f"disputed: {', '.join(weak)}. Look for independent corroboration "
                "(search_web, fetch_source, search_sources) and add it with add_evidence, or "
                "record the disagreement. If none can be found, call finish_research again; "
                "those claims will be labelled as single-source or disputed."
            )
        if not any(project.assess(c).supporting_sources for c in project.claims()):
            raise ToolError(
                "cannot finish: no claim is supported by a quote from a fetched source. "
                "Research must be grounded in sources: fetch_source the relevant pages, then "
                "record_claim your findings with quoted evidence before finishing."
            )
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
                "Fetch a web page or PDF, store it as a source and return its id, reliability "
                "tier and an excerpt. Only fetched sources can be cited."
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
            name="search_sources",
            description=(
                "Find passages mentioning given words across fetched sources. Use it to locate "
                "exact wording to quote, and to cross-check a claim in other sources."
            ),
            args_model=SearchSourcesArgs,
            handler=search_sources,
        ),
        Tool(
            name="assess_source",
            description=(
                "Set a source's reliability tier when the domain-based default is wrong, e.g. "
                "mark a company's own documentation as official."
            ),
            args_model=AssessSourceArgs,
            handler=assess_source,
        ),
        Tool(
            name="record_claim",
            description=(
                "Record one claim with quoted evidence. Every quote is checked against the "
                "stored source text and the claim is rejected if a quote is not found. Status "
                "is computed: verified (2+ independent sources), single_source, disputed (a "
                "source contradicts it) or unsupported."
            ),
            args_model=RecordClaimArgs,
            handler=record_claim,
        ),
        Tool(
            name="add_evidence",
            description=(
                "Add quoted evidence to an existing claim, e.g. corroboration from another "
                "source or a contradicting passage."
            ),
            args_model=AddEvidenceArgs,
            handler=add_evidence,
        ),
        Tool(
            name="get_claim",
            description="Show a claim with all its evidence side by side, to compare sources.",
            args_model=GetClaimArgs,
            handler=get_claim,
        ),
        Tool(
            name="list_claims",
            description="Page through claims, optionally only those with a given status.",
            args_model=ListClaimsArgs,
            handler=list_claims,
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


def _evidence(items: list[EvidenceArgs]) -> list[EvidenceInput]:
    return [EvidenceInput(source_id=e.source_id, quote=e.quote, stance=e.stance) for e in items]


def _source_text(project: Project, source_id: str) -> str:
    try:
        return project.read_source_text(source_id)
    except UnknownSourceError:
        raise ToolError(f"unknown source_id '{source_id}'") from None


def _describe_source(source: Source) -> dict[str, Any]:
    described: dict[str, Any] = {
        "source_id": source.id,
        "url": source.url,
        "tier": source.tier,
        "published": source.published,
    }
    if source.duplicate_of:
        described["duplicate_of"] = source.duplicate_of
    return described


def _describe_claim(project: Project, claim_id: str) -> dict[str, Any]:
    assessment = project.assess(project.get_claim(claim_id))
    return {
        "claim_id": claim_id,
        "status": assessment.status,
        "independent_supporting_sources": assessment.supporting_sources,
        "independent_contradicting_sources": assessment.contradicting_sources,
    }
