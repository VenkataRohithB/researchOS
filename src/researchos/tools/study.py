"""`study_source`: a dedicated, thorough reading pass over one source.

The main agent decides which sources deserve study. The study itself is a focused model call
(see `researchos.focused`) that reads the source text and returns claims with quotes,
corroborations of claims already recorded from other sources, concepts with their
relationships, and open questions. The harness then checks every quote against the stored
source text before anything is recorded, so the reader cannot invent evidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from researchos.evidence import ClaimKind, Stance, words
from researchos.focused import FocusedCallError, FocusedModel, load_prompt
from researchos.project import EvidenceError, EvidenceInput, Project, UnknownClaimError
from researchos.tools.registry import ToolError
from researchos.untrusted import untrusted

STUDY_CHARS = 40_000
"""Source text per study call; longer sources are studied in several passes."""
MAX_KNOWN_CLAIMS = 80
MAX_KNOWN_CONCEPTS = 80


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StudiedClaim(_Model):
    text: str = Field(min_length=1, max_length=800)
    kind: ClaimKind = "fact"
    quote: str = Field(min_length=1, max_length=1_500)
    concepts: list[str] = Field(default_factory=list, max_length=8)


class Corroboration(_Model):
    claim_id: str
    quote: str = Field(min_length=1, max_length=1_500)
    stance: Stance = "supports"


class StudiedConcept(_Model):
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(default="", max_length=600)
    parent: str | None = Field(default=None, max_length=120)
    prerequisites: list[str] = Field(default_factory=list, max_length=8)
    related: list[str] = Field(default_factory=list, max_length=8)
    aliases: list[str] = Field(default_factory=list, max_length=6)


class StudyResult(_Model):
    claims: list[StudiedClaim] = Field(default_factory=list, max_length=30)
    corroborations: list[Corroboration] = Field(default_factory=list, max_length=30)
    concepts: list[StudiedConcept] = Field(default_factory=list, max_length=20)
    open_questions: list[str] = Field(default_factory=list, max_length=5)


class StudySourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Id returned by fetch_source.")
    focus: str | None = Field(
        default=None,
        max_length=500,
        description="What to look for in particular; defaults to the open agenda items.",
    )
    offset: int = Field(
        default=0, ge=0, description="Where to start, for sources longer than one pass."
    )


def study_source(project: Project, model: FocusedModel, args: StudySourceArgs) -> dict[str, Any]:
    try:
        source = project.get_source(args.source_id)
        text = project.read_source_text(args.source_id)
    except KeyError:
        raise ToolError(f"unknown source_id '{args.source_id}'") from None
    chunk = text[args.offset : args.offset + STUDY_CHARS]
    if not chunk.strip():
        raise ToolError(
            f"nothing to study at offset {args.offset}; the source has {len(text)} chars"
        )
    end = args.offset + len(chunk)

    try:
        result = model.submit(
            purpose="study",
            system=load_prompt("study.md"),
            user=_study_request(project, args, source.title, source.url, source.tier, chunk),
            tool_name="submit_study",
            tool_description="Submit everything extracted from the document.",
            result_model=StudyResult,
        )
    except FocusedCallError as exc:
        raise ToolError(str(exc)) from None

    return {
        **_apply(project, args.source_id, result),
        "next_offset": end if end < len(text) else None,
    }


def _study_request(
    project: Project, args: StudySourceArgs, title: str, url: str, tier: str, chunk: str
) -> str:
    request = project.metadata.request
    focus = args.focus or "; ".join(
        item.text for item in project.state.agenda if item.status == "open"
    )
    claims = project.claims()[-MAX_KNOWN_CLAIMS:]
    concepts = project.concepts()[-MAX_KNOWN_CONCEPTS:]
    lines = [
        f"Research topic: {request.topic}",
        f"Goal: {request.goal or 'not stated'}",
        f"Reader's level: {request.knowledge_level or 'not stated'}",
        f"Focus: {focus or 'the topic as a whole'}",
        "",
        "Claims already recorded (corroborate or contradict these instead of repeating them):",
        *([f"- {c.id}: {c.text}" for c in claims] or ["- none yet"]),
        "",
        "Concepts already known (reuse these titles):",
        *([f"- {c.title}" for c in concepts] or ["- none yet"]),
        "",
        f"Document: {title} <{url}> (tier: {tier})",
        untrusted(chunk, args.source_id),
    ]
    return "\n".join(lines)


def _apply(project: Project, source_id: str, result: StudyResult) -> dict[str, Any]:
    known = {" ".join(words(c.text)): c.id for c in project.claims()}
    recorded: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    corroborated: list[dict[str, Any]] = []

    def corroborate(claim_id: str, quote: str, stance: Stance) -> None:
        try:
            claim = project.add_evidence(claim_id, [EvidenceInput(source_id, quote, stance)])
        except UnknownClaimError:
            rejected.append({"claim": claim_id, "reason": "unknown claim id"})
            return
        except EvidenceError as exc:
            rejected.append({"claim": claim_id, "reason": str(exc)})
            return
        corroborated.append(
            {"claim_id": claim.id, "stance": stance, "status": project.assess(claim).status}
        )

    for corroboration in result.corroborations:
        corroborate(corroboration.claim_id, corroboration.quote, corroboration.stance)

    for item in result.claims:
        duplicate_of = known.get(" ".join(words(item.text)))
        if duplicate_of:
            corroborate(duplicate_of, item.quote, "supports")
            continue
        try:
            claim = project.add_claim(
                item.text, item.kind, [EvidenceInput(source_id, item.quote, "supports")]
            )
        except EvidenceError as exc:
            rejected.append({"claim": item.text[:120], "reason": str(exc)})
            continue
        known[" ".join(words(claim.text))] = claim.id
        for title in item.concepts:
            project.merge_concept(title, claim_ids=[claim.id])
        recorded.append({"claim_id": claim.id, "text": claim.text[:160]})

    created: list[str] = []
    problems: list[str] = []
    for concept in result.concepts:
        change = project.merge_concept(
            concept.title,
            summary=concept.summary or None,
            parent=concept.parent,
            prerequisites=concept.prerequisites,
            related=concept.related,
            aliases=concept.aliases,
        )
        if change.created:
            created.append(change.concept.id)
        problems += change.problems

    # Readers often leave a claim's concepts empty; link every claim to the concepts it names,
    # so each concept gathers the evidence its explanations are written from. Linking is
    # idempotent, and covering all claims also links older ones to newly found concepts.
    for claim in project.claims():
        for concept_ref in mentioned_concepts(project, claim.text):
            project.merge_concept(concept_ref, claim_ids=[claim.id])

    return {
        "claims_recorded": len(recorded),
        "claims": recorded,
        "corroborated": corroborated,
        "rejected": rejected[:8],
        "rejected_count": len(rejected),
        "concepts_created": created,
        "concept_problems": problems,
        "open_questions": result.open_questions,
    }


def mentioned_concepts(project: Project, text: str) -> list[str]:
    """Ids of known concepts whose title or an alias appears in `text` as whole words."""
    haystack = f" {' '.join(words(text))} "
    found = []
    for concept in project.concepts():
        names = [concept.title, *concept.aliases]
        if any(
            len(name) >= 3 and f" {' '.join(words(name))} " in haystack
            for name in names
            if words(name)
        ):
            found.append(concept.id)
    return found
