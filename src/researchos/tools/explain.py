"""`explain_concepts`: a dedicated writing pass that explains concepts at five depths.

Each concept is explained by a focused model call that sees only the concept, its neighbours
and the claims recorded about it (and about its children). The harness then checks the
output: citations must name claims the writer was given, and [[links]] must name known
concepts. Anything else is removed and reported, so explanations stay grounded in evidence.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from researchos.focused import FocusedCallError, FocusedModel, load_prompt
from researchos.knowledge import LEVELS, Concept, Level
from researchos.project import Project, UnknownConceptError
from researchos.tools.registry import ToolError

MAX_CLAIMS_PER_CONCEPT = 60
MAX_KNOWN_CONCEPTS = 80

CITATION = re.compile(r"\[\s*(claim-\d{4}(?:\s*[,;]\s*claim-\d{4})*)\s*\]")
LINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_CLAIM_ID = re.compile(r"claim-\d{4}")


class ExplanationSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=1_500)
    beginner: str = Field(min_length=1, max_length=8_000)
    intermediate: str = Field(min_length=1, max_length=10_000)
    deep: str = Field(min_length=1, max_length=12_000)
    expert: str = Field(min_length=1, max_length=12_000)


class ExplainConceptsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concepts: list[str] = Field(
        min_length=1, max_length=5, description="Concept ids or titles to explain."
    )


def explain_concepts(
    project: Project, model: FocusedModel, args: ExplainConceptsArgs
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for ref in args.concepts:
        try:
            concept = project.get_concept(ref)
        except UnknownConceptError:
            results.append({"concept": ref, "error": "unknown concept; see list_concepts"})
            continue
        claim_ids = _claims_for(project, concept)
        if not claim_ids:
            results.append(
                {"concept": concept.id, "error": "no claims about it yet; study sources first"}
            )
            continue
        try:
            written = model.submit(
                purpose="explain",
                system=load_prompt("explain.md"),
                user=_explain_request(project, concept, claim_ids),
                tool_name="submit_explanations",
                tool_description="Submit the explanations at all five depths.",
                result_model=ExplanationSet,
            )
        except FocusedCallError as exc:
            results.append({"concept": concept.id, "error": str(exc)})
            continue
        explanations, report = check_explanations(
            written, allowed_claims=set(claim_ids), concepts=project.concepts()
        )
        project.set_explanations(concept.id, explanations)
        results.append({"concept": concept.id, **report})
    if not any("error" not in r for r in results):
        raise ToolError(f"nothing was explained: {results}")
    return {"explained": results}


def check_explanations(
    written: ExplanationSet, *, allowed_claims: set[str], concepts: list[Concept]
) -> tuple[dict[Level, str], dict[str, Any]]:
    """Drop citations of claims the writer was not given and links to unknown concepts.
    Returns the cleaned text per level and a report of what was kept and removed."""
    titles = {c.title.lower(): c.title for c in concepts}
    for concept in concepts:
        for alias in concept.aliases:
            titles.setdefault(alias.lower(), concept.title)
    kept = removed = links = unlinked = 0

    def fix_citation(match: re.Match[str]) -> str:
        nonlocal kept, removed
        ids = _CLAIM_ID.findall(match.group(1))
        valid = [i for i in dict.fromkeys(ids) if i in allowed_claims]
        kept += len(valid)
        removed += len(ids) - len(valid)
        return f"[{', '.join(valid)}]" if valid else ""

    def fix_link(match: re.Match[str]) -> str:
        nonlocal links, unlinked
        title = titles.get(match.group(1).strip().lower())
        if title is None:
            unlinked += 1
            return match.group(1)
        links += 1
        return f"[[{title}]]"

    cleaned: dict[Level, str] = {}
    uncited = 0
    for level in LEVELS:
        text = LINK.sub(fix_link, CITATION.sub(fix_citation, getattr(written, level)))
        cleaned[level] = re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()
        if level in ("intermediate", "deep", "expert"):
            uncited += sum(
                1 for p in cleaned[level].split("\n\n") if p.strip() and not CITATION.search(p)
            )
    return cleaned, {
        "levels": list(LEVELS),
        "citations": kept,
        "removed_citations": removed,
        "links": links,
        "unlinked_mentions": unlinked,
        "uncited_paragraphs": uncited,
    }


def _claims_for(project: Project, concept: Concept) -> list[str]:
    ids = list(concept.claim_ids)
    for child in project.children(concept.id):
        ids += child.claim_ids
    return list(dict.fromkeys(ids))[:MAX_CLAIMS_PER_CONCEPT]


def _explain_request(project: Project, concept: Concept, claim_ids: list[str]) -> str:
    request = project.metadata.request

    def titles(ids: list[str]) -> str:
        names = [project.get_concept(i).title for i in ids]
        return ", ".join(names) if names else "none"

    parent = project.get_concept(concept.parent).title if concept.parent else "none"
    children = [c.id for c in project.children(concept.id)]
    claim_lines = []
    for claim_id in claim_ids:
        claim = project.get_claim(claim_id)
        assessment = project.assess(claim)
        status = {
            "verified": f"verified by {assessment.supporting_sources} independent sources",
            "single_source": "single-source",
            "disputed": "disputed: sources disagree",
            "unsupported": "unsupported",
        }[assessment.status]
        note = f" Note: {claim.note}" if claim.note else ""
        claim_lines.append(f"- {claim.id} ({claim.kind}, {status}): {claim.text}{note}")
    known = [c.title for c in project.concepts()][-MAX_KNOWN_CONCEPTS:]
    return "\n".join(
        [
            f"Research topic: {request.topic}",
            f"Learner's goal: {request.goal or 'not stated'}",
            f"Learner's level: {request.knowledge_level or 'not stated'}",
            "",
            f"Concept: {concept.title}",
            f"Summary so far: {concept.summary or 'none'}",
            f"Part of: {parent}",
            f"Prerequisites: {titles(concept.prerequisites)}",
            f"Sub-concepts: {titles(children)}",
            f"Related: {titles(concept.related)}",
            "",
            "Claims (the only facts you may state):",
            *claim_lines,
            "",
            "Known concepts you can link with [[Title]]:",
            ", ".join(known),
        ]
    )
