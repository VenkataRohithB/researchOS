from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helpers import ScriptedLLM, call

from researchos.focused import FocusedModel
from researchos.knowledge import Concept
from researchos.llm import ToolCall
from researchos.project import EvidenceInput, Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.tools.explain import ExplanationSet, check_explanations
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.user import NoUser
from researchos.web import FetchedPage, MockFetcher, MockSearch

NOW = datetime.now(UTC)
CONCEPTS = [
    Concept(id="memory-bandwidth", title="Memory bandwidth", aliases=["bandwidth"],
            created_at=NOW, updated_at=NOW),
]  # fmt: skip


def written(**overrides: str) -> ExplanationSet:
    levels = {
        level: f"{level} text."
        for level in ("summary", "beginner", "intermediate", "deep", "expert")
    }
    return ExplanationSet(**{**levels, **overrides})


def test_citations_and_links_are_checked() -> None:
    text = (
        "Bandwidth is 120 GB/s [claim-0001, claim-0009]. Invented [claim-0042]. "
        "See [[memory bandwidth]] and [[Bandwidth]] but not [[Quantum tunnelling]].\n\n"
        "An uncited paragraph."
    )

    cleaned, report = check_explanations(
        written(deep=text), allowed_claims={"claim-0001"}, concepts=CONCEPTS
    )

    assert cleaned["deep"] == (
        "Bandwidth is 120 GB/s [claim-0001]. Invented. See [[Memory bandwidth]] and "
        "[[Memory bandwidth]] but not Quantum tunnelling.\n\nAn uncited paragraph."
    )
    assert (report["citations"], report["removed_citations"]) == (1, 2)
    assert (report["links"], report["unlinked_mentions"]) == (2, 1)
    # intermediate and expert are uncited single paragraphs, plus one in deep.
    assert report["uncited_paragraphs"] == 3


def run_explain(project: Project, writer: ScriptedLLM, concepts: list[str]) -> dict[str, Any]:
    meter = UsageMeter(
        limits=Limits(max_steps=5, max_cost_usd=5, max_wall_seconds=60),
        pricing=Pricing(input_per_million=0, output_per_million=0),
        events_path=project.events_path,
        run_id="r",
    )
    tools = ToolRegistry(
        build_research_tools(
            project, MockSearch(), MockFetcher(), focused=FocusedModel(writer, meter), user=NoUser()
        )
    )
    outcome = tools.execute(
        ToolCall(id="c", name="explain_concepts", arguments=json.dumps({"concepts": concepts}))
    )
    result: dict[str, Any] = json.loads(outcome.content)
    return result


def test_explain_concepts_stores_grounded_levels(tmp_path: Path) -> None:
    project = Project.create(
        tmp_path, ResearchRequest(topic="Unified memory", knowledge_level="new")
    )
    source = project.add_source(FetchedPage(
        url="https://a.example/x", title="A", text="The CPU and GPU share one pool of memory today."
    ))  # fmt: skip
    claim = project.add_claim(
        "CPU and GPU share memory.",
        "fact",
        [EvidenceInput(source.id, "The CPU and GPU share one pool of memory", "supports")],
    )
    project.merge_concept("Unified memory", parent="Apple silicon", claim_ids=[claim.id])
    project.merge_concept("Apple silicon")
    reply = call(
        "submit_explanations", **written(beginner="Shared desk [claim-0001].").model_dump()
    )
    writer = ScriptedLLM([reply, reply])

    result = run_explain(project, writer, ["Unified memory", "Apple silicon", "nope"])

    first, parent, unknown = result["explained"]
    assert first["concept"] == "unified-memory" and first["citations"] == 1
    # The parent is explained from its children's claims, so it is written too.
    assert parent["concept"] == "apple-silicon"
    assert "unknown concept" in unknown["error"]
    stored = Project(project.root).get_concept("unified-memory").explanations
    assert stored["beginner"] == "Shared desk [claim-0001]."
    request = writer.requests[0][1].content or ""
    assert "- claim-0001 (fact, single-source): CPU and GPU share memory." in request
    assert "Learner's level: new" in request


def test_concepts_without_claims_are_not_explained(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    project.merge_concept("Lonely")

    result = run_explain(project, ScriptedLLM([]), ["Lonely"])

    assert "nothing was explained" in result["error"]
    assert "study sources first" in result["error"]
