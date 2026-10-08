from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchos.focused import FocusedModel
from researchos.llm import ToolCall
from researchos.project import Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.user import NoUser
from researchos.web import FetchedPage, MockFetcher, MockSearch


class _Unused:
    model = "unused"

    def chat(self, messages: Any, tools: Any) -> Any:
        raise AssertionError("these tools make no model calls")

    def close(self) -> None:
        pass


WIRING: dict[str, Any] = {
    "focused": FocusedModel(
        _Unused(),
        UsageMeter(
            limits=Limits(max_steps=1, max_cost_usd=1, max_wall_seconds=1),
            pricing=Pricing(input_per_million=0, output_per_million=0),
            events_path=Path("/dev/null"),
            run_id="t",
        ),
    ),
    "user": NoUser(),
}

TEXT_A = (
    "Unified memory lets the CPU and GPU share one pool of memory.\n"
    "This avoids copying data between separate memory pools.\n\n"
    "Bandwidth reaches 120 GB/s on the base chip."
)
TEXT_B = "Apple says the CPU and GPU share one pool of memory on every M-series chip."


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path, ResearchRequest(topic="Unified memory"))
    project.add_source(FetchedPage(url="https://a.example/um", title="A", text=TEXT_A))
    project.add_source(FetchedPage(url="https://b.example/um", title="B", text=TEXT_B))
    return project


def ids(project: Project) -> tuple[str, str]:
    a, b = project.sources()
    return a.id, b.id


def run(project: Project, name: str, **arguments: Any) -> dict[str, Any]:
    tools = ToolRegistry(build_research_tools(project, MockSearch(), MockFetcher(), **WIRING))
    outcome = tools.execute(ToolCall(id="c", name=name, arguments=json.dumps(arguments)))
    result: dict[str, Any] = json.loads(outcome.content)
    return result


def test_record_claim_then_corroborate_it(project: Project) -> None:
    a, b = ids(project)
    quote = "the CPU and GPU share one pool of memory"

    recorded = run(
        project, "record_claim", text="CPU and GPU share memory",
        evidence=[{"source_id": a, "quote": quote}],
    )  # fmt: skip
    corroborated = run(
        project, "add_evidence", claim_id=recorded["claim_id"],
        evidence=[{"source_id": b, "quote": quote}],
    )  # fmt: skip

    assert recorded["status"] == "single_source"
    assert corroborated["status"] == "verified"
    assert corroborated["independent_supporting_sources"] == 2


def test_rejected_quote_explains_how_to_fix_it(project: Project) -> None:
    a, _ = ids(project)

    result = run(
        project, "record_claim", text="x",
        evidence=[{"source_id": a, "quote": "memory is shared by every chip ever made"}],
    )  # fmt: skip

    assert "claim not recorded" in result["error"]
    assert "use read_source" in result["error"]
    assert project.claims() == []


def test_a_fact_needs_evidence_but_an_inference_does_not(project: Project) -> None:
    assert "needs at least one quoted passage" in run(project, "record_claim", text="x")["error"]
    assert run(project, "record_claim", text="y", kind="inference")["status"] == "unsupported"


def test_contradicting_evidence_marks_a_claim_disputed(project: Project) -> None:
    a, b = ids(project)
    qualifying = {
        "source_id": a,
        "quote": "avoids copying data between separate memory pools",
        "stance": "qualifies",
    }
    contradicting = {
        "source_id": b,
        "quote": "share one pool of memory on every M-series chip",
        "stance": "contradicts",
    }
    claim = run(
        project,
        "record_claim",
        text="Only some M-series chips share memory",
        kind="interpretation",
        evidence=[qualifying],
    )

    result = run(project, "add_evidence", claim_id=claim["claim_id"], evidence=[contradicting])

    assert result["status"] == "disputed"


def test_get_claim_shows_evidence_side_by_side_as_untrusted(project: Project) -> None:
    a, b = ids(project)
    quote = "the CPU and GPU share one pool of memory"
    claim = run(
        project, "record_claim", text="shared memory",
        evidence=[{"source_id": a, "quote": quote}, {"source_id": b, "quote": quote}],
    )  # fmt: skip

    shown = run(project, "get_claim", claim_id=claim["claim_id"])

    assert [e["source_id"] for e in shown["evidence"]] == [a, b]
    assert all(e["quote"].startswith("<<<UNTRUSTED_CONTENT") for e in shown["evidence"])
    assert shown["evidence"][0]["tier"] == "other"


def test_search_sources_ranks_passages_by_matching_words(project: Project) -> None:
    a, b = ids(project)

    result = run(project, "search_sources", query="memory bandwidth pool")
    only_b = run(project, "search_sources", query="memory", source_ids=[b])

    passages = [(p["source_id"], p["offset"]) for p in result["passages"]]
    # Two passages match two words each; the bandwidth paragraph matches one and ranks last.
    assert passages[:2] == [(a, 0), (b, 0)]
    assert passages[2] == (a, TEXT_A.index("Bandwidth"))
    assert [p["source_id"] for p in only_b["passages"]] == [b]
    assert (
        "unknown source_id" in run(project, "search_sources", query="x", source_ids=["no"])["error"]
    )


def test_assess_source_overrides_the_default_tier(project: Project) -> None:
    _, b = ids(project)

    result = run(
        project, "assess_source", source_id=b, tier="official", rationale="Apple's own docs"
    )

    assert result["tier"] == "official"
    assert project.get_source(b).tier_rationale == "Apple's own docs"


class PagesFetcher:
    """Serves fixed pages by URL."""

    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url=url, title=url, text=self._pages[url])

    def close(self) -> None:
        pass


def test_fetching_a_copy_of_a_source_warns_that_it_is_not_independent(tmp_path: Path) -> None:
    article = " ".join(f"Paragraph {i} explains unified memory on Apple chips." for i in range(80))
    fetcher = PagesFetcher(
        {"https://arxiv.org/abs/1": article, "https://mirror.example/1": "Mirror. " + article}
    )
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    tools = ToolRegistry(build_research_tools(project, MockSearch(), fetcher, **WIRING))

    def fetch(url: str) -> dict[str, Any]:
        outcome = tools.execute(
            ToolCall(id="c", name="fetch_source", arguments=json.dumps({"url": url}))
        )
        result: dict[str, Any] = json.loads(outcome.content)
        return result

    original = fetch("https://arxiv.org/abs/1")
    copy = fetch("https://mirror.example/1")

    assert "warning" not in original
    assert copy["duplicate_of"] == original["source_id"]
    assert "not count as an independent source" in copy["warning"]
