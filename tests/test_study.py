from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import ScriptedLLM, call
from pydantic import BaseModel

from researchos.focused import FocusedCallError, FocusedModel
from researchos.llm import Message, ToolCall
from researchos.project import EvidenceInput, Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.tools.study import STUDY_CHARS
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.user import NoUser
from researchos.web import FetchedPage, MockFetcher, MockSearch

APPLE_DOC = (
    "Apple silicon uses a unified memory architecture. The CPU and GPU share a single pool of "
    "high-bandwidth memory, so data does not need to be copied between them. The M4 chip "
    "supports up to 120 GB/s of memory bandwidth."
)
REVIEW_DOC = (
    "In unified memory designs the CPU and GPU share a single pool of memory. Reviewers "
    "measured the base M4 at close to 120 GB/s of memory bandwidth in practice."
)


def meter(project: Project) -> UsageMeter:
    return UsageMeter(
        limits=Limits(max_steps=10, max_cost_usd=10, max_wall_seconds=60),
        pricing=Pricing(input_per_million=0, output_per_million=0),
        events_path=project.events_path,
        run_id="r",
    )


def submit(**result: Any) -> Message:
    return call("submit_study", **result)


def study(project: Project, reader: ScriptedLLM, **arguments: Any) -> dict[str, Any]:
    tools = ToolRegistry(
        build_research_tools(
            project,
            MockSearch(),
            MockFetcher(),
            focused=FocusedModel(reader, meter(project)),
            user=NoUser(),
        )
    )
    outcome = tools.execute(ToolCall(id="c", name="study_source", arguments=json.dumps(arguments)))
    result: dict[str, Any] = json.loads(outcome.content)
    return result


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path, ResearchRequest(topic="Apple unified memory"))
    project.add_source(FetchedPage(url="https://www.apple.com/m4", title="M4", text=APPLE_DOC))
    return project


def test_study_records_quoted_claims_and_builds_the_concept_graph(project: Project) -> None:
    (source,) = project.sources()
    reader = ScriptedLLM(
        [
            submit(
                claims=[
                    {
                        "text": "Apple silicon's CPU and GPU share one pool of memory.",
                        "quote": "The CPU and GPU share a single pool of high-bandwidth memory",
                        "concepts": ["Unified memory architecture"],
                    },
                    {
                        "text": "The M4 supports 120 GB/s of memory bandwidth.",
                        "quote": "The M4 chip supports up to 120 GB/s of memory bandwidth.",
                        "concepts": ["Memory bandwidth"],
                    },
                    {"text": "Invented fact", "quote": "a sentence the document never says"},
                ],
                concepts=[
                    {
                        "title": "Unified memory architecture",
                        "summary": "One memory pool shared by all processors.",
                        "parent": "Apple silicon",
                        "prerequisites": ["System on a chip"],
                    },
                    {"title": "Memory bandwidth", "parent": "Unified memory architecture"},
                ],
                open_questions=["How does this compare with discrete GPUs?"],
            )
        ]
    )

    result = study(project, reader, source_id=source.id)

    assert result["claims_recorded"] == 2
    assert result["rejected_count"] == 1
    assert "quote not found" in result["rejected"][0]["reason"]
    assert result["next_offset"] is None
    assert result["open_questions"] == ["How does this compare with discrete GPUs?"]
    uma = project.get_concept("Unified memory architecture")
    assert uma.parent == "apple-silicon"
    assert uma.prerequisites == ["system-on-a-chip"]
    assert uma.claim_ids == ["claim-0001"]
    assert project.get_concept("Apple silicon").summary == ""  # created as a stub
    assert [c.id for c in project.children("apple-silicon")] == ["unified-memory-architecture"]
    assert project.get_concept("memory-bandwidth").parent == "unified-memory-architecture"
    # The reader was told what is already known and what to focus on.
    request = reader.requests[0][1].content or ""
    assert "Research topic: Apple unified memory" in request
    assert request.count("<<<UNTRUSTED_CONTENT") == 1


def test_studying_a_second_source_cross_checks_existing_claims(project: Project) -> None:
    (apple,) = project.sources()
    project.add_claim(
        "Apple silicon's CPU and GPU share one pool of memory.",
        "fact",
        [
            EvidenceInput(
                apple.id, "The CPU and GPU share a single pool of high-bandwidth memory", "supports"
            )
        ],
    )
    review = project.add_source(
        FetchedPage(url="https://reviews.example/m4", title="Review", text=REVIEW_DOC)
    )
    reader = ScriptedLLM(
        [
            submit(
                corroborations=[
                    {
                        "claim_id": "claim-0001",
                        "quote": "the CPU and GPU share a single pool of memory",
                    },
                    {
                        "claim_id": "claim-0099",
                        "quote": "the CPU and GPU share a single pool of memory",
                    },
                ],
                claims=[
                    {
                        # Same claim text as an existing one: recorded as corroboration.
                        "text": "Apple silicon's CPU and GPU share one pool of memory.",
                        "quote": "In unified memory designs the CPU and GPU share a single pool",
                    }
                ],
            )
        ]
    )

    result = study(project, reader, source_id=review.id)

    assert result["claims_recorded"] == 0
    assert [c["claim_id"] for c in result["corroborated"]] == ["claim-0001", "claim-0001"]
    assert result["corroborated"][0]["status"] == "verified"
    assert result["rejected"] == [{"claim": "claim-0099", "reason": "unknown claim id"}]
    assert len(project.get_claim("claim-0001").evidence) == 3


def test_long_sources_are_studied_in_passes(project: Project) -> None:
    long_text = "Unified memory removes copies between processors. " * 2000
    source = project.add_source(
        FetchedPage(url="https://long.example/doc", title="Long", text=long_text)
    )
    reader = ScriptedLLM([submit(), submit()])

    first = study(project, reader, source_id=source.id)
    second = study(project, reader, source_id=source.id, offset=first["next_offset"])

    assert first["next_offset"] == STUDY_CHARS
    assert second["next_offset"] == 2 * STUDY_CHARS
    assert "nothing to study" in study(project, reader, source_id=source.id, offset=10**7)["error"]


def test_study_reports_a_reader_that_never_submits(project: Project) -> None:
    (source,) = project.sources()
    reader = ScriptedLLM([Message(role="assistant", content="I read it.")] * 2)

    result = study(project, reader, source_id=source.id)

    assert "study produced no valid result" in result["error"]


class Answer(BaseModel):
    value: int


def test_focused_call_retries_after_invalid_arguments(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    llm = ScriptedLLM([call("answer", value="not a number"), call("answer", value=42)])
    model = FocusedModel(llm, meter(project))

    result = model.submit(
        purpose="test",
        system="s",
        user="u",
        tool_name="answer",
        tool_description="d",
        result_model=Answer,
    )

    assert result.value == 42
    retry = llm.requests[1]
    assert retry[-1].role == "tool" and "Invalid arguments" in (retry[-1].content or "")


def test_focused_call_answers_unknown_tool_calls_before_retrying(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    llm = ScriptedLLM([call("something_else"), call("answer", value=1)])

    FocusedModel(llm, meter(project)).submit(
        purpose="test", system="s", user="u", tool_name="answer", tool_description="d",
        result_model=Answer,
    )  # fmt: skip

    retry = llm.requests[1]
    assert [m.role for m in retry[-2:]] == ["assistant", "tool"]
    assert "Unknown tool" in (retry[-1].content or "")


def test_focused_call_gives_up_after_two_attempts(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    llm = ScriptedLLM([Message(role="assistant", content="no")] * 2)

    with pytest.raises(FocusedCallError, match="did not call answer"):
        FocusedModel(llm, meter(project)).submit(
            purpose="test", system="s", user="u", tool_name="answer", tool_description="d",
            result_model=Answer,
        )  # fmt: skip
