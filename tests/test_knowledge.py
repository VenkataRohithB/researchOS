from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from helpers import ScriptedLLM, call, make_agent, tool_results

from researchos.knowledge import concept_id
from researchos.project import Project, ResearchRequest, UnknownConceptError
from researchos.state import RunStatus


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path, ResearchRequest(topic="t"))


def test_concept_ids_are_stable_slugs() -> None:
    assert concept_id("Gate-All-Around (GAA) Transistor") == "gate-all-around-gaa-transistor"
    assert concept_id("Café façade") == "cafe-facade"


def test_merge_adds_without_overwriting(project: Project) -> None:
    project.merge_concept("GAA transistor", summary="First summary.", related=["FinFET"])

    change = project.merge_concept(
        "gaa transistor",
        summary="A different summary.",
        related=["Nanosheet"],
        aliases=["Gate-all-around"],
    )

    concept = change.concept
    assert not change.created
    assert concept.summary == "First summary."
    assert concept.related == ["finfet", "nanosheet"]
    assert project.get_concept("Gate-all-around").id == "gaa-transistor"
    assert {c.id for c in project.concepts()} == {"gaa-transistor", "finfet", "nanosheet"}


def test_parent_links_cannot_form_cycles(project: Project) -> None:
    project.merge_concept("Nanosheet", parent="GAA transistor")
    project.merge_concept("GAA transistor", parent="Transistor")

    change = project.update_concept("Transistor", parent="Nanosheet")

    assert "would form a cycle" in change.problems[0]
    assert project.get_concept("transistor").parent is None
    assert project.update_concept("Nanosheet", parent="").concept.parent is None
    assert "cannot be its own parent" in project.merge_concept("Lone", parent="Lone").problems[0]


def test_concepts_persist(project: Project) -> None:
    project.merge_concept("Unified memory", summary="Shared pool.", parent="Apple silicon")
    project.set_explanations("unified-memory", {"beginner": "Like one shared desk."})

    reopened = Project(project.root)

    concept = reopened.get_concept("unified-memory")
    assert (concept.parent, concept.explanations) == (
        "apple-silicon",
        {"beginner": "Like one shared desk."},
    )
    with pytest.raises(UnknownConceptError):
        reopened.get_concept("nothing")


def test_concept_tools_show_and_correct_the_graph(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            call("update_concept", concept="nanosheet", parent="Gate-all-around"),
            call("get_concept", concept="Gate-all-around"),
            call("list_concepts"),
            call("get_concept", concept="nope"),
            call("finish_research", summary="done"),
        ]
    )
    agent, project = make_agent(tmp_path, llm)
    project.merge_concept("Nanosheet", summary="A thin channel.", claim_ids=["claim-0001"])

    agent.run()

    updated, shown, listed, missing = tool_results(llm.requests[-1])[-4:]
    assert updated["parent"] == "gate-all-around"
    assert shown["children"] == ["nanosheet"]
    assert listed["total"] == 2
    assert "unknown concept 'nope'" in missing["error"]
    snapshot = llm.requests[-1][1].content or ""
    assert "- Gate-all-around (0 claims)\n  - Nanosheet (1 claims)" in snapshot


class FakeUser:
    def __init__(self, answers: list[str] | None) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def ask(self, questions: Sequence[str]) -> list[str] | None:
        self.asked += questions
        return self.answers


def _ask_run(
    tmp_path: Path, user: FakeUser, calls: int = 1
) -> tuple[list[dict[str, Any]], Project]:
    questions = ["Which JEV do you mean: the virus or the vehicle?"]
    llm = ScriptedLLM(
        [call("ask_user", questions=questions)] * calls + [call("finish_research", summary="d")]
    )
    agent, project = make_agent(tmp_path, llm, user=user)
    assert agent.run().status is RunStatus.COMPLETED
    return tool_results(llm.requests[-1]), project


def test_answers_reach_the_agent_and_stay_in_its_context(tmp_path: Path) -> None:
    user = FakeUser(["The Japanese encephalitis virus"])

    (result,), project = _ask_run(tmp_path, user)

    assert result["answered"] is True
    assert result["answers"][0]["answer"] == "The Japanese encephalitis virus"
    assert user.asked == ["Which JEV do you mean: the virus or the vehicle?"]
    state = Project(project.root).state
    assert state.clarifications[0].answer == "The Japanese encephalitis virus"


def test_unattended_runs_proceed_on_assumptions(tmp_path: Path) -> None:
    (result,), _ = _ask_run(tmp_path, FakeUser(None))

    assert result["answered"] is False
    assert "assumptions" in result["note"]


def test_questions_per_project_are_limited(tmp_path: Path) -> None:
    results, _ = _ask_run(tmp_path, FakeUser(["a"]), calls=10)

    assert "question limit reached" in json.dumps(results[-1])
