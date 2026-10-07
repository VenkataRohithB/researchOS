from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import ScriptedLLM, agent_for, call, make_agent, tool_results

from researchos.context import Transcript
from researchos.llm import Message, MockLLM
from researchos.project import Project, ProjectLockedError, ResearchRequest
from researchos.state import Phase, RunStatus


def reopen(project: Project) -> Project:
    """Load the project from disk, as a new process would."""
    return Project(project.root)


def test_state_and_transcript_are_persisted(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, MockLLM())
    agent.run()

    state = reopen(project).state
    assert state.status is RunStatus.COMPLETED
    assert state.step == 5
    assert state.agenda[0].text == "Understand the basics of test topic"
    assert state.summary == "Mock research on 'test topic' complete."
    assert state.runs[0].usage is not None and state.runs[0].usage.llm_calls == 5
    assert len(Transcript(project.transcript_path)) == 5


def test_resume_continues_where_the_previous_run_stopped(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, MockLLM(), max_steps=2)
    assert agent.run().status is RunStatus.BUDGET_EXHAUSTED

    project = reopen(project)
    result = agent_for(project, MockLLM(), run_id="second").run()

    assert result.status is RunStatus.COMPLETED
    assert result.usage.steps == 3  # fetch, note, finish: plan and search were not redone
    state = reopen(project).state
    assert [r.status for r in state.runs] == [RunStatus.BUDGET_EXHAUSTED, RunStatus.COMPLETED]
    assert state.step == 5
    assert len(project.sources()) == 1


def test_run_left_running_by_a_crash_is_marked_interrupted(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="test topic"))
    project.state.begin_run("crashed")
    project.save_state()

    agent_for(reopen(project), MockLLM(), run_id="next").run()

    runs = reopen(project).state.runs
    assert [r.status for r in runs] == [RunStatus.INTERRUPTED, RunStatus.COMPLETED]


def test_ctrl_c_saves_state_as_interrupted(tmp_path: Path) -> None:
    llm = ScriptedLLM([call("update_agenda", add=["learn x"]), KeyboardInterrupt()])
    agent, project = make_agent(tmp_path, llm)

    with pytest.raises(KeyboardInterrupt):
        agent.run()

    state = reopen(project).state
    assert state.status is RunStatus.INTERRUPTED
    assert [item.text for item in state.agenda] == ["learn x"]


def test_a_project_cannot_be_run_by_two_processes_at_once(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, MockLLM())

    with project.lock(), pytest.raises(ProjectLockedError):
        agent.run()


def test_context_is_rebuilt_from_state_with_a_bounded_window(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [call("search_web", query=f"q{i}") for i in range(5)]
        + [call("finish_research", summary="done")]
    )
    agent, project = make_agent(tmp_path, llm, context_turns=2)
    project.add_directive("Focus on memory architecture")

    agent.run()

    last = llm.requests[-1]
    assert [m.role for m in last] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    snapshot = last[1].content or ""
    assert snapshot.startswith("Topic: test topic")
    assert "Focus on memory architecture" in snapshot
    assert "Step 6 of at most 20" in snapshot


def test_agenda_and_phase_tools_update_state(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            call("update_agenda", add=["what is X", "how X works"]),
            call(
                "update_agenda",
                update=[{"id": "a1", "status": "done", "note": "covered"}],
                add=["X vs Y"],
            ),
            call("update_agenda", update=[{"id": "a99", "status": "done"}]),
            call("set_phase", phase="cross_checking", reason="verifying key claims"),
            call("finish_research", summary="done"),
        ]
    )
    agent, project = make_agent(tmp_path, llm)

    agent.run()

    state = reopen(project).state
    assert [(i.id, i.status) for i in state.agenda] == [
        ("a1", "done"),
        ("a2", "open"),
        ("a3", "open"),
    ]
    assert state.phase is Phase.CROSS_CHECKING
    errors = [r for r in tool_results(llm.requests[-1]) if "error" in r]
    assert "unknown agenda item(s): a99" in errors[0]["error"]
    assert "a1 [done] what is X — covered" in (llm.requests[-1][1].content or "")


def test_read_notes_pages_through_notes(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            call("save_note", text="first"),
            call("save_note", text="second"),
            call("read_notes", offset=1, limit=5),
            call("finish_research", summary="done"),
        ]
    )
    agent, _ = make_agent(tmp_path, llm)

    agent.run()

    page = tool_results(llm.requests[-1])[-1]
    assert page["total"] == 2
    assert [n["text"] for n in page["notes"]] == ["second"]


def test_transcript_ignores_a_partially_written_last_line(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, ScriptedLLM([call("finish_research", summary="x")]))
    agent.run()
    with project.transcript_path.open("a") as f:
        f.write('{"step": 2, "messa')

    assert len(Transcript(project.transcript_path)) == 1


def test_source_titles_in_snapshot_are_untrusted(tmp_path: Path) -> None:
    url = "https://mock.researchos.invalid/test%20topic/1"
    llm = ScriptedLLM([call("fetch_source", url=url), call("finish_research", summary="x")])
    agent, _ = make_agent(tmp_path, llm)

    agent.run()

    snapshot = llm.requests[-1][1].content or ""
    sources_section = snapshot.split("## Sources fetched (1)\n", 1)[1]
    assert sources_section.startswith("<<<UNTRUSTED_CONTENT origin=source-titles>>>")


def test_nudge_is_part_of_the_recorded_turn(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [Message(role="assistant", content="hmm"), call("finish_research", summary="x")]
    )
    agent, project = make_agent(tmp_path, llm)

    agent.run()

    first_turn = json.loads(project.transcript_path.read_text().splitlines()[0])
    assert [m["role"] for m in first_turn["messages"]] == ["assistant", "user"]
