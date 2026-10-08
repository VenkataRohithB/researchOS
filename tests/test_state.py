from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import ScriptedLLM, agent_for, call, make_agent, tool_results

from researchos.context import Transcript
from researchos.llm import Message, MockLLM, ToolCall
from researchos.project import Project, ProjectLockedError, ResearchRequest
from researchos.state import Phase, RunStatus


def reopen(project: Project) -> Project:
    """Load the project from disk, as a new process would."""
    return Project(project.root)


def test_state_and_transcript_are_persisted(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, MockLLM(), grounded=False)
    agent.run()

    state = reopen(project).state
    assert state.status is RunStatus.COMPLETED
    assert state.agenda[0].text == "Understand the basics of test topic"
    assert state.agenda[0].status == "done"
    assert state.summary == "Mock research on 'test topic' complete."
    assert state.breadth_reviewed and state.reviewed_claims == ["claim-0001"]
    # Every agent step is a transcript turn; tools made one study and one explain call.
    assert len(Transcript(project.transcript_path)) == state.step
    assert state.runs[0].usage is not None
    assert state.runs[0].usage.llm_calls == state.step + 2


def test_resume_continues_where_the_previous_run_stopped(tmp_path: Path) -> None:
    agent, project = make_agent(tmp_path, MockLLM(), grounded=False, max_steps=2)
    assert agent.run().status is RunStatus.BUDGET_EXHAUSTED

    project = reopen(project)
    result = agent_for(project, MockLLM(), run_id="second").run()

    assert result.status is RunStatus.COMPLETED
    state = reopen(project).state
    assert [r.status for r in state.runs] == [RunStatus.BUDGET_EXHAUSTED, RunStatus.COMPLETED]
    assert state.step == 2 + result.usage.steps
    # The resumed run continued: planning and searching were not repeated.
    calls = [
        c.name for turn in Transcript(project.transcript_path).recent(99) for m in turn.messages
        for c in m.tool_calls
    ]  # fmt: skip
    assert calls.count("update_agenda") == 2  # the plan, then closing it
    assert calls.count("search_web") == 1
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
                update=[
                    {"id": "a1", "status": "done", "note": "covered", "claim_ids": ["claim-0001"]}
                ],
                add=["X vs Y"],
            ),
            call(
                "update_agenda",
                update=[
                    {"id": "a2", "status": "dropped", "note": "out of scope"},
                    {"id": "a3", "status": "dropped", "note": "no sources found"},
                ],
            ),
            call("set_phase", phase="cross_checking", reason="verifying key claims"),
            call("finish_research", summary="done"),
        ]
    )
    agent, project = make_agent(tmp_path, llm)

    # The seeded claim is verified by three sites, so finishing needs no review.
    assert agent.run().status is RunStatus.COMPLETED

    state = reopen(project).state
    assert [(i.id, i.status, i.claim_ids) for i in state.agenda] == [
        ("a1", "done", ["claim-0001"]),
        ("a2", "dropped", []),
        ("a3", "dropped", []),
    ]
    assert state.phase is Phase.CROSS_CHECKING
    assert "a1 [done] what is X (claims: claim-0001) — covered" in (
        llm.requests[-1][1].content or ""
    )


def test_agenda_items_close_only_with_quoted_claims_and_finish_needs_none_open(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(
        [
            call("update_agenda", add=["what is X", "how X works"]),
            call("record_claim", text="my guess", kind="inference"),
            call("update_agenda", update=[{"id": "a1", "status": "done"}]),
            call(
                "update_agenda",
                update=[{"id": "a1", "status": "done", "claim_ids": ["claim-0002"]}],
            ),
            call(
                "update_agenda",
                update=[{"id": "a1", "status": "done", "claim_ids": ["claim-0099"]}],
            ),
            call("update_agenda", update=[{"id": "a2", "status": "dropped"}]),
            call("update_agenda", update=[{"id": "a99", "status": "done"}]),
            call("finish_research", summary="done"),
            call("list_claims"),  # lets the test observe the rejected finish
        ]
    )
    agent, project = make_agent(tmp_path, llm, max_steps=9)

    assert agent.run().status is RunStatus.BUDGET_EXHAUSTED

    errors = [r["error"] for r in tool_results(llm.requests[-1]) if "error" in r]
    assert "list the claim_ids that cover it" in errors[0]
    assert "claim-0002 has no supporting quote" in errors[1]
    assert "unknown claim 'claim-0099'" in errors[2]
    assert "give a reason in note to drop it" in errors[3]
    assert "unknown agenda item(s): a99" in errors[4]
    assert "agenda items still open: a1, a2" in errors[5]
    assert all(item.status == "open" for item in reopen(project).state.agenda)


def test_list_claims_pages_and_filters_by_status(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            call("record_claim", text="first", kind="inference"),
            call("record_claim", text="second", kind="interpretation"),
            call("list_claims", offset=1, limit=5),
            call("list_claims", status="unsupported"),
            call("finish_research", summary="done"),
        ]
    )
    agent, _ = make_agent(tmp_path, llm)

    agent.run()

    paged, unsupported = tool_results(llm.requests[-1])[-2:]
    assert paged["total"] == 3
    assert [c["text"] for c in paged["claims"]] == ["first", "second"]
    assert [c["text"] for c in unsupported["claims"]] == ["first", "second"]


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
    sources_section = snapshot.split("## Sources fetched (4)\n", 1)[1]
    assert sources_section.startswith("<<<UNTRUSTED_CONTENT origin=source-titles>>>")


def test_nudge_is_part_of_the_recorded_turn(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [Message(role="assistant", content="hmm"), call("finish_research", summary="x")]
    )
    agent, project = make_agent(tmp_path, llm)

    agent.run()

    first_turn = json.loads(project.transcript_path.read_text().splitlines()[0])
    assert [m["role"] for m in first_turn["messages"]] == ["assistant", "user"]


def test_transcript_preserves_provider_data_on_tool_calls(tmp_path: Path) -> None:
    signed = Message(
        role="assistant",
        tool_calls=(
            ToolCall(
                id="c1",
                name="finish_research",
                arguments='{"summary": "x"}',
                provider_data={"extra_content": {"google": {"thought_signature": "sig"}}},
            ),
        ),
    )
    agent, project = make_agent(tmp_path, ScriptedLLM([signed]))
    agent.run()

    (turn,) = Transcript(project.transcript_path).recent(1)
    assert turn.messages[0].tool_calls == signed.tool_calls
