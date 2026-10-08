"""Offline stand-in for a real model.

`MockLLM` does not reason. As the agent it follows a fixed plan -> search -> fetch -> study ->
explain -> close -> finish trajectory, deciding each step from the state snapshot. Asked to
study a source it returns one claim quoting the mock page; asked to explain a concept it
returns placeholder text citing the concept's claims. This exercises the real harness end to
end (tool validation, quote checks, persistence, budget accounting) without credentials or
network access.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from itertools import count
from typing import Any

from researchos.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

# A passage every page served by `MockFetcher` contains, so mock claims carry a real quote.
MOCK_PAGE_QUOTE = "It exists only to exercise the research harness offline"

_CHARS_PER_TOKEN = 4


class MockLLM:
    model = "mock"

    def __init__(self) -> None:
        self._ids = count(1)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        if any(tool.name == "submit_study" for tool in tools):
            return self._respond(messages, self._study(messages))
        if any(tool.name == "submit_explanations" for tool in tools):
            return self._respond(messages, self._explain(messages))
        return self._respond(messages, self._next_step(messages))

    def _next_step(self, messages: Sequence[Message]) -> Message:
        """Choose the next action from the state snapshot, as a real model would."""
        snapshot = next((m.content for m in messages if m.role == "user" and m.content), "")
        topic = snapshot.splitlines()[0].removeprefix("Topic:").strip() if snapshot else ""
        last_result = _last_tool_result(messages)
        open_items = re.findall(r"^- (a\d+) \[open\]", snapshot, re.MULTILINE)
        unstudied = re.findall(r"(src-[0-9a-f]+) \[[a-z]+, not studied yet", snapshot)
        claims = re.findall(r"^- (claim-\d+) \[", snapshot, re.MULTILINE)
        unexplained = re.findall(r"^\s*- (.+) \([1-9]\d* claims\)$", snapshot, re.MULTILINE)
        no_sources = "## Sources fetched (0)" in snapshot
        finish = self._call("finish_research", summary=f"Mock research on '{topic}' complete.")

        if last_result is not None and "error" in last_result:
            # Usually the pre-finish review; this script cannot act on errors, so it finishes.
            return finish
        if "## Agenda\nEmpty." in snapshot:
            return self._call("update_agenda", add=[f"Understand the basics of {topic}"])
        if no_sources:
            results = (last_result or {}).get("results")
            if results:
                return self._call("fetch_source", url=results[0]["url"])
            if results is not None:
                return self._call("finish_research", summary="No search results were found.")
            return self._call("search_web", query=topic, max_results=3)
        if unstudied:
            return self._call("study_source", source_id=unstudied[0])
        if unexplained:
            return self._call("explain_concepts", concepts=unexplained[:5])
        if open_items and claims:
            covered = [{"id": item, "status": "done", "claim_ids": claims} for item in open_items]
            return self._call("update_agenda", update=covered)
        return finish

    def close(self) -> None:
        pass

    def _study(self, messages: Sequence[Message]) -> Message:
        request = next((m.content for m in messages if m.role == "user" and m.content), "")
        topic = request.splitlines()[0].removeprefix("Research topic:").strip()
        return self._call(
            "submit_study",
            claims=[
                {
                    "text": f"The source about {topic} is synthetic test content.",
                    "quote": MOCK_PAGE_QUOTE,
                    "concepts": [topic],
                }
            ],
            concepts=[{"title": topic, "summary": f"The subject being researched: {topic}."}],
        )

    def _explain(self, messages: Sequence[Message]) -> Message:
        request = next((m.content for m in messages if m.role == "user" and m.content), "")
        cite = "[" + ", ".join(re.findall(r"^- (claim-\d+) \(", request, re.MULTILINE)) + "]"
        concept = re.search(r"^Concept: (.+)$", request, re.MULTILINE)
        name = concept.group(1) if concept else "the concept"
        return self._call(
            "submit_explanations",
            **{
                level: f"A {level} explanation of {name} for testing. {cite}"
                for level in ("summary", "beginner", "intermediate", "deep", "expert")
            },
        )

    def _respond(self, messages: Sequence[Message], reply: Message) -> LLMResponse:
        return LLMResponse(message=reply, usage=_estimate_usage(messages, reply), model=self.model)

    def _call(self, name: str, **arguments: Any) -> Message:
        call = ToolCall(id=f"mock-{next(self._ids)}", name=name, arguments=json.dumps(arguments))
        return Message(role="assistant", tool_calls=(call,))


def _last_tool_result(messages: Sequence[Message]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.role == "tool":
            try:
                parsed = json.loads(message.content or "")
            except ValueError:
                return {"error": "unparseable tool result"}
            return parsed if isinstance(parsed, dict) else None
    return None


def _estimate_usage(messages: Sequence[Message], reply: Message) -> Usage:
    prompt_chars = sum(len(m.content or "") for m in messages)
    reply_chars = sum(len(call.arguments) for call in reply.tool_calls)
    return Usage(
        input_tokens=prompt_chars // _CHARS_PER_TOKEN,
        output_tokens=reply_chars // _CHARS_PER_TOKEN,
    )
