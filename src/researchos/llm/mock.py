"""Offline stand-in for a real model.

`MockLLM` does not reason. It follows a fixed plan -> search -> fetch -> claim -> close -> finish
trajectory, reading the previous tool results so it exercises the real harness end to end
(tool validation, persistence, budget accounting) without credentials or network access.
"""

from __future__ import annotations

import json
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
        names = [call.name for m in messages if m.role == "assistant" for call in m.tool_calls]
        called = set(names)
        last_result = _last_tool_result(messages)
        topic = _topic(messages)

        if last_result is not None and "error" in last_result:
            # Usually the pre-finish review; this script cannot act on errors, so it finishes.
            message = self._call("finish_research", summary=f"Mock research on '{topic}' complete.")
        elif "update_agenda" not in called:
            message = self._call("update_agenda", add=[f"Understand the basics of {topic}"])
        elif "search_web" not in called:
            message = self._call("search_web", query=topic, max_results=3)
        elif "fetch_source" not in called:
            results = (last_result or {}).get("results") or []
            if not results:
                message = self._call("finish_research", summary="No search results were found.")
            else:
                message = self._call("fetch_source", url=results[0]["url"])
        elif "record_claim" not in called:
            source_id = (last_result or {}).get("source_id")
            message = self._call(
                "record_claim",
                text=f"The fetched page about {topic} is synthetic test content.",
                kind="fact",
                evidence=[{"source_id": source_id, "quote": MOCK_PAGE_QUOTE}] if source_id else [],
            )
        elif names.count("update_agenda") < 2 and "claim_id" in (last_result or {}):
            covered = {"id": "a1", "status": "done", "claim_ids": [(last_result or {})["claim_id"]]}
            message = self._call("update_agenda", update=[covered])
        else:
            message = self._call("finish_research", summary=f"Mock research on '{topic}' complete.")

        return LLMResponse(
            message=message, usage=_estimate_usage(messages, message), model=self.model
        )

    def close(self) -> None:
        pass

    def _call(self, name: str, **arguments: Any) -> Message:
        call = ToolCall(id=f"mock-{next(self._ids)}", name=name, arguments=json.dumps(arguments))
        return Message(role="assistant", tool_calls=(call,))


def _topic(messages: Sequence[Message]) -> str:
    first_user = next((m.content for m in messages if m.role == "user" and m.content), "")
    return first_user.splitlines()[0].removeprefix("Topic:").strip() if first_user else ""


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
