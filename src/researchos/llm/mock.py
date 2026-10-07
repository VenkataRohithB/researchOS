"""Offline stand-in for a real model.

`MockLLM` does not reason. It follows a fixed search -> fetch -> note -> finish trajectory,
reading the previous tool results so it exercises the real harness end to end (tool
validation, persistence, budget accounting) without credentials or network access.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import count
from typing import Any

from researchos.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

_CHARS_PER_TOKEN = 4


class MockLLM:
    model = "mock"

    def __init__(self) -> None:
        self._ids = count(1)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        called = {call.name for m in messages if m.role == "assistant" for call in m.tool_calls}
        last_result = _last_tool_result(messages)

        if last_result is not None and "error" in last_result:
            message = self._call("finish_research", summary="Stopped: a tool reported an error.")
        elif "search_web" not in called:
            message = self._call("search_web", query=_topic(messages), max_results=3)
        elif "fetch_source" not in called:
            results = (last_result or {}).get("results") or []
            if not results:
                message = self._call("finish_research", summary="No search results were found.")
            else:
                message = self._call("fetch_source", url=results[0]["url"])
        elif "save_note" not in called:
            source = last_result or {}
            message = self._call(
                "save_note",
                text=f"Key points from '{source.get('title', 'source')}'.",
                source_ids=[source["source_id"]] if "source_id" in source else [],
            )
        else:
            message = self._call(
                "finish_research", summary=f"Mock research on '{_topic(messages)}' complete."
            )

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
