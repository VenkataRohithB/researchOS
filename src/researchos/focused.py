"""Focused model calls made by tools: one prompt in, one structured result out.

The main agent decides *what* to do; heavy, well-defined jobs such as reading a whole source or
writing explanations run as separate calls with their own prompt and a small, relevant
context. The result is returned through a single tool whose arguments are validated against a
pydantic model, so the output is structured without relying on provider-specific JSON modes
or forced tool choice (which some models reject).
"""

from __future__ import annotations

import time
from importlib import resources
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from researchos.llm import LLMClient, LLMError, Message, ToolSpec
from researchos.usage import UsageMeter

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 2


class FocusedCallError(Exception):
    """The model did not produce a valid result; the message is safe to show the agent."""


def load_prompt(name: str) -> str:
    return resources.files("researchos.prompts").joinpath(name).read_text("utf-8")


class FocusedModel:
    def __init__(self, llm: LLMClient, meter: UsageMeter) -> None:
        self._llm = llm
        self._meter = meter

    def submit(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        result_model: type[T],
    ) -> T:
        """Ask the model to call `tool_name` with arguments matching `result_model`. A missing
        or invalid call is answered with the error and retried once."""
        spec = ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters=result_model.model_json_schema(),
        )
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        problem = "no result"
        for _ in range(MAX_ATTEMPTS):
            started = time.monotonic()
            try:
                response = self._llm.chat(messages, [spec])
            except LLMError as exc:
                self._meter.record_llm_failure(
                    model=self._llm.model,
                    error=str(exc),
                    latency_seconds=time.monotonic() - started,
                    purpose=purpose,
                )
                raise FocusedCallError(f"the {purpose} model call failed: {exc}") from exc
            self._meter.record_llm_call(
                model=response.model,
                usage=response.usage,
                latency_seconds=time.monotonic() - started,
                purpose=purpose,
            )
            reply = response.message
            call = next((c for c in reply.tool_calls if c.name == tool_name), None)
            if call is not None:
                try:
                    return result_model.model_validate_json(call.arguments)
                except ValidationError as exc:
                    problem = f"invalid {tool_name} arguments: {exc.error_count()} errors"
                    feedback = exc.json(include_url=False, include_input=False)[:2000]
                    retry = f"Invalid arguments, fix them and call {tool_name} again: {feedback}"
            else:
                problem = f"the model did not call {tool_name}"
                retry = f"Call {tool_name} now with your complete result."
            # Every tool call needs a result before the conversation can continue.
            messages.append(reply)
            messages += [
                Message(
                    role="tool",
                    content=retry if c is call else f"Unknown tool. {retry}",
                    tool_call_id=c.id,
                )
                for c in reply.tool_calls
            ]
            if not reply.tool_calls:
                messages.append(Message(role="user", content=retry))
        raise FocusedCallError(f"{purpose} produced no valid result ({problem})")
