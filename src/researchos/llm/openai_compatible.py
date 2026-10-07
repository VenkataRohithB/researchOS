"""Client for any server that speaks the OpenAI Chat Completions protocol with tool calling.

This covers hosted providers, LiteLLM, and HTTP gateways placed in front of them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from researchos.llm.types import (
    LLMAuthorizationError,
    LLMResponse,
    LLMResponseError,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_AUTHORIZATION_STATUS = frozenset({401, 402, 403})
_MAX_BACKOFF_SECONDS = 30.0


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._max_retries = max_retries
        self._http = http_client or httpx.Client(timeout=timeout_seconds)
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_encode_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [_encode_tool(t) for t in tools]
            payload["tool_choice"] = "auto"
        return _decode_response(self._post(payload), self.model)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        for attempt in range(self._max_retries + 1):
            delay: float | None = None
            try:
                response = self._http.post(self._url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                failure = f"transport error: {exc.__class__.__name__}"
            else:
                status = response.status_code
                if status < 400:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise LLMResponseError("response body is not valid JSON") from exc
                    if not isinstance(body, dict):
                        raise LLMResponseError("response body is not a JSON object")
                    return body
                detail = _error_detail(response)
                if status in _AUTHORIZATION_STATUS:
                    raise LLMAuthorizationError(f"HTTP {status}: {detail}")
                if status not in _RETRYABLE_STATUS:
                    raise LLMResponseError(f"HTTP {status}: {detail}")
                failure = f"HTTP {status}: {detail}"
                delay = _retry_after_seconds(response)

            if attempt == self._max_retries:
                raise LLMUnavailableError(f"giving up after {attempt + 1} attempts ({failure})")
            delay = min(delay if delay is not None else 2.0**attempt, _MAX_BACKOFF_SECONDS)
            logger.warning("LLM request failed (%s); retrying in %.1fs", failure, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")


def _encode_message(message: Message) -> dict[str, Any]:
    encoded: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        encoded["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        encoded["tool_call_id"] = message.tool_call_id
    return encoded


def _encode_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _decode_response(body: dict[str, Any], requested_model: str) -> LLMResponse:
    try:
        raw = body["choices"][0]["message"]
        tool_calls = tuple(
            ToolCall(
                id=str(call["id"]),
                name=str(call["function"]["name"]),
                arguments=str(call["function"].get("arguments") or "{}"),
            )
            for call in raw.get("tool_calls") or ()
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(f"unexpected response shape: {exc!r}") from exc

    usage = body.get("usage") or {}
    return LLMResponse(
        message=Message(role="assistant", content=raw.get("content"), tool_calls=tool_calls),
        usage=Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        ),
        model=str(body.get("model") or requested_model),
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _error_detail(response: httpx.Response, limit: int = 300) -> str:
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        text = str(message) if message else response.text
    except ValueError:
        text = response.text
    return text[:limit] or response.reason_phrase
