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
_MAX_SERVER_DELAY_SECONDS = 120.0
_STANDARD_CALL_KEYS = frozenset({"id", "type", "function"})


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
            delay = (
                min(delay, _MAX_SERVER_DELAY_SECONDS)
                if delay is not None
                else min(2.0**attempt, _MAX_BACKOFF_SECONDS)
            )
            logger.warning("LLM request failed (%s); retrying in %.1fs", failure, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")


def _encode_message(message: Message) -> dict[str, Any]:
    encoded: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        encoded["tool_calls"] = [
            {
                **(call.provider_data or {}),
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
                provider_data={k: v for k, v in call.items() if k not in _STANDARD_CALL_KEYS}
                or None,
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
    """The server's requested wait: the Retry-After header, or Google's RetryInfo
    (`error.details[].retryDelay`, e.g. "59s") in the error body."""
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return max(0.0, float(header))
        except ValueError:
            return None
    error = _error_object(response)
    details = error.get("details") if error else None
    for detail in details if isinstance(details, list) else ():
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            delay = str(detail.get("retryDelay", ""))
            try:
                return max(0.0, float(delay.removesuffix("s")))
            except ValueError:
                return None
    return None


def _error_detail(response: httpx.Response, limit: int = 300) -> str:
    error = _error_object(response)
    message = error.get("message") if error else None
    text = str(message) if message else response.text
    return text[:limit] or response.reason_phrase


def _error_object(response: httpx.Response) -> dict[str, Any] | None:
    """The `error` object of an error response, if the body has one."""
    try:
        body = response.json()
    except ValueError:
        return None
    # Some providers (e.g. Gemini) wrap the error object in a one-element list.
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else None
