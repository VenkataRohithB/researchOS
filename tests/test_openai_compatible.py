from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from researchos.llm import (
    LLMAuthorizationError,
    LLMResponseError,
    LLMUnavailableError,
    Message,
    OpenAICompatibleClient,
    ToolCall,
    ToolSpec,
)

TOOL = ToolSpec(name="search_web", description="d", parameters={"type": "object"})
OK_BODY = {
    "model": "served-model",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": '{"query": "x"}'},
                    }
                ],
            }
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def client(
    handler: Callable[[httpx.Request], httpx.Response], sleeps: list[float] | None = None
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://llm.example/v1/",
        api_key=SecretStr("sk-secret"),
        model="test-model",
        timeout_seconds=5,
        max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_sends_openai_request_and_parses_tool_calls() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK_BODY)

    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=(ToolCall("c0", "search_web", "{}"),)),
        Message(role="tool", content="{}", tool_call_id="c0"),
    ]
    response = client(handler).chat(history, [TOOL])

    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-secret"
    assert body["model"] == "test-model"
    assert body["tools"][0]["function"]["name"] == "search_web"
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert body["messages"][2]["tool_call_id"] == "c0"
    assert response.message.tool_calls == (ToolCall("c1", "search_web", '{"query": "x"}'),)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 3)
    assert response.model == "served-model"


def test_retries_rate_limits_honouring_retry_after() -> None:
    responses = [
        httpx.Response(429, headers={"retry-after": "7"}),
        httpx.Response(200, json=OK_BODY),
    ]
    sleeps: list[float] = []

    client(lambda _: responses.pop(0), sleeps).chat([Message(role="user", content="x")], [])

    assert sleeps == [7.0]


def test_gives_up_after_max_retries() -> None:
    sleeps: list[float] = []
    with pytest.raises(LLMUnavailableError):
        client(lambda _: httpx.Response(503), sleeps).chat([Message(role="user", content="x")], [])
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_authorization_failures_are_not_retried(status: int) -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "budget exhausted"}})

    with pytest.raises(LLMAuthorizationError, match="budget exhausted") as exc_info:
        client(handler).chat([Message(role="user", content="x")], [])
    assert len(calls) == 1
    assert "sk-secret" not in str(exc_info.value)


def test_malformed_response_is_rejected() -> None:
    with pytest.raises(LLMResponseError):
        client(lambda _: httpx.Response(200, json={"choices": []})).chat(
            [Message(role="user", content="x")], []
        )


def test_error_wrapped_in_a_list_is_unwrapped() -> None:
    body = [{"error": {"code": 403, "message": "project denied"}}]
    with pytest.raises(LLMAuthorizationError, match=r"HTTP 403: project denied$"):
        client(lambda _: httpx.Response(403, json=body)).chat(
            [Message(role="user", content="x")], []
        )


def test_provider_specific_tool_call_fields_round_trip() -> None:
    signature = {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    body = json.loads(json.dumps(OK_BODY))
    body["choices"][0]["message"]["tool_calls"][0]["extra_content"] = signature
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    llm = client(handler)
    first = llm.chat([Message(role="user", content="x")], [TOOL])
    llm.chat([Message(role="user", content="x"), first.message], [TOOL])

    assert first.message.tool_calls[0].provider_data == {"extra_content": signature}
    echoed = sent[1]["messages"][1]["tool_calls"][0]
    assert echoed["extra_content"] == signature
    assert echoed["function"]["name"] == "search_web"
