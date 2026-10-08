"""Provider-neutral chat types shared by the agent and every LLM client."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """Raw JSON text produced by the model; validated by the tool registry, never trusted."""
    provider_data: dict[str, Any] | None = None
    """Extra provider-specific fields on the call (e.g. Gemini thought signatures in
    `extra_content`). Opaque to the agent; sent back to the provider unchanged."""


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default=())
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema of the tool's arguments."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    message: Message
    usage: Usage
    model: str


class LLMClient(Protocol):
    @property
    def model(self) -> str:
        """The model currently in use (it can change if the client falls back to another)."""
        ...

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse: ...

    def close(self) -> None: ...


class LLMError(Exception):
    """Base class for LLM client failures."""


class LLMUnavailableError(LLMError):
    """Transient failure (network, rate limit, 5xx) that persisted after all retries."""


class LLMAuthorizationError(LLMError):
    """The provider (or a gateway in front of it) refused the request: bad key, revoked
    access, exhausted budget. Retrying will not help."""


class LLMResponseError(LLMError):
    """The provider rejected the request or returned something that is not a valid reply."""
