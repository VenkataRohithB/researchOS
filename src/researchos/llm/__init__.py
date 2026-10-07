"""LLM access. The agent depends only on `LLMClient`; concrete clients are chosen by config."""

from __future__ import annotations

from researchos.config import Settings
from researchos.llm.mock import MockLLM
from researchos.llm.openai_compatible import OpenAICompatibleClient
from researchos.llm.types import (
    LLMAuthorizationError,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLM()
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


__all__ = [
    "LLMAuthorizationError",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LLMResponseError",
    "LLMUnavailableError",
    "Message",
    "MockLLM",
    "OpenAICompatibleClient",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "create_llm_client",
]
