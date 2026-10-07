"""Runtime configuration, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai_compatible", "mock"] = "mock"
    llm_model: str = "mock"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_price_in: float = Field(default=0.0, ge=0, description="USD per 1M input tokens")
    llm_price_out: float = Field(default=0.0, ge=0, description="USD per 1M output tokens")
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    search_provider: Literal["tavily", "mock"] = "mock"
    search_api_key: SecretStr = SecretStr("")

    max_steps: int = Field(default=150, gt=0)
    max_cost_usd: float = Field(default=5.0, gt=0)
    max_wall_seconds: float = Field(default=3600.0, gt=0)

    workspace_dir: Path = Path("workspace")

    @model_validator(mode="after")
    def _require_credentials(self) -> Self:
        if self.llm_provider != "mock" and not self.llm_api_key.get_secret_value():
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER is not 'mock'")
        if self.search_provider != "mock" and not self.search_api_key.get_secret_value():
            raise ValueError("SEARCH_API_KEY is required when SEARCH_PROVIDER is not 'mock'")
        return self
