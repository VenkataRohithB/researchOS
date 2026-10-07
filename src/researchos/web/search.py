"""Web search providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx
from pydantic import SecretStr

TAVILY_URL = "https://api.tavily.com/search"
MOCK_HOST = "mock.researchos.invalid"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(Exception):
    """Search failed; the message is safe to show to the model."""


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...

    def close(self) -> None: ...


class TavilySearch:
    def __init__(
        self,
        api_key: SecretStr,
        *,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._http = http_client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._http.close()

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            response = self._http.post(
                TAVILY_URL,
                json={"query": query, "max_results": max_results, "search_depth": "basic"},
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            )
        except httpx.TransportError as exc:
            raise SearchError(f"search request failed: {exc.__class__.__name__}") from exc
        if response.status_code != 200:
            raise SearchError(f"search provider returned HTTP {response.status_code}")
        try:
            items = response.json()["results"]
            return [
                SearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item["url"]),
                    snippet=str(item.get("content") or ""),
                )
                for item in items
            ][:max_results]
        except (ValueError, KeyError, TypeError) as exc:
            raise SearchError("search provider returned an unexpected response") from exc


class MockSearch:
    """Deterministic offline results. URLs point at `MOCK_HOST`, served by `MockFetcher`."""

    def close(self) -> None:
        pass

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"{query} — overview part {i}",
                url=f"https://{MOCK_HOST}/{quote(query, safe='')}/{i}",
                snippet=f"Mock search result {i} for '{query}'.",
            )
            for i in range(1, max_results + 1)
        ]
