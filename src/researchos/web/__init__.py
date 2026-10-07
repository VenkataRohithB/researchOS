"""Web access: search providers and page fetching."""

from __future__ import annotations

from researchos.config import Settings
from researchos.web.fetch import FetchedPage, Fetcher, FetchError, HttpFetcher, MockFetcher
from researchos.web.search import (
    MockSearch,
    SearchError,
    SearchProvider,
    SearchResult,
    SearXNGSearch,
    TavilySearch,
)


def create_search_provider(settings: Settings) -> SearchProvider:
    match settings.search_provider:
        case "mock":
            return MockSearch()
        case "searxng":
            return SearXNGSearch(settings.searxng_url)
        case "tavily":
            return TavilySearch(settings.search_api_key)


def create_fetcher(settings: Settings) -> Fetcher:
    # Mock search returns mock URLs, so fetching must be mocked alongside it.
    if settings.search_provider == "mock":
        return MockFetcher()
    return HttpFetcher()


__all__ = [
    "FetchError",
    "FetchedPage",
    "Fetcher",
    "HttpFetcher",
    "MockFetcher",
    "MockSearch",
    "SearXNGSearch",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "TavilySearch",
    "create_fetcher",
    "create_search_provider",
]
