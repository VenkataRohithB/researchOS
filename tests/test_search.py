from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from researchos.web import SearchError, SearchResult, SearXNGSearch, TavilySearch

Handler = Callable[[httpx.Request], httpx.Response]


def searxng(handler: Handler) -> SearXNGSearch:
    return SearXNGSearch(
        "http://127.0.0.1:8888/",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_searxng_queries_json_api_and_parses_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "RAG", "url": "https://example.org/rag", "content": "About RAG"},
                    {"title": "Bad", "url": "javascript:alert(1)", "content": "x"},
                    {"title": "Paper", "url": "https://arxiv.org/abs/2005.11401"},
                    {"title": "Third", "url": "https://example.org/3", "content": "c"},
                ],
                "unresponsive_engines": [["duckduckgo", "CAPTCHA"]],
            },
        )

    results = searxng(handler).search("what is rag", max_results=2)

    assert seen[0].url.path == "/search"
    assert seen[0].url.params["q"] == "what is rag"
    assert seen[0].url.params["format"] == "json"
    assert results == [
        SearchResult(title="RAG", url="https://example.org/rag", snippet="About RAG"),
        SearchResult(title="Paper", url="https://arxiv.org/abs/2005.11401", snippet=""),
    ]


def test_searxng_reports_blocked_engines_when_nothing_comes_back() -> None:
    body = {"results": [], "unresponsive_engines": [["google", "CAPTCHA"], ["brave", "timeout"]]}

    with pytest.raises(SearchError, match="engines unavailable: google, brave"):
        searxng(lambda _: httpx.Response(200, json=body)).search("q", max_results=5)


def test_searxng_empty_results_without_engine_errors_is_not_an_error() -> None:
    body: dict[str, list[str]] = {"results": [], "unresponsive_engines": []}

    assert searxng(lambda _: httpx.Response(200, json=body)).search("q", max_results=5) == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(403), "HTTP 403"),
        (httpx.Response(200, text="<html>not json</html>"), "unexpected response"),
    ],
)
def test_searxng_failures_become_search_errors(response: httpx.Response, message: str) -> None:
    with pytest.raises(SearchError, match=message):
        searxng(lambda _: response).search("q", max_results=5)


def test_searxng_unreachable_service_is_a_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SearchError, match="unreachable"):
        searxng(handler).search("q", max_results=5)


def test_tavily_sends_key_and_parses_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"results": [{"title": "T", "url": "https://t.example", "content": "s"}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = TavilySearch(SecretStr("tvly-key"), http_client=client).search("q", max_results=3)

    assert seen[0].headers["authorization"] == "Bearer tvly-key"
    assert results == [SearchResult(title="T", url="https://t.example", snippet="s")]
