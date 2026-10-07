from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from researchos.web import FetchError, HttpFetcher
from researchos.web.fetch import MAX_BYTES, ensure_public_http_url

PUBLIC = "http://93.184.215.14"  # literal public IP: no DNS needed

ARTICLE = """<html><head><title>Gate-all-around transistors</title></head><body>
<nav>menu menu menu</nav>
<article><h1>Gate-all-around transistors</h1>
<p>In a gate-all-around transistor the gate surrounds the channel on all sides, which gives
better electrostatic control than a FinFET and reduces leakage at small dimensions.</p>
<p>Nanosheets are stacked horizontal channels used to implement this structure in
modern process nodes.</p></article></body></html>"""


def fetcher(handler: Callable[[httpx.Request], httpx.Response]) -> HttpFetcher:
    return HttpFetcher(http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://93.184.215.14/x",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://user:pw@93.184.215.14/",
    ],
)
def test_rejects_non_public_or_non_http_urls(url: str) -> None:
    with pytest.raises(FetchError):
        ensure_public_http_url(url)


def test_extracts_main_text_and_title_from_html() -> None:
    page = fetcher(lambda _: httpx.Response(200, html=ARTICLE)).fetch(f"{PUBLIC}/gaa")

    assert page.title == "Gate-all-around transistors"
    assert "surrounds the channel on all sides" in page.text
    assert "menu menu" not in page.text


def test_redirect_to_private_address_is_blocked() -> None:
    redirect = httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
    with pytest.raises(FetchError, match="non-public"):
        fetcher(lambda _: redirect).fetch(f"{PUBLIC}/start")


def test_follows_public_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, text="plain body", headers={"content-type": "text/plain"})

    page = fetcher(handler).fetch(f"{PUBLIC}/old")

    assert page.url == f"{PUBLIC}/new"
    assert page.text == "plain body"


def test_rejects_unsupported_content_and_oversized_bodies() -> None:
    binary = httpx.Response(200, content=b"\x00", headers={"content-type": "image/png"})
    with pytest.raises(FetchError, match="unsupported content type"):
        fetcher(lambda _: binary).fetch(PUBLIC)

    huge = httpx.Response(
        200, content=b"a" * (MAX_BYTES + 1), headers={"content-type": "text/plain"}
    )
    with pytest.raises(FetchError, match="larger than"):
        fetcher(lambda _: huge).fetch(PUBLIC)


def test_http_errors_become_fetch_errors() -> None:
    with pytest.raises(FetchError, match="HTTP 404"):
        fetcher(lambda _: httpx.Response(404)).fetch(PUBLIC)
