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


def make_pdf(text: str, title: str) -> bytes:
    """A minimal valid single-page PDF with one line of text and document metadata."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title ({title}) /CreationDate (D:20240115093000Z) >>".encode(),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def test_extracts_text_title_and_date_from_pdf() -> None:
    pdf = make_pdf("Nanosheets improve gate control.", "GAA Paper")
    response = httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

    page = fetcher(lambda _: response).fetch(f"{PUBLIC}/paper.pdf")

    assert "Nanosheets improve gate control." in page.text
    assert page.title == "GAA Paper"
    assert page.published == "2024-01-15"


def test_broken_pdf_is_a_fetch_error() -> None:
    response = httpx.Response(
        200, content=b"%PDF-1.4 garbage", headers={"content-type": "application/pdf"}
    )

    with pytest.raises(FetchError, match=r"could not read the PDF|no extractable text"):
        fetcher(lambda _: response).fetch(f"{PUBLIC}/broken.pdf")


def test_html_publication_date_is_extracted() -> None:
    dated = ARTICLE.replace(
        "<head>", '<head><meta property="article:published_time" content="2025-03-04">'
    )

    page = fetcher(lambda _: httpx.Response(200, html=dated)).fetch(f"{PUBLIC}/gaa")

    assert page.published == "2025-03-04"
