"""Fetching and text extraction for web pages.

`HttpFetcher` only reaches public internet addresses: every URL, including each redirect hop,
is resolved and rejected if any address is private, loopback, link-local or otherwise
non-global. Responses are size-capped and only textual content types are accepted.
"""

from __future__ import annotations

import io
import ipaddress
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import trafilatura
from pypdf import PdfReader

from researchos.web.search import MOCK_HOST

MAX_BYTES = 15 * 1024 * 1024
MAX_REDIRECTS = 5
USER_AGENT = "ResearchOS/0.1 (+https://github.com/VenkataRohithB/researchOS)"
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_TYPES = frozenset({"text/plain", "text/markdown"})
_PDF_TYPE = "application/pdf"


@dataclass(frozen=True)
class FetchedPage:
    url: str
    """Final URL after redirects."""
    title: str
    text: str
    published: str | None = None
    """Publication date stated by the page (ISO format where known)."""


class FetchError(Exception):
    """Fetching failed; the message is safe to show to the model."""


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...

    def close(self) -> None: ...


def ensure_public_http_url(url: str) -> None:
    """Raise `FetchError` unless `url` is http(s) and every address its host resolves to is
    globally routable."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"unsupported URL scheme: {parts.scheme or '(none)'}")
    if not parts.hostname:
        raise FetchError("URL has no host")
    if parts.username or parts.password:
        raise FetchError("URLs with embedded credentials are not allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise FetchError(f"cannot resolve host {parts.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise FetchError(f"host {parts.hostname} resolves to a non-public address")


class HttpFetcher:
    # ponytail: the address check and the connection resolve DNS separately, so a DNS-rebinding
    # host could swap addresses in between; pin the checked IP in the transport if this ever
    # runs somewhere with sensitive internal services.
    def __init__(
        self, *, timeout_seconds: float = 30.0, http_client: httpx.Client | None = None
    ) -> None:
        self._http = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        self._http.close()

    def fetch(self, url: str) -> FetchedPage:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            ensure_public_http_url(current)
            try:
                with self._http.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("redirect without a Location header")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise FetchError(f"HTTP {response.status_code}")
                    content_type = _media_type(response.headers.get("content-type", ""))
                    body = _read_capped(response)
                    encoding = response.encoding or "utf-8"
            except httpx.TransportError as exc:
                raise FetchError(f"request failed: {exc.__class__.__name__}") from exc
            return _to_page(current, content_type, body, encoding)
        raise FetchError(f"more than {MAX_REDIRECTS} redirects")


class MockFetcher:
    """Serves synthetic pages for `MockSearch` URLs; refuses everything else."""

    def close(self) -> None:
        pass

    def fetch(self, url: str) -> FetchedPage:
        parts = urlsplit(url)
        if parts.hostname != MOCK_HOST:
            raise FetchError("mock fetcher only serves mock search results")
        topic, _, index = unquote(parts.path).strip("/").rpartition("/")
        return FetchedPage(
            url=url,
            title=f"{topic} — overview part {index}",
            text=(
                f"# {topic}\n\nThis is synthetic mock content (part {index}) about {topic}. "
                "It exists only to exercise the research harness offline and contains no "
                "real information."
            ),
        )


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _read_capped(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_BYTES:
            raise FetchError(f"response larger than {MAX_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _to_page(url: str, content_type: str, body: bytes, encoding: str) -> FetchedPage:
    if content_type == _PDF_TYPE or (not content_type and body.startswith(b"%PDF-")):
        return _pdf_page(url, body)
    text = body.decode(encoding, errors="replace")
    if content_type in _TEXT_TYPES:
        return FetchedPage(url=url, title=_name_from_url(url), text=text)
    if content_type not in _HTML_TYPES:
        raise FetchError(f"unsupported content type: {content_type or 'unknown'}")
    extracted = trafilatura.extract(text, url=url, include_comments=False, include_tables=True)
    if not extracted:
        raise FetchError("no readable text could be extracted from the page")
    metadata = trafilatura.extract_metadata(text, default_url=url)
    title = (metadata.title if metadata is not None else None) or url
    published = metadata.date if metadata is not None else None
    return FetchedPage(
        url=url,
        title=str(title),
        text=str(extracted),
        published=str(published) if published else None,
    )


def _pdf_page(url: str, body: bytes) -> FetchedPage:
    try:
        reader = PdfReader(io.BytesIO(body))
        pages = [page.extract_text() or "" for page in reader.pages]
        info = reader.metadata
    except Exception as exc:  # pypdf raises many exception types on malformed input
        raise FetchError(f"could not read the PDF: {exc.__class__.__name__}") from exc
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise FetchError("the PDF has no extractable text (it may be scanned images)")
    title = (info.title if info is not None else None) or _name_from_url(url)
    created = info.creation_date if info is not None else None
    published = created.date().isoformat() if created is not None else None
    return FetchedPage(url=url, title=str(title), text=text, published=published)


def _name_from_url(url: str) -> str:
    return unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or url
