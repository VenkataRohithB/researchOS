"""Evidence rules: how sources are identified and classified, how quotes are checked against
source text, and how a claim's status follows from its evidence.

Everything here is deterministic. The model proposes claims and quotes; these rules decide
whether the evidence holds up, so a claim's status never rests on the model's word.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

SourceTier = Literal[
    "paper", "academic", "standard", "government", "official", "reference", "news",
    "community", "other",
]  # fmt: skip
Stance = Literal["supports", "contradicts", "qualifies"]
ClaimKind = Literal["fact", "interpretation", "inference"]
ClaimStatus = Literal["verified", "single_source", "disputed", "unsupported"]

MIN_QUOTE_WORDS = 5
MAX_QUOTE_WORDS = 120


# Sources --------------------------------------------------------------------------------

_TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid"}
)


def canonical_url(url: str) -> str:
    """Normalise a URL so trivially different links to one page compare equal: lowercase
    scheme and host, no default port, fragment or tracking parameters, sorted query, and no
    trailing slash."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = host if parts.port in (None, default_port) else f"{host}:{parts.port}"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
        )
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


_SECOND_LEVEL = frozenset({"co", "ac", "gov", "edu", "org", "com", "net", "nic", "res"})


def site_of(url: str) -> str:
    """The registrable domain of a URL, e.g. docs.python.org -> python.org,
    www.cam.ac.uk -> cam.ac.uk. Used to judge whether two sources are independent."""
    # ponytail: heuristic instead of the Public Suffix List; misjudges rare multi-part
    # suffixes. Swap in a PSL library if independence decisions start depending on them.
    labels = (urlsplit(url).hostname or url).lower().removeprefix("www.").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


_PAPER_SITES = frozenset(
    {
        "arxiv.org", "doi.org", "acm.org", "springer.com", "nature.com", "sciencedirect.com",
        "aclanthology.org", "openreview.net", "neurips.cc", "semanticscholar.org", "nih.gov",
        "plos.org", "wiley.com", "pnas.org", "science.org", "jmlr.org", "mlr.press",
        "biorxiv.org", "medrxiv.org", "ssrn.com",
    }
)  # fmt: skip
_STANDARD_SITES = frozenset({"iso.org", "ietf.org", "rfc-editor.org", "w3.org", "whatwg.org"})
_REFERENCE_SITES = frozenset({"wikipedia.org", "britannica.com", "wiktionary.org"})
_COMMUNITY_SITES = frozenset(
    {
        "reddit.com", "stackoverflow.com", "stackexchange.com", "quora.com", "medium.com",
        "dev.to", "ycombinator.com", "substack.com", "hashnode.dev", "x.com", "twitter.com",
        "youtube.com", "facebook.com", "linkedin.com",
    }
)  # fmt: skip


def default_tier(url: str, topic: str | None = None) -> SourceTier:
    """A first guess at a source's reliability tier from its domain. A site named in the
    research topic (apple.com for "Apple Silicon") is treated as the subject's own, official
    source. The agent can override the guess."""
    host = (urlsplit(url).hostname or "").lower()
    site = site_of(url)
    if topic and site.split(".")[0] in words(topic):
        return "official"
    if site in _PAPER_SITES or host.startswith("ieeexplore.") or "arxiv" in host:
        return "paper"
    if site in _STANDARD_SITES or host == "standards.ieee.org":
        return "standard"
    if re.search(r"\.(gov|mil)(\.[a-z]{2})?$", host) or host.endswith("europa.eu"):
        return "government"
    if re.search(r"\.(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$", host):
        return "academic"
    if site in _REFERENCE_SITES:
        return "reference"
    if site in _COMMUNITY_SITES:
        return "community"
    return "other"


# Quotes and duplicates -------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Lowercased word tokens, ignoring punctuation, case and Unicode presentation forms, so
    a quote matches its source despite differences in quote marks, dashes or line breaks."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower())


class QuoteError(ValueError):
    """Why a quote cannot be accepted; the message is shown to the model."""


def check_quote(quote: str, source_text: str) -> None:
    """Raise `QuoteError` unless `quote` appears verbatim (modulo punctuation and case) in
    `source_text`."""
    if "..." in quote or "…" in quote:
        raise QuoteError("quote one contiguous passage; ellipses are not allowed")
    tokens = words(quote)
    if len(tokens) < MIN_QUOTE_WORDS:
        raise QuoteError(f"quote is too short; use at least {MIN_QUOTE_WORDS} words")
    if len(tokens) > MAX_QUOTE_WORDS:
        raise QuoteError(f"quote is too long; use at most {MAX_QUOTE_WORDS} words")
    if f" {' '.join(tokens)} " not in f" {' '.join(words(source_text))} ":
        raise QuoteError(
            "quote not found in the source text; copy it exactly from the source "
            "(use read_source to see the passage)"
        )


_SHINGLE = 6
_FINGERPRINT_WORDS = 4000
_MIN_SHINGLES = 40
DUPLICATE_THRESHOLD = 0.6


def fingerprint(text: str) -> frozenset[int]:
    """Hashed word 6-grams of the start of a document, for near-duplicate detection."""
    tokens = words(text)[:_FINGERPRINT_WORDS]
    return frozenset(
        zlib.crc32(" ".join(tokens[i : i + _SHINGLE]).encode())
        for i in range(len(tokens) - _SHINGLE + 1)
    )


def is_near_duplicate(a: frozenset[int], b: frozenset[int]) -> bool:
    """True when most of the shorter document also appears in the longer one, e.g. the same
    paper on arxiv.org and ar5iv.org, or a page and its print version."""
    if min(len(a), len(b)) < _MIN_SHINGLES:
        return False
    return len(a & b) / min(len(a), len(b)) >= DUPLICATE_THRESHOLD


# Claims ------------------------------------------------------------------------------------


class Evidence(BaseModel):
    source_id: str
    quote: str
    stance: Stance
    added_at: datetime


class Claim(BaseModel):
    id: str
    text: str
    kind: ClaimKind
    evidence: list[Evidence] = Field(default_factory=list)
    note: str | None = None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SourceIdentity:
    """What independence is judged on: the source's site and, for copies of the same
    document, the original it duplicates."""

    site: str
    original_id: str


@dataclass(frozen=True)
class Assessment:
    status: ClaimStatus
    supporting_sources: int
    """Independent sources that support the claim."""
    contradicting_sources: int


def assess(claim: Claim, identities: Mapping[str, SourceIdentity]) -> Assessment:
    """Derive a claim's status from its evidence. Sources count as independent only if they
    are on different sites and are not copies of the same document."""

    def independent(evidence: Iterable[Evidence]) -> int:
        groups: list[SourceIdentity] = []
        for item in evidence:
            identity = identities[item.source_id]
            if not any(
                g.site == identity.site or g.original_id == identity.original_id for g in groups
            ):
                groups.append(identity)
        return len(groups)

    supporting = independent(e for e in claim.evidence if e.stance == "supports")
    contradicting = independent(e for e in claim.evidence if e.stance == "contradicts")
    status: ClaimStatus
    if contradicting:
        status = "disputed"
    elif supporting >= 2:
        status = "verified"
    elif supporting == 1:
        status = "single_source"
    else:
        status = "unsupported"
    return Assessment(
        status=status, supporting_sources=supporting, contradicting_sources=contradicting
    )
