from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from researchos.evidence import (
    Claim,
    Evidence,
    QuoteError,
    SourceIdentity,
    assess,
    canonical_url,
    check_quote,
    default_tier,
    fingerprint,
    is_near_duplicate,
    site_of,
)
from researchos.project import EvidenceError, EvidenceInput, Project, ResearchRequest
from researchos.web import FetchedPage

ARTICLE = " ".join(
    f"Sentence {i} explains how gate all around transistors control leakage at small nodes."
    for i in range(60)
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "HTTPS://Example.ORG:443/a/b/?utm_source=x&b=2&a=1#frag",
            "https://example.org/a/b?a=1&b=2",
        ),
        ("http://example.org:8080/", "http://example.org:8080/"),
        ("https://example.org/page?fbclid=abc", "https://example.org/page"),
        ("https://example.org", "https://example.org/"),
    ],
)
def test_canonical_url(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


@pytest.mark.parametrize(
    ("url", "site"),
    [
        ("https://docs.python.org/3/", "python.org"),
        ("https://www.cam.ac.uk/research", "cam.ac.uk"),
        ("https://ar5iv.labs.arxiv.org/html/2312.10997", "arxiv.org"),
        ("https://bbc.co.uk/news", "bbc.co.uk"),
    ],
)
def test_site_of(url: str, site: str) -> None:
    assert site_of(url) == site


@pytest.mark.parametrize(
    ("url", "tier"),
    [
        ("https://arxiv.org/abs/2005.11401", "paper"),
        ("https://ar5iv.labs.arxiv.org/html/2312.10997", "paper"),
        ("https://www.rfc-editor.org/rfc/rfc9110", "standard"),
        ("https://www.nist.gov/topics", "government"),
        ("https://cs.stanford.edu/people", "academic"),
        ("https://www.iitb.ac.in/page", "academic"),
        ("https://en.wikipedia.org/wiki/RAG", "reference"),
        ("https://www.reddit.com/r/x", "community"),
        ("https://www.apple.com/newsroom", "other"),
    ],
)
def test_default_tier(url: str, tier: str) -> None:
    assert default_tier(url) == tier


def test_quotes_match_despite_case_punctuation_and_line_breaks() -> None:
    source = "The “gate-all-around” design — introduced at 2nm —\nimproves electrostatic control."

    check_quote("the gate all around design, introduced at 2nm, improves", source)


@pytest.mark.parametrize(
    ("quote", "message"),
    [
        ("design introduced at 2nm improves electrostatic", "not found"),
        ("gate all around", "too short"),
        ("the gate-all-around design ... improves electrostatic control", "ellipses"),
        (" ".join(["word"] * 121), "too long"),
    ],
)
def test_quotes_that_are_not_verbatim_are_rejected(quote: str, message: str) -> None:
    source = "The gate-all-around design introduced at 3nm improves electrostatic control."
    with pytest.raises(QuoteError, match=message):
        check_quote(quote, source)


def test_quote_must_match_whole_words() -> None:
    with pytest.raises(QuoteError):
        check_quote("ate all around design introduced", "the gate all around design introduced")


def test_near_duplicates_are_detected_but_different_documents_are_not() -> None:
    copy_with_chrome = "Skip to content. Menu. " + ARTICLE + " Footer links."
    unrelated = " ".join(f"Line {i} describes a recipe for bread and butter." for i in range(60))

    assert is_near_duplicate(fingerprint(ARTICLE), fingerprint(copy_with_chrome))
    assert not is_near_duplicate(fingerprint(ARTICLE), fingerprint(unrelated))
    assert not is_near_duplicate(fingerprint("short page"), fingerprint("short page"))


def _claim(*evidence: tuple[str, str]) -> Claim:
    now = datetime.now(UTC)
    return Claim(
        id="claim-0001",
        text="t",
        kind="fact",
        evidence=[
            Evidence(source_id=sid, quote="q", stance=stance, added_at=now)  # type: ignore[arg-type]
            for sid, stance in evidence
        ],
        created_at=now,
        updated_at=now,
    )


IDENTITIES = {
    "a": SourceIdentity(site="arxiv.org", original_id="a"),
    "a-copy": SourceIdentity(site="ar5iv.org", original_id="a"),
    "a-sibling": SourceIdentity(site="arxiv.org", original_id="a-sibling"),
    "b": SourceIdentity(site="ieee.org", original_id="b"),
}


@pytest.mark.parametrize(
    ("evidence", "status", "supporting"),
    [
        ([("a", "supports"), ("b", "supports")], "verified", 2),
        ([("a", "supports"), ("a-copy", "supports")], "single_source", 1),
        ([("a", "supports"), ("a-sibling", "supports")], "single_source", 1),
        ([("a", "supports"), ("b", "contradicts")], "disputed", 1),
        ([("a", "qualifies")], "unsupported", 0),
        ([], "unsupported", 0),
    ],
)
def test_status_counts_only_independent_sources(
    evidence: list[tuple[str, str]], status: str, supporting: int
) -> None:
    result = assess(_claim(*evidence), IDENTITIES)

    assert result.status == status
    assert result.supporting_sources == supporting


# Project storage -------------------------------------------------------------------------


def make_project(tmp_path: Path) -> Project:
    return Project.create(tmp_path, ResearchRequest(topic="t"))


def test_sources_are_keyed_by_canonical_url_and_duplicates_marked(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    original = project.add_source(
        FetchedPage(url="https://arxiv.org/abs/1?utm_source=x", title="Paper", text=ARTICLE)
    )
    refetched = project.add_source(
        FetchedPage(url="https://ARXIV.org/abs/1/", title="Paper", text=ARTICLE)
    )
    mirror = project.add_source(
        FetchedPage(url="https://ar5iv.org/html/1", title="Paper (HTML)", text="Nav. " + ARTICLE)
    )

    assert refetched.id == original.id
    assert len(project.sources()) == 2
    assert original.url == "https://arxiv.org/abs/1"
    assert mirror.duplicate_of == original.id
    assert Project(project.root).get_source(mirror.id).duplicate_of == original.id


def test_agent_assigned_tier_survives_a_refetch(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    page = FetchedPage(url="https://www.apple.com/newsroom/m4", title="M4", text="text")
    source = project.add_source(page)
    project.set_source_tier(source.id, "official", "Apple describing its own chip")

    refreshed = project.add_source(page)

    assert (refreshed.tier, refreshed.tier_rationale) == (
        "official",
        "Apple describing its own chip",
    )


def test_claims_reject_all_evidence_when_any_item_is_invalid(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    source = project.add_source(FetchedPage(url="https://a.example/x", title="A", text=ARTICLE))
    good = EvidenceInput(source.id, "explains how gate all around transistors control", "supports")
    bad = EvidenceInput(source.id, "a sentence that the page never contained", "supports")
    unknown = EvidenceInput("src-nope", "explains how gate all around transistors", "supports")

    with pytest.raises(EvidenceError) as exc_info:
        project.add_claim("claim", "fact", [good, bad, unknown])

    assert "evidence 2" in str(exc_info.value) and "evidence 3" in str(exc_info.value)
    assert project.claims() == []


def test_adding_independent_evidence_verifies_a_claim(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    first = project.add_source(FetchedPage(url="https://a.example/x", title="A", text=ARTICLE))
    second = project.add_source(
        FetchedPage(
            url="https://b.example/y",
            title="B",
            text="Engineers agree that gate all around transistors control leakage at small nodes.",
        )
    )
    quote = "gate all around transistors control leakage at small nodes"
    claim = project.add_claim(
        "GAA reduces leakage", "fact", [EvidenceInput(first.id, quote, "supports")]
    )
    assert project.assess(claim).status == "single_source"

    claim = project.add_evidence(claim.id, [EvidenceInput(second.id, quote, "supports")])
    claim = project.add_evidence(claim.id, [EvidenceInput(second.id, quote, "supports")])

    assert len(claim.evidence) == 2  # the repeated item is not added twice
    assert project.assess(Project(project.root).get_claim(claim.id)).status == "verified"


def test_legacy_notes_are_migrated_to_claims(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    (project.root / "knowledge" / "claims.json").unlink()
    (project.root / "notes").mkdir()
    note = {
        "id": "note-0001",
        "text": "Old finding",
        "source_ids": ["src-1"],
        "created_at": "2026-10-07T10:00:00+00:00",
    }
    (project.root / "notes" / "notes.jsonl").write_text(json.dumps(note) + "\n")

    (claim,) = Project(project.root).claims()

    assert (claim.id, claim.text, claim.evidence) == ("claim-0001", "Old finding", [])
    assert claim.note is not None and "src-1" in claim.note


def test_a_site_named_in_the_topic_is_official() -> None:
    assert default_tier("https://developer.apple.com/metal", "Apple Silicon memory") == "official"
    assert default_tier("https://docs.python.org/3/", "Python asyncio internals") == "official"
    assert default_tier("https://www.howtogeek.com/x", "Apple Silicon memory") == "other"
    # When the subject is the site itself, its pages are official even if usually papers.
    assert default_tier("https://arxiv.org/help", "How arXiv moderation works") == "official"
    assert default_tier("https://arxiv.org/abs/1", "Retrieval-augmented generation") == "paper"
