from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import re
from pathlib import Path

from researchos.project import EvidenceInput, Project, ResearchRequest
from researchos.site import build_site
from researchos.state import RunStatus
from researchos.usage import UsageSummary
from researchos.web import FetchedPage

PAPER_TEXT = (
    "In a gate-all-around transistor the gate wraps around the channel on all sides. "
    "In these devices stacked horizontal nanosheet channels replace fins."
)
OTHER_TEXT = "Unlike FinFETs, nanosheets are stacked horizontal silicon channels."

USAGE = UsageSummary(
    steps=3,
    llm_calls=3,
    tool_calls=3,
    tool_errors=0,
    input_tokens=1200,
    output_tokens=300,
    cost_usd=0.0123,
    elapsed_seconds=75,
)


def make_project(tmp_path: Path, *, topic: str = "Gate-all-around transistors") -> Project:
    project = Project.create(tmp_path, ResearchRequest(topic=topic, goal="Understand GAA"))
    paper = project.add_source(
        FetchedPage(url="https://www.example.org/gaa", title="GAA explained", text=PAPER_TEXT)
    )
    other = project.add_source(
        FetchedPage(url="https://ieee.example/nanosheet", title="Nanosheets", text=OTHER_TEXT)
    )
    project.add_claim(
        "The gate surrounds the channel on **all four sides**.",
        "fact",
        [EvidenceInput(paper.id, "the gate wraps around the channel on all sides", "supports")],
    )
    project.add_claim(
        "Nanosheets stack horizontal channels.",
        "fact",
        [
            EvidenceInput(
                other.id, "nanosheets are stacked horizontal silicon channels", "supports"
            ),
            EvidenceInput(
                paper.id, "stacked horizontal nanosheet channels replace fins", "supports"
            ),
        ],
    )
    project.state.add_agenda_items(["What is GAA?", "Why nanosheets?"])
    project.state.set_agenda_status("a1", "done", "Covered by note-0001 and note-0002.")
    project.state.set_agenda_status("a2", "done", None, ["claim-0002"])
    project.state.summary = (
        "# GAA\n\nIntro.\n\n## How it works\n\nDetails [docs](https://x.example)."
    )
    project.state.begin_run("run-1")
    project.state.end_run(RunStatus.COMPLETED, "done", USAGE)
    project.save_state()
    return project


def build(project: Project) -> str:
    return build_site(project).read_text(encoding="utf-8")


def test_renders_findings_sources_agenda_and_log(tmp_path: Path) -> None:
    html = build(make_project(tmp_path))

    assert "<title>Gate-all-around transistors</title>" in html
    assert "<strong>all four sides</strong>" in html
    # Sources are numbered by first citation; the second finding cites both.
    assert re.search(r'id="source-1".*?GAA explained', html, re.S)
    assert re.search(r'id="source-2".*?Nanosheets', html, re.S)
    assert 'href="#source-2" aria-label="Source 2: Nanosheets"' in html
    assert 'Cited in findings <a href="#claim-0001">1</a>, <a href="#claim-0002">2</a>' in html
    # Agenda notes link to findings by the label readers see; legacy note ids map to claims.
    assert '<a href="#claim-0001">Finding 1</a> and <a href="#claim-0002">Finding 2</a>' in html
    # Status comes from evidence: two independent sites verify the second claim.
    assert "Single source" in html
    assert "Verified by 2 independent sources" in html
    assert "1 of them is confirmed by independent sources." in html
    assert "<blockquote>nanosheets are stacked horizontal silicon channels</blockquote>" in html
    assert "Web page from example.org" in html
    assert "2 of 2 research questions covered" in html
    assert 'Covered by <a href="#claim-0002">Finding 2</a>.' in html
    assert "1,500" in html and "$0.0123" in html and "1m 15s" in html


def test_summary_headings_get_anchors_listed_in_contents(tmp_path: Path) -> None:
    html = build(make_project(tmp_path))

    assert '<h3 id="gaa">GAA</h3>' in html
    assert '<h4 id="how-it-works">How it works</h4>' in html
    assert '<a href="#how-it-works">How it works</a>' in html
    assert 'href="https://x.example" target="_blank" rel="noopener noreferrer"' in html


def test_untrusted_content_cannot_inject_markup_or_script(tmp_path: Path) -> None:
    project = make_project(tmp_path, topic="<script>alert('topic')</script>")
    project.add_claim(
        '<img src=x onerror="alert(1)"> </script><script>alert(2)</script>', "inference", []
    )
    project.state.summary = "[click](javascript:alert(3)) <b onclick=x>raw</b>"
    project.state.set_agenda_status("a2", "open", '<a href="javascript:alert(4)">x</a>')
    project.save_state()

    html = build(project)

    assert "<script>alert" not in html
    assert "<img src=x" not in html
    assert "<b onclick" not in html
    assert 'href="javascript:' not in html
    assert "&lt;script&gt;alert(&#39;topic&#39;)&lt;/script&gt;" in html
    # Exactly two script elements: the search data and the page script.
    assert html.count("<script") == 2
    data = re.search(r'<script type="application/json" id="search-data">(.*?)</script>', html, re.S)
    assert data is not None
    assert any("</script>" in entry["text"] for entry in json.loads(data.group(1)))


def test_csp_pins_the_inline_style_and_script_by_hash(tmp_path: Path) -> None:
    html = build(make_project(tmp_path))

    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    style = re.search(r"<style>(.*?)</style>", html, re.S)
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert csp and style and script

    def digest(text: str) -> str:
        return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"

    policy = html_lib.unescape(csp.group(1))  # as the browser reads the attribute
    assert "default-src 'none'" in policy
    assert f"style-src {digest(style.group(1))}" in policy
    assert f"script-src {digest(script.group(1))}" in policy


def test_unfinished_research_says_how_to_continue(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.state.begin_run("run-2")
    project.state.end_run(RunStatus.BUDGET_EXHAUSTED, "step limit of 5 reached", USAGE)
    project.save_state()

    html = build(project)

    assert "This research is unfinished." in html
    assert "Stopped at budget limit (step limit of 5 reached)" in html
    assert f"researchos resume {project.metadata.id}" in html


def test_empty_project_builds_with_guidance(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="Empty"))

    html = build(project)

    assert "No findings yet." in html
    assert "No sources yet." in html
    assert 'id="summary"' not in html
