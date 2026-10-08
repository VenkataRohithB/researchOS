from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

from researchos.project import EvidenceInput, Project, ResearchRequest
from researchos.site import build_site, collect
from researchos.site.layout import ROOT, learning_order, radial_map
from researchos.state import RunStatus
from researchos.usage import UsageSummary
from researchos.web import FetchedPage

PAPER = (
    "In a gate-all-around transistor the gate wraps around the channel on all sides. "
    "Stacked horizontal nanosheet channels replace fins."
)
OTHER = "Unlike FinFETs, nanosheets are stacked horizontal silicon channels."
USAGE = UsageSummary(
    steps=3, llm_calls=3, tool_calls=3, tool_errors=0, input_tokens=1200, output_tokens=300,
    cost_usd=0.0123, elapsed_seconds=75,
)  # fmt: skip


def make_project(tmp_path: Path, topic: str = "Gate-all-around transistors") -> Project:
    project = Project.create(tmp_path, ResearchRequest(topic=topic, goal="Understand GAA"))
    paper = project.add_source(
        FetchedPage(url="https://www.example.org/gaa", title="GAA", text=PAPER)
    )
    other = project.add_source(FetchedPage(url="https://ieee.example/ns", title="NS", text=OTHER))
    wrap = project.add_claim(
        "The gate surrounds the channel on **all sides**.",
        "fact",
        [EvidenceInput(paper.id, "the gate wraps around the channel on all sides", "supports")],
    )
    stack = project.add_claim(
        "Nanosheets are stacked horizontal channels.",
        "fact",
        [
            EvidenceInput(
                other.id, "nanosheets are stacked horizontal silicon channels", "supports"
            ),
            EvidenceInput(
                paper.id, "Stacked horizontal nanosheet channels replace fins", "supports"
            ),
        ],
    )
    project.merge_concept("GAA transistor", summary="A transistor whose gate wraps the channel.",
                          prerequisites=["FinFET"], claim_ids=[wrap.id])  # fmt: skip
    project.merge_concept("Nanosheet", parent="GAA transistor", claim_ids=[stack.id])
    project.merge_concept("FinFET", summary="The previous transistor design.")
    project.set_explanations(
        "gaa-transistor",
        {
            "beginner": "Like a hand gripping a hose [claim-0001]. See [[Nanosheet]] and "
            "[[Quantum tunnelling]]. Bogus [claim-0099].",
            "expert": "## Detail\n\nElectrostatics improve [claim-0001, claim-0002].",
        },
    )
    project.state.add_agenda_items(["What is GAA?"])
    project.state.set_agenda_status("a1", "done", "Covered by claim-0001.", ["claim-0001"])
    project.state.summary = "GAA wraps the gate around the channel [claim-0001]."
    project.state.begin_run("run-1")
    project.state.end_run(RunStatus.COMPLETED, "done", USAGE)
    project.save_state()
    return project


def page(project: Project) -> str:
    return build_site(project).read_text(encoding="utf-8")


def embedded(html: str) -> dict[str, Any]:
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert match is not None
    data: dict[str, Any] = json.loads(match.group(1))
    return data


def test_data_describes_concepts_claims_sources_and_path(tmp_path: Path) -> None:
    data = embedded(page(make_project(tmp_path)))

    gaa = data["concepts"]["gaa-transistor"]
    assert gaa["children"] == ["nanosheet"] and gaa["prerequisites"] == ["finfet"]
    assert set(gaa["levels"]) == {"beginner", "expert"}
    assert data["concepts"]["nanosheet"]["parent"] == "gaa-transistor"
    # GAA before its sub-concept; FinFET (its prerequisite) has no claims or explanations,
    # so it is left off the path.
    assert data["path"] == ["gaa-transistor", "nanosheet"]
    assert data["claims"]["claim-0002"]["status"] == "verified"
    assert data["claims"]["claim-0002"]["statusLabel"] == "Verified by 2 independent sources"
    assert data["claims"]["claim-0001"]["concepts"] == ["gaa-transistor"]
    assert [s["number"] for s in data["sources"]] == [1, 2]
    assert data["sources"][0]["claims"] == ["claim-0001", "claim-0002"]
    assert data["statusCounts"] == {
        "verified": 1,
        "single_source": 1,
        "disputed": 0,
        "unsupported": 0,
    }
    assert data["agenda"][0]["claims"] == ["claim-0001"]


def test_citations_become_evidence_chips_and_links_navigate(tmp_path: Path) -> None:
    concepts = embedded(page(make_project(tmp_path)))["concepts"]
    beginner = concepts["gaa-transistor"]["levels"]["beginner"]

    assert 'class="cite" data-claim="claim-0001"' in beginner
    assert "claim-0099" not in beginner
    assert '<a class="concept-link" href="#/concept/nanosheet">Nanosheet</a>' in beginner
    assert "Quantum tunnelling" in beginner and "[[" not in beginner


def test_untrusted_text_cannot_inject_markup_or_script(tmp_path: Path) -> None:
    project = make_project(tmp_path, topic="<script>alert('topic')</script>")
    project.merge_concept("</script><script>alert(1)</script>", summary='<img src=x onerror="x()">')
    project.state.summary = "[click](javascript:alert(3)) <b onclick=x>raw</b>"
    project.save_state()

    html = page(project)

    assert html.count("<script") == 2  # the data block and the page script
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(&#39;topic&#39;)&lt;/script&gt;" in html
    data = embedded(html)
    rogue = next(c for c in data["concepts"].values() if "alert(1)" in c["title"])
    assert rogue["summaryHtml"] == "&lt;img src=x onerror=&quot;x()&quot;&gt;"
    assert 'href="javascript:' not in data["summaryHtml"]
    assert "<b onclick" not in data["summaryHtml"]


def test_csp_pins_the_inline_style_and_script_by_hash(tmp_path: Path) -> None:
    html = page(make_project(tmp_path))

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


def test_unfinished_research_is_flagged(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.state.begin_run("run-2")
    project.state.end_run(RunStatus.BUDGET_EXHAUSTED, "step limit of 5 reached", USAGE)
    project.save_state()

    assert collect(project)["stopped"] == "Stopped at budget limit (step limit of 5 reached)"


def test_empty_project_builds(tmp_path: Path) -> None:
    data = embedded(page(Project.create(tmp_path, ResearchRequest(topic="Empty"))))

    assert data["concepts"] == {} and data["path"] == [] and data["summaryHtml"] is None
    assert [n["id"] for n in data["map"]["nodes"]] == [ROOT]


def test_map_places_subtrees_on_rings_without_overlap(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    for name in ("A", "B", "C"):
        project.merge_concept(f"Child {name}", parent="Nanosheet")

    concept_map = radial_map("Topic", project.concepts())

    nodes = {n.id: n for n in concept_map.nodes}
    distance = {i: round((n.x**2 + n.y**2) ** 0.5) for i, n in nodes.items()}
    assert distance[ROOT] == 0
    assert distance["gaa-transistor"] == distance["finfet"] == 190
    assert distance["nanosheet"] == 380 and distance["child-a"] == 570
    assert len({(n.x, n.y) for n in concept_map.nodes}) == len(concept_map.nodes)
    kinds = {(e.source, e.target, e.kind) for e in concept_map.edges}
    assert ("finfet", "gaa-transistor", "prerequisite") in kinds
    assert ("gaa-transistor", "nanosheet", "part") in kinds


def test_learning_order_survives_prerequisite_cycles(tmp_path: Path) -> None:
    project = Project.create(tmp_path, ResearchRequest(topic="t"))
    project.merge_concept("A", prerequisites=["B"])
    project.merge_concept("B", prerequisites=["A"])
    for concept_ref in ("a", "b"):
        project.set_explanations(concept_ref, {"summary": "s"})

    assert sorted(learning_order(project.concepts())) == ["a", "b"]
