"""The project website: an interactive, self-contained learning site.

`build_site` writes `website/index.html`, one file that works offline. It embeds:

- the concept graph, with each concept's explanations at five depths,
- every claim with its status and quoted evidence,
- the sources, the research plan and the run log,
- a precomputed concept-map layout.

All text is rendered here: Markdown with raw HTML disabled, citations turned into evidence
chips and [[links]] into navigation. The page script only places this pre-rendered,
escaped markup and wires up navigation, so model- and web-derived text can never inject
markup or script. A Content-Security-Policy pins the inline style and script by hash.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markdown_it import MarkdownIt
from markdown_it.token import Token
from markupsafe import Markup, escape

from researchos.knowledge import LEVELS
from researchos.project import Project
from researchos.site.layout import ROOT, learning_order, radial_map
from researchos.state import RunStatus

_CITATION = re.compile(r"\[\s*(claim-\d{4}(?:\s*[,;]\s*claim-\d{4})*)\s*\]")
_CLAIM_ID = re.compile(r"claim-\d{4}")
_LINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_FINDING_REF = re.compile(r"\b(?:claim|note)-(\d{4})\b")
# Valid in JSON strings but line terminators in JavaScript source.
_LINE_SEPARATOR = chr(0x2028)
_PARAGRAPH_SEPARATOR = chr(0x2029)

LEVEL_LABELS = {
    "summary": "Summary",
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "deep": "Deep dive",
    "expert": "Expert",
}

TIER_LABELS = {
    "paper": "Research paper",
    "academic": "Academic",
    "standard": "Standard",
    "government": "Government",
    "official": "Official",
    "reference": "Reference work",
    "news": "News",
    "community": "Community",
    "other": "Web page",
}

STATUS_LABELS = {
    "verified": "Verified",
    "single_source": "Single source",
    "disputed": "Disputed",
    "unsupported": "Unsupported",
}

RUN_STATUS_LABELS = {
    RunStatus.RUNNING: "Running",
    RunStatus.COMPLETED: "Completed",
    RunStatus.INTERRUPTED: "Interrupted",
    RunStatus.BUDGET_EXHAUSTED: "Stopped at budget limit",
    RunStatus.STALLED: "Stopped: agent stalled",
    RunStatus.LLM_REFUSED: "Stopped: model access refused",
    RunStatus.LLM_FAILED: "Stopped: model unavailable",
}


def build_site(project: Project) -> Path:
    """Render the project's website and return the path of `index.html`."""
    html = render(collect(project), project.metadata.request.topic)
    out_dir = project.root / "website"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "index.html"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, path)
    return path


def collect(project: Project) -> dict[str, Any]:
    """Everything the page shows, as JSON-ready data with HTML fields pre-rendered."""
    request = project.metadata.request
    state = project.state
    concepts = project.concepts()
    claims = project.claims()
    sources = project.sources()
    md = _markdown()

    claim_numbers = {c.id: n for n, c in enumerate(claims, start=1)}
    concept_ids = {c.id for c in concepts}
    titles = {c.title.lower(): c.id for c in concepts}
    for concept in concepts:
        for alias in concept.aliases:
            titles.setdefault(alias.lower(), concept.id)

    def rich(markdown_text: str, *, inline: bool = False) -> str:
        html = md.renderInline(markdown_text) if inline else md.render(markdown_text)
        html = _CITATION.sub(lambda m: _chips(m.group(1), claim_numbers), html)
        return _LINK.sub(lambda m: _concept_link(m.group(1), titles), html)

    # Sources numbered by first use as evidence.
    order: list[str] = []
    for claim in claims:
        order += [e.source_id for e in claim.evidence if e.source_id not in order]
    order += [s.id for s in sources if s.id not in order]
    source_numbers = {sid: n for n, sid in enumerate(order, start=1)}
    by_source: dict[str, list[str]] = {sid: [] for sid in order}
    for claim in claims:
        for sid in dict.fromkeys(e.source_id for e in claim.evidence):
            by_source[sid].append(claim.id)

    claim_concepts: dict[str, list[str]] = {}
    for concept in concepts:
        for claim_id in concept.claim_ids:
            claim_concepts.setdefault(claim_id, []).append(concept.id)

    children = {c.id: [k.id for k in concepts if k.parent == c.id] for c in concepts}
    concept_map = radial_map(request.topic, concepts)

    last_run = state.runs[-1] if state.runs else None
    stopped = None
    if state.status is not None and state.status is not RunStatus.COMPLETED:
        stopped = RUN_STATUS_LABELS[state.status] + (
            f" ({last_run.reason})" if last_run and last_run.reason else ""
        )
    researched = (last_run.ended_at or last_run.started_at) if last_run else None
    generated = datetime.now(UTC)
    assessments = {c.id: project.assess(c) for c in claims}
    status_counts = dict.fromkeys(STATUS_LABELS, 0)
    for assessment in assessments.values():
        status_counts[assessment.status] += 1

    return {
        "topic": request.topic,
        "goal": request.goal,
        "level": request.knowledge_level,
        "projectId": project.metadata.id,
        "researchedOn": _date(researched or generated),
        "generatedAt": f"{_date(generated)}, {generated:%H:%M} UTC",
        "stopped": stopped,
        "summaryHtml": rich(state.summary) if state.summary else None,
        "levels": [{"id": level, "label": LEVEL_LABELS[level]} for level in LEVELS],
        "path": learning_order(concepts),
        "concepts": {
            c.id: {
                "id": c.id,
                "title": c.title,
                "summaryHtml": rich(c.summary, inline=True) if c.summary else "",
                "parent": c.parent if c.parent in concept_ids else None,
                "children": children[c.id],
                "prerequisites": [p for p in c.prerequisites if p in concept_ids],
                "related": [r for r in c.related if r in concept_ids],
                "claims": c.claim_ids,
                "levels": {level: rich(text) for level, text in c.explanations.items()},
                "researched": bool(c.claim_ids),
            }
            for c in concepts
        },
        "claims": {
            c.id: {
                "number": claim_numbers[c.id],
                "textHtml": rich(c.text, inline=True),
                "kind": c.kind,
                "status": assessments[c.id].status,
                "statusLabel": _status_label(
                    assessments[c.id].status, assessments[c.id].supporting_sources
                ),
                "note": c.note,
                "concepts": claim_concepts.get(c.id, []),
                "evidence": [
                    {"quote": e.quote, "stance": e.stance, "source": source_numbers[e.source_id]}
                    for e in c.evidence
                ],
            }
            for c in claims
        },
        "statusCounts": status_counts,
        "sources": [
            {
                "number": source_numbers[s.id],
                "title": s.title,
                "url": s.url,
                "domain": (urlsplit(s.url).hostname or s.url).removeprefix("www."),
                "tier": TIER_LABELS[s.tier],
                "published": s.published,
                "fetched": _date(s.fetched_at, month="%b"),
                "duplicateOf": source_numbers.get(s.duplicate_of or ""),
                "claims": by_source[s.id],
            }
            for s in sorted(sources, key=lambda s: source_numbers[s.id])
        ],
        "agenda": [
            {
                "status": item.status,
                "text": item.text,
                "noteHtml": _agenda_note(item.note, claim_numbers),
                "claims": [cid for cid in item.claim_ids if cid in claim_numbers],
            }
            for item in state.agenda
        ],
        "runs": [
            {
                "started": f"{_date(run.started_at, month='%b')}, {run.started_at:%H:%M}",
                "status": RUN_STATUS_LABELS[run.status],
                "steps": run.usage.steps if run.usage else 0,
                "tokens": (run.usage.input_tokens + run.usage.output_tokens) if run.usage else 0,
                "cost": f"${run.usage.cost_usd:.4f}" if run.usage else "$0.0000",
            }
            for run in state.runs
        ],
        "map": {
            "root": ROOT,
            "viewBox": concept_map.view_box,
            "nodes": [vars(n) for n in concept_map.nodes],
            "edges": [vars(e) for e in concept_map.edges],
        },
    }


def render(data: dict[str, Any], title: str) -> str:
    static = resources.files("researchos.site").joinpath("static")
    css = static.joinpath("app.css").read_text("utf-8")
    js = static.joinpath("app.js").read_text("utf-8")
    csp = "; ".join(
        [
            "default-src 'none'",
            f"style-src '{_sha256(css)}'",
            f"script-src '{_sha256(js)}'",
            "img-src https: data:",
            "base-uri 'none'",
            "form-action 'none'",
        ]
    )
    env = Environment(
        loader=PackageLoader("researchos.site", "templates"),
        autoescape=select_autoescape(default=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("index.html.j2").render(
        title=title,
        data=data,
        css=_trusted(css),
        js=_trusted(js),
        csp=csp,
        data_json=_trusted(_script_safe_json(data)),
    )


# Rendering helpers ------------------------------------------------------------------------


def _markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": True})
    md.enable(["table", "strikethrough"])

    def link_open(self: Any, tokens: list[Token], idx: int, options: Any, env: Any) -> str:
        href = str(tokens[idx].attrGet("href") or "")
        if href.startswith(("http://", "https://")):
            tokens[idx].attrSet("target", "_blank")
            tokens[idx].attrSet("rel", "noopener noreferrer")
        return str(self.renderToken(tokens, idx, options, env))

    md.add_render_rule("link_open", link_open)
    return md


def _chips(group: str, claim_numbers: dict[str, int]) -> str:
    """Citation chips for claim ids found in already-escaped HTML."""
    chips = [
        f'<button type="button" class="cite" data-claim="{cid}" '
        f'aria-label="Evidence for claim {claim_numbers[cid]}">{claim_numbers[cid]}</button>'
        for cid in dict.fromkeys(_CLAIM_ID.findall(group))
        if cid in claim_numbers
    ]
    return f'<span class="cites">{"".join(chips)}</span>' if chips else ""


def _concept_link(escaped_title: str, titles: dict[str, str]) -> str:
    """A link for a [[Title]] found in already-escaped HTML; plain text if unknown."""
    key = escaped_title.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'")
    concept_id = titles.get(key.strip().lower())
    if concept_id is None:
        return escaped_title
    return f'<a class="concept-link" href="#/concept/{concept_id}">{escaped_title}</a>'


def _agenda_note(note: str | None, claim_numbers: dict[str, int]) -> str:
    if not note:
        return ""
    pieces: list[Markup] = []
    last = 0
    for match in _FINDING_REF.finditer(note):
        pieces.append(escape(note[last : match.start()]))
        claim_id = f"claim-{match.group(1)}"
        if claim_id in claim_numbers:
            pieces.append(
                Markup('<button type="button" class="cite" data-claim="{}">{}</button>').format(
                    claim_id, claim_numbers[claim_id]
                )
            )
        else:
            pieces.append(escape(match.group(0)))
        last = match.end()
    pieces.append(escape(note[last:]))
    return str(Markup("").join(pieces))


def _status_label(status: str, supporting: int) -> str:
    if status == "verified":
        return f"Verified by {supporting} independent sources"
    if status == "single_source":
        return "From a single source"
    if status == "disputed":
        return "Sources disagree"
    return "Not backed by a quoted source"


def _date(moment: datetime, month: str = "%B") -> str:
    return f"{moment.day} {moment.strftime(month)} {moment.year}"


def _trusted(html: str) -> Markup:
    """Mark markup as safe to embed. Only for content produced by this module: the bundled
    static assets and script-safe JSON."""
    return Markup(html)  # noqa: S704


def _sha256(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


def _script_safe_json(value: Any) -> str:
    """JSON that cannot terminate the <script> element it is embedded in."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(_LINE_SEPARATOR, "\\u2028")
        .replace(_PARAGRAPH_SEPARATOR, "\\u2029")
    )
