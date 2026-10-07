"""Static website generation for a research project.

`build_site` renders everything the project knows into a single self-contained
`website/index.html`: styles, scripts and search data are inlined so the file works offline
and can be opened directly or shared. All model- and web-derived text is escaped or rendered
through Markdown with raw HTML disabled, and a Content-Security-Policy pins the inline style
and script by hash.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markdown_it import MarkdownIt
from markdown_it.token import Token
from markupsafe import Markup, escape

from researchos.project import Note, Project
from researchos.state import RunStatus

_NOTE_REF = re.compile(r"\bnote-\d{4}\b")
# Valid in JSON strings but line terminators in JavaScript source.
_LINE_SEPARATOR = chr(0x2028)
_PARAGRAPH_SEPARATOR = chr(0x2029)

_RUN_STATUS_LABELS = {
    RunStatus.RUNNING: "Running",
    RunStatus.COMPLETED: "Completed",
    RunStatus.INTERRUPTED: "Interrupted",
    RunStatus.BUDGET_EXHAUSTED: "Stopped at budget limit",
    RunStatus.STALLED: "Stopped: agent stalled",
    RunStatus.LLM_REFUSED: "Stopped: model access refused",
    RunStatus.LLM_FAILED: "Stopped: model unavailable",
}


@dataclass(frozen=True)
class TocEntry:
    anchor: str
    title: str
    level: int


@dataclass(frozen=True)
class SummarySection:
    anchor: str
    title: str
    text: str


@dataclass(frozen=True)
class SiteSource:
    number: int
    anchor: str
    title: str
    url: str
    domain: str
    fetched: str
    cited_by: list[tuple[str, int]]
    """(finding anchor, finding number) for each finding that cites this source."""


@dataclass(frozen=True)
class Citation:
    number: int
    anchor: str
    title: str
    domain: str


@dataclass(frozen=True)
class SiteFinding:
    anchor: str
    label: str
    html: Markup
    citations: list[Citation]


@dataclass(frozen=True)
class SiteAgendaItem:
    status: str
    html: Markup


@dataclass(frozen=True)
class SiteRun:
    started: str
    status: str
    steps: int
    tokens: int
    cost: str
    duration: str


@dataclass
class SiteData:
    topic: str
    goal: str | None
    level: str | None
    project_id: str
    researched_on: str
    generated_at: str
    stopped_reason: str | None
    summary_html: Markup | None
    summary_toc: list[TocEntry]
    findings: list[SiteFinding]
    sources: list[SiteSource]
    agenda: list[SiteAgendaItem]
    runs: list[SiteRun]
    search_index: list[dict[str, str]] = field(default_factory=list)


def build_site(project: Project) -> Path:
    """Render the project's website and return the path of `index.html`."""
    data = collect(project)
    html = render(data)
    out_dir = project.root / "website"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "index.html"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, path)
    return path


def collect(project: Project) -> SiteData:
    """Gather the project's research into template-ready, already-escaped values."""
    request = project.metadata.request
    state = project.state
    notes = project.notes()
    sources = project.sources()

    # Number sources in order of first citation, then the uncited ones.
    order: list[str] = []
    for note in notes:
        order += [sid for sid in note.source_ids if sid not in order]
    order += [s.id for s in sources if s.id not in order]
    by_id = {s.id: s for s in sources}
    numbers = {sid: n for n, sid in enumerate(order, start=1)}

    finding_numbers = {note.id: n for n, note in enumerate(notes, start=1)}
    cited_by: dict[str, list[tuple[str, int]]] = {sid: [] for sid in order}
    for note in notes:
        for sid in dict.fromkeys(note.source_ids):
            cited_by[sid].append((note.id, finding_numbers[note.id]))

    site_sources = [
        SiteSource(
            number=numbers[sid],
            anchor=f"source-{numbers[sid]}",
            title=by_id[sid].title,
            url=by_id[sid].url,
            domain=_domain(by_id[sid].url),
            fetched=_date(by_id[sid].fetched_at, month="%b"),
            cited_by=cited_by[sid],
        )
        for sid in order
    ]

    md = _markdown()
    findings = [
        SiteFinding(
            anchor=note.id,
            label=f"Finding {finding_numbers[note.id]}",
            html=_trusted(md.renderInline(note.text)),
            citations=[
                Citation(
                    number=numbers[sid],
                    anchor=f"source-{numbers[sid]}",
                    title=by_id[sid].title,
                    domain=_domain(by_id[sid].url),
                )
                for sid in dict.fromkeys(note.source_ids)
            ],
        )
        for note in notes
    ]

    summary_html, toc, sections = (
        (None, [], []) if not state.summary else _render_summary(md, state.summary)
    )

    status = state.status
    last_run = state.runs[-1] if state.runs else None
    stopped_reason = None
    if status is not None and status is not RunStatus.COMPLETED:
        stopped_reason = f"{_RUN_STATUS_LABELS[status]}" + (
            f" ({last_run.reason})" if last_run and last_run.reason else ""
        )

    researched = (last_run.ended_at or last_run.started_at) if last_run else None
    generated = datetime.now(UTC)
    data = SiteData(
        topic=request.topic,
        goal=request.goal,
        level=request.knowledge_level,
        project_id=project.metadata.id,
        researched_on=_date(researched or generated),
        generated_at=f"{_date(generated)}, {generated:%H:%M} UTC",
        stopped_reason=stopped_reason,
        summary_html=summary_html,
        summary_toc=toc,
        findings=findings,
        sources=site_sources,
        agenda=[
            SiteAgendaItem(
                status=item.status, html=_agenda_html(item.text, item.note, finding_numbers)
            )
            for item in state.agenda
        ],
        runs=[
            SiteRun(
                started=f"{_date(run.started_at, month='%b')}, {run.started_at:%H:%M}",
                status=_RUN_STATUS_LABELS[run.status],
                steps=run.usage.steps if run.usage else 0,
                tokens=(run.usage.input_tokens + run.usage.output_tokens) if run.usage else 0,
                cost=f"${run.usage.cost_usd:.4f}" if run.usage else "$0.0000",
                duration=_duration(run.usage.elapsed_seconds) if run.usage else "",
            )
            for run in state.runs
        ],
    )
    data.search_index = _search_index(sections, findings, notes, site_sources)
    return data


def render(data: SiteData) -> str:
    static = resources.files("researchos.site").joinpath("static")
    css = static.joinpath("site.css").read_text("utf-8")
    js = static.joinpath("site.js").read_text("utf-8")
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
    template = env.get_template("index.html.j2")
    return template.render(
        site=data,
        css=_trusted(css),
        js=_trusted(js),
        csp=csp,
        search_json=_trusted(_script_safe_json(data.search_index)),
        agenda_done=sum(1 for item in data.agenda if item.status == "done"),
    )


# Markdown -------------------------------------------------------------------------------


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


def _render_summary(
    md: MarkdownIt, text: str
) -> tuple[Markup, list[TocEntry], list[SummarySection]]:
    """Render the summary with stable heading anchors, its top heading level shifted to h3
    (the page supplies the h2). Also returns its table of contents and its text split into
    sections for search."""
    tokens = md.parse(text)
    top = min((int(t.tag[1]) for t in tokens if t.type == "heading_open"), default=3)
    toc: list[TocEntry] = []
    sections = [SummarySection(anchor="summary", title="Summary", text="")]
    used: set[str] = set()
    for i, token in enumerate(tokens):
        if token.type in ("heading_open", "heading_close"):
            level = min(int(token.tag[1]) - top + 3, 6)
            token.tag = f"h{level}"
            if token.type == "heading_open":
                title = _plain(tokens[i + 1].content)
                anchor = _unique_slug(title, used)
                token.attrSet("id", anchor)
                if level <= 4:
                    toc.append(TocEntry(anchor=anchor, title=title, level=level))
                sections.append(SummarySection(anchor=anchor, title=title, text=""))
        elif token.type == "inline" and tokens[i - 1].type != "heading_open":
            last = sections[-1]
            sections[-1] = SummarySection(
                anchor=last.anchor, title=last.title, text=f"{last.text} {_plain(token.content)}"
            )
    html = _trusted(md.renderer.render(tokens, md.options, {}))
    return html, toc, [s for s in sections if s.text.strip()]


def _agenda_html(text: str, note: str | None, finding_numbers: dict[str, int]) -> Markup:
    """Escape an agenda item; in its note, turn references like "note-0003" into links
    labelled with the finding's number as shown on the page."""
    if not note:
        return escape(text)
    pieces: list[Markup] = []
    last = 0
    for match in _NOTE_REF.finditer(note):
        pieces.append(escape(note[last : match.start()]))
        ref = match.group(0)
        if ref in finding_numbers:
            label = f"Finding {finding_numbers[ref]}"
            pieces.append(Markup('<a href="#{}">{}</a>').format(ref, label))
        else:
            pieces.append(escape(ref))
        last = match.end()
    pieces.append(escape(note[last:]))
    return Markup('{}<span class="agenda-note">{}</span>').format(text, Markup("").join(pieces))


# Helpers --------------------------------------------------------------------------------


def _search_index(
    sections: list[SummarySection],
    findings: list[SiteFinding],
    notes: list[Note],
    sources: list[SiteSource],
) -> list[dict[str, str]]:
    index = [
        {"kind": "Summary", "title": sec.title, "text": sec.text.strip(), "href": f"#{sec.anchor}"}
        for sec in sections
    ]
    index += [
        {"kind": "Finding", "title": finding.label, "text": note.text, "href": f"#{note.id}"}
        for finding, note in zip(findings, notes, strict=True)
    ]
    index += [
        {
            "kind": "Source",
            "title": source.title,
            "text": f"{source.domain} {source.url}",
            "href": f"#{source.anchor}",
        }
        for source in sources
    ]
    return index


def _unique_slug(title: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", _plain(title).lower()).strip("-")[:60] or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def _plain(markdown_text: str) -> str:
    text = re.sub(r"[*_`#>\[\]]|\(https?://[^)]*\)", "", markdown_text)
    return " ".join(text.split())


def _date(moment: datetime, month: str = "%B") -> str:
    return f"{moment.day} {moment.strftime(month)} {moment.year}"


def _domain(url: str) -> str:
    host = urlsplit(url).hostname or url
    return host.removeprefix("www.")


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _trusted(html: str) -> Markup:
    """Mark HTML as safe to embed. Only for markup produced by this module: Markdown rendered
    with raw HTML disabled, the bundled static assets, and script-safe JSON."""
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
