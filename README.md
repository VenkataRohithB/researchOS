# ResearchOS

An autonomous AI research and learning agent. Give it a topic; it researches the
topic with tools, verifies claims against sources, builds a knowledge graph, and
publishes an explorable static learning website with multiple depth levels and
presentation modes.

> Status: early development (Phase 3: source verification).

## Setup

Requires Python 3.11+ on Linux or macOS.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

The defaults in `.env.example` use mock providers, so everything runs offline without
credentials.

**Model:** set `LLM_PROVIDER=openai_compatible` with `LLM_MODEL`, `LLM_API_KEY` and
`LLM_BASE_URL`. Any OpenAI Chat Completions-compatible endpoint works, for example Gemini
(`https://generativelanguage.googleapis.com/v1beta/openai/`) or Claude
(`https://api.anthropic.com/v1/`).

**Web search:** either run SearXNG locally (no account needed; requires Docker):

```bash
echo "SEARXNG_SECRET=$(openssl rand -hex 32)" >> .env   # once
docker compose up -d                                   # serves http://127.0.0.1:8888
```

and set `SEARCH_PROVIDER=searxng`, or use Tavily with `SEARCH_PROVIDER=tavily` and
`SEARCH_API_KEY`.

## Usage

With the virtualenv activated (`source .venv/bin/activate`):

```bash
researchos new "How retrieval-augmented generation works" \
    --goal "understand it well enough to build one" \
    --level "software engineer, new to ML"

researchos list                              # all projects and their status
researchos status <project>                  # phase, agenda, progress, usage
researchos steer <project> "go deeper on reranking"   # works while a run is in progress
researchos resume <project> ["optional instruction"]  # continue after a stop or Ctrl-C
researchos build <project> --open            # regenerate the website and open it
```

Every run ends by generating the project's website at `workspace/<project-id>/website/index.html`:
a single self-contained file (works offline, safe to share) with the summary and its contents,
findings with their sources in the margin, the research plan, numbered sources, a run log,
search (press `/`), and light/dark themes.

Each project lives under `workspace/<project-id>/`:

```
metadata/project.json       the request
metadata/state.json         phase, agenda, summary, every run with its status and usage
metadata/transcript.jsonl   every agent step: the model's reply and the tool results
metadata/directives.jsonl   instructions given with steer/resume
metadata/events.jsonl       every LLM and tool call, with tokens, cost and latency
sources/sources.json        fetched sources: canonical URL, tier, publication date, duplicates
sources/snapshots/          extracted text of each source (HTML or PDF)
knowledge/claims.json       claims with quoted evidence
report.md                   report
website/index.html          the project website
```

Exit codes: `0` research finished, `1` run stopped early (budget, model failure, stalled),
`2` usage or configuration error, `130` interrupted.

## How it works

The model drives a tool-calling loop: each step it chooses tools, the harness validates the
arguments and executes them, and the results are fed back. There is no fixed sequence.

| Tools | Purpose |
|---|---|
| `update_agenda`, `set_phase` | Plan the research and record its stage |
| `search_web`, `fetch_source`, `read_source`, `search_sources` | Find, fetch and read sources (HTML or PDF) |
| `record_claim`, `add_evidence`, `get_claim`, `list_claims` | Record claims backed by quotes; cross-check them |
| `assess_source` | Correct a source's reliability tier |
| `finish_research` | End with a synthesis |

**Evidence rules.** Findings are recorded as claims, labelled fact, interpretation or inference,
and backed by passages quoted from sources. Every quote is checked word for word against the
stored source text; quotes that are not found are rejected. A claim's status is computed, not
asserted: *verified* needs two independent sources (different sites, not copies of the same
document, which are detected by content), *single source* has one, *disputed* has a
contradicting source. Research cannot finish until at least one claim is supported by a quote.

Each step the model sees a fresh snapshot of the research state plus only its last few steps,
so the prompt stays roughly constant in size however long the research runs. State is saved
after every step: a run stopped by the budget, an error, Ctrl-C or a crash resumes where it
left off. A per-project lock prevents two processes from running the same project.

Safety boundaries:

- Tools are the only way the model affects anything; arguments are schema-validated.
- Fetching is limited to public http(s) addresses, size-capped, text content only.
- Web content is wrapped as untrusted data, and the model is told never to follow it.
- Claims can only cite sources that were actually fetched, with quotes found in them.
- Step, cost and wall-clock limits end the run gracefully.

## Development

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```
