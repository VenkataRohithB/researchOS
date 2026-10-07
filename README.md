# ResearchOS

An autonomous AI research and learning agent. Give it a topic; it researches the
topic with tools, verifies claims against sources, builds a knowledge graph, and
publishes an explorable static learning website with multiple depth levels and
presentation modes.

> Status: early development (Phase 2: resumable research state).

## Setup

Requires Python 3.11+ on Linux or macOS.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

The defaults in `.env.example` use mock providers, so everything runs offline without
credentials. To use a real model, set `LLM_PROVIDER=openai_compatible` with `LLM_MODEL`,
`LLM_API_KEY` and `LLM_BASE_URL`; for real web search set `SEARCH_PROVIDER=tavily` and
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
```

Each project lives under `workspace/<project-id>/`:

```
metadata/project.json       the request
metadata/state.json         phase, agenda, summary, every run with its status and usage
metadata/transcript.jsonl   every agent step: the model's reply and the tool results
metadata/directives.jsonl   instructions given with steer/resume
metadata/events.jsonl       every LLM and tool call, with tokens, cost and latency
sources/sources.json        fetched sources
sources/snapshots/          extracted text of each source
notes/notes.jsonl           findings, each citing source ids
report.md                   report
```

Exit codes: `0` research finished, `1` run stopped early (budget, model failure, stalled),
`2` usage or configuration error, `130` interrupted.

## How it works

The model drives a tool-calling loop: each step it chooses tools (`update_agenda`,
`set_phase`, `search_web`, `fetch_source`, `read_source`, `save_note`, `read_notes`,
`finish_research`), the harness validates the arguments and executes them, and the results are
fed back. There is no fixed sequence.

Each step the model sees a fresh snapshot of the research state plus only its last few steps,
so the prompt stays roughly constant in size however long the research runs. State is saved
after every step: a run stopped by the budget, an error, Ctrl-C or a crash resumes where it
left off. A per-project lock prevents two processes from running the same project.

Safety boundaries:

- Tools are the only way the model affects anything; arguments are schema-validated.
- Fetching is limited to public http(s) addresses, size-capped, text content only.
- Web content is wrapped as untrusted data, and the model is told never to follow it.
- Notes can only cite sources that were actually fetched.
- Step, cost and wall-clock limits end the run gracefully.

## Development

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```
