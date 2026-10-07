# ResearchOS

An autonomous AI research and learning agent. Give it a topic; it researches the
topic with tools, verifies claims against sources, builds a knowledge graph, and
publishes an explorable static learning website with multiple depth levels and
presentation modes.

> Status: early development (Phase 1: core agent loop).

## Setup

Requires Python 3.11+.

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

```bash
.venv/bin/researchos "How retrieval-augmented generation works" \
    --goal "understand it well enough to build one" \
    --level "software engineer, new to ML"
```

Each run creates a project under `workspace/`:

```
workspace/<project-id>/
  metadata/project.json   the request
  metadata/events.jsonl   every LLM and tool call, with tokens, cost and latency
  sources/sources.json    fetched sources
  sources/snapshots/      extracted text of each source
  notes/notes.jsonl       findings, each citing source ids
  report.md               final report
```

The exit code is `0` when the agent finishes the research, `1` when the run stops early
(budget, model failure, stalled), and `2` on configuration errors.

## How it works

The model drives a tool-calling loop: each step it chooses a tool (`search_web`,
`fetch_source`, `read_source`, `save_note`, `finish_research`), the harness validates the
arguments and executes it, and the result is fed back. There is no fixed sequence.

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
