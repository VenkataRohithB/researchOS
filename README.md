# ResearchOS

An autonomous AI research and learning agent. Give it a topic; it researches the
topic with tools, verifies claims against sources, builds a knowledge graph, and
publishes an explorable static learning website with multiple depth levels and
presentation modes.

> Status: early development. Research, verification, concept graph, multi-depth explanations and the interactive site work end to end.

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
researchos "How does Apple Silicon's unified memory work"   # research a topic
researchos                                                    # or be asked what to research
```

The agent may ask you a few questions first (an ambiguous topic, your level or goal). When
it finishes it prints the website's path and asks what you would like to improve; describe
it ("go deeper on the GPU side", "explain it more simply") and it continues the same
project, building on what is already there. Press Enter to stop.

```bash
researchos list                                   # your projects
researchos improve <project> "add a comparison with discrete GPUs"
researchos build <project> --open                 # open the website
researchos status <project>                       # progress, claims, concepts, usage
researchos resume <project>                       # continue after a stop or Ctrl-C
researchos steer <project> "focus on bandwidth"   # instruct a run in progress
```

### The website

Every run generates `workspace/<project-id>/website/index.html`, one self-contained file that
works offline:

- **Learn**: concepts in learning order (prerequisites first), each with a depth scale
  (Summary, Beginner, Intermediate, Deep, Expert), "Learn first", "Go deeper" and related
  concepts, and breadcrumbs back up.
- **Evidence everywhere**: every cited claim opens the passages quoted from its sources and
  whether it is verified, single-source or disputed.
- **Map**: an interactive concept map; **Read**: everything as one article at your chosen
  depth; **Revise**: flashcards and facts confirmed by independent sources;
  **Sources**: every page read, the research plan and the run log. Search with `/`.

### Project files

```
metadata/      project.json, state.json, transcript.jsonl, directives.jsonl, events.jsonl
sources/       sources.json and the extracted text of every source (HTML or PDF)
knowledge/     claims.json (quoted evidence) and concepts.json (the knowledge graph)
website/       index.html
report.md      plain-text report
```

Exit codes: `0` research finished, `1` run stopped early (budget, model failure, stalled),
`2` usage or configuration error, `130` interrupted.

## How it works

A tool-calling agent loop decides what to do next; heavy jobs run as focused model calls
inside tools:

| Tool | What it does |
|---|---|
| `ask_user` | Clarifying questions when the request is ambiguous |
| `update_agenda`, `set_phase` | Plan the research and record its stage |
| `search_web`, `fetch_source` | Find and fetch sources (HTML or PDF) |
| `study_source` | A dedicated reading pass: claims with quotes, corroboration of claims from other sources, concepts and their relationships |
| `explain_concepts` | A dedicated writing pass: each concept at five depths, citing its claims |
| `search_sources`, `read_source`, `record_claim`, `add_evidence`, `get_claim`, `list_claims` | Targeted look-ups and manual evidence |
| `list_concepts`, `get_concept`, `update_concept` | Inspect and correct the concept graph |
| `assess_source` | Correct a source's reliability tier |
| `finish_research` | End with a synthesis |

**Evidence rules.** Every quote is checked word for word against the stored source text.
A claim is *verified* when two independent sources support it (different sites, not copies,
which are detected by content), *single-source* with one, *disputed* when a source
contradicts it. The harness, not the model, enforces the quality bar: plan items close only
with the claims that cover them; research cannot finish with open items, unexplained
concepts, or (without one review round) evidence from fewer than three sites or
single-source claims; citations in explanations must name the claims the writer was given.

**State and resilience.** Each step the model sees a snapshot of the research state plus its
last few steps, so prompts stay small however long the research runs. State is saved after
every step, so runs resume after a budget stop, error, Ctrl-C or crash. `LLM_MODEL` can list
several models (`model-a,model-b`); when one's quota is exhausted or it keeps failing, the
next takes over.

Safety boundaries:

- Tools are the only way the model affects anything; arguments are schema-validated.
- Fetching is limited to public http(s) addresses, size-capped, text content only.
- Web content is wrapped as untrusted data, and the model is told never to follow it.
- Claims can only cite sources that were actually fetched, with quotes found in them.
- The website escapes all model and web text and pins its inline code with a strict CSP.
- Step, cost and wall-clock limits end the run gracefully.

## Development

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```
