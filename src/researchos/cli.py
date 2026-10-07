"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Sequence
from contextlib import closing

from pydantic import ValidationError

from researchos.agent import ResearchAgent, RunResult, RunStatus
from researchos.config import Settings
from researchos.llm import create_llm_client
from researchos.project import Project, ResearchRequest
from researchos.tools import ToolRegistry, build_research_tools
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.web import create_fetcher, create_search_provider

logger = logging.getLogger("researchos")

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_CONFIG_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    request = ResearchRequest(topic=args.topic, goal=args.goal, knowledge_level=args.level)
    result = run_research(settings, request)
    _print_result(result)
    return EXIT_OK if result.status is RunStatus.COMPLETED else EXIT_INCOMPLETE


def run_research(settings: Settings, request: ResearchRequest) -> RunResult:
    if settings.llm_provider == "mock":
        logger.warning("LLM_PROVIDER=mock: using the offline mock model; no real research")
    if settings.search_provider == "mock":
        logger.warning("SEARCH_PROVIDER=mock: using synthetic search results")

    project = Project.create(settings.workspace_dir, request)
    logger.info("project: %s", project.root)
    meter = UsageMeter(
        limits=Limits(
            max_steps=settings.max_steps,
            max_cost_usd=settings.max_cost_usd,
            max_wall_seconds=settings.max_wall_seconds,
        ),
        pricing=Pricing(
            input_per_million=settings.llm_price_in,
            output_per_million=settings.llm_price_out,
        ),
        events_path=project.events_path,
        run_id=uuid.uuid4().hex,
    )
    with (
        closing(create_llm_client(settings)) as llm,
        closing(create_search_provider(settings)) as search,
        closing(create_fetcher(settings)) as fetcher,
    ):
        tools = ToolRegistry(build_research_tools(project, search, fetcher))
        agent = ResearchAgent(llm=llm, tools=tools, meter=meter, project=project)
        return agent.run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchos", description="Autonomously research a topic."
    )
    parser.add_argument("topic", help='what to research, e.g. "how RAG systems work"')
    parser.add_argument("--goal", help="what you want to be able to do or understand")
    parser.add_argument("--level", help="your current knowledge of the topic")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the result")
    return parser


def _print_result(result: RunResult) -> None:
    usage = result.usage
    print(
        f"\nStatus:  {result.status.value} ({result.reason})\n"
        f"Report:  {result.report_path}\n"
        f"Usage:   {usage.steps} steps, {usage.llm_calls} LLM calls, "
        f"{usage.tool_calls} tool calls ({usage.tool_errors} failed), "
        f"{usage.input_tokens + usage.output_tokens} tokens, "
        f"${usage.cost_usd:.4f}, {usage.elapsed_seconds:.1f}s"
    )
