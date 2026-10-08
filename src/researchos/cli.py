"""Command-line interface.

    researchos                         ask what to research, then research it
    researchos "Apple's M5 chip"        research a topic
    researchos improve <project> "..."  build on an existing project
    researchos list | status | resume | steer | build

In an interactive terminal the agent can ask clarifying questions while it works, and every
run ends by offering to improve the result, which continues the same project.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
import webbrowser
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing

from pydantic import ValidationError

from researchos.agent import ResearchAgent, RunResult
from researchos.config import Settings
from researchos.focused import FocusedModel
from researchos.llm import create_llm_client
from researchos.project import Project, ProjectLockedError, ProjectNotFoundError, ResearchRequest
from researchos.site import build_site
from researchos.state import RunStatus
from researchos.tools import ToolRegistry, build_research_tools
from researchos.usage import Limits, Pricing, UsageMeter
from researchos.user import NoUser, TerminalUser, UserChannel
from researchos.web import create_fetcher, create_search_provider

logger = logging.getLogger("researchos")

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130

COMMANDS = frozenset({"new", "improve", "resume", "steer", "build", "status", "list"})

Prompt = Callable[[str], str]


def main(
    argv: Sequence[str] | None = None,
    *,
    prompt: Prompt | None = None,
    user: UserChannel | None = None,
) -> int:
    """`prompt` and `user` default to the terminal when one is attached; tests pass fakes."""
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    prompt = prompt or (input if interactive else None)
    user = user or (TerminalUser() if interactive else NoUser())

    args = _parser().parse_args(_with_default_command(list(sys.argv[1:] if argv is None else argv)))
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        if args.command == "new":
            topic = args.topic or _ask(prompt, "What would you like to learn about?")
            if not topic:
                print('Nothing to research. Usage: researchos "<topic>"', file=sys.stderr)
                return EXIT_USAGE_ERROR
            request = ResearchRequest(topic=topic, goal=args.goal, knowledge_level=args.level)
            project = Project.create(settings.workspace_dir, request)
            print(f"Project: {project.metadata.id}")
            return _session(settings, project, user, prompt)
        if args.command == "list":
            return _list(settings)

        project = Project.open(settings.workspace_dir, args.project)
        if args.command == "improve":
            project.add_directive(f"Improvement requested: {args.request}")
            return _session(settings, project, user, prompt)
        if args.command == "resume":
            if args.instruction:
                project.add_directive(args.instruction)
            return _session(settings, project, user, prompt)
        if args.command == "build":
            path = build_site(project)
            print(f"Website: {path}")
            if args.open:
                webbrowser.open(path.resolve().as_uri())
            return EXIT_OK
        if args.command == "steer":
            project.add_directive(args.instruction)
            print("Instruction recorded. A running agent sees it on its next step;")
            print(f"otherwise run: researchos resume {project.metadata.id}")
            return EXIT_OK
        return _status(project)
    except (ProjectNotFoundError, ProjectLockedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; continue with: researchos resume <project>")
        return EXIT_INTERRUPTED


def run_research(settings: Settings, project: Project, user: UserChannel) -> RunResult:
    if settings.llm_provider == "mock":
        logger.warning("LLM_PROVIDER=mock: using the offline mock model; no real research")
    if settings.search_provider == "mock":
        logger.warning("SEARCH_PROVIDER=mock: using synthetic search results")

    run_id = uuid.uuid4().hex
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
        run_id=run_id,
    )
    with (
        closing(create_llm_client(settings)) as llm,
        closing(create_search_provider(settings)) as search,
        closing(create_fetcher(settings)) as fetcher,
    ):
        tools = build_research_tools(
            project, search, fetcher, focused=FocusedModel(llm, meter), user=user
        )
        agent = ResearchAgent(
            llm=llm,
            tools=ToolRegistry(tools),
            meter=meter,
            project=project,
            run_id=run_id,
            context_turns=settings.context_turns,
        )
        return agent.run()


def _session(settings: Settings, project: Project, user: UserChannel, prompt: Prompt | None) -> int:
    """Run the research, then keep improving it for as long as the person asks."""
    while True:
        exit_code = _run(settings, project, user)
        if prompt is None:
            return exit_code
        request = _ask(
            prompt,
            "What would you like to improve? Describe it, or press Enter to finish.",
        )
        if not request:
            print(f'Improve it later with: researchos improve {project.metadata.id} "..."')
            return exit_code
        project.add_directive(f"Improvement requested: {request}")


def _run(settings: Settings, project: Project, user: UserChannel) -> int:
    result = run_research(settings, project, user)
    website = build_site(project)
    usage = result.usage
    print(
        f"\nStatus:  {result.status.value} ({result.reason})\n"
        f"Website: {website}\n"
        f"Report:  {result.report_path}\n"
        f"Usage:   {usage.steps} steps, {usage.llm_calls} LLM calls, "
        f"{usage.tool_calls} tool calls ({usage.tool_errors} failed), "
        f"{usage.input_tokens + usage.output_tokens} tokens, "
        f"${usage.cost_usd:.4f}, {usage.elapsed_seconds:.1f}s"
    )
    if result.status is not RunStatus.COMPLETED:
        print(f"Resume:  researchos resume {project.metadata.id}")
        return EXIT_INCOMPLETE
    return EXIT_OK


def _ask(prompt: Prompt | None, question: str) -> str:
    if prompt is None:
        return ""
    try:
        return prompt(f"\n{question}\n> ").strip()
    except EOFError:
        return ""


def _status(project: Project) -> int:
    state = project.state
    request = project.metadata.request
    runs = state.runs
    cost = sum(r.usage.cost_usd for r in runs if r.usage)
    tokens = sum(r.usage.input_tokens + r.usage.output_tokens for r in runs if r.usage)
    print(f"Project:  {project.metadata.id}")
    print(f"Topic:    {request.topic}")
    print(f"Status:   {state.status.value if state.status else 'not started'}")
    print(f"Phase:    {state.phase.value}")
    print(
        f"Progress: {state.step} steps over {len(runs)} run(s), {len(project.sources())} sources,"
    )
    claims = Counter(project.assess(c).status for c in project.claims())
    print(
        f"          {sum(claims.values())} claims ({claims['verified']} verified, "
        f"{claims['single_source']} single-source, {claims['disputed']} disputed), "
        f"{len(project.concepts())} concepts, {tokens} tokens, ${cost:.4f}"
    )
    if state.agenda:
        print("Agenda:")
        for item in state.agenda:
            print(f"  {item.id} [{item.status}] {item.text}")
    directives = project.directives()
    if directives:
        print("Instructions:")
        for directive in directives:
            print(f"  - {directive.text}")
    return EXIT_OK


def _list(settings: Settings) -> int:
    ids = Project.list_ids(settings.workspace_dir)
    if not ids:
        print(f"No research projects in {settings.workspace_dir}")
    for project_id in ids:
        project = Project.open(settings.workspace_dir, project_id)
        status = project.state.status
        print(f"{project_id}  [{status.value if status else 'not started'}]")
    return EXIT_OK


_GLOBAL_FLAGS = frozenset({"-q", "--quiet", "-h", "--help"})


def _with_default_command(argv: list[str]) -> list[str]:
    """Make `researchos "topic"` and bare `researchos` mean `researchos new ...`."""
    first = next((i for i, arg in enumerate(argv) if arg not in _GLOBAL_FLAGS), None)
    if first is None:
        return argv if {"-h", "--help"} & set(argv) else [*argv, "new"]
    if argv[first] in COMMANDS:
        return argv
    return [*argv[:first], "new", *argv[first:]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchos",
        description='Research a topic: researchos "how Apple\'s M5 chip works"',
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print results")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="research a new topic (the default command)")
    new.add_argument("topic", nargs="?", help="what to research; asked for if omitted")
    new.add_argument("--goal", help="what you want to be able to do or understand")
    new.add_argument("--level", help="your current knowledge of the topic")

    improve = commands.add_parser("improve", help="build on an existing project")
    improve.add_argument("project", help="project id or path")
    improve.add_argument("request", help='e.g. "go deeper on the GPU" or "explain it simpler"')

    resume = commands.add_parser("resume", help="continue a project that stopped early")
    resume.add_argument("project", help="project id or path")
    resume.add_argument("instruction", nargs="?", help="optional instruction for this run")

    steer = commands.add_parser("steer", help="give an instruction to a running project")
    steer.add_argument("project", help="project id or path")
    steer.add_argument("instruction", help='e.g. "focus on memory bandwidth"')

    build = commands.add_parser("build", help="regenerate a project's website")
    build.add_argument("project", help="project id or path")
    build.add_argument("--open", action="store_true", help="open the website in a browser")

    status = commands.add_parser("status", help="show a project's state")
    status.add_argument("project", help="project id or path")

    commands.add_parser("list", help="list research projects")
    return parser
