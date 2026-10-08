"""Usage accounting and budget enforcement for a research run.

Every LLM and tool call is appended to an events log (JSON Lines) so usage can be audited
afterwards and later reconciled against an AI gateway's own metering.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchos.llm import Usage


@dataclass(frozen=True)
class Limits:
    max_steps: int
    max_cost_usd: float
    max_wall_seconds: float


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    output_per_million: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class UsageSummary:
    steps: int
    llm_calls: int
    tool_calls: int
    tool_errors: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_seconds: float


class BudgetExceeded(Exception):
    pass


class UsageMeter:
    def __init__(
        self,
        *,
        limits: Limits,
        pricing: Pricing,
        events_path: Path,
        run_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits
        self._pricing = pricing
        self._events_path = events_path
        self._run_id = run_id
        self._clock = clock
        self._started = clock()
        self._steps = 0
        self._llm_calls = 0
        self._tool_calls = 0
        self._tool_errors = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost = 0.0

    def start_step(self) -> int:
        """Begin a new agent step, raising `BudgetExceeded` if any limit is already reached.

        Cost is only known after a call returns, so a single call can overshoot the cost limit;
        the next step is then refused."""
        summary = self.summary()
        if summary.steps >= self.limits.max_steps:
            raise BudgetExceeded(f"step limit of {self.limits.max_steps} reached")
        if summary.cost_usd >= self.limits.max_cost_usd:
            raise BudgetExceeded(f"cost limit of ${self.limits.max_cost_usd:.2f} reached")
        if summary.elapsed_seconds >= self.limits.max_wall_seconds:
            raise BudgetExceeded(f"time limit of {self.limits.max_wall_seconds:.0f}s reached")
        self._steps += 1
        return self._steps

    def record_llm_call(
        self, *, model: str, usage: Usage, latency_seconds: float, purpose: str = "agent"
    ) -> None:
        """Record a completed model call. `purpose` says what it was for ("agent" for the main
        loop, "study" or "explain" for focused calls made by tools)."""
        cost = self._pricing.cost(usage)
        self._llm_calls += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._cost += cost
        self._log(
            "llm_call",
            purpose=purpose,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            latency_seconds=latency_seconds,
        )

    def record_llm_failure(
        self, *, model: str, error: str, latency_seconds: float, purpose: str = "agent"
    ) -> None:
        self._log(
            "llm_error", purpose=purpose, model=model, error=error, latency_seconds=latency_seconds
        )

    def record_tool_call(self, *, name: str, ok: bool, latency_seconds: float) -> None:
        self._tool_calls += 1
        self._tool_errors += 0 if ok else 1
        self._log("tool_call", tool=name, ok=ok, latency_seconds=latency_seconds)

    def record_run_end(self, *, status: str, reason: str) -> None:
        self._log("run_end", status=status, reason=reason, summary=asdict(self.summary()))

    def summary(self) -> UsageSummary:
        return UsageSummary(
            steps=self._steps,
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            tool_errors=self._tool_errors,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost,
            elapsed_seconds=self._clock() - self._started,
        )

    def _log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self._run_id,
            "step": self._steps,
            "event": event,
            **fields,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
