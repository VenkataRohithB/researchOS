"""How the agent reaches the person it is researching for."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Protocol, TextIO


class UserChannel(Protocol):
    def ask(self, questions: Sequence[str]) -> list[str] | None:
        """Ask questions and return one answer per question (possibly empty), or None if
        nobody is available to answer."""
        ...


class TerminalUser:
    """Asks on the terminal. Only usable when a person is attached to it."""

    def __init__(
        self,
        read: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
    ) -> None:
        self._read = read
        self._out = out

    def ask(self, questions: Sequence[str]) -> list[str] | None:
        self._out.write("\nThe agent has a question for you (press Enter to skip):\n")
        answers = []
        for question in questions:
            try:
                answers.append(self._read(f"\n  {question}\n  > ").strip())
            except EOFError:
                return None
        self._out.write("\n")
        return answers


class NoUser:
    """For unattended runs: questions go unanswered and the agent proceeds on assumptions."""

    def ask(self, questions: Sequence[str]) -> list[str] | None:
        return None


def terminal_user_if_interactive() -> UserChannel:
    return TerminalUser() if sys.stdin.isatty() and sys.stdout.isatty() else NoUser()
