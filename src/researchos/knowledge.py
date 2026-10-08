"""Concepts: the units a topic is broken into, and the relationships between them.

Concepts form the knowledge graph used by the agent, the website and the Obsidian export:

- `parent`: the broader concept this one is part of (the "go back" direction). Children of a
  concept are the "go deeper" direction.
- `prerequisites`: concepts to understand first.
- `related`: concepts worth comparing or connecting.

Each concept collects the ids of the claims about it, and later its explanations at several
depths. Concept ids are slugs of their titles, so the same idea named twice maps to one node.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["summary", "beginner", "intermediate", "deep", "expert"]
LEVELS: tuple[Level, ...] = ("summary", "beginner", "intermediate", "deep", "expert")


class Concept(BaseModel):
    id: str
    title: str
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    explanations: dict[Level, str] = Field(default_factory=dict)
    """Markdown per depth level, citing claims as [claim-0001]."""
    created_at: datetime
    updated_at: datetime


def concept_id(title: str) -> str:
    """Stable id for a concept title: lowercase ASCII slug ("Gate-All-Around (GAA)" ->
    "gate-all-around-gaa")."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:80] or "concept"
