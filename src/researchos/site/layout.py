"""Deterministic layout for the concept map: a radial tree around the research topic.

Top-level concepts sit on the first ring around the topic, their sub-concepts further out,
each subtree given an arc proportional to its number of leaves so branches never overlap.
Computing the layout here keeps the page's script small and the map identical on every load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from researchos.knowledge import Concept

RING = 190.0
"""Distance between rings, in SVG units."""
MARGIN_X = 230.0
"""Room for labels, which extend sideways from their nodes."""
MARGIN_Y = 70.0


@dataclass(frozen=True)
class MapNode:
    id: str
    title: str
    x: float
    y: float
    r: float
    depth: int
    researched: bool


@dataclass(frozen=True)
class MapEdge:
    source: str
    target: str
    kind: str
    """"part" for parent-child links, "prerequisite" for learn-this-first links."""


@dataclass(frozen=True)
class ConceptMap:
    nodes: list[MapNode]
    edges: list[MapEdge]
    view_box: tuple[float, float, float, float]


ROOT = "__topic__"


def radial_map(topic: str, concepts: list[Concept]) -> ConceptMap:
    ids = {c.id for c in concepts}
    children: dict[str, list[Concept]] = {}
    for concept in sorted(concepts, key=lambda c: c.title.lower()):
        parent = concept.parent if concept.parent in ids else ROOT
        children.setdefault(parent, []).append(concept)

    leaves: dict[str, int] = {}

    def count_leaves(node: str) -> int:
        kids = children.get(node, [])
        leaves[node] = sum(count_leaves(k.id) for k in kids) or 1
        return leaves[node]

    count_leaves(ROOT)
    nodes = [MapNode(id=ROOT, title=topic, x=0.0, y=0.0, r=34.0, depth=0, researched=True)]

    def place(node: str, start: float, span: float, depth: int) -> None:
        angle = start
        for child in children.get(node, []):
            share = span * leaves[child.id] / leaves[node]
            middle = angle + share / 2
            radius = RING * depth
            nodes.append(
                MapNode(
                    id=child.id,
                    title=child.title,
                    x=round(radius * math.cos(middle), 1),
                    y=round(radius * math.sin(middle), 1),
                    r=round(9 + 3 * math.sqrt(len(child.claim_ids)), 1),
                    depth=depth,
                    researched=bool(child.claim_ids),
                )
            )
            place(child.id, angle, share, depth + 1)
            angle += share

    place(ROOT, -math.pi / 2, 2 * math.pi, 1)

    edges = [
        MapEdge(source=c.parent if c.parent in ids else ROOT, target=c.id, kind="part")
        for c in concepts
    ]
    edges += [
        MapEdge(source=p, target=c.id, kind="prerequisite")
        for c in concepts
        for p in c.prerequisites
        if p in ids and p != c.parent
    ]

    left = min(n.x for n in nodes) - MARGIN_X
    top = min(n.y for n in nodes) - MARGIN_Y
    width = max(n.x for n in nodes) + MARGIN_X - left
    height = max(n.y for n in nodes) + MARGIN_Y - top
    return ConceptMap(nodes=nodes, edges=edges, view_box=(left, top, width, height))


def learning_order(concepts: list[Concept]) -> list[str]:
    """An order for learning concepts: prerequisites before what needs them, and a broad
    concept before its sub-concepts. Concepts without claims are left out."""
    by_id = {c.id: c for c in concepts}
    children: dict[str | None, list[Concept]] = {}
    for concept in sorted(concepts, key=lambda c: c.title.lower()):
        parent = concept.parent if concept.parent in by_id else None
        children.setdefault(parent, []).append(concept)

    order: list[str] = []
    visiting: set[str] = set()

    def visit(concept_id: str) -> None:
        if concept_id in order or concept_id in visiting:
            return
        visiting.add(concept_id)
        concept = by_id[concept_id]
        for prerequisite in concept.prerequisites:
            if prerequisite in by_id:
                visit(prerequisite)
        order.append(concept_id)
        for child in children.get(concept_id, []):
            visit(child.id)
        visiting.discard(concept_id)

    for top in children.get(None, []):
        visit(top.id)
    return [cid for cid in order if by_id[cid].claim_ids or by_id[cid].explanations]
