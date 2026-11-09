"""Which claims hold, and what breaks when a source is taken away.

A claim holds if at least one of its supports holds. A support either rests
directly on a source or depends on another claim that must itself hold.
Support spreads outward from the direct ones, so articles that only cite each
other never support anything, however many citations they contain.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from .store import EXTERNAL, MATCHED, SELF, Index

INTACT = "intact"
WEAKENED = "weakened"
LOST = "lost"


@dataclass(frozen=True)
class Support:
    source: str | None  # article id or external target; None for a conclusion's premise
    depends_on: int | None = None  # claim that must hold for this support to hold


@dataclass(frozen=True)
class ConclusionImpact:
    claim_id: int
    status: str  # WEAKENED or LOST
    supports_before: int
    supports_after: int
    path: tuple[int, ...]  # claims leading from the conclusion to the removed source
    removed_source: str


@dataclass(frozen=True)
class Impact:
    removed: frozenset[str]
    conclusions: list[ConclusionImpact]
    claims_lost: list[int]  # every claim that held before and doesn't now


@dataclass(frozen=True)
class SourceRisk:
    source: str
    lost: int
    weakened: int
    claims_lost: int


@dataclass
class SupportGraph:
    article_of: dict[int, str]
    supports: dict[int, list[Support]]
    conclusions: set[int]
    _dependents: dict[int, list[tuple[int, Support]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dependents = defaultdict(list)
        for claim, supports in self.supports.items():
            for support in supports:
                if support.depends_on is not None:
                    self._dependents[support.depends_on].append((claim, support))

    @classmethod
    def from_index(cls, index: Index) -> "SupportGraph":
        claims = index.claims()
        citations = index.citations()
        premises = index.premises()
        # A cited sentence, or a conclusion drawn from premises, needs support of
        # its own. Any other sentence is simply what the source says.
        needs_support = {c.claim_id for c in citations if c.status != SELF}
        needs_support |= {p.conclusion_id for p in premises}

        supports: dict[int, list[Support]] = defaultdict(list)
        for citation in citations:
            if citation.status == MATCHED:
                evidence = citation.evidence_claim_id
                depends_on = evidence if evidence in needs_support else None
                support = Support(citation.target_article_id, depends_on)
            elif citation.status == EXTERNAL:
                support = Support(citation.target)
            else:
                continue
            if support not in supports[citation.claim_id]:
                supports[citation.claim_id].append(support)
        for premise in premises:
            supports[premise.conclusion_id].append(Support(None, premise.claim_id))
        return cls(
            article_of={c.id: c.article_id for c in claims},
            supports=dict(supports),
            conclusions={c.id for c in claims if c.is_conclusion},
        )

    def holding(self, removed: Collection[str] = frozenset()) -> set[int]:
        """Return the claims that hold once the given sources are gone."""
        held: set[int] = set()
        queue: deque[int] = deque()
        for claim, supports in self.supports.items():
            if self.article_of[claim] in removed:
                continue
            if any(s.depends_on is None and s.source not in removed for s in supports):
                held.add(claim)
                queue.append(claim)
        while queue:
            for dependent, support in self._dependents[queue.popleft()]:
                if (
                    dependent not in held
                    and self.article_of[dependent] not in removed
                    and support.source not in removed
                ):
                    held.add(dependent)
                    queue.append(dependent)
        return held

    def _live(self, claim: int, held: set[int], removed: Collection[str]) -> list[Support]:
        return [
            s
            for s in self.supports.get(claim, [])
            if s.source not in removed and (s.depends_on is None or s.depends_on in held)
        ]

    def _path_to_removed(
        self, claim: int, before: set[int], after: set[int], removed: Collection[str]
    ) -> tuple[tuple[int, ...], str]:
        """Find the shortest chain from a claim to a source that was removed."""
        parents: dict[int, int | None] = {claim: None}
        queue = deque([claim])
        while queue:
            current = queue.popleft()
            for support in self.supports.get(current, []):
                if support.source in removed:
                    chain = [current]
                    while parents[chain[-1]] is not None:
                        chain.append(parents[chain[-1]])
                    return tuple(reversed(chain)), support.source
                next_claim = support.depends_on
                if (
                    next_claim is not None
                    and next_claim in before
                    and next_claim not in after
                    and next_claim not in parents
                ):
                    parents[next_claim] = current
                    queue.append(next_claim)
        raise AssertionError(f"Claim {claim} lost support without reaching a removed source")

    def impact(self, removed: Iterable[str]) -> Impact:
        """Compare the graph with and without the given sources."""
        removed = frozenset(removed)
        before = self.holding()
        after = self.holding(removed)
        conclusions = []
        for claim in sorted(self.conclusions):
            if self.article_of[claim] in removed:
                continue
            live_before = self._live(claim, before, frozenset())
            live_after = self._live(claim, after, removed)
            if len(live_after) == len(live_before):
                continue
            path, source = self._path_to_removed(claim, before, after, removed)
            status = LOST if not live_after else WEAKENED
            conclusions.append(
                ConclusionImpact(claim, status, len(live_before), len(live_after), path, source)
            )
        conclusions.sort(key=lambda c: (c.status != LOST, c.claim_id))
        claims_lost = sorted(c for c in before - after if self.article_of[c] not in removed)
        return Impact(removed, conclusions, claims_lost)

    def sources(self) -> set[str]:
        return {
            support.source
            for supports in self.supports.values()
            for support in supports
            if support.source is not None
        }

    def conclusion_status(self, claim: int) -> tuple[int, bool]:
        """Return how many supports a conclusion has and whether it holds."""
        held = self.holding()
        return len(self._live(claim, held, frozenset())), claim in held

    def source_risks(self) -> list[SourceRisk]:
        """Rank sources by how much would fall if each one disappeared."""
        risks = []
        for source in self.sources():
            impact = self.impact([source])
            lost = sum(c.status == LOST for c in impact.conclusions)
            weakened = sum(c.status == WEAKENED for c in impact.conclusions)
            risks.append(SourceRisk(source, lost, weakened, len(impact.claims_lost)))
        risks.sort(key=lambda r: (-r.lost, -r.weakened, -r.claims_lost, r.source))
        return risks


def reachable(edges: Mapping[str, Iterable[str]], starts: Iterable[str]) -> set[str]:
    """Return the starting nodes and everything they lead to."""
    seen = set(starts)
    queue = deque(seen)
    while queue:
        for neighbor in edges.get(queue.popleft(), ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def citation_loops(edges: Mapping[str, Iterable[str]]) -> list[list[str]]:
    """Return groups of articles that cite each other, in or across several hops.

    Tarjan's strongly connected components algorithm, written iteratively so a
    long citation chain can't hit Python's recursion limit.
    """
    nodes = set(edges) | {n for targets in edges.values() for n in targets}
    order: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    loops: list[list[str]] = []

    def enter(node: str) -> None:
        order[node] = low[node] = len(order)
        stack.append(node)
        on_stack.add(node)

    for root in sorted(nodes):
        if root in order:
            continue
        enter(root)
        work = [(root, iter(sorted(edges.get(root, ()))))]
        while work:
            node, neighbors = work[-1]
            for neighbor in neighbors:
                if neighbor not in order:
                    enter(neighbor)
                    work.append((neighbor, iter(sorted(edges.get(neighbor, ())))))
                    break
                if neighbor in on_stack:
                    low[node] = min(low[node], order[neighbor])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == order[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        loops.append(sorted(component))
    return sorted(loops)
