"""Search passages and show who cites them and who merely resembles them."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .embed import Embedder
from .link import PassageMatcher
from .store import MATCHED, Index, PassageRow


@dataclass(frozen=True)
class Reference:
    article_id: str
    claim: str
    score: float


@dataclass(frozen=True)
class SearchHit:
    passage: PassageRow
    score: float
    direct: list[Reference]  # claims that cite this article and match this passage
    repetition: list[Reference]  # claims that resemble it with no citation path


def search(index: Index, embedder: Embedder, query: str, limit: int = 5) -> list[SearchHit]:
    if not query.strip():
        raise ValueError("The search query is empty.")
    vector = embedder.encode([query])[0]
    matcher = PassageMatcher(index)
    matches = [m for article in index.articles() for m in matcher.rank(vector, article.id)]
    matches.sort(key=lambda m: (-m.score, m.passage_id))

    claims = {claim.id: claim for claim in index.claims()}
    passages = {passage.id: passage for passage in index.passages()}
    direct: dict[int, list[Reference]] = defaultdict(list)
    for citation in index.citations():
        if citation.status == MATCHED:
            claim = claims[citation.claim_id]
            direct[citation.passage_id].append(Reference(claim.article_id, claim.text, citation.score))
    repetition: dict[int, list[Reference]] = defaultdict(list)
    for link in index.similar_links():
        claim = claims[link.claim_id]
        repetition[link.passage_id].append(Reference(claim.article_id, claim.text, link.score))

    hits = []
    for match in matches[:limit]:
        hits.append(
            SearchHit(
                passages[match.passage_id],
                match.score,
                sorted(direct[match.passage_id], key=lambda r: -r.score),
                sorted(repetition[match.passage_id], key=lambda r: -r.score),
            )
        )
    return hits
