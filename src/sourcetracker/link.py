"""Connect claims to the passages they rely on and to text that resembles them."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .embed import Thresholds
from .graph import reachable
from .parse import url_key
from .store import EXTERNAL, MATCHED, SELF, UNMATCHED, ArticleRow, CitationRow, ClaimRow, Index

MIN_REPETITION_WORDS = 6  # shorter sentences resemble each other by chance
MAX_REPETITIONS_PER_CLAIM = 5


@dataclass(frozen=True)
class Match:
    passage_id: int
    score: float
    evidence_claim_id: int  # the sentence in the passage closest to the claim


class PassageMatcher:
    """Rank the passages of an article against a claim."""

    def __init__(self, index: Index) -> None:
        passage_ids, self._passages = index.passage_vectors()
        claim_ids, self._claims = index.claim_vectors()
        self._passage_position = {p: i for i, p in enumerate(passage_ids)}
        self._claim_position = {c: i for i, c in enumerate(claim_ids)}
        self._article_passages: dict[str, list[int]] = defaultdict(list)
        for passage in index.passages():
            self._article_passages[passage.article_id].append(passage.id)
        self._passage_claims: dict[int, list[int]] = defaultdict(list)
        for claim in index.claims():
            self._passage_claims[claim.passage_id].append(claim.id)

    def claim_vector(self, claim_id: int) -> np.ndarray:
        return self._claims[self._claim_position[claim_id]]

    def rank(self, vector: np.ndarray, article_id: str) -> list[Match]:
        """Return the article's passages, best first.

        A passage scores as high as its best sentence, because one matching
        sentence can be diluted when the whole passage is embedded together.
        """
        matches = []
        for passage_id in self._article_passages.get(article_id, []):
            claim_ids = self._passage_claims[passage_id]
            sentence_scores = self._claims[[self._claim_position[c] for c in claim_ids]] @ vector
            best = int(np.argmax(sentence_scores))
            passage_score = float(self._passages[self._passage_position[passage_id]] @ vector)
            score = max(passage_score, float(sentence_scores[best]))
            matches.append(Match(passage_id, score, claim_ids[best]))
        matches.sort(key=lambda m: (-m.score, m.passage_id))
        return matches


def resolve_targets(articles: list[ArticleRow]) -> dict[str, str]:
    """Map every way of naming an article to its id."""
    names = {}
    for article in articles:
        names[article.id] = article.id
        if article.url:
            names[url_key(article.url)] = article.id
    return names


def citation_edges(claims: list[ClaimRow], citations: list[CitationRow]) -> dict[str, set[str]]:
    """Build the article-level citation graph from resolved citations."""
    article_of = {claim.id: claim.article_id for claim in claims}
    edges: dict[str, set[str]] = defaultdict(set)
    for citation in citations:
        if citation.target_article_id and citation.status != SELF:
            edges[article_of[citation.claim_id]].add(citation.target_article_id)
    return dict(edges)


def _link_citations(index: Index, matcher: PassageMatcher, threshold: float) -> None:
    names = resolve_targets(index.articles())
    claims = {claim.id: claim for claim in index.claims()}
    for citation in index.citations():
        target_article = names.get(citation.target)
        if target_article is None:
            index.resolve_citation(citation.id, None, EXTERNAL)
        elif target_article == claims[citation.claim_id].article_id:
            index.resolve_citation(citation.id, target_article, SELF)
        else:
            ranked = matcher.rank(matcher.claim_vector(citation.claim_id), target_article)
            best = ranked[0] if ranked else None
            if best is not None and best.score >= threshold:
                index.resolve_citation(
                    citation.id, target_article, MATCHED, best.passage_id,
                    best.evidence_claim_id, best.score,
                )
            else:
                score = best.score if best else None
                index.resolve_citation(citation.id, target_article, UNMATCHED, score=score)


def _explained_by_citations(
    a: ClaimRow, b: ClaimRow, reach: Mapping[int, set[str]]
) -> bool:
    """Tell whether a citation path accounts for two claims saying the same thing."""
    reach_a = reach.get(a.id, set())
    reach_b = reach.get(b.id, set())
    return b.article_id in reach_a or a.article_id in reach_b or bool(reach_a & reach_b)


def _link_repetition(index: Index, threshold: float) -> None:
    """Record text that resembles a claim where no citation path explains it."""
    claims = index.claims()
    citations = index.citations()
    if not claims:
        return
    _, vectors = index.claim_vectors()
    edges = citation_edges(claims, citations)
    cited: dict[int, set[str]] = defaultdict(set)
    for citation in citations:
        if citation.target_article_id and citation.status != SELF:
            cited[citation.claim_id].add(citation.target_article_id)
    reach = {claim_id: reachable(edges, targets) for claim_id, targets in cited.items()}

    eligible = np.array([len(c.text.split()) >= MIN_REPETITION_WORDS for c in claims])
    for i, claim in enumerate(claims):
        if not eligible[i]:
            continue
        scores = vectors @ vectors[i]
        best_by_passage: dict[int, float] = {}
        for j in np.flatnonzero((scores >= threshold) & eligible):
            other = claims[int(j)]
            if other.article_id == claim.article_id:
                continue
            if _explained_by_citations(claim, other, reach):
                continue
            score = float(scores[j])
            if score > best_by_passage.get(other.passage_id, -1.0):
                best_by_passage[other.passage_id] = score
        ranked = sorted(best_by_passage.items(), key=lambda item: (-item[1], item[0]))
        for passage_id, score in ranked[:MAX_REPETITIONS_PER_CLAIM]:
            index.add_similar_link(claim.id, passage_id, score)


def _link_premises(index: Index, threshold: float) -> None:
    """Tie each conclusion to the earlier cited claims that bear on it."""
    claims = index.claims()
    citing = {c.claim_id for c in index.citations() if c.status != SELF}
    claim_ids, vectors = index.claim_vectors()
    position = {claim_id: i for i, claim_id in enumerate(claim_ids)}
    by_article: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in claims:
        by_article[claim.article_id].append(claim)
    for claim in claims:
        if not claim.is_conclusion:
            continue
        for earlier in by_article[claim.article_id]:
            if earlier.ord >= claim.ord or earlier.id not in citing:
                continue
            score = float(vectors[position[claim.id]] @ vectors[position[earlier.id]])
            if score >= threshold:
                index.add_premise(claim.id, earlier.id, score)


def link_index(index: Index, thresholds: Thresholds) -> None:
    matcher = PassageMatcher(index)
    _link_citations(index, matcher, thresholds.match)
    _link_repetition(index, thresholds.similar)
    _link_premises(index, thresholds.premise)
