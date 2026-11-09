"""Score the index against labels written by a person who read the articles.

A label file looks like this. List every citation in each labeled article,
because anything the tool found beyond those counts as a false positive.

    {
      "articles": {
        "some-article": {
          "citations": [
            {"claim": "unique words from the sentence", "target": "cited-article",
             "label": "supported", "gold_quote": "words from the passage backing it"},
            {"claim": "another sentence", "target": "cited-article", "label": "not_supported"}
          ],
          "repetitions": [{"claim": "uncited sentence", "source": "article-it-repeats"}]
        }
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .link import PassageMatcher, resolve_targets
from .parse import citation_key
from .store import EXTERNAL, MATCHED, UNMATCHED, ClaimRow, Index

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float | None:
        found = self.true_positive + self.false_positive
        return self.true_positive / found if found else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass(frozen=True)
class Report:
    articles: int
    citation_links: Counts
    passage_examples: int
    passage_top1: float | None
    passage_mrr: float | None
    unsupported_flags: Counts
    repetition: Counts
    misses: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        def ratio(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        def line(name: str, counts: Counts) -> str:
            return (
                f"{name:<27} precision {ratio(counts.precision)}  recall {ratio(counts.recall)}  "
                f"f1 {ratio(counts.f1)}  (tp {counts.true_positive}, fp {counts.false_positive}, "
                f"fn {counts.false_negative})"
            )

        lines = [
            f"Labeled articles: {self.articles}",
            line("Citation links", self.citation_links),
            f"{'Passage matches':<27} top-1 {ratio(self.passage_top1)}  MRR {ratio(self.passage_mrr)}  "
            f"(n {self.passage_examples})",
            line("Unsupported-citation flags", self.unsupported_flags),
            line("Uncited repetition", self.repetition),
        ]
        if self.misses:
            lines += ["", "Disagreements with the labels:"] + [f"  {miss}" for miss in self.misses]
        return "\n".join(lines)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _find_claim(claims: list[ClaimRow], article_id: str, fragment: str) -> ClaimRow:
    wanted = _normalize(fragment)
    if not wanted:
        raise LabelError(f"A label in {article_id} has an empty claim.")
    found = [c for c in claims if c.article_id == article_id and wanted in _normalize(c.text)]
    if len(found) != 1:
        problem = "matches no sentence" if not found else f"matches {len(found)} sentences"
        raise LabelError(f"Claim {fragment!r} in {article_id} {problem}.")
    return found[0]


def _counts(gold: set, predicted: set) -> Counts:
    return Counts(len(gold & predicted), len(predicted - gold), len(gold - predicted))


def _target_name(names: dict[str, str], target: str) -> str:
    """Resolve a label's target the same way a citation's target is resolved."""
    if target in names:
        return names[target]
    key = citation_key(target)
    return names.get(key, key) if key else target


def evaluate(index: Index, labels: dict) -> Report:
    articles = labels.get("articles")
    if not articles:
        raise LabelError("The label file has no articles.")
    claims = index.claims()
    known = {article.id for article in index.articles()}
    names = resolve_targets(index.articles())
    passages = {passage.id: passage for passage in index.passages()}
    matcher = PassageMatcher(index)
    predicted_by_key = {
        (c.claim_id, c.target_article_id or c.target): c for c in index.citations()
    }
    article_of = {claim.id: claim.article_id for claim in claims}
    text_of = {claim.id: claim.text for claim in claims}

    gold_links: set[tuple[int, str]] = set()
    gold_repetition: set[tuple[int, str]] = set()
    gold_unsupported: set[tuple[int, str]] = set()
    judged: set[tuple[int, str]] = set()
    reciprocal_ranks: list[float] = []
    misses: list[str] = []

    for article_id, entry in articles.items():
        if article_id not in known:
            raise LabelError(f"Labeled article {article_id!r} is not in the index.")
        for label in entry.get("citations", []):
            verdict = label.get("label")
            if verdict not in (SUPPORTED, NOT_SUPPORTED):
                raise LabelError(
                    f"Citation {label.get('claim')!r} in {article_id} needs a label of "
                    f"{SUPPORTED!r} or {NOT_SUPPORTED!r}."
                )
            claim = _find_claim(claims, article_id, label.get("claim", ""))
            target = _target_name(names, label.get("target", ""))
            key = (claim.id, target)
            gold_links.add(key)
            if verdict == NOT_SUPPORTED:
                gold_unsupported.add(key)
            citation = predicted_by_key.get(key)
            if citation is None or citation.status == EXTERNAL:
                continue
            if citation.status not in (MATCHED, UNMATCHED):
                continue
            judged.add(key)
            if verdict == SUPPORTED:
                quote = _normalize(label.get("gold_quote", ""))
                if not quote:
                    raise LabelError(f"Supported citation {label['claim']!r} needs a gold_quote.")
                ranked = matcher.rank(matcher.claim_vector(claim.id), target)
                ranks = [
                    position
                    for position, match in enumerate(ranked, start=1)
                    if quote in _normalize(passages[match.passage_id].text)
                ]
                if not ranks:
                    raise LabelError(f"gold_quote {label['gold_quote']!r} is not in {target}.")
                reciprocal_ranks.append(1 / ranks[0])
                if ranks[0] != 1:
                    misses.append(f"passage ranked {ranks[0]}: {claim.text}")
        for label in entry.get("repetitions", []):
            claim = _find_claim(claims, article_id, label.get("claim", ""))
            gold_repetition.add((claim.id, _target_name(names, label.get("source", ""))))

    predicted_links = {
        key for key in predicted_by_key if article_of[key[0]] in articles
    }
    predicted_unsupported = {
        key for key in judged if predicted_by_key[key].status == UNMATCHED
    }
    for key in sorted((gold_unsupported & judged) - predicted_unsupported):
        misses.append(f"unsupported citation was matched: {text_of[key[0]]} -> {key[1]}")
    for key in sorted(predicted_unsupported - gold_unsupported):
        misses.append(f"supported citation was flagged: {text_of[key[0]]} -> {key[1]}")

    passage_article = {p.id: p.article_id for p in passages.values()}
    predicted_repetition = {
        (link.claim_id, passage_article[link.passage_id])
        for link in index.similar_links()
        if article_of[link.claim_id] in articles
    }
    for key in sorted(gold_repetition - predicted_repetition):
        misses.append(f"repetition not found: {text_of[key[0]]} -> {key[1]}")
    for key in sorted(predicted_repetition - gold_repetition):
        misses.append(f"unlabeled repetition reported: {text_of[key[0]]} -> {key[1]}")

    count = len(reciprocal_ranks)
    return Report(
        articles=len(articles),
        citation_links=_counts(gold_links, predicted_links),
        passage_examples=count,
        passage_top1=sum(r == 1 for r in reciprocal_ranks) / count if count else None,
        passage_mrr=sum(reciprocal_ranks) / count if count else None,
        unsupported_flags=_counts(gold_unsupported & judged, predicted_unsupported),
        repetition=_counts(gold_repetition, predicted_repetition),
        misses=misses,
    )


def label_template(index: Index, article_ids: list[str]) -> dict:
    """Write out each citation with blank labels for a person to fill in.

    Scores and suggested passages are left out on purpose, so the tool's own
    answers can't steer the person labeling it.
    """
    known = {article.id for article in index.articles()}
    unknown = sorted(set(article_ids) - known)
    if unknown:
        raise LabelError(f"Not in the index: {', '.join(unknown)}")
    claims = {claim.id: claim for claim in index.claims()}
    output: dict = {
        "_instructions": (
            "Read each article and its sources yourself. Set label to 'supported' or "
            "'not_supported'. For supported citations, copy a phrase from the backing "
            "passage into gold_quote. Add any citation the tool missed, delete any "
            "that isn't real, and list uncited repetition under repetitions."
        ),
        "articles": {a: {"citations": [], "repetitions": []} for a in article_ids},
    }
    for citation in index.citations():
        claim = claims[citation.claim_id]
        if claim.article_id in output["articles"]:
            output["articles"][claim.article_id]["citations"].append(
                {
                    "claim": claim.text,
                    "target": citation.target_article_id or citation.target,
                    "label": "",
                    "gold_quote": "",
                }
            )
    return output
