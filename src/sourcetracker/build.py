"""Build a searchable index from a directory of articles."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from .embed import Embedder, Thresholds
from .link import link_index
from .parse import load_corpus
from .store import Index


@dataclass(frozen=True)
class BuildSummary:
    articles: int
    passages: int
    claims: int
    citations: int


def build_index(
    corpus: Path, database: Path, embedder: Embedder, thresholds: Thresholds | None = None
) -> BuildSummary:
    """Index a corpus, replacing the database only if the whole build succeeds."""
    thresholds = thresholds or embedder.thresholds
    articles = load_corpus(corpus)
    if not articles:
        raise ValueError(f"No articles found in {corpus}")

    passages = [p for article in articles for p in article.passages]
    sentences = [s for passage in passages for s in passage.sentences]
    passage_vectors = iter(embedder.encode([p.text for p in passages]))
    sentence_vectors = iter(embedder.encode([s.text for s in sentences]))

    temporary = database.with_name(database.name + ".building")
    temporary.unlink(missing_ok=True)
    index = Index.create(temporary)
    citations = 0
    try:
        index.set_meta("embedder", embedder.name)
        index.set_meta("thresholds", json.dumps(asdict(thresholds)))
        for article in articles:
            index.add_article(article.id, article.title, article.url, article.published, article.path)
            claim_order = 0
            for passage_order, passage in enumerate(article.passages):
                passage_id = index.add_passage(
                    article.id, passage_order, passage.section, passage.text, next(passage_vectors)
                )
                for sentence in passage.sentences:
                    claim_id = index.add_claim(
                        article.id, passage_id, claim_order, sentence.text,
                        sentence.is_conclusion, next(sentence_vectors),
                    )
                    claim_order += 1
                    for citation in sentence.citations:
                        index.add_citation(claim_id, citation.target, citation.anchor)
                        citations += 1
        link_index(index, thresholds)
        index.commit()
    except BaseException:
        index.close()
        temporary.unlink(missing_ok=True)
        raise
    index.close()
    os.replace(temporary, database)
    return BuildSummary(len(articles), len(passages), len(sentences), citations)
