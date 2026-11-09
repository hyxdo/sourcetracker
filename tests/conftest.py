from pathlib import Path

import pytest

from sourcetracker.build import build_index
from sourcetracker.embed import HashingEmbedder
from sourcetracker.store import Index


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def make_index(tmp_path, embedder):
    """Build an index from {article id: markdown body} and open it."""
    opened = []

    def build(articles: dict[str, str]) -> Index:
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        for article_id, body in articles.items():
            (corpus / f"{article_id}.md").write_text(body, encoding="utf-8")
        database = tmp_path / "index.db"
        build_index(corpus, database, embedder)
        index = Index.open(database)
        opened.append(index)
        return index

    yield build
    for index in opened:
        index.close()


@pytest.fixture
def example_corpus() -> Path:
    return Path(__file__).parent.parent / "examples" / "corpus"
