"""SQLite storage for articles, embeddings, and the links between them.

Direct references and text-similarity matches live in different tables
(``citations`` and ``similar_links``), so a similarity match can never be
counted as support by accident.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1"

MATCHED = "matched"  # the cited article contains a passage backing the claim
UNMATCHED = "unmatched"  # the cited article is indexed but no passage backs the claim
EXTERNAL = "external"  # the target is outside the corpus, so it can't be checked
SELF = "self"  # the article cites itself

_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE articles(
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    published TEXT,
    path TEXT NOT NULL
);

CREATE TABLE passages(
    id INTEGER PRIMARY KEY,
    article_id TEXT NOT NULL REFERENCES articles(id),
    ord INTEGER NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE claims(
    id INTEGER PRIMARY KEY,
    article_id TEXT NOT NULL REFERENCES articles(id),
    passage_id INTEGER NOT NULL REFERENCES passages(id),
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    is_conclusion INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE citations(
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    target TEXT NOT NULL,
    anchor TEXT NOT NULL,
    target_article_id TEXT REFERENCES articles(id),
    status TEXT CHECK(status IN ('matched', 'unmatched', 'external', 'self')),
    passage_id INTEGER REFERENCES passages(id),
    evidence_claim_id INTEGER REFERENCES claims(id),
    score REAL
);

CREATE TABLE similar_links(
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    passage_id INTEGER NOT NULL REFERENCES passages(id),
    score REAL NOT NULL,
    PRIMARY KEY(claim_id, passage_id)
);

CREATE TABLE premises(
    conclusion_id INTEGER NOT NULL REFERENCES claims(id),
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    score REAL NOT NULL,
    PRIMARY KEY(conclusion_id, claim_id)
);

CREATE INDEX idx_passages_article ON passages(article_id);
CREATE INDEX idx_claims_article ON claims(article_id);
CREATE INDEX idx_citations_claim ON citations(claim_id);
CREATE INDEX idx_citations_passage ON citations(passage_id);
CREATE INDEX idx_similar_passage ON similar_links(passage_id);
"""


@dataclass(frozen=True)
class ArticleRow:
    id: str
    title: str
    url: str | None
    published: str | None
    path: str


@dataclass(frozen=True)
class PassageRow:
    id: int
    article_id: str
    ord: int
    section: str
    text: str


@dataclass(frozen=True)
class ClaimRow:
    id: int
    article_id: str
    passage_id: int
    ord: int
    text: str
    is_conclusion: bool


@dataclass(frozen=True)
class CitationRow:
    id: int
    claim_id: int
    target: str
    anchor: str
    target_article_id: str | None
    status: str | None
    passage_id: int | None
    evidence_claim_id: int | None
    score: float | None


@dataclass(frozen=True)
class SimilarLink:
    claim_id: int
    passage_id: int
    score: float


@dataclass(frozen=True)
class Premise:
    conclusion_id: int
    claim_id: int
    score: float


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class Index:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(cls, path: Path) -> "Index":
        connection = sqlite3.connect(path)
        connection.executescript(_SCHEMA)
        index = cls(connection)
        index.set_meta("schema_version", SCHEMA_VERSION)
        return index

    @classmethod
    def open(cls, path: Path) -> "Index":
        if not Path(path).is_file():
            raise FileNotFoundError(f"No index at {path}. Run `sourcetracker build` first.")
        index = cls(sqlite3.connect(path))
        try:
            version = index.meta("schema_version")
        except (sqlite3.DatabaseError, KeyError) as error:
            index.close()
            raise ValueError(f"{path} is not a SourceTracker index.") from error
        if version != SCHEMA_VERSION:
            index.close()
            raise ValueError(f"{path} uses schema {version}; rebuild it with `sourcetracker build`.")
        return index

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def commit(self) -> None:
        self.connection.commit()

    # Metadata --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def meta(self, key: str) -> str:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    # Writing ---------------------------------------------------------------

    def add_article(self, id: str, title: str, url: str | None, published: str | None, path: str) -> None:
        self.connection.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?)", (id, title, url, published, path)
        )

    def add_passage(self, article_id: str, ord: int, section: str, text: str, vector: np.ndarray) -> int:
        cursor = self.connection.execute(
            "INSERT INTO passages(article_id, ord, section, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (article_id, ord, section, text, _to_blob(vector)),
        )
        return int(cursor.lastrowid)

    def add_claim(
        self, article_id: str, passage_id: int, ord: int, text: str, is_conclusion: bool, vector: np.ndarray
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO claims(article_id, passage_id, ord, text, is_conclusion, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (article_id, passage_id, ord, text, int(is_conclusion), _to_blob(vector)),
        )
        return int(cursor.lastrowid)

    def add_citation(self, claim_id: int, target: str, anchor: str) -> None:
        self.connection.execute(
            "INSERT INTO citations(claim_id, target, anchor) VALUES (?, ?, ?)",
            (claim_id, target, anchor),
        )

    def resolve_citation(
        self,
        citation_id: int,
        target_article_id: str | None,
        status: str,
        passage_id: int | None = None,
        evidence_claim_id: int | None = None,
        score: float | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE citations SET target_article_id = ?, status = ?, passage_id = ?, "
            "evidence_claim_id = ?, score = ? WHERE id = ?",
            (target_article_id, status, passage_id, evidence_claim_id, score, citation_id),
        )

    def add_similar_link(self, claim_id: int, passage_id: int, score: float) -> None:
        self.connection.execute(
            "INSERT INTO similar_links VALUES (?, ?, ?)", (claim_id, passage_id, score)
        )

    def add_premise(self, conclusion_id: int, claim_id: int, score: float) -> None:
        self.connection.execute(
            "INSERT INTO premises VALUES (?, ?, ?)", (conclusion_id, claim_id, score)
        )

    # Reading ---------------------------------------------------------------

    def articles(self) -> list[ArticleRow]:
        rows = self.connection.execute(
            "SELECT id, title, url, published, path FROM articles ORDER BY id"
        )
        return [ArticleRow(*row) for row in rows]

    def passages(self) -> list[PassageRow]:
        rows = self.connection.execute(
            "SELECT id, article_id, ord, section, text FROM passages ORDER BY id"
        )
        return [PassageRow(*row) for row in rows]

    def claims(self) -> list[ClaimRow]:
        rows = self.connection.execute(
            "SELECT id, article_id, passage_id, ord, text, is_conclusion FROM claims ORDER BY id"
        )
        return [ClaimRow(*row[:5], bool(row[5])) for row in rows]

    def citations(self) -> list[CitationRow]:
        rows = self.connection.execute(
            "SELECT id, claim_id, target, anchor, target_article_id, status, passage_id, "
            "evidence_claim_id, score FROM citations ORDER BY id"
        )
        return [CitationRow(*row) for row in rows]

    def similar_links(self) -> list[SimilarLink]:
        rows = self.connection.execute(
            "SELECT claim_id, passage_id, score FROM similar_links ORDER BY claim_id, score DESC"
        )
        return [SimilarLink(*row) for row in rows]

    def premises(self) -> list[Premise]:
        rows = self.connection.execute(
            "SELECT conclusion_id, claim_id, score FROM premises ORDER BY conclusion_id, claim_id"
        )
        return [Premise(*row) for row in rows]

    def _vectors(self, table: str) -> tuple[list[int], np.ndarray]:
        rows = self.connection.execute(f"SELECT id, embedding FROM {table} ORDER BY id").fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        ids = [row[0] for row in rows]
        return ids, np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])

    def passage_vectors(self) -> tuple[list[int], np.ndarray]:
        return self._vectors("passages")

    def claim_vectors(self) -> tuple[list[int], np.ndarray]:
        return self._vectors("claims")
