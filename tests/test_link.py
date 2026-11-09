import sqlite3

import numpy as np
import pytest

from sourcetracker.build import build_index
from sourcetracker.graph import SupportGraph
from sourcetracker.store import EXTERNAL, MATCHED, SELF, UNMATCHED, Index

STUDY = "Urchin numbers doubled near the northern harbor during the survey.\n\nWater temperature stayed stable through every sampling season."


def by_text(index: Index, fragment: str):
    matches = [c for c in index.claims() if fragment in c.text]
    assert len(matches) == 1, f"{fragment!r} matched {len(matches)} claims"
    return matches[0]


def citations_of(index: Index, fragment: str):
    claim = by_text(index, fragment)
    return [c for c in index.citations() if c.claim_id == claim.id]


def passage_text(index: Index, passage_id: int) -> str:
    return next(p.text for p in index.passages() if p.id == passage_id)


def test_citation_links_to_the_passage_that_backs_the_claim(make_index):
    index = make_index(
        {"study": STUDY, "report": "Urchin numbers doubled near the northern harbor [@study]."}
    )
    citation = citations_of(index, "Urchin numbers doubled near the northern harbor.")
    citation = [c for c in citation if c.target == "study"][0]
    assert citation.status == MATCHED
    assert citation.target_article_id == "study"
    assert "Urchin numbers doubled" in passage_text(index, citation.passage_id)
    assert citation.score > 0.25


def test_citation_to_a_source_that_says_something_else_is_unmatched(make_index):
    index = make_index(
        {"study": STUDY, "report": "The mayor approved parking meters downtown yesterday [@study]."}
    )
    citation = citations_of(index, "parking meters")[0]
    assert citation.status == UNMATCHED
    assert citation.passage_id is None
    assert citation.target_article_id == "study"


def test_citation_outside_the_corpus_is_external(make_index):
    index = make_index({"report": "Kelp canopy cover fell sharply [source](https://elsewhere.org/paper)."})
    citation = citations_of(index, "Kelp canopy")[0]
    assert (citation.status, citation.target_article_id) == (EXTERNAL, None)
    assert citation.target == "elsewhere.org/paper"


def test_citation_resolves_by_url(make_index):
    index = make_index(
        {
            "study": f"---\nurl: https://www.example.org/study/\n---\n{STUDY}",
            "report": "Urchin numbers doubled near the northern harbor [link](http://example.org/study).",
        }
    )
    citations = [c for c in index.citations() if c.target_article_id == "study"]
    assert [c.status for c in citations] == [MATCHED]


def test_self_citation_gives_no_support(make_index):
    index = make_index({"report": "Kelp canopy cover fell sharply this decade [@report]."})
    citation = citations_of(index, "Kelp canopy")[0]
    assert citation.status == SELF
    assert SupportGraph.from_index(index).holding() == set()


def repeated(index: Index) -> set[tuple[str, str]]:
    claims = {c.id: c for c in index.claims()}
    passages = {p.id: p for p in index.passages()}
    return {
        (claims[link.claim_id].article_id, passages[link.passage_id].article_id)
        for link in index.similar_links()
    }


def test_uncited_repetition_is_kept_apart_from_citations(make_index):
    index = make_index(
        {
            "study": STUDY,
            "blog": "Urchin numbers doubled near the northern harbor during the survey, friends.",
        }
    )
    assert ("blog", "study") in repeated(index)
    assert index.citations() == []
    assert SupportGraph.from_index(index).holding() == set()


def dated(date: str, body: str) -> str:
    return f"---\npublished: {date}\n---\n{body}"


REPEATED = "Urchin numbers doubled near the northern harbor during the survey."


def test_only_the_later_article_is_reported_as_repeating(make_index):
    index = make_index(
        {"early": dated("2023-03-14", REPEATED), "late": dated("2023-05-11", REPEATED)}
    )
    assert repeated(index) == {("late", "early")}


def test_repetition_runs_both_ways_when_the_order_is_unknown(make_index):
    index = make_index({"early": dated("2023-03-14", REPEATED), "undated": REPEATED})
    assert repeated(index) == {("early", "undated"), ("undated", "early")}


def test_repetition_runs_both_ways_on_the_same_day(make_index):
    index = make_index(
        {"one": dated("2023-03-14", REPEATED), "two": dated("2023-03-14T09:00:00Z", REPEATED)}
    )
    assert repeated(index) == {("one", "two"), ("two", "one")}


def test_cited_text_is_not_reported_as_repetition(make_index):
    index = make_index(
        {
            "study": STUDY,
            "report": "Urchin numbers doubled near the northern harbor during the survey [@study].",
        }
    )
    assert repeated(index) == set()


def test_text_explained_by_a_citation_chain_is_not_repetition(make_index):
    claim = "Urchin numbers doubled near the northern harbor during the survey"
    index = make_index(
        {"study": STUDY, "middle": f"{claim} [@study].", "last": f"{claim} [@middle]."}
    )
    assert repeated(index) == set()


def test_two_articles_citing_one_source_are_not_repeating_each_other(make_index):
    claim = "Urchin numbers doubled near the northern harbor during the survey"
    index = make_index(
        {"study": STUDY, "one": f"{claim} [@study].", "two": f"{claim} [@study]."}
    )
    assert repeated(index) == set()


def test_a_correctly_cited_claim_is_not_repeating_anyone(make_index):
    index = make_index(
        {"study": REPEATED, "report": f"{REPEATED[:-1]} [@study].", "blog": REPEATED}
    )
    assert ("report", "blog") not in repeated(index)


def test_a_claim_citing_the_wrong_source_can_still_be_repetition(make_index):
    index = make_index(
        {
            "unrelated": "The mayor approved parking meters downtown yesterday afternoon.",
            "original": REPEATED,
            "copier": f"{REPEATED[:-1]} [@unrelated].",
        }
    )
    assert ("copier", "original") in repeated(index)


def test_uncited_text_is_traced_to_the_source_rather_than_its_relay(make_index):
    index = make_index(
        {"study": REPEATED, "relay": f"{REPEATED[:-1]} [@study].", "blog": REPEATED}
    )
    assert repeated(index) == {("blog", "study"), ("study", "blog")}


def test_a_relay_is_reported_when_its_source_does_not_resemble_the_claim(make_index):
    index = make_index(
        {
            "study": "Sampling covered forty fixed transects across the whole northern reef system.",
            "relay": f"{REPEATED[:-1]} [@study].",
            "blog": REPEATED,
        }
    )
    assert ("blog", "relay") in repeated(index)


def test_short_sentences_are_not_treated_as_repetition(make_index):
    index = make_index({"a": "Kelp grew back.", "b": "Kelp grew back."})
    assert repeated(index) == set()


def premises_of(index: Index, fragment: str) -> list[str]:
    conclusion = by_text(index, fragment)
    claims = {c.id: c for c in index.claims()}
    return [claims[p.claim_id].text for p in index.premises() if p.conclusion_id == conclusion.id]


def test_premises_are_earlier_cited_claims_on_the_same_topic(make_index):
    index = make_index(
        {
            "study": STUDY,
            "brief": (
                "Urchin numbers doubled near the northern harbor [@study]. "
                "Parking revenue rose downtown last spring [@study]. "
                "Urchin numbers near the harbor went unmentioned here. "
                "Therefore urchin numbers near the northern harbor need monitoring. "
                "Urchin numbers near the northern harbor doubled again later [@study]."
            ),
        }
    )
    assert premises_of(index, "Therefore urchin") == [
        "Urchin numbers doubled near the northern harbor."
    ]


def test_conclusion_loses_support_through_a_chain(make_index):
    index = make_index(
        {
            "study": STUDY,
            "middle": "Urchin numbers doubled near the northern harbor during the survey [@study].",
            "brief": (
                "Urchin numbers doubled near the northern harbor [@middle]. "
                "Therefore urchin numbers near the northern harbor need monitoring."
            ),
        }
    )
    graph = SupportGraph.from_index(index)
    conclusion = by_text(index, "Therefore urchin")
    assert conclusion.id in graph.holding()
    impact = graph.impact(["study"])
    assert [c.claim_id for c in impact.conclusions] == [conclusion.id]
    assert impact.conclusions[0].status == "lost"
    assert impact.conclusions[0].removed_source == "study"


def test_citing_a_conclusion_depends_on_what_that_conclusion_rests_on(make_index):
    index = make_index(
        {
            "study": STUDY,
            "brief": (
                "Urchin numbers doubled near the northern harbor [@study]. "
                "Therefore urchin numbers near the northern harbor need monitoring."
            ),
            "column": "Urchin numbers near the northern harbor need monitoring [@brief].",
        }
    )
    graph = SupportGraph.from_index(index)
    column = next(c for c in index.claims() if c.article_id == "column")
    assert column.id in graph.holding()
    assert column.id not in graph.holding({"study"})


def test_a_failed_citation_breaks_the_chain_above_it(make_index):
    claim = "Urchin numbers doubled near the northern harbor during the survey"
    index = make_index(
        {
            "unrelated": "The mayor approved parking meters downtown yesterday afternoon.",
            "middle": f"{claim} [@unrelated].",
            "last": f"{claim} [@middle].",
        }
    )
    assert SupportGraph.from_index(index).holding() == set()


def test_uncited_source_sentence_grounds_the_claim(make_index):
    index = make_index(
        {"study": STUDY, "report": "Urchin numbers doubled near the northern harbor [@study]."}
    )
    graph = SupportGraph.from_index(index)
    report = by_text(index, "doubled near the northern harbor.")
    assert report.id in graph.holding()
    assert graph.holding({"study"}) == set()


def test_external_citation_holds_and_can_be_removed_by_url(make_index):
    index = make_index({"report": "Kelp canopy cover fell sharply [source](https://elsewhere.org/paper)."})
    graph = SupportGraph.from_index(index)
    assert len(graph.holding()) == 1
    assert graph.holding({"elsewhere.org/paper"}) == set()


def test_failed_build_keeps_the_existing_index(tmp_path, embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Alpha text sits here.", encoding="utf-8")
    database = tmp_path / "index.db"
    build_index(corpus, database, embedder)

    class Broken:
        name = "broken"
        thresholds = embedder.thresholds

        def encode(self, texts):
            return np.zeros((0, 4), dtype=np.float32)

    with pytest.raises(StopIteration):
        build_index(corpus, database, Broken())

    with Index.open(database) as index:
        assert index.meta("embedder") == "hashing"
    assert list(tmp_path.glob("*.building")) == []


def test_empty_corpus_is_an_error(tmp_path, embedder):
    (tmp_path / "corpus").mkdir()
    with pytest.raises(ValueError, match="No articles"):
        build_index(tmp_path / "corpus", tmp_path / "index.db", embedder)


def test_open_rejects_missing_and_foreign_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="sourcetracker build"):
        Index.open(tmp_path / "missing.db")
    foreign = tmp_path / "foreign.db"
    connection = sqlite3.connect(foreign)
    connection.execute("CREATE TABLE other(x)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="not a SourceTracker index"):
        Index.open(foreign)


def test_vectors_round_trip_through_sqlite(make_index, embedder):
    index = make_index({"a": "Alpha text sits here."})
    ids, vectors = index.claim_vectors()
    assert len(ids) == 1
    assert np.allclose(vectors[0], embedder.encode(["Alpha text sits here."])[0])
