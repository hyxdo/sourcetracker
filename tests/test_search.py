import pytest

from sourcetracker.search import search

STUDY = (
    "Urchin numbers doubled near the northern harbor during the survey.\n\n"
    "Water temperature stayed stable through every sampling season."
)
ARTICLES = {
    "study": STUDY,
    "report": "Urchin numbers doubled near the northern harbor during the survey [@study].",
    "blog": "Urchin numbers doubled near the northern harbor during the survey, friends.",
}


def test_search_separates_citations_from_repetition(make_index, embedder):
    index = make_index(ARTICLES)
    hits = search(index, embedder, "urchin numbers doubled northern harbor")
    study_hit = next(hit for hit in hits if hit.passage.article_id == "study")
    assert [r.article_id for r in study_hit.direct] == ["report"]
    assert [r.article_id for r in study_hit.repetition] == ["blog"]


def test_search_ranks_the_relevant_passage_first(make_index, embedder):
    index = make_index(ARTICLES)
    top = search(index, embedder, "water temperature sampling season")[0]
    assert top.passage.text.startswith("Water temperature")
    assert top.direct == [] and top.repetition == []


def test_search_respects_the_limit(make_index, embedder):
    index = make_index(ARTICLES)
    assert len(search(index, embedder, "urchin", limit=2)) == 2


def test_empty_query_is_an_error(make_index, embedder):
    index = make_index(ARTICLES)
    with pytest.raises(ValueError, match="empty"):
        search(index, embedder, "   ")
