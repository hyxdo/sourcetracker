from pathlib import Path

import pytest

from sourcetracker.parse import (
    citation_key,
    is_conclusion,
    load_corpus,
    parse_html,
    parse_markdown,
    sentence_spans,
    url_key,
)


def md(body: str):
    return parse_markdown(body, Path("sample.md"))


def split(text: str) -> list[str]:
    return [text[start:end] for start, end in sentence_spans(text)]


def test_url_key_treats_spellings_of_one_page_as_equal():
    first = url_key("https://www.Example.org/a/b/?utm_source=x&id=2#frag")
    second = url_key("http://example.org/a/b?id=2")
    assert first == second == "example.org/a/b?id=2"


@pytest.mark.parametrize(
    "href, expected",
    [
        ("https://example.org/a/", "example.org/a"),
        ("other-article", "other-article"),
        ("../notes/other-article.md#part", "other-article"),
        ("#footnote-1", None),
        ("mailto:someone@example.org", None),
        ("", None),
    ],
)
def test_citation_key(href, expected):
    assert citation_key(href) == expected


def test_sentences_survive_abbreviations_and_decimals():
    text = "Dr. Lee et al. measured 3.5 units, e.g. in June. Counts then fell! Did they recover?"
    assert split(text) == [
        "Dr. Lee et al. measured 3.5 units, e.g. in June.",
        "Counts then fell!",
        "Did they recover?",
    ]


def test_sentence_break_needs_a_plausible_next_sentence():
    assert split("It rose by 4 p.m. on most days.") == ["It rose by 4 p.m. on most days."]


def test_frontmatter_sets_identity():
    article = md('---\nid: my-id\ntitle: "A Title"\nurl: https://example.org/a\npublished: 2024-01-02\n---\n\nBody text here.')
    assert (article.id, article.title, article.url, article.published) == (
        "my-id",
        "A Title",
        "https://example.org/a",
        "2024-01-02",
    )


def test_id_and_title_fall_back_to_filename_and_heading():
    article = md("# Heading Title\n\nBody text here.")
    assert article.id == "sample"
    assert article.title == "Heading Title"


def test_citations_attach_to_the_sentence_that_makes_them():
    article = md("First claim stands alone. Second claim cites [a study](source-a) directly. Third is plain.")
    sentences = article.sentences
    assert [s.text for s in sentences] == [
        "First claim stands alone.",
        "Second claim cites a study directly.",
        "Third is plain.",
    ]
    assert [[c.target for c in s.citations] for s in sentences] == [[], ["source-a"], []]
    assert sentences[1].citations[0].anchor == "a study"


@pytest.mark.parametrize(
    "body",
    ["Claim one [@a]. Claim two.", "Claim one. [@a] Claim two.", "Claim one.[@a] Claim two."],
)
def test_bare_marker_belongs_to_the_preceding_sentence(body):
    sentences = md(body).sentences
    assert [s.text for s in sentences] == ["Claim one.", "Claim two."]
    assert [[c.target for c in s.citations] for s in sentences] == [["a"], []]


def test_marker_can_list_several_sources_and_repeats_collapse():
    sentence = md("One claim here [@a; @b] and [@a].").sentences[0]
    assert [c.target for c in sentence.citations] == ["a", "b"]


def test_non_citation_links_keep_their_text():
    sentence = md("Write to [the desk](mailto:x@example.org) or see [notes](#n).").sentences[0]
    assert sentence.text == "Write to the desk or see notes."
    assert sentence.citations == []


def test_markdown_structure():
    article = md(
        "# Title\n\nIntro *with* **emphasis**.\n\n## Findings\n\n- First item here.\n- Second item here.\n\n"
        "> Quoted line\n> continues.\n\n---\n\n```\nnot prose.\n```\n\n![alt](image.png)\n"
    )
    assert [(p.section, p.text) for p in article.passages] == [
        ("", "Intro with emphasis."),
        ("Findings", "First item here."),
        ("Findings", "Second item here."),
        ("Findings", "Quoted line continues."),
    ]


def test_long_paragraphs_become_several_passages():
    article = md("One here. Two here. Three here. Four here.")
    assert [p.text for p in article.passages] == ["One here. Two here. Three here.", "Four here."]


@pytest.mark.parametrize(
    "text, section, expected",
    [
        ("Therefore the program should expand.", "", True),
        ("These findings suggest a link.", "", True),
        ("Thusly is not a real cue.", "", False),
        ("The kelp grew quickly last year.", "", False),
        ("The kelp grew quickly last year.", "Conclusions", True),
        ("Thanks for reading.", "Conclusion", False),
    ],
)
def test_conclusion_detection(text, section, expected):
    assert is_conclusion(text, section) is expected


def test_conclusions_are_marked_on_sentences():
    sentences = md("A fact appears here. Therefore a result follows from it.").sentences
    assert [s.is_conclusion for s in sentences] == [False, True]


HTML = """<!doctype html>
<html><head>
<title>Site | Page</title>
<link rel="canonical" href="https://example.org/page">
<meta property="article:published_time" content="2023-05-06">
<style>p { color: red; }</style>
</head><body>
<nav><ul><li><a href="https://example.org/">Home link text.</a></li></ul></nav>
<p>Outside the article body.</p>
<article>
  <h1>Real Headline</h1>
  <p>The survey found <a href="https://www.example.org/study/">fewer urchins</a> near
     the harbor. Numbers   recovered later.</p>
  <h2>Conclusion</h2>
  <p>Kelp can return within a decade.<a href="source-b"></a></p>
  <script>var ignored = "Not prose.";</script>
</article>
<footer><p>Copyright notice text.</p></footer>
</body></html>"""


def test_html_extracts_the_article_body_and_its_citations():
    article = parse_html(HTML, Path("page.html"))
    assert article.title == "Real Headline"
    assert article.url == "https://example.org/page"
    assert article.published == "2023-05-06"
    sentences = article.sentences
    assert [s.text for s in sentences] == [
        "The survey found fewer urchins near the harbor.",
        "Numbers recovered later.",
        "Kelp can return within a decade.",
    ]
    assert [[c.target for c in s.citations] for s in sentences] == [
        ["example.org/study"],
        [],
        ["source-b"],
    ]
    assert sentences[2].is_conclusion


def test_html_without_a_main_element_keeps_all_blocks():
    article = parse_html("<p>Only paragraph here.</p>", Path("bare.html"))
    assert [s.text for s in article.sentences] == ["Only paragraph here."]


def test_load_corpus_reads_both_formats_and_ignores_others(tmp_path):
    (tmp_path / "a.md").write_text("Alpha text here.", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.html").write_text("<p>Beta text here.</p>", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert [a.id for a in load_corpus(tmp_path)] == ["a", "b"]


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a.md").write_text("Alpha text.", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\nid: a\n---\nBeta text.", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate article id"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_shared_urls(tmp_path):
    (tmp_path / "a.md").write_text("---\nurl: https://example.org/x\n---\nAlpha.", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\nurl: http://www.example.org/x/\n---\nBeta.", encoding="utf-8")
    with pytest.raises(ValueError, match="share the URL"):
        load_corpus(tmp_path)


def test_load_corpus_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
