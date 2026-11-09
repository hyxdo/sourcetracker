import pytest

from sourcetracker.evaluate import Counts, LabelError, evaluate, label_template

STUDY = (
    "Urchin numbers doubled near the northern harbor during the survey.\n\n"
    "Water temperature stayed stable through every sampling season."
)
ARTICLES = {
    "study": STUDY,
    "report": (
        "Urchin numbers doubled near the northern harbor [@study].\n\n"
        "The mayor approved parking meters downtown yesterday [@study].\n\n"
        "Kelp canopy cover fell sharply [source](https://elsewhere.org/paper)."
    ),
    "blog": "Urchin numbers doubled near the northern harbor during the survey, friends.",
}


def citation(claim, target="study", label="supported", quote="Urchin numbers doubled"):
    return {"claim": claim, "target": target, "label": label, "gold_quote": quote}


CORRECT = {
    "articles": {
        "report": {
            "citations": [
                citation("Urchin numbers doubled"),
                citation("parking meters", label="not_supported", quote=""),
                citation("Kelp canopy", target="https://elsewhere.org/paper", quote=""),
            ],
        },
        "blog": {
            "citations": [],
            "repetitions": [{"claim": "friends", "source": "study"}],
        },
    }
}


def test_counts_compute_the_standard_ratios():
    counts = Counts(true_positive=3, false_positive=1, false_negative=3)
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.5)
    assert counts.f1 == pytest.approx(0.6)


def test_counts_are_undefined_without_examples():
    assert Counts().precision is None and Counts().recall is None and Counts().f1 is None


def test_agreeing_labels_score_perfectly(make_index):
    report = evaluate(make_index(ARTICLES), CORRECT)
    assert report.citation_links == Counts(3, 0, 0)
    assert report.passage_examples == 1
    assert report.passage_top1 == 1.0
    assert report.unsupported_flags == Counts(1, 0, 0)
    assert report.repetition == Counts(1, 0, 0)
    assert report.misses == []


def test_missing_labels_count_as_false_positives(make_index):
    labels = {
        "articles": {
            "report": {"citations": [citation("Urchin numbers doubled")]},
        }
    }
    assert evaluate(make_index(ARTICLES), labels).citation_links == Counts(1, 2, 0)


def test_a_citation_the_tool_never_found_is_a_false_negative(make_index):
    labels = {
        "articles": {
            "blog": {"citations": [citation("friends", quote="")]},
        }
    }
    assert evaluate(make_index(ARTICLES), labels).citation_links == Counts(0, 0, 1)


def test_wrong_passage_lowers_top1_and_mrr(make_index):
    labels = {
        "articles": {
            "report": {
                "citations": [citation("Urchin numbers doubled", quote="Water temperature stayed")],
            }
        }
    }
    report = evaluate(make_index(ARTICLES), labels)
    assert report.passage_top1 == 0.0
    assert report.passage_mrr == pytest.approx(0.5)
    assert any("passage ranked 2" in miss for miss in report.misses)


def test_a_wrongly_flagged_citation_is_reported(make_index):
    labels = {
        "articles": {
            "report": {
                "citations": [citation("parking meters", quote="Water temperature stayed")],
            }
        }
    }
    report = evaluate(make_index(ARTICLES), labels)
    assert report.unsupported_flags == Counts(0, 1, 0)
    assert any("supported citation was flagged" in miss for miss in report.misses)


def test_an_unsupported_citation_the_tool_accepted_is_reported(make_index):
    labels = {
        "articles": {
            "report": {
                "citations": [citation("Urchin numbers doubled", label="not_supported", quote="")],
            }
        }
    }
    report = evaluate(make_index(ARTICLES), labels)
    assert report.unsupported_flags == Counts(0, 0, 1)
    assert any("unsupported citation was matched" in miss for miss in report.misses)


@pytest.mark.parametrize(
    "labels, message",
    [
        ({"articles": {}}, "no articles"),
        ({"articles": {"nope": {}}}, "not in the index"),
        ({"articles": {"report": {"citations": [citation("zebra")]}}}, "matches no sentence"),
        ({"articles": {"report": {"citations": [citation("Urchin", label="")]}}}, "needs a label"),
        ({"articles": {"report": {"citations": [citation("Urchin numbers doubled", quote="")]}}}, "gold_quote"),
        ({"articles": {"report": {"citations": [citation("Urchin numbers doubled", quote="absent words")]}}}, "is not in study"),
        ({"articles": {"report": {"citations": [citation("", quote="")]}}}, "empty claim"),
    ],
)
def test_bad_labels_are_rejected(make_index, labels, message):
    with pytest.raises(LabelError, match=message):
        evaluate(make_index(ARTICLES), labels)


def test_ambiguous_claim_is_rejected(make_index):
    index = make_index({"a": "Kelp grew quickly here.\n\nKelp grew slowly there."})
    labels = {"articles": {"a": {"citations": [citation("Kelp grew", quote="")]}}}
    with pytest.raises(LabelError, match="matches 2 sentences"):
        evaluate(index, labels)


def test_template_lists_citations_with_blank_labels(make_index):
    template = label_template(make_index(ARTICLES), ["report"])
    entries = template["articles"]["report"]["citations"]
    assert [e["target"] for e in entries] == ["study", "study", "elsewhere.org/paper"]
    assert all(e["label"] == "" and e["gold_quote"] == "" for e in entries)


def test_template_hides_the_tool_s_answers(make_index):
    entry = label_template(make_index(ARTICLES), ["report"])["articles"]["report"]["citations"][0]
    assert set(entry) == {"claim", "target", "label", "gold_quote"}


def test_an_unfilled_template_can_t_be_scored(make_index):
    index = make_index(ARTICLES)
    with pytest.raises(LabelError, match="needs a label"):
        evaluate(index, label_template(index, ["report"]))


def test_template_rejects_unknown_articles(make_index):
    with pytest.raises(LabelError, match="Not in the index: nope"):
        label_template(make_index(ARTICLES), ["nope"])
