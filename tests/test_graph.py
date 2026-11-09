import pytest

from sourcetracker.graph import LOST, WEAKENED, Support, SupportGraph, citation_loops, reachable


def graph(articles: dict[int, str], supports: dict[int, list[Support]], conclusions=()):
    return SupportGraph(articles, supports, set(conclusions))


def test_claim_holds_while_any_source_remains():
    g = graph({1: "a"}, {1: [Support("x"), Support("y")]})
    assert g.holding() == {1}
    assert g.holding({"x"}) == {1}
    assert g.holding({"x", "y"}) == set()


def test_uncited_claim_never_holds():
    assert graph({1: "a"}, {}).holding() == set()


def test_removal_follows_a_chain_of_citations():
    # 3 relies on 2, which relies on 1, which rests on the primary source.
    g = graph(
        {1: "b", 2: "c", 3: "d"},
        {1: [Support("primary")], 2: [Support("b", 1)], 3: [Support("c", 2)]},
    )
    assert g.holding() == {1, 2, 3}
    assert g.holding({"primary"}) == set()


def test_removing_a_middle_article_keeps_what_is_upstream():
    g = graph(
        {1: "b", 2: "c", 3: "d"},
        {1: [Support("primary")], 2: [Support("b", 1)], 3: [Support("c", 2)]},
    )
    assert g.holding({"c"}) == {1}


def test_a_removed_article_loses_its_own_claims():
    g = graph({1: "a"}, {1: [Support("x")]})
    assert g.holding({"a"}) == set()


def test_mutual_citation_supports_nothing():
    g = graph({1: "a", 2: "b"}, {1: [Support("b", 2)], 2: [Support("a", 1)]})
    assert g.holding() == set()


def test_loop_with_an_outside_source_holds_until_that_source_goes():
    g = graph(
        {1: "a", 2: "b"},
        {1: [Support("b", 2), Support("primary")], 2: [Support("a", 1)]},
    )
    assert g.holding() == {1, 2}
    assert g.holding({"primary"}) == set()


def conclusion_graph():
    """Conclusion 5 in article r rests on claims 3 and 4, each citing one source."""
    return graph(
        {3: "r", 4: "r", 5: "r"},
        {3: [Support("x")], 4: [Support("y")], 5: [Support(None, 3), Support(None, 4)]},
        conclusions=[5],
    )


def test_losing_one_premise_weakens_a_conclusion():
    result = conclusion_graph().impact(["x"])
    assert [(c.claim_id, c.status) for c in result.conclusions] == [(5, WEAKENED)]
    assert (result.conclusions[0].supports_before, result.conclusions[0].supports_after) == (2, 1)
    assert result.claims_lost == [3]


def test_losing_every_premise_loses_the_conclusion():
    result = conclusion_graph().impact(["x", "y"])
    assert [(c.claim_id, c.status) for c in result.conclusions] == [(5, LOST)]
    assert result.claims_lost == [3, 4, 5]


def test_unrelated_removal_changes_nothing():
    result = conclusion_graph().impact(["unrelated"])
    assert result.conclusions == []
    assert result.claims_lost == []


def test_conclusion_without_support_is_not_reported_as_a_loss():
    g = graph({1: "a", 2: "a"}, {1: [Support("x")]}, conclusions=[2])
    assert g.impact(["x"]).conclusions == []


def test_conclusions_in_the_removed_article_are_not_reported():
    g = graph({1: "a"}, {1: [Support("x")]}, conclusions=[1])
    assert g.impact(["a"]).conclusions == []


def test_impact_explains_the_shortest_path_to_the_removed_source():
    g = graph(
        {1: "b", 2: "c", 3: "r"},
        {1: [Support("primary")], 2: [Support("b", 1)], 3: [Support("c", 2)]},
        conclusions=[3],
    )
    item = g.impact(["primary"]).conclusions[0]
    assert item.path == (3, 2, 1)
    assert item.removed_source == "primary"
    assert item.status == LOST


def test_impact_path_handles_loops_with_an_escape():
    g = graph(
        {1: "a", 2: "b", 3: "r"},
        {
            1: [Support("b", 2), Support("primary")],
            2: [Support("a", 1)],
            3: [Support("a", 1)],
        },
        conclusions=[3],
    )
    item = g.impact(["primary"]).conclusions[0]
    assert item.status == LOST
    assert item.path == (3, 1)


def test_lost_conclusions_are_listed_before_weakened_ones():
    g = graph(
        {1: "r", 2: "r", 3: "r"},
        {1: [Support("x"), Support("y")], 2: [Support("x")], 3: [Support("x")]},
        conclusions=[1, 2],
    )
    assert [c.status for c in g.impact(["x"]).conclusions] == [LOST, WEAKENED]


def test_source_risks_rank_the_most_damaging_source_first():
    g = graph(
        {1: "r", 2: "r", 3: "s"},
        {1: [Support("big")], 2: [Support("big"), Support("small")], 3: [Support("big")]},
        conclusions=[1, 2, 3],
    )
    risks = g.source_risks()
    assert [(r.source, r.lost, r.weakened) for r in risks] == [("big", 2, 1), ("small", 0, 1)]


def test_premises_are_not_listed_as_sources():
    assert conclusion_graph().sources() == {"x", "y"}


def test_reachable_includes_starts_and_handles_cycles():
    edges = {"a": ["b"], "b": ["c", "a"], "d": ["e"]}
    assert reachable(edges, ["a"]) == {"a", "b", "c"}


@pytest.mark.parametrize(
    "edges, expected",
    [
        ({"a": ["b"], "b": ["c"]}, []),
        ({"a": ["b"], "b": ["a"]}, [["a", "b"]]),
        ({"a": ["b"], "b": ["c"], "c": ["a"], "c2": ["a"]}, [["a", "b", "c"]]),
        ({"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"], "b2": ["c"]}, [["a", "b"], ["c", "d"]]),
        # One component reached through two entry points inside it.
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["a"]}, [["a", "b", "c", "d"]]),
        ({}, []),
    ],
)
def test_citation_loops(edges, expected):
    assert citation_loops(edges) == expected


def test_citation_loops_handles_chains_deeper_than_the_recursion_limit():
    edges = {str(i): [str(i + 1)] for i in range(5000)}
    edges["5000"] = ["0"]
    loops = citation_loops(edges)
    assert len(loops) == 1 and len(loops[0]) == 5001
