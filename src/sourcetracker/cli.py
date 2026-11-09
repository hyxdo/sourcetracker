"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from .build import build_index
from .embed import EMBEDDERS, EmbedderUnavailable, get_embedder
from .evaluate import LabelError, evaluate, label_template
from .graph import LOST, SupportGraph, citation_loops, reachable
from .link import citation_edges, resolve_targets
from .parse import citation_key
from .search import search
from .store import EXTERNAL, MATCHED, SELF, UNMATCHED, Index

DEFAULT_DATABASE = "sourcetracker.db"


def _clip(text: str, width: int = 110) -> str:
    return text if len(text) <= width else text[: width - 3].rstrip() + "..."


def _cmd_build(args: argparse.Namespace) -> int:
    embedder = get_embedder(args.embedder)
    thresholds = embedder.thresholds
    overrides = {
        "match": args.match_threshold,
        "similar": args.similar_threshold,
        "premise": args.premise_threshold,
    }
    thresholds = replace(thresholds, **{k: v for k, v in overrides.items() if v is not None})
    summary = build_index(Path(args.corpus), Path(args.db), embedder, thresholds)
    print(
        f"Indexed {summary.articles} articles: {summary.passages} passages, "
        f"{summary.claims} sentences, {summary.citations} citations -> {args.db}"
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    with Index.open(Path(args.db)) as index:
        embedder = get_embedder(index.meta("embedder"))
        hits = search(index, embedder, args.query, args.limit)
    for number, hit in enumerate(hits, start=1):
        print(f"{number}. [{hit.score:.3f}] {hit.passage.article_id}: {_clip(hit.passage.text, 200)}")
        for reference in hit.direct:
            print(f"     cited by   {reference.article_id}: {_clip(reference.claim)}")
        for reference in hit.repetition:
            print(f"     resembles  {reference.article_id}: {_clip(reference.claim)} (no citation)")
    return 0


def _resolve_sources(index: Index, graph: SupportGraph, requested: list[str]) -> list[str]:
    """Accept article ids and URLs, and refuse names that match nothing.

    A misspelled source must not look like a source nothing depends on.
    """
    names = resolve_targets(index.articles())
    known = graph.sources() | {article.id for article in index.articles()}
    resolved = []
    for name in requested:
        key = citation_key(name) or name
        source = names.get(name) or names.get(key)
        if source is None:
            source = next((candidate for candidate in (name, key) if candidate in known), None)
        if source is None:
            raise ValueError(f"Unknown source {name!r}. See `sourcetracker audit` for sources.")
        resolved.append(source)
    return resolved


def _cmd_remove(args: argparse.Namespace) -> int:
    with Index.open(Path(args.db)) as index:
        graph = SupportGraph.from_index(index)
        removed = _resolve_sources(index, graph, args.sources)
        impact = graph.impact(removed)
        claims = {claim.id: claim for claim in index.claims()}
    print(f"Without: {', '.join(sorted(impact.removed))}")
    if not impact.conclusions:
        print("No conclusion changes.")
    for status, label in ((LOST, "Lost all cited support"), ("weakened", "Weakened")):
        items = [c for c in impact.conclusions if c.status == status]
        if not items:
            continue
        print(f"\n{label} ({len(items)}):")
        for item in items:
            claim = claims[item.claim_id]
            print(f"  {claim.article_id}: {_clip(claim.text)}")
            print(f"    supports: {item.supports_before} -> {item.supports_after}")
            for step in item.path[1:]:
                print(f"    relied on {claims[step].article_id}: {_clip(claims[step].text, 90)}")
            print(f"    which relied on {item.removed_source}")
    print(f"\nClaims no longer supported: {len(impact.claims_lost)}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    with Index.open(Path(args.db)) as index:
        claims = {claim.id: claim for claim in index.claims()}
        citations = index.citations()
        passages = {passage.id: passage for passage in index.passages()}
        similar = index.similar_links()
        graph = SupportGraph.from_index(index)
        edges = citation_edges(list(claims.values()), citations)
        article_count = len(index.articles())
    statuses = Counter(citation.status for citation in citations)
    print(
        f"{article_count} articles, {len(claims)} sentences, {len(citations)} citations: "
        f"{statuses[MATCHED]} matched, {statuses[UNMATCHED]} unmatched, "
        f"{statuses[EXTERNAL]} outside the corpus, {statuses[SELF]} self-citations"
    )

    unmatched = [c for c in citations if c.status == UNMATCHED]
    if unmatched:
        print(f"\nCited source has no passage backing the claim ({len(unmatched)}):")
        for citation in unmatched:
            claim = claims[citation.claim_id]
            print(f"  {claim.article_id} -> {citation.target_article_id}: {_clip(claim.text)}")

    loops = citation_loops(edges)
    if loops:
        print(f"\nCitation loops ({len(loops)}):")
        for loop in loops:
            print(f"  {' <-> '.join(loop)}")

    held = graph.holding()
    conclusions = [claims[c] for c in sorted(graph.conclusions)]
    unsupported = [c for c in conclusions if c.id not in held]
    print(f"\nConclusions: {len(conclusions)}, without cited support: {len(unsupported)}")
    for claim in unsupported:
        print(f"  {claim.article_id}: {_clip(claim.text)}")

    risks = [r for r in graph.source_risks() if r.lost or r.weakened or r.claims_lost]
    if risks:
        print("\nSources with the most resting on them:")
        for risk in risks[: args.limit]:
            print(
                f"  {risk.source}: {risk.lost} conclusions lost, {risk.weakened} weakened, "
                f"{risk.claims_lost} claims unsupported"
            )

    if similar:
        print(f"\nPossible repetition with no citation path ({len(similar)}):")
        for link in similar[: args.limit]:
            claim = claims[link.claim_id]
            print(
                f"  [{link.score:.3f}] {claim.article_id} resembles "
                f"{passages[link.passage_id].article_id}: {_clip(claim.text)}"
            )
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    with Index.open(Path(args.db)) as index:
        articles = {article.id: article for article in index.articles()}
        if args.article not in articles:
            raise ValueError(f"Unknown article {args.article!r}.")
        claims = {claim.id: claim for claim in index.claims()}
        citations = index.citations()
        edges = citation_edges(list(claims.values()), citations)
    print(f"{args.article}: {articles[args.article].title}")

    outgoing: dict[str, Counter] = defaultdict(Counter)
    incoming: dict[str, Counter] = defaultdict(Counter)
    for citation in citations:
        source = claims[citation.claim_id].article_id
        target = citation.target_article_id or citation.target
        if source == args.article:
            outgoing[target][citation.status] += 1
        if target == args.article and source != args.article:
            incoming[source][citation.status] += 1

    def show(title: str, groups: dict[str, Counter]) -> None:
        print(f"\n{title}:" if groups else f"\n{title}: none")
        for name, counts in sorted(groups.items()):
            summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
            print(f"  {name} ({summary})")

    show("Cites", outgoing)
    show("Cited by", incoming)
    upstream = reachable(edges, edges.get(args.article, ())) - {args.article}
    print(f"\nAll sources upstream: {', '.join(sorted(upstream)) or 'none'}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LabelError(f"{args.labels} is not valid JSON: {error}") from error
    with Index.open(Path(args.db)) as index:
        print(evaluate(index, labels).to_text())
    return 0


def _cmd_label_template(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists() and not args.force:
        raise ValueError(f"{output} already exists. Use --force to replace it.")
    with Index.open(Path(args.db)) as index:
        template = label_template(index, args.articles)
    output.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}. Fill in the labels, then run `sourcetracker evaluate {output}`.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sourcetracker",
        description="Trace claims to the passages they cite and test what depends on each source.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help)
        sub.add_argument("--db", default=DEFAULT_DATABASE, help="index file (default: %(default)s)")
        sub.set_defaults(handler=handler)
        return sub

    build = command("build", _cmd_build, "Index a directory of Markdown and HTML articles.")
    build.add_argument("corpus")
    build.add_argument("--embedder", choices=sorted(EMBEDDERS), default="minilm")
    build.add_argument("--match-threshold", type=float, help="score for a cited passage to back a claim")
    build.add_argument("--similar-threshold", type=float, help="score for uncited text to count as repetition")
    build.add_argument("--premise-threshold", type=float, help="score for a cited claim to support a conclusion")

    find = command("search", _cmd_search, "Find passages and show who cites or repeats them.")
    find.add_argument("query")
    find.add_argument("--limit", type=int, default=5)

    remove = command("remove", _cmd_remove, "Show which conclusions depend on the given sources.")
    remove.add_argument("sources", nargs="+", metavar="SOURCE", help="article id or URL")

    audit = command("audit", _cmd_audit, "Report weak citations, loops, and the riskiest sources.")
    audit.add_argument("--limit", type=int, default=10)

    trace = command("trace", _cmd_trace, "Map an article's citations in both directions.")
    trace.add_argument("article")

    score = command("evaluate", _cmd_evaluate, "Score the index against labeled articles.")
    score.add_argument("labels")

    template = command("label-template", _cmd_label_template, "Create a blank label file to fill in by hand.")
    template.add_argument("articles", nargs="+", metavar="ARTICLE")
    template.add_argument("-o", "--output", default="labels.json")
    template.add_argument("--force", action="store_true", help="replace an existing file")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, EmbedderUnavailable) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
