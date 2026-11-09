# SourceTracker

[![CI](https://github.com/hyxdo/sourcetracker/actions/workflows/ci.yml/badge.svg)](https://github.com/hyxdo/sourcetracker/actions/workflows/ci.yml)

An article can cite a source that doesn't say what the article claims, or repeat a source it never cites. A conclusion can look well supported when every citation under it traces back to one report. SourceTracker links each claim to the passage it relies on, keeps real citations separate from repetition that only looks related, and shows which conclusions lose their support when a source is removed.

Built with Python, NumPy, sentence embeddings, graph algorithms, and SQLite.

> Early development. The pipeline works end to end on the included sample, but it hasn't been measured on real articles yet. See [Measuring accuracy](#measuring-accuracy).

## Try it

Python 3.10 or newer.

```bash
git clone https://github.com/hyxdo/sourcetracker.git
cd sourcetracker
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[embeddings,dev]"
sourcetracker build examples/corpus
sourcetracker audit
sourcetracker remove harbor-survey
```

The first build downloads the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model. To skip it, install without the `embeddings` extra and pass `--embedder hashing`. That embedder only compares shared words, so it can't recognize a paraphrase.

`examples/corpus` is a fictional town's reef, invented to contain one of each problem SourceTracker looks for.

## What happens when a source goes away

```text
$ sourcetracker remove harbor-survey
Without: harbor-survey

Lost all cited support (1):
  policy-brief: The northern reef needs urchin removal before its kelp canopy disappears.
    supports: 2 -> 0
    relied on policy-brief: Urchin density on the northern reef nearly tripled between 2019 and 2022.
    relied on tribune-report: Urchin density on the northern reef nearly tripled between 2019 and 2022, according to...
    which relied on harbor-survey

Weakened (1):
  policy-brief: Fewer urchins on the northern reef, whether from otters or from removal, would let the kelp canopy recover.
    supports: 2 -> 1
    relied on policy-brief: The kelp canopy above the northern reef shrank by sixty percent.
    relied on tribune-report: The kelp canopy above the northern reef shrank by sixty percent over those four years.
    which relied on harbor-survey

Claims no longer supported: 5
```

The policy brief never cites the survey. It cites a newspaper, and the newspaper cites the survey, so the dependency only shows up by following the chain. The brief's third conclusion, about otters, rests on a different study and isn't affected.

## Citations versus repetition

These are stored in separate tables, so text similarity can never be counted as support.

| Relationship | Meaning | Counts as support? |
|---|---|---|
| Matched citation | The claim cites an article, and a passage there backs it | Yes |
| Unmatched citation | The claim cites an article, but no passage there backs it | No, and it gets flagged |
| Outside the corpus | The cited page isn't indexed, so it can't be checked | Assumed, and reported separately |
| Repetition | Similar text with no citation path connecting the two | Never |

Repetition is only reported when citations don't explain it. SourceTracker skips a pair if either side's citations lead to the other or both trace to a common source, if the claim already cites a source that backs it, or if the supposed original was published later.

```text
$ sourcetracker search "how much did the urchin population grow"
1. [0.631] harbor-survey: Divers counted purple urchins on forty fixed transects between 2019 and 2022. Urchin density on the northern reef rose from four to eleven animals...
     cited by   tribune-report: Urchin density on the northern reef nearly tripled between 2019 and 2022, according to a survey by the mari...
     resembles  coastal-blog: Urchin density on the northern reef rose from four to eleven animals per square meter over the survey perio... (no citation)
```

## How support is decided

A claim holds if at least one of its supports holds. A support is either a passage in a source or an earlier claim that must itself hold, which is how a chain of citations gets followed. If the matched sentence in a source has citations of its own, the claim depends on those too. If it has none, the chain stops there and that source counts as the origin.

Support spreads outward from those origins with a breadth-first search, in time linear in the size of the graph. Starting from the origins is what makes circular citation fail: two newsletters that only cite each other never receive support, however many citations they contain. `sourcetracker audit` names those loops using Tarjan's strongly connected components algorithm, and it ranks sources by how much would fall without each one.

A conclusion is a sentence opening with a phrase like "therefore" or "these findings suggest", or any sentence under a heading like "Conclusion". Its premises are the earlier cited claims in the same article that are similar enough to it.

## Commands

| Command | |
|---|---|
| `build CORPUS` | Index a folder of Markdown and HTML articles |
| `search QUERY` | Find passages and show who cites or repeats them |
| `remove SOURCE...` | Show what depends on an article or URL, without changing the index |
| `audit` | Report unmatched citations, loops, unsupported conclusions, and the riskiest sources |
| `trace ARTICLE` | Map an article's citations in both directions |
| `label-template ARTICLE...` | Create a blank label file to fill in by hand |
| `evaluate LABELS` | Score the index against labeled articles |

Markdown files can start with `id`, `title`, `url`, and `published` frontmatter. Citations are links, `[anchor](target)`, or bare markers, `[@id]` and `[@a; @b]`. A target is a URL or another article's id. HTML files are read from their `<article>` or `<main>` element when they have one.

## Measuring accuracy

`evaluate` compares the index with labels written by someone who read the articles:

| Metric | Question it answers |
|---|---|
| Citation links | Were the right claim-to-source pairs extracted? |
| Passage matches (top-1, MRR) | Was the backing passage ranked first? |
| Unsupported-citation flags | Are flagged citations really unsupported, and are any missed? |
| Uncited repetition | Is reported repetition real, and is any missed? |

```bash
sourcetracker label-template some-article another-article -o labels.json
# Fill in labels.json by hand
sourcetracker evaluate labels.json
```

The template leaves out scores and suggested passages so the tool's answers can't steer the labels.

`examples/labels.json` only demonstrates the format. The sample corpus was written alongside the code, so scores on it show that the pipeline runs and say nothing about accuracy.

## Known limits

- **Similarity isn't entailment.** Embeddings measure whether two sentences are about the same thing. A source saying the opposite of a claim can still score as a match.
- **Premise selection is a heuristic.** It relies on similarity alone, so a conclusion can be tied to a claim it doesn't logically use, or miss one it does.
- **The thresholds aren't validated.** The current values for `minilm` were chosen by inspecting scores on the small sample corpus. They should be tuned on labeled real articles, and all three can be set on `build`.
- **Some citation styles aren't read yet.** Numbered footnotes that link to a reference list within the same page are ignored.

## License

MIT
