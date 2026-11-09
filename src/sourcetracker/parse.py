"""Turn Markdown and HTML article files into sentences, passages, and citations.

Citations are recorded per sentence, so later stages can ask what a specific
claim cites rather than what an article cites somewhere in its body.

Markdown citations are ordinary links, ``[anchor](target)``, or bare markers,
``[@id]`` and ``[@a; @b]``. A target is a URL or the id of another article.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlsplit

MAX_PASSAGE_SENTENCES = 3
MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}
HTML_SUFFIXES = {".html", ".htm"}


@dataclass(frozen=True)
class Citation:
    target: str
    anchor: str = ""


@dataclass
class Sentence:
    text: str
    citations: list[Citation] = field(default_factory=list)
    is_conclusion: bool = False


@dataclass
class Passage:
    section: str
    sentences: list[Sentence]

    @property
    def text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)


@dataclass
class Article:
    id: str
    title: str
    url: str | None
    published: str | None
    path: str
    passages: list[Passage]

    @property
    def sentences(self) -> list[Sentence]:
        return [s for passage in self.passages for s in passage.sentences]


@dataclass
class _Block:
    section: str
    text: str
    marks: list[tuple[int, Citation]]
    in_main: bool = False


def url_key(url: str) -> str:
    """Reduce a URL to a form that matches other spellings of the same page."""
    parts = urlsplit(url if "//" in url else f"//{url}")
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    kept = sorted(
        (k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")
    )
    query = urlencode(kept)
    return f"{host}{path}?{query}" if query else f"{host}{path}"


def citation_key(href: str) -> str | None:
    """Return a citation target for a link, or None if it can't be a citation."""
    href = href.strip()
    if not href or href.startswith("#"):
        return None
    scheme = urlsplit(href).scheme.lower()
    if scheme in ("http", "https") or href.startswith("//"):
        return url_key(href)
    if scheme:  # mailto:, tel:, javascript:
        return None
    stem = PurePosixPath(href.split("#", 1)[0].split("?", 1)[0]).stem
    return stem or None


_BREAK = re.compile(r"""[.!?]+["'”’)\]]*\s+""")
_ABBREVIATIONS = {
    "al", "approx", "cf", "dr", "e.g", "etc", "fig", "i.e", "inc", "jr", "ltd",
    "mr", "mrs", "ms", "no", "prof", "sr", "st", "vs",
}
_SENTENCE_OPENERS = "\"'“‘(["


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into (start, end) sentence spans."""
    spans = []
    start = 0
    for match in _BREAK.finditer(text):
        following = text[match.end()] if match.end() < len(text) else ""
        if following and not (
            following.isupper() or following.isdigit() or following in _SENTENCE_OPENERS
        ):
            continue
        words = text[start : match.start()].split()
        word = words[-1].lstrip(_SENTENCE_OPENERS).lower() if words else ""
        if text[match.start()] == "." and (
            word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())
        ):
            continue
        spans.append((start, match.start() + len(match.group().rstrip())))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


_CONCLUSION_SECTION = re.compile(
    r"\b(conclusions?|takeaways?|bottom line|what this means|implications)\b", re.I
)
_CONCLUSION_CUE = re.compile(
    r"^(?:so,?\s+|and\s+)?(?:therefore|thus|hence|consequently|in conclusion|in summary"
    r"|overall|taken together|we conclude"
    r"|this (?:means|shows|suggests|proves|demonstrates|indicates)"
    r"|these (?:findings|results|data) (?:show|suggest|indicate|mean|demonstrate)"
    r"|the (?:evidence|data|results) (?:shows?|suggests?|indicates?))\b",
    re.I,
)


def is_conclusion(text: str, section: str) -> bool:
    if _CONCLUSION_CUE.match(text):
        return True
    return bool(_CONCLUSION_SECTION.search(section)) and len(text.split()) >= 5


def _sentences(block: _Block) -> list[Sentence]:
    spans = sentence_spans(block.text)
    assigned: list[list[Citation]] = [[] for _ in spans]
    for offset, citation in block.marks:
        # A mark in the gap between two sentences belongs to the earlier one.
        index = max((i for i, (start, _) in enumerate(spans) if start <= offset), default=0)
        if citation.target not in {c.target for c in assigned[index]}:
            assigned[index].append(citation)
    sentences = []
    for (start, end), citations in zip(spans, assigned):
        text = block.text[start:end].strip()
        sentences.append(Sentence(text, citations, is_conclusion(text, block.section)))
    return sentences


def _build_article(
    path: Path,
    article_id: str,
    title: str | None,
    url: str | None,
    published: str | None,
    blocks: list[_Block],
) -> Article:
    passages = []
    for block in blocks:
        sentences = _sentences(block)
        for i in range(0, len(sentences), MAX_PASSAGE_SENTENCES):
            passages.append(Passage(block.section, sentences[i : i + MAX_PASSAGE_SENTENCES]))
    return Article(article_id, title or article_id, url, published, str(path), passages)


# Markdown ------------------------------------------------------------------

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INLINE = re.compile(
    r"\[(?P<anchor>[^\[\]]*)\]\((?P<href>[^()\s]+)(?:\s+\"[^\"]*\")?\)"
    r"|\[(?P<marker>@[^\[\]]+)\]"
)
_EMPHASIS = re.compile(r"\*\*|\*|`")
_HEADING = re.compile(r"(#{1,6})\s+(.*?)\s*#*$")
_LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s+(.*)")
_RULE = re.compile(r"[-*_]{3,}")


def _frontmatter(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            meta = {}
            for entry in lines[1:end]:
                key, separator, value = entry.partition(":")
                if separator:
                    meta[key.strip().lower()] = value.strip().strip("\"'")
            return meta, "\n".join(lines[end + 1 :])
    return {}, raw


def _inline_markdown(raw: str) -> tuple[str, list[tuple[int, Citation]]]:
    raw = _IMAGE.sub("", raw)
    text = ""
    marks: list[tuple[int, Citation]] = []
    position = 0
    for match in _INLINE.finditer(raw):
        text += _EMPHASIS.sub("", raw[position : match.start()])
        position = match.end()
        if match.group("marker"):
            text = text.rstrip()
            for part in match.group("marker").split(";"):
                target = part.strip().lstrip("@").strip()
                if target:
                    marks.append((max(len(text) - 1, 0), Citation(target)))
        else:
            anchor = _EMPHASIS.sub("", match.group("anchor"))
            key = citation_key(match.group("href"))
            if key is not None:
                marks.append((len(text), Citation(key, anchor)))
            text += anchor
    text += _EMPHASIS.sub("", raw[position:])
    return text, marks


def parse_markdown(raw: str, path: Path) -> Article:
    meta, body = _frontmatter(raw)
    title = meta.get("title")
    section = ""
    blocks: list[_Block] = []
    lines: list[str] = []
    fenced = False

    def flush() -> None:
        if lines:
            text, marks = _inline_markdown(" ".join(lines))
            if text.strip():
                blocks.append(_Block(section, text.strip(), marks))
            lines.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = _HEADING.fullmatch(stripped)
        item = _LIST_ITEM.fullmatch(stripped)
        if not stripped or _RULE.fullmatch(stripped):
            flush()
        elif heading:
            flush()
            if len(heading.group(1)) == 1 and title is None:
                title = heading.group(2)
            else:
                section = heading.group(2)
        elif item:
            flush()
            lines.append(item.group(1))
            flush()
        else:
            lines.append(stripped.lstrip("> "))
    flush()
    return _build_article(
        path, meta.get("id") or path.stem, title, meta.get("url"), meta.get("published"), blocks
    )


# HTML ----------------------------------------------------------------------


class _HTMLExtractor(HTMLParser):
    BLOCKS = {"p", "li", "blockquote"}
    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    SKIPPED = {"script", "style", "nav", "footer", "aside", "header", "form", "noscript", "svg"}
    MAIN = {"article", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.h1: str | None = None
        self.url: str | None = None
        self.published: str | None = None
        self.blocks: list[_Block] = []
        self._section = ""
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False
        self._title_text = ""
        self._heading: str | None = None
        self._heading_text = ""
        self._in_block = False
        self._text = ""
        self._marks: list[tuple[int, Citation]] = []
        self._link: tuple[str, int] | None = None

    def _flush(self) -> None:
        if self._in_block and self._text.strip():
            self.blocks.append(
                _Block(self._section, self._text.strip(), self._marks, self._main_depth > 0)
            )
        self._in_block = False
        self._text = ""
        self._marks = []
        self._link = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag in self.SKIPPED:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (values.get("property") or values.get("name") or "").lower()
            if name in ("article:published_time", "date", "dc.date"):
                self.published = self.published or values.get("content")
            elif name == "og:url":
                self.url = self.url or values.get("content")
        elif tag == "link" and values.get("rel", "").lower() == "canonical":
            self.url = values.get("href") or self.url
        if self._skip_depth:
            return
        if tag in self.MAIN:
            self._main_depth += 1
        elif tag in self.HEADINGS:
            self._flush()
            self._heading = tag
            self._heading_text = ""
        elif tag in self.BLOCKS:
            self._flush()
            self._in_block = True
        elif tag == "a" and self._in_block:
            key = citation_key(values.get("href", ""))
            if key is not None:
                self._link = (key, len(self._text))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIPPED:
            self._skip_depth = max(self._skip_depth - 1, 0)
        elif tag == "title":
            self._in_title = False
            self.title = self.title or " ".join(self._title_text.split()) or None
        if self._skip_depth:
            return
        if tag in self.MAIN:
            self._flush()
            self._main_depth = max(self._main_depth - 1, 0)
        elif tag == self._heading:
            text = " ".join(self._heading_text.split())
            if tag == "h1" and self.h1 is None:
                self.h1 = text
            else:
                self._section = text
            self._heading = None
        elif tag in self.BLOCKS:
            self._flush()
        elif tag == "a" and self._link:
            key, start = self._link
            anchor = self._text[start:].strip()
            offset = start if anchor else max(len(self._text.rstrip()) - 1, 0)
            self._marks.append((offset, Citation(key, anchor)))
            self._link = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_text += data
        if self._skip_depth:
            return
        if self._heading:
            self._heading_text += data
        elif self._in_block:
            data = re.sub(r"\s+", " ", data)
            if not self._text or self._text.endswith(" "):
                data = data.lstrip()
            self._text += data


def parse_html(raw: str, path: Path) -> Article:
    extractor = _HTMLExtractor()
    extractor.feed(raw)
    extractor.close()
    extractor._flush()
    blocks = extractor.blocks
    if any(block.in_main for block in blocks):
        blocks = [block for block in blocks if block.in_main]
    title = extractor.h1 or extractor.title
    return _build_article(path, path.stem, title, extractor.url, extractor.published, blocks)


# Loading -------------------------------------------------------------------


def load_article(path: Path) -> Article:
    raw = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return parse_markdown(raw, path)
    if suffix in HTML_SUFFIXES:
        return parse_html(raw, path)
    raise ValueError(f"Unsupported article format: {path}")


def load_corpus(directory: Path) -> list[Article]:
    """Load every article under a directory, rejecting clashing ids and URLs."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    suffixes = MARKDOWN_SUFFIXES | HTML_SUFFIXES
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    articles = [load_article(path) for path in paths]
    ids: dict[str, str] = {}
    urls: dict[str, str] = {}
    for article in articles:
        if article.id in ids:
            raise ValueError(
                f"Duplicate article id {article.id!r}: {ids[article.id]} and {article.path}"
            )
        ids[article.id] = article.path
        if article.url:
            key = url_key(article.url)
            if key in urls:
                raise ValueError(f"Two articles share the URL {article.url}: {urls[key]} and {article.path}")
            urls[key] = article.path
    return articles
