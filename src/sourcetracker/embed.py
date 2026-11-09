"""Text embedders. Every embedder returns L2-normalized float32 row vectors."""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """Cosine-similarity cutoffs. Their scale depends on the embedder."""

    match: float  # a cited passage counts as backing the claim
    similar: float  # uncited text counts as possible repetition
    premise: float  # an earlier cited claim counts as a conclusion's premise


class Embedder(Protocol):
    name: str
    thresholds: Thresholds

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class EmbedderUnavailable(RuntimeError):
    pass


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have in into is it its of on or "
    "that the their this to was were which with".split()
)


class HashingEmbedder:
    """Deterministic bag of hashed words and word pairs.

    It only sees shared vocabulary, so it can't recognize a paraphrase. It needs
    no model download, which makes it useful for tests and quick offline runs.
    """

    name = "hashing"
    thresholds = Thresholds(match=0.25, similar=0.5, premise=0.1)

    def __init__(self, dimensions: int = 4096) -> None:
        self.dimensions = dimensions

    def _features(self, text: str) -> Counter[str]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        return Counter(words) + Counter(f"{a} {b}" for a, b in zip(words, words[1:]))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                digest = zlib.crc32(feature.encode("utf-8"))
                sign = 1.0 if digest & 0x80000000 else -1.0
                vectors[row, digest % self.dimensions] += sign * (1.0 + np.log(count))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEmbedder:
    """Sentence embeddings from a pretrained model, downloaded on first use."""

    name = "minilm"
    thresholds = Thresholds(match=0.5, similar=0.85, premise=0.5)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbedderUnavailable(
                "The minilm embedder needs sentence-transformers. Install it with "
                "pip install -e \".[embeddings]\", or pass --embedder hashing."
            ) from error
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


EMBEDDERS = {"minilm": SentenceTransformerEmbedder, "hashing": HashingEmbedder}


def get_embedder(name: str) -> Embedder:
    try:
        factory = EMBEDDERS[name]
    except KeyError:
        raise ValueError(f"Unknown embedder {name!r}. Choose from {sorted(EMBEDDERS)}.") from None
    return factory()
