import builtins

import numpy as np
import pytest

from sourcetracker.embed import EmbedderUnavailable, HashingEmbedder, SentenceTransformerEmbedder, get_embedder


def test_hashing_vectors_are_unit_length_and_repeatable():
    embedder = HashingEmbedder()
    texts = ["Sea urchins graze on kelp forests.", "Budget talks resumed on Tuesday."]
    first = embedder.encode(texts)
    assert first.dtype == np.float32
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.array_equal(first, HashingEmbedder().encode(texts))


def test_hashing_ranks_shared_wording_above_unrelated_text():
    embedder = HashingEmbedder()
    claim, near, far = embedder.encode(
        [
            "Urchin numbers doubled near the harbor.",
            "Near the harbor, urchin numbers doubled last year.",
            "The council approved a new parking budget.",
        ]
    )
    assert claim @ near > 0.5 > claim @ far


def test_hashing_handles_text_with_no_words():
    assert np.allclose(HashingEmbedder().encode(["..."]), 0.0)


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder("nope")


def test_missing_library_explains_how_to_continue(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbedderUnavailable, match="--embedder hashing"):
        SentenceTransformerEmbedder()
