"""
Requires Ollama running locally with nomic-embed-text pulled — these hit the
real embedding endpoint rather than mocking it (see tests/README.md).
"""
import math

import pytest

from src.embedder.embedder import embed


def test_embed_returns_one_vector_per_input():
    vectors = embed(["hello world", "another sentence"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_embed_output_is_normalized():
    vector = embed(["some text"])[0]

    norm = math.sqrt(sum(x * x for x in vector))

    assert norm == pytest.approx(1.0, abs=1e-3)
