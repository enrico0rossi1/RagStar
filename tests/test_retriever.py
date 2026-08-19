"""
Requires Ollama running locally and `src/ingest.py` already run (the real
vector_db/chunks table must exist) — see tests/README.md.
"""
from src.retriever.retriever import TOP_K, retrieve


def test_retrieve_returns_top_k_results():
    results = retrieve("What is retrieval-augmented generation?")

    assert len(results) == TOP_K


def test_retrieve_finds_relevant_source():
    results = retrieve("What is retrieval-augmented generation?")

    assert results[0].source in {"gao2024_rag_survey.pdf", "lewis2020_rag.pdf"}


def test_retrieve_results_are_ranked_by_distance():
    results = retrieve("What is retrieval-augmented generation?")

    distances = [r.distance for r in results]
    assert distances == sorted(distances)
