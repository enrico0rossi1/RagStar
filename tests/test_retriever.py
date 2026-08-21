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


def test_retrieve_results_have_no_duplicate_chunks():
    # Each chunk is indexed under multiple embeddings (itself + its Reverse
    # HyDE questions), so raw search hits can repeat a chunk; retrieve()
    # must fuse/dedupe those down to distinct chunks before returning k.
    results = retrieve("What is retrieval-augmented generation?")

    keys = [(r.source, r.chunk_index) for r in results]
    assert len(keys) == len(set(keys))
