"""
Requires Ollama running locally and `src/ingest.py` already run — see
tests/README.md. These are the closest thing to end-to-end tests: real
retrieval, real prompt, real generation.
"""
from src.query import query


def test_query_answers_from_corpus():
    result = query("What is retrieval-augmented generation?")

    assert result.answer.strip()
    assert {c.source for c in result.chunks} & {"gao2024_rag_survey.pdf", "lewis2020_rag.pdf"}


def test_query_rejects_out_of_scope_question():
    # Negative rejection (naive-rag.md's robustness checklist): a question
    # with no answer in the corpus should trigger the strict-grounding
    # fallback instead of a confidently wrong answer from prior knowledge.
    result = query("What is the capital of France?")

    assert "don't know" in result.answer.lower()
