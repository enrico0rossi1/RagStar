"""
Requires Ollama running locally (hits the real chat model) — see tests/README.md.
"""
from src.enrichment.enrichment import generate_questions


def test_generate_questions_returns_questions():
    text = (
        "LanceDB is an embedded vector database with no separate server "
        "process. It stores data in a local directory and supports cosine "
        "similarity search."
    )

    questions = generate_questions(text)

    assert questions
    assert all(q.endswith("?") for q in questions)


def test_generate_questions_respects_n():
    text = "The chunk size is 2000 characters with 200 characters of overlap."

    questions = generate_questions(text, n=1)

    assert len(questions) <= 1
