"""
Reverse HyDE: at index time, ask the LLM what questions a chunk answers,
so those questions can be embedded and stored alongside the chunk itself.

Why: a user's question and the declarative prose that answers it sit
further apart in embedding space than two passages phrased the same way
(the query/document vocabulary gap — see diario_di_bordo.md, 2026-08-20).
Indexing a chunk's own likely questions gives retrieval a second, closer
target to match against: query-to-question instead of only query-to-prose.
"""
import re

if __package__ in (None, ""):
    # Allows running this file directly (`python src/enrichment/enrichment.py`)
    # in addition to importing it as part of the `src` package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.generator.generator import generate
else:
    from ..generator.generator import generate

N_QUESTIONS = 3

_PROMPT = """Given the following text, write {n} distinct, specific questions that this text directly and fully answers. One question per line, no numbering, no bullets, no extra commentary — just the questions.

Text:
{text}"""


def _clean_line(line: str) -> str:
    line = line.strip()
    return re.sub(r"^[\d\.\)\-\*\s]+", "", line).strip()


def generate_questions(chunk_text: str, n: int = N_QUESTIONS) -> list[str]:
    """Ask the LLM what questions `chunk_text` answers.

    Best-effort: returns whatever parses as a question (ends with "?"),
    up to n. Can return fewer than n, or an empty list, if the model's
    output doesn't parse cleanly — callers should treat this as optional
    enrichment, not something every chunk must produce.
    """
    result = generate([{"role": "user", "content": _PROMPT.format(n=n, text=chunk_text)}])
    lines = [_clean_line(line) for line in result.text.splitlines()]
    questions = [line for line in lines if line.endswith("?")]
    return questions[:n]


if __name__ == "__main__":
    # Self-check: generate questions for a known piece of text.
    # Run with: .venv/Scripts/python.exe src/enrichment/enrichment.py
    text = (
        "Retrieval-Augmented Generation (RAG) combines a retriever, which fetches "
        "relevant text chunks from a knowledge base, with a generator, which uses "
        "those chunks as context to produce an answer."
    )
    questions = generate_questions(text)
    assert questions, "expected at least one question back"
    for q in questions:
        assert q.endswith("?"), f"not a question: {q!r}"
    print(f"Generated {len(questions)} questions:")
    for q in questions:
        print(f"  - {q}")
