"""
Assembles the final prompt sent to the LLM: strict-grounding system
instructions + retrieved context (best-first, with source citations) +
the user's question.

Strict grounding (as opposed to letting the model fall back on its own
knowledge) is what makes "faithfulness" measurable later in eval — an
answer that's allowed to use prior knowledge can't be checked against the
retrieved context alone.
"""
if __package__ in (None, ""):
    # Allows running this file directly (`python src/prompt/prompt.py`)
    # in addition to importing it as part of the `src` package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.retriever.retriever import RetrievedChunk
else:
    from ..retriever.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the provided context. If the answer is not in the context, "
    'say exactly: "I don\'t know based on the provided documents." Do not use '
    "prior knowledge."
)


def assemble_prompt(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Build chat messages (system + user) from retrieved chunks and a query.

    `chunks` are assumed already ordered best-first, as returned by
    retrieve(). Returns a messages list ready for the chat completions API.
    """
    context = "\n---\n".join(f"{c.text}\n[Source: {c.source}]" for c in chunks)
    user_message = f"Context:\n---\n{context}\n---\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


if __name__ == "__main__":
    # Self-check: retrieve real chunks and assemble a prompt from them.
    # Run with: .venv/Scripts/python.exe src/prompt/prompt.py
    from src.retriever.retriever import retrieve

    query = "What is retrieval-augmented generation?"
    chunks = retrieve(query)
    messages = assemble_prompt(query, chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]

    print(messages[0]["content"])
    print()
    print(messages[1]["content"][:800])
