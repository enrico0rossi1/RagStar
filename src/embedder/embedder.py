"""
Turns text into vectors, using a local embedding model served by Ollama.

Ollama exposes an OpenAI-compatible /v1/embeddings endpoint, the same way it
exposes /v1/chat/completions for the chat model. So we reuse the exact same
`openai` client library and BASE_URL as the chat model — only the endpoint
called and the model name differ.

Why embeddings at all: an embedding model turns text into a list of numbers
(a vector) that captures its meaning. Two chunks of text with similar meaning
end up as vectors that are close together in that vector space. That's what
lets retrieval later find "chunks similar to this query" via math (cosine
similarity) instead of keyword matching.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(base_url=os.environ["BASE_URL"], api_key="ollama")
_MODEL = os.environ["EMBED_MODEL"]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed one or more strings in a single call.

    Always pass a list, even for one string, e.g. embed([query])[0] for a
    single query vector. Batching chunks together at index time is faster
    than one call per chunk.

    Returns one vector per input string, in the same order as `texts`.
    """
    response = _client.embeddings.create(model=_MODEL, input=texts)
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    # Self-check: embed a known string and sanity-check the shape of the result.
    # Run with: .venv/Scripts/python.exe src/embedder/embedder.py
    vectors = embed(["hello world"])
    assert len(vectors) == 1, "expected one vector back for one input string"
    assert len(vectors[0]) > 0, "vector should not be empty"
    print(f"Embedded 1 text into a {len(vectors[0])}-dim vector.")
    print(f"First 5 values: {vectors[0][:5]}")
