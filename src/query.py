"""
End-to-end querying pipeline: retrieve top-K chunks, assemble a grounded
prompt, generate an answer.

    .venv/Scripts/python.exe src/query.py "What is retrieval-augmented generation?"
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator.generator import generate
from src.prompt.prompt import assemble_prompt
from src.retriever.retriever import RetrievedChunk, retrieve


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    completion_tokens: int


def query(question: str) -> QueryResult:
    chunks = retrieve(question)
    messages = assemble_prompt(question, chunks)
    result = generate(messages)
    return QueryResult(
        question=question,
        answer=result.text,
        chunks=chunks,
        completion_tokens=result.completion_tokens,
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    result = query(question)

    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for c in result.chunks:
        print(f"  - {c.source} (chunk {c.chunk_index}, dist={c.distance:.4f})")
