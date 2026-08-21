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
from src.reranker.reranker import rerank
from src.retriever.retriever import TOP_K, RetrievedChunk, retrieve

CANDIDATE_K = 30  # over-fetch this many before reranking down to TOP_K
# Swept 5/10/15/20/30/35/40/45/50 against the eval harness (2026-08-21): quality
# peaks at 30 (faithfulness 0.97, context relevance 0.615, answer relevance
# 0.94 — all better than 20's 0.92/0.575/0.89), then plateaus/declines by 40.
# Latency scales ~linearly up to 40 (~0.4s/candidate); 45+ hit a non-linear
# cliff (80s+) that reproduced across two consecutive values and reads as
# resource exhaustion (thermal/RAM) on this machine, not a real cost curve —
# stayed well clear of it. See diario_di_bordo.md, 2026-08-21.


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    completion_tokens: int


def query(question: str) -> QueryResult:
    candidates = retrieve(question, k=CANDIDATE_K)
    chunks = rerank(question, candidates, top_k=TOP_K)
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
