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

CANDIDATE_K = 40  # over-fetch this many before reranking down to TOP_K
# Re-swept after switching the reranker to MiniLM (much cheaper per candidate
# than the earlier BAAI/bge-reranker-base, see src/reranker/reranker.py):
# quality is byte-identical from K=40 through K=80 (same top-5 chunks win
# regardless of pool size beyond 40), while latency keeps climbing for no
# benefit. K=40 matches/beats K=30 on every quality metric at effectively the
# same latency as 30 (MiniLM's extra candidates are nearly free). See
# diario_di_bordo.md, 2026-08-21.


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
