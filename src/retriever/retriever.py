"""
Retrieves the top-K most similar chunks to a query from the vector store
built by ingest.py. Uses the same embedder as indexing — query and chunks
must live in the same vector space or similarity is meaningless.

Each chunk is indexed under multiple embeddings: its own text, plus its
Reverse HyDE hypothetical questions (src/enrichment/enrichment.py). A raw
single similarity search over both mixed together doesn't work — a first
measured attempt (see diario_di_bordo.md, 2026-08-20) showed
question-embeddings systematically out-rank real chunk-embeddings on raw
cosine distance, because two questions share interrogative phrasing
regardless of whether their answers are actually related. That let
irrelevant chunks with generic-sounding hypothetical questions beat the
genuinely relevant chunk on distance alone, and it measurably hurt every
quality metric.

Fix: search chunk-kind and question-kind rows as two SEPARATE ranked lists,
then fuse with Reciprocal Rank Fusion (RRF) — rank position within each
list, not raw distance, decides the combined score. This is the standard
fix for combining scores from different embedding populations (also the
roadmap's Phase 3.4 answer for sparse+dense fusion); pulled forward here to
solve the same apples-to-oranges problem between chunk and question hits.

A third list (query-time HyDE — embedding a generated hypothetical answer
and searching it against chunk rows, same vocabulary-gap fix as Reverse
HyDE but from the query side) was tried and reverted: it added ~2.75s
latency per query for a regression in answer relevance below even the
naive baseline, with no compensating gain elsewhere. The reranker already
does the work of separating good candidates from noise regardless of which
list surfaced them, so a third, noisier candidate source had nothing left
to contribute. See diario_di_bordo.md, 2026-08-20, for the measured numbers.
"""
from dataclasses import dataclass

import lancedb

if __package__ in (None, ""):
    # Allows running this file directly (`python src/retriever/retriever.py`)
    # in addition to importing it as part of the `src` package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.embedder.embedder import embed
else:
    from ..embedder.embedder import embed

VECTOR_DB_DIR = "vector_db"
TABLE_NAME = "chunks"
TOP_K = 5
PER_LIST_MULTIPLIER = 4  # candidates pulled from each of the chunk/question lists, per requested k
# RRF_K=60 is the standard web-search constant (lists of thousands of results,
# so rank 1 vs rank 60 is a big gap). This corpus is ~157 chunks, single-topic
# (everything is about RAG), so almost every chunk appears somewhere in BOTH
# lists within a generous window — RRF's "credit for appearing in both lists"
# then rewards two mediocre placements over one genuinely strong match,
# pushing the single best chunk out of the fused top-k entirely (measured:
# see diario_di_bordo.md, 2026-08-20). A small RRF_K makes rank position
# matter much more steeply, so one strong single-list rank beats two weak
# ones again.
RRF_K = 5


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float  # cosine distance of the winning row: 0 = identical, 2 = opposite
    kind: str = "chunk"  # which search list produced the winning match: chunk or question
    rerank_score: float | None = None  # set by src/reranker/reranker.py; higher = more relevant


def _search(table, query_vector, kind: str, limit: int) -> list[dict]:
    return (
        table.search(query_vector)
        .metric("cosine")
        .where(f"kind = '{kind}'")
        .limit(limit)
        .to_list()
    )


def retrieve(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    db = lancedb.connect(VECTOR_DB_DIR)
    table = db.open_table(TABLE_NAME)
    query_vector = embed([query])[0]

    per_list_limit = k * PER_LIST_MULTIPLIER
    lists = {
        "chunk": _search(table, query_vector, "chunk", per_list_limit),
        "question": _search(table, query_vector, "question", per_list_limit),
    }

    fused = {}  # (source, chunk_index) -> {"score": float, "best_row": row, "best_via": list name}
    for list_name, hits in lists.items():
        for rank, r in enumerate(hits):
            key = (r["source"], r["chunk_index"])
            entry = fused.setdefault(key, {"score": 0.0, "best_row": r, "best_via": list_name})
            entry["score"] += 1.0 / (RRF_K + rank)
            if r["_distance"] < entry["best_row"]["_distance"]:
                entry["best_row"] = r
                entry["best_via"] = list_name

    ranked = sorted(fused.values(), key=lambda e: -e["score"])[:k]
    return [
        RetrievedChunk(
            text=e["best_row"]["text"],
            source=e["best_row"]["source"],
            chunk_index=e["best_row"]["chunk_index"],
            distance=e["best_row"]["_distance"],
            kind=e["best_via"],
        )
        for e in ranked
    ]


if __name__ == "__main__":
    # Self-check: retrieve against a known question and sanity-check the results.
    # Run with: .venv/Scripts/python.exe src/retriever/retriever.py
    results = retrieve("What is retrieval-augmented generation?")
    assert results, "no results retrieved — did you run ingest.py first?"
    assert len(results) == TOP_K
    for r in results:
        print(f"{r.source} (chunk {r.chunk_index}, dist={r.distance:.4f}, via={r.kind}): {r.text[:80]!r}")
