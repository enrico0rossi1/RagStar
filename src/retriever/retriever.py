"""
Retrieves the top-K most similar chunks to a query from the vector store
built by ingest.py. Uses the same embedder as indexing — query and chunks
must live in the same vector space or similarity is meaningless.
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


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float  # cosine distance: 0 = identical, 2 = opposite. Lower is more similar.


def retrieve(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    db = lancedb.connect(VECTOR_DB_DIR)
    table = db.open_table(TABLE_NAME)
    query_vector = embed([query])[0]
    results = table.search(query_vector).metric("cosine").limit(k).to_list()
    return [
        RetrievedChunk(
            text=r["text"],
            source=r["source"],
            chunk_index=r["chunk_index"],
            distance=r["_distance"],
        )
        for r in results
    ]


if __name__ == "__main__":
    # Self-check: retrieve against a known question and sanity-check the results.
    # Run with: .venv/Scripts/python.exe src/retriever/retriever.py
    results = retrieve("What is retrieval-augmented generation?")
    assert results, "no results retrieved — did you run ingest.py first?"
    assert len(results) == TOP_K
    for r in results:
        print(f"{r.source} (chunk {r.chunk_index}, dist={r.distance:.4f}): {r.text[:80]!r}")
