"""
Post-retrieval reranking: re-scores a candidate pool of chunks against the
query with a cross-encoder, which judges (query, chunk) pairs jointly
instead of comparing two independently-computed embeddings. That's slower
than bi-encoder distance but far more precise — aimed specifically at the
garbled-bibliography-chunk problem noted in diario_di_bordo.md (2026-08-20):
chunks that only *look* relevant via a generic embedding match should score
poorly once a model actually reads the query against that chunk's real text.

BAAI/bge-reranker-base, via sentence-transformers' CrossEncoder — the
reranker ROADMAP.md names for this. Downloaded once, cached locally
(~sentence-transformers's default cache dir), then runs on CPU; no server,
consistent with this project's local-first stance.
"""
if __package__ in (None, ""):
    # Allows running this file directly (`python src/reranker/reranker.py`)
    # in addition to importing it as part of the `src` package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.retriever.retriever import RetrievedChunk
else:
    from ..retriever.retriever import RetrievedChunk

MODEL_NAME = "BAAI/bge-reranker-base"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score `chunks` against `query` with the cross-encoder, return the
    best `top_k` in descending relevance order."""
    if not chunks:
        return []

    scores = _get_model().predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:top_k]
    return [
        RetrievedChunk(
            text=c.text,
            source=c.source,
            chunk_index=c.chunk_index,
            distance=c.distance,
            kind=c.kind,
            rerank_score=float(score),
        )
        for c, score in ranked
    ]


if __name__ == "__main__":
    # Self-check: rerank real retrieved candidates and confirm scores are
    # actually descending (not just passed through in retrieval order).
    # Run with: .venv/Scripts/python.exe src/reranker/reranker.py
    from src.retriever.retriever import retrieve

    query = "What is retrieval-augmented generation?"
    candidates = retrieve(query, k=20)
    results = rerank(query, candidates, top_k=5)

    assert len(results) == 5
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True), "results should be sorted by rerank_score descending"

    for r in results:
        print(f"{r.source} (chunk {r.chunk_index}, rerank_score={r.rerank_score:.4f}): {r.text[:80]!r}")
