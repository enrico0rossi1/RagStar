# Advanced RAG — Architecture & Decision Guide

**Status:** Complete (2026-08-21)
**Prerequisite:** [naive-rag.md](./naive-rag.md) phase gate satisfied, baseline recorded in [memory.md](./memory.md)
**Decision reasoning / dev diary:** [diario_di_bordo.md](./diario_di_bordo.md)

---

Advanced RAG adds pre-retrieval (indexing-time and query-time) and post-retrieval improvements around the Naive RAG core, each measured against the Phase 1 baseline. Three techniques were attempted; two were kept.

## 2.1 — Indexing-time preprocessing: Reverse HyDE

**Decision:** Hypothetical questions per chunk (3/chunk), generated at ingest time and indexed alongside each chunk's own embedding — [src/enrichment/](../src/enrichment/). Retrieval fuses chunk-embedding and question-embedding search results via Reciprocal Rank Fusion (RRF), not a single raw-distance ranking — [src/retriever/retriever.py](../src/retriever/retriever.py).

Picked over metadata attachment (nothing downstream filters on metadata, so it wouldn't move retrieval quality), Small2Big (solves a chunk-size problem never diagnosed here), and a KG index (roadmap's own advice: defer unless the corpus is highly structured, which this one isn't) — Reverse HyDE goes directly at the query/document vocabulary gap the Naive RAG baseline's low context relevance (0.4975) pointed at.

**First attempt regressed every quality metric** — question-embeddings systematically outrank chunk-embeddings on raw cosine distance regardless of true relevance, since two questions share interrogative phrasing. Fixed with RRF fusion (rank position decides the combined score, not raw distance). Reasoning and full numbers: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

## 2.2 — Query optimization: tried, reverted

**Decision:** No query-time preprocessing in the current pipeline.

Tried query-time HyDE (embed a generated hypothetical answer, search it as a third fused list alongside the chunk/question lists — [src/hyde/](../src/hyde/), since removed). Measured negative: answer relevance dropped to 0.84 (below the 0.86 naive baseline), context relevance barely moved, and it added ~2.75s latency per query. The reranker (2.3) was already doing the work of separating good candidates from noise regardless of which list surfaced them — a third, noisier candidate source had nothing left to contribute. Reverted cleanly (code, tests, and the `generate()` parameter it needed). Full reasoning: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

## 2.3 — Post-retrieval: reranking

**Decision:** Cross-encoder reranking, `BAAI/bge-reranker-base` via `sentence-transformers` — [src/reranker/](../src/reranker/). `retrieve()` over-fetches a candidate pool (`CANDIDATE_K=30`), the cross-encoder re-scores each `(query, chunk)` pair jointly, top `TOP_K=5` go to the prompt.

Picked over an LLM-based reranker (would've stayed Ollama-only/zero new deps, but slower per query) and MMR (free, but optimizes for result diversity, not for filtering low-quality content — the actual problem here was garbled PDF-bibliography chunks scoring well on topical relevance despite being unusable text). Cross-encoder correctly separates real content from that noise (0.82 vs 0.23 score on a real traced example) once it actually gets to see the right candidates — which required first recalibrating `RRF_K` (60 → 5; the web-scale default was starving the single best chunk-embedding match out of the candidate pool in this small, single-topic corpus). Reasoning: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

**`CANDIDATE_K` tuning (2026-08-21):** swept 5 through 50 against the eval harness to address the latency this step adds. Quality is not monotonic below 20 (every value under 20 lost real quality for modest savings) and actually *peaks* at 30 — better than the initial default of 20 on every metric, at the cost of *more* latency, not less. K=45+ hit a latency cliff read as this machine's resource limits, not a real cost curve, and was excluded from the decision. Landed on `CANDIDATE_K=30`: not a latency fix, a deliberate choice to keep the quality-optimal point since no smaller K offered a good trade. Full reasoning: [diario_di_bordo.md — 2026-08-21](./diario_di_bordo.md#2026-08-21-the-candidate_k-sweep-more-candidates-isnt-free-but-it-isnt-the-cost-you-think).

---

## Phase gate — ✅ SATISFIED (2026-08-21)

- [x] All three Advanced RAG buckets attempted (indexing-time, query-time, post-retrieval) — 2 kept, 1 measured negative and reverted
- [x] Every change re-run through the eval harness, before/after recorded, no assumption taken on faith
- [x] `CANDIDATE_K` tuned against the same harness rather than guessed
- [x] Final scores vs. Naive RAG baseline recorded in [memory.md](./memory.md)

**Final scores (naive baseline → Advanced RAG, CANDIDATE_K=30):**

| Metric | Naive RAG | Advanced RAG |
|---|---|---|
| Faithfulness | 0.85 | 0.97 |
| Context relevance | 0.4975 | 0.615 |
| Answer relevance | 0.86 | 0.94 |
| Rejection accuracy | 1.0 | 1.0 |
| Avg latency/query | 6.87s | 20.19s |

Every quality metric improved substantially; latency roughly tripled — an accepted, measured tradeoff, not an oversight. Next: [modular-rag.md](./modular-rag.md).
