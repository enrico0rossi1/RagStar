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

**Decision:** Cross-encoder reranking, `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` — [src/reranker/](../src/reranker/). `retrieve()` over-fetches a candidate pool (`CANDIDATE_K=40`), the cross-encoder re-scores each `(query, chunk)` pair jointly, top `TOP_K=5` go to the prompt.

Picked cross-encoder reranking over an LLM-based reranker (would've stayed Ollama-only/zero new deps, but slower per query) and MMR (free, but optimizes for result diversity, not for filtering low-quality content — the actual problem here was garbled PDF-bibliography chunks scoring well on topical relevance despite being unusable text). A cross-encoder correctly separates real content from that noise once it actually gets to see the right candidates — which required first recalibrating `RRF_K` (60 → 5; the web-scale default was starving the single best chunk-embedding match out of the candidate pool in this small, single-topic corpus). Reasoning: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

**`CANDIDATE_K` tuning, round 1 (2026-08-21), with `BAAI/bge-reranker-base`:** swept 5 through 50. Quality is not monotonic below 20 (every value under 20 lost real quality for modest savings) and actually *peaked* at 30 — better than the initial default of 20 on every metric, at the cost of *more* latency (20.19s), not less.

**Reranker model swap (2026-08-21):** rather than accept 20.19s, checked the actual hardware — an RTX 4050 (6GB VRAM) is present, but `sentence-transformers` had pulled CPU-only PyTorch, so the reranker never touched the GPU. The GPU itself was also nearly full: Ollama's `qwen2.5:7b` already uses 4.3GB, leaving only ~1.6GB free — too tight against BGE's ~1.1GB of weights to add safely (risked pushing Ollama further onto CPU, trading a reranking speedup for a generation slowdown). Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) instead — ~6x cheaper per candidate even on CPU, no VRAM risk. Re-swept `CANDIDATE_K` (30/40/50/60/80) since reranking got so much cheaper: quality plateaus at 40 (byte-identical through 80), so kept 40. A rejection-accuracy dip at K≥50 traced back to a measurement artifact in `eval.py`'s substring check, not a real failure — the model's actual answer was a correctly-phrased refusal. Full reasoning: [diario_di_bordo.md — 2026-08-21](./diario_di_bordo.md#2026-08-21-the-candidate_k-sweep-more-candidates-isnt-free-but-it-isnt-the-cost-you-think).

---

## Phase gate — ✅ SATISFIED (2026-08-21)

- [x] All three Advanced RAG buckets attempted (indexing-time, query-time, post-retrieval) — 2 kept, 1 measured negative and reverted
- [x] Every change re-run through the eval harness, before/after recorded, no assumption taken on faith
- [x] `CANDIDATE_K` tuned against the same harness rather than guessed
- [x] Final scores vs. Naive RAG baseline recorded in [memory.md](./memory.md)

**Final scores (naive baseline → Advanced RAG, MiniLM reranker, CANDIDATE_K=40):**

| Metric | Naive RAG | Advanced RAG |
|---|---|---|
| Faithfulness | 0.85 | 0.94 |
| Context relevance | 0.4975 | 0.7025 |
| Answer relevance | 0.86 | 0.89 |
| Rejection accuracy | 1.0 | 1.0 |
| Avg latency/query | 6.87s | 8.87s |

Every quality metric beats baseline, and latency is back near where Naive RAG started despite running the full Reverse HyDE + RRF + reranking pipeline — the initial reranker choice (`BAAI/bge-reranker-base`) measured well on quality but cost 20.19s; swapping to a lighter, better-suited model (`ms-marco-MiniLM-L-6-v2`) recovered almost all of that latency without giving back the quality gain. Next: [modular-rag.md](./modular-rag.md).
