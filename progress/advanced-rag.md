# Advanced RAG — Architecture & Decision Guide

**Status:** Complete (2026-08-21)
**Prerequisite:** [naive-rag.md](./naive-rag.md) phase gate satisfied, baseline recorded in [memory.md](./memory.md)
**Decision reasoning / dev diary:** [diario_di_bordo.md](./diario_di_bordo.md)

---

Advanced RAG adds pre-retrieval (indexing-time and query-time) and post-retrieval improvements around the Naive RAG core, each measured against the Phase 1 baseline. Three techniques were attempted; two were kept. This doc covers *how* each piece works, not just *what* was chosen — the diary covers the day-by-day *why*, including every dead end.

```
INDEXING (offline)                          QUERYING (online)
  Chunk ─┬─► Embed chunk text ──────────┐     Query ─► embed(query)
         └─► Enrichment: LLM writes     │            │
             3 hypothetical questions   │            ▼
             per chunk ─► Embed each ───┤     ┌── retrieve() ──────────────┐
                                        ▼     │ search kind=chunk (top N)    │
                                    LanceDB    │ search kind=question (top N) │
                                  (chunk +     │        │         │          │
                                   question    │        └── RRF ──┘          │
                                   rows)       │     fuse by rank, not       │
                                               │     raw distance           │
                                               └────────────┬───────────────┘
                                                             ▼ (40 candidates)
                                                    ┌── rerank() ──────────┐
                                                    │ cross-encoder scores │
                                                    │ (query, chunk) pairs │
                                                    │ jointly, top 5 win   │
                                                    └──────────┬───────────┘
                                                                ▼
                                                    assemble_prompt ─► generate
```

## The problem this phase targets

Naive RAG's baseline had context relevance of 0.4975 — retrieved chunks were only loosely related to the question about half the time, even when the answer was clearly present somewhere in the corpus. Root cause, confirmed by tracing real examples: a **bi-encoder** embeds the query and each chunk *independently* (that's what makes it fast — chunk vectors are precomputed once at index time, and only the query needs embedding per request), then compares the two vectors by cosine distance. But a short interrogative question ("what license is X released under?") and the long declarative prose that answers it are phrased too differently to land close together in embedding space, even when the prose is exactly right. Both fixes below attack this same gap from different sides.

## 2.1 — Indexing-time preprocessing: Reverse HyDE

**Decision:** Hypothetical questions per chunk (3/chunk), generated at ingest time and indexed alongside each chunk's own embedding — [src/enrichment/](../src/enrichment/). Retrieval fuses chunk-embedding and question-embedding search results via Reciprocal Rank Fusion, not a single raw-distance ranking — [src/retriever/retriever.py](../src/retriever/retriever.py).

**How it works:** at ingest time, for every chunk, an LLM call asks "what questions does this text answer?" and the 2–3 questions it writes get embedded and stored as their *own* searchable rows — `kind="question"` — alongside the chunk's own `kind="chunk"` row. Both kinds always carry the same `text` field (the real chunk), so retrieval always returns real content no matter which embedding matched. At query time, a real user question can now match against a *generated question* — interrogative-to-interrogative, much tighter in embedding space — instead of only against declarative prose.

Picked over metadata attachment (nothing downstream filters on metadata, so it wouldn't move retrieval quality), Small2Big (solves a chunk-size problem never diagnosed here), and a KG index (roadmap's own advice: defer unless the corpus is highly structured, which this one isn't).

**First attempt regressed every quality metric.** Pooling chunk-embeddings and question-embeddings into one raw-distance-ranked list let the questions systematically win: two questions share interrogative phrasing regardless of whether their answers are related, so a generic-sounding generated question could beat a genuinely relevant chunk on distance alone. Confirmed with a traced example: a query about a specific license landed **zero** chunk-embedding hits in its top-5 — all five were question-embedding hits, and the actual answer-bearing chunk wasn't even in the candidate pool.

**Fix — Reciprocal Rank Fusion.** Search `kind="chunk"` and `kind="question"` as two *separate* ranked lists, then combine by rank position, not raw distance:

```
score(chunk) = Σ over each list it appears in:  1 / (RRF_K + rank_in_that_list)
```

A chunk's combined score only depends on *where it placed* in each list, never on the two lists' raw distance values being compared directly — which is exactly what breaks when the two embedding populations (chunk prose vs. generated questions) have different distance distributions. `RRF_K=5`, not the textbook web-search default of 60: that constant assumes lists of thousands of results, where rank 1 vs. rank 60 is a big gap. In this ~157-chunk, single-topic corpus (everything is about RAG), almost every chunk shows up in *both* lists within a generous window, so a large `RRF_K` ends up rewarding two mediocre placements over one genuinely strong single-list match — it actually pushed the single best chunk out of the fused top-20 entirely in one traced case. A small `RRF_K` makes rank position matter more steeply, restoring the strong single match's advantage.

Full numbers and the traced regression: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

## 2.2 — Query optimization: tried, reverted

**Decision:** No query-time preprocessing in the current pipeline.

**What was tried:** query-time HyDE — the query-side mirror of 2.1. Instead of (or alongside) embedding the raw question, ask the LLM to write a short hypothetical *answer* passage, embed that, and search it as a third fused list against chunk rows. Same underlying logic as Reverse HyDE (declarative prose embeds closer to declarative prose than a question does), just closing the gap from the query side instead of the indexing side.

**Measured negative:** answer relevance dropped to 0.84 — below the 0.86 naive baseline, not just below the reranking-only checkpoint — while context relevance barely moved and latency grew ~2.75s/query for a generated passage that only gets embedded, never shown to anyone. Read as: the reranker (2.3) already separates good candidates from noise regardless of which list surfaced them, so a third, *noisier* candidate source (a generated passage can be subtly wrong even while being declarative-prose-shaped) had nothing left to contribute — it just gave the reranker more chances to be fooled, for a real latency cost. Reverted cleanly: the third RRF list, the standalone `src/hyde/` module, its test, and the `generate()` parameter it needed all came back out.

This is the project's one-variable-at-a-time discipline in practice: a technique being "part of the roadmap" isn't a reason to keep it once it's measured worse than not having it. Full reasoning: [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive).

## 2.3 — Post-retrieval: reranking

**Decision:** Cross-encoder reranking, `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` — [src/reranker/](../src/reranker/). `retrieve()` over-fetches a candidate pool (`CANDIDATE_K=40`), the cross-encoder re-scores each `(query, chunk)` pair jointly, top `TOP_K=5` go to the prompt.

**How it works, and why it succeeds where bi-encoder distance fails:** a bi-encoder (used everywhere above) encodes the query and a chunk *separately*, with no interaction between them — fast, since chunk vectors are precomputed, but the model never actually reads the query against the chunk. A **cross-encoder** concatenates the query and chunk into one input and runs them through the transformer *together*, so every token of the query can attend directly to every token of the chunk. It can judge "does this passage actually answer this question," not just "are these two independently-computed summaries in the same neighborhood." Concretely, this is what separated a garbled, PDF-extraction-mangled bibliography chunk from the real definition chunk it was being confused with: bi-encoder distances put them almost indistinguishably close (0.2794 vs. 0.2168 — the *garbled* chunk actually scored better), while the cross-encoder scored them 4.97 vs. 3.58 (MiniLM) — a clean, correct separation. The cost is structural, not incidental: nothing about a cross-encoder score can be precomputed at index time, because it doesn't exist until both texts are present — every query re-runs a full forward pass per candidate, which is why this step, not retrieval, dominates latency.

Picked cross-encoder reranking over an LLM-based reranker (would've stayed Ollama-only, zero new dependencies, but a slower, more expensive call per query) and MMR (free, rule-based, but optimizes for *diversity* among results, not for filtering low-quality content — the actual problem here was garbled chunks scoring well on topical relevance despite being unusable text, which MMR doesn't address).

Getting the cross-encoder to actually work required first fixing something upstream: `RRF_K=60` (the web-search default from 2.1) was itself starving the single best chunk-embedding match out of the reranker's candidate pool before the reranker ever saw it — recalibrating to `RRF_K=5` was the prerequisite, not the reranker itself.

### `CANDIDATE_K` tuning, round 1 — with `BAAI/bge-reranker-base`

Swept 5 through 50 candidates against the full eval harness (2026-08-21):

| CANDIDATE_K | Faithfulness | Context relevance | Answer relevance | Avg latency |
|---|---|---|---|---|
| 5 | 0.74 | 0.5475 | 0.73 | 9.11s |
| 10 | 0.82 | 0.5025 | 0.79 | 11.93s |
| 15 | 0.77 | 0.51 | 0.74 | 13.56s |
| 20 | 0.92 | 0.575 | 0.89 | 15.44s |
| **30** | **0.97** | **0.615** | **0.94** | 20.19s |
| 35 | 0.97 | 0.6025 | 0.94 | 20.90s |
| 40 | 0.93 | 0.6025 | 0.89 | 22.15s |
| 45† | 0.93 | 0.5625 | 0.88 | 80.28s |
| 50† | 0.93 | 0.5625 | 0.93 | 135.31s |

†45 and 50 hit a non-linear latency cliff inconsistent with the clean ~linear scaling below 40 (~0.4s/extra candidate) — read as this machine's own resource limits (thermal/RAM) under 40+ minutes of sustained load, not a real cost curve, and excluded from the decision.

Quality was *not* monotonic below 20 — every value under 20 lost real quality for savings that didn't come cheap. It peaked at 30, better than the default of 20 on every metric, at the cost of *more* latency (20.19s), not less — inverting the whole premise of shrinking `K` to save time.

### Reranker model swap

Rather than accept 20.19s, checked the actual hardware instead of tuning further: an RTX 4050 (6GB VRAM) is present, but `sentence-transformers`' installed PyTorch was the CPU-only build, so the reranker never touched the GPU at all. Checking whether the GPU was even usable mattered too — `ollama ps` showed `qwen2.5:7b` already running on it (82%/18% GPU/CPU split), using 4.3 of the 6GB, leaving only ~1.6GB free. `bge-reranker-base`'s ~1.1GB of weights would be tight against that: if it didn't fit cleanly, Ollama would silently offload more of its own layers to CPU to make room, trading a reranking speedup for a generation slowdown — not a net win, and not obviously safe to try blind.

Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, ~90MB) instead of pursuing CUDA — sidesteps the VRAM question entirely, and is ~6x cheaper per candidate even while staying on CPU (reranking 30 candidates: ~11s proportional for BGE → 1.85s for MiniLM). Re-verified the smaller model still separates real content from garbled noise correctly before trusting it (4.97 vs. 3.58, see above).

First full eval at the same K=30 was a mixed result, not a clean win: context relevance improved further (0.7175) but faithfulness and answer relevance both dropped versus BGE, with answer relevance (0.84) landing just below the naive baseline's 0.86.

### `CANDIDATE_K` tuning, round 2 — with MiniLM

Since MiniLM is so much cheaper per candidate, there was real headroom to spend on a bigger pool without approaching BGE's cost. Swept 30/40/50/60/80:

| CANDIDATE_K | Faithfulness | Context relevance | Answer relevance | Rejection* | Avg latency |
|---|---|---|---|---|---|
| 30 | 0.89 | 0.7175 | 0.84 | 1.0 | 8.85s |
| **40** | **0.94** | 0.7025 | **0.89** | 1.0 | **8.87s** |
| 50 | 0.94 | 0.7025 | 0.89 | 1.0 | 10.57s |
| 60 | 0.94 | 0.7025 | 0.89 | 1.0 | 11.32s |
| 80 | 0.94 | 0.7025 | 0.89 | 1.0 | 12.36s |

\* The raw eval run reported rejection accuracy 0.6667 at K≥50 — traced before trusting it, and it wasn't real. The "failed" case was the FIFA World Cup question; the model's actual answer was *"The document provided does not contain any information about the 2022 FIFA World Cup winner"* — a correct refusal, just phrased differently than the exact substring `eval.py`'s check looks for (`"don't know" in answer.lower()`). A measurement blind spot in the harness, not a retrieval failure.

Quality is byte-identical from K=40 through K=80 — the same top-5 chunks win the rerank regardless of pool size beyond that point, so nothing is gained by going higher. **`CANDIDATE_K=40`** matches or beats the K=30 numbers on every quality metric at effectively the same latency (8.87s vs. 8.85s, since MiniLM's extra 10 candidates cost almost nothing).

Full reasoning for both rounds: [diario_di_bordo.md — 2026-08-21](./diario_di_bordo.md#2026-08-21-the-candidate_k-sweep-more-candidates-isnt-free-but-it-isnt-the-cost-you-think).

---

## Phase gate — ✅ SATISFIED (2026-08-21)

- [x] All three Advanced RAG buckets attempted (indexing-time, query-time, post-retrieval) — 2 kept, 1 measured negative and reverted
- [x] Every change re-run through the eval harness, before/after recorded, no assumption taken on faith
- [x] `CANDIDATE_K` and the reranker model tuned against the same harness rather than guessed
- [x] Final scores vs. Naive RAG baseline recorded in [memory.md](./memory.md)

**Final scores (naive baseline → Advanced RAG, MiniLM reranker, CANDIDATE_K=40):**

| Metric | Naive RAG | Advanced RAG |
|---|---|---|
| Faithfulness | 0.85 | 0.94 |
| Context relevance | 0.4975 | 0.7025 |
| Answer relevance | 0.86 | 0.89 |
| Rejection accuracy | 1.0 | 1.0 |
| Avg latency/query | 6.87s | 8.87s |

Every quality metric beats baseline, and latency is back near where Naive RAG started despite running the full Reverse HyDE + RRF + reranking pipeline. The lesson worth carrying forward: the initial reranker choice (`BAAI/bge-reranker-base`) hit good quality numbers but tripled latency, and no amount of `CANDIDATE_K` tuning fixed that — what fixed it was recognizing the *model* was the wrong tool for this hardware and swapping it, not tuning the existing tool's parameters further. Next: [modular-rag.md](./modular-rag.md).
