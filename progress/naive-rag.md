# Naive RAG — Architecture & Decision Guide

**Status:** Complete (2026-08-19)
**Gate:** Complete eval harness and record baseline scores before moving to [advanced-rag.md](./advanced-rag.md) — satisfied
**Progress:** [memory.md](./memory.md)
**Decision reasoning / dev diary:** [diario_di_bordo.md](./diario_di_bordo.md)

---

## What Naive RAG is

The simplest working shape of RAG: find relevant chunks, paste them into a prompt, ask the LLM to answer from that context. No query rewriting, no reranking, no iterative retrieval, no compression, no routing, no memory. This is the baseline every later phase (see [advanced-rag.md](./advanced-rag.md)) is measured against.

```
INDEXING (offline, once):
  Documents ─► Loader ─► Chunker ─► Embedding model ─► Vector Store

QUERYING (online, per request):
  Query ─► Embedding model ─► Vector Store (similarity search, top-K)
        ─► Prompt assembler ─► LLM ─► Answer
```

## Components

Each component below: what it does, and the decision actually made. Full option comparisons and reasoning behind each choice live in the diary, not here — this doc is the *what*, not the *why*.

| # | Component | What it does | Decision |
|---|---|---|---|
| 1 | Document loading | Reads `.txt`/`.md`/`.pdf` from disk into raw text | `.txt`, `.pdf` (`pdfplumber`), `.md` — [src/loader/](../src/loader/) |
| 2 | Chunking | Splits raw text into retrieval-sized pieces | Recursive character split, hand-rolled, 2000/200 char (~512/50 tok) — [src/chunker/](../src/chunker/) |
| 3 | Embedding | Text → dense vector | `nomic-embed-text` via Ollama, 768-dim — [src/embedder/](../src/embedder/) |
| 4 | Vector store | Persists embeddings, runs similarity search | LanceDB, embedded, full rebuild per `ingest.py` run — [src/ingest.py](../src/ingest.py) |
| 5 | Retrieval | query → top-K similar chunks | Cosine metric (explicit), top-K=5 — [src/retriever/](../src/retriever/) |
| 6 | Prompt assembly | query + chunks → chat messages | Strict grounding system prompt, best-first order, `[Source: ...]` citations — [src/prompt/](../src/prompt/) |
| 7 | Generation | chat messages → answer | `qwen2.5:7b`, temperature 0.0, max 512 tokens — [src/generator/](../src/generator/) |
| 8 | Evaluation | Scores the whole pipeline | Hand-rolled LLM-as-judge (not RAGAS — broken dependency chain, see diary), 23 Q&A pairs — [src/eval.py](../src/eval.py) |

Full reasoning for every row above, in the order each decision actually got made: [diario_di_bordo.md — 2026-08-17 through 2026-08-19](./diario_di_bordo.md).

## Phase gate — ✅ SATISFIED (2026-08-19)

- [x] `ingest.py` runs end-to-end without errors
- [x] `query.py` returns a grounded answer for a test question
- [x] `eval.py` produces scores for all 3 quality metrics
- [x] Negative rejection tested — 3/3 out-of-scope questions correctly refused
- [x] Baseline scores recorded in [memory.md](./memory.md)

**Baseline (2026-08-19):** faithfulness 0.85, context relevance 0.4975, answer relevance 0.86, rejection accuracy 1.0, avg latency 6.87s, 4.74 tok/s. Context relevance landing in the predicted 0.5–0.7 "naive RAG weak point" range is what [advanced-rag.md](./advanced-rag.md) targets next.
