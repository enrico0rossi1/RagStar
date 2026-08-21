# RagStar

A local-first RAG (Retrieval-Augmented Generation) system, built from scratch as a learning project: **Naive RAG → Advanced RAG → Modular RAG**, each phase measured against the last with a real evaluation harness instead of "it seems better."

Everything runs on a local Ollama backend today, built against the standard OpenAI-compatible chat-completions API so it can later point at [DwarfStar (`ds4`)](https://github.com/antirez/ds4) — antirez's local inference engine — by swapping one env var, with zero code changes. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for why that boundary exists.

## Highlights

- **Full RAG pipeline built from first principles** — loading, chunking, embedding, vector search, prompt engineering, and generation, each as an independently tested module, not assembled from a framework's black box.
- **Real evaluation, not vibes** — a hand-rolled LLM-as-judge harness (faithfulness, context relevance, answer relevance, latency, throughput) run against a hand-written, corpus-grounded Q&A set. Every architectural decision downstream is measured against this baseline.
- **Diagnosed and worked around a real upstream bug**, not just an integration — traced a broken `ragas` install to an unpinned dependency and a since-deprecated transitive package, confirmed it against the project's own open GitHub issue, and made a documented build-vs-buy call instead of patching around it blindly.
- **Engineering discipline throughout**: a full pytest suite (22/22 passing) hitting the real local stack, an architecture decision record (ADR) justifying a key design boundary, and a running, dated decision log for every non-trivial choice — chunking strategy, similarity metric, prompt design, and why.
- **Verified failure modes, not just the happy path** — the system correctly refuses to answer out-of-scope questions ("I don't know based on the provided documents") instead of hallucinating, confirmed with a real automated test.

## Status

**Phase 1 (Naive RAG) and Phase 2 (Advanced RAG): both complete.** Full pipeline works end-to-end, every technique measured against the previous baseline with a real before/after, phase gates satisfied. Next: Phase 3 (Modular RAG).

| Stage | Status |
|---|---|
| Naive RAG core — loading, chunking, embedding, vector store, prompt assembly, generation | ✅ full detail: [progress/naive-rag.md](./progress/naive-rag.md) |
| **Indexing-time preprocessing — Reverse HyDE** | ✅ 3 hypothetical questions/chunk generated at index time, `src/enrichment/` |
| **Indexing pipeline** (`ingest.py`) | ✅ 157 chunks + 459 questions indexed, verified with a real search |
| **Retrieval — RRF-fused chunk + question search** | ✅ `src/retriever/`, Reciprocal Rank Fusion across two embedding populations |
| **Reranking — cross-encoder** | ✅ `ms-marco-MiniLM-L-6-v2`, `src/reranker/`, 40-candidate pool → top-5 |
| **Querying pipeline** (`query.py`) | ✅ verified end-to-end, including correctly refusing off-topic questions instead of guessing |
| Evaluation harness | ✅ hand-rolled LLM-as-judge (`src/eval.py`) — see baseline below |
| Tests | ✅ 22/22 passing (`pytest`) |

**Naive RAG baseline** (2026-08-19) **→ Advanced RAG, final** (2026-08-21), 23 real Q&A pairs against the actual corpus:

| Metric | Naive RAG | Advanced RAG |
|---|---|---|
| Faithfulness | 0.85 | **0.94** |
| Answer relevance | 0.86 | **0.89** |
| Context relevance | 0.4975 | **0.7025** |
| Rejection accuracy (out-of-scope questions) | 1.0 | 1.0 |
| Avg latency / query | 6.87s (p95 10.72s) | 8.87s (p95 ~9.8s) |
| Generation speed | 4.74 tok/s | ~2.9 tok/s* |

\* completion tokens ÷ full pipeline latency (retrieval + rerank + generation), not pure LLM decode speed.

Naive RAG's context relevance landed almost exactly where this project's own planning docs predicted the weak point would be (0.5-0.7) — Advanced RAG (Reverse HyDE + RRF fusion + cross-encoder reranking) moved every quality metric past baseline. A first reranker choice (`BAAI/bge-reranker-base`) hit those numbers too but at 20.19s — tripling latency. Rather than accept that, checked the actual hardware (a free GPU, but the wrong PyTorch build; a nearly-full 6GB VRAM budget once Ollama's own model was accounted for) and swapped to a smaller, better-suited cross-encoder (`ms-marco-MiniLM-L-6-v2`) instead of fighting for GPU memory — latency dropped back to nearly Naive RAG's own speed without giving back the quality gain. Full reasoning, including the dead ends: [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

Full phase checklist and running notes: [progress/memory.md](./progress/memory.md). Full decision reasoning, day by day: [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

### How Advanced RAG actually works

**The problem:** a bi-encoder (used for retrieval) embeds the query and each chunk *independently*, then compares the two vectors — fast, since chunk vectors are precomputed at index time, but the model never reads the query against the chunk together. A short question and the long declarative prose that answers it are phrased too differently to land close together in that vector space, even when the prose is exactly right.

**Reverse HyDE** attacks this from the indexing side: at ingest time, an LLM writes 2–3 questions each chunk answers, and those get embedded and indexed too (`kind="question"`, alongside the chunk's own `kind="chunk"` row — both always point back to the real chunk text). A real user's question can now match a *generated question* — interrogative-to-interrogative, much tighter — instead of only matching prose. Mixing the two embedding populations into one raw-distance ranking backfired at first (questions systematically outrank prose regardless of relevance, since they share phrasing) — fixed by searching them as separate ranked lists and fusing by **rank position** via Reciprocal Rank Fusion, not raw distance:

```
score(chunk) = Σ over each list it appears in:  1 / (RRF_K + rank_in_that_list)
```

**Reranking** attacks it from the other side, after retrieval: a **cross-encoder** concatenates the query and a candidate chunk into *one* input and runs them through the transformer together, so query and chunk tokens attend to each other directly — it can judge relevance instead of just measuring embedding proximity. That's precise but expensive (nothing about it can be precomputed, since the score doesn't exist until both texts are present), so it only runs on the 40 candidates retrieval narrows down to, not the whole index.

Full mechanics, every parameter's reasoning, and the two sweeps behind `CANDIDATE_K=40`: [progress/advanced-rag.md](./progress/advanced-rag.md).

## Pipeline

```
INDEXING (offline, ✅ done)
  data/knowledge/ (.txt/.md/.pdf)
      │
      ▼
  Loader ──► Chunker ──► ┬──► Embedder ─────────────────► LanceDB (vector_db/)
                          └──► Enrichment (Reverse HyDE:  ──► Embedder ──► LanceDB
                               generate 3 questions/chunk)


QUERYING (online, ✅ done)
  User query ──► Retriever (RRF-fused chunk + question search, 40 candidates)
             ──► Reranker (cross-encoder, ms-marco-MiniLM-L-6-v2, → top-5)
             ──► Prompt assembly ──► Generator (qwen2.5:7b) ──► Answer + sources
```

## Key decisions so far

Every choice below was deliberate, not default-because-nobody-looked — reasoning behind each is in [progress/naive-rag.md](./progress/naive-rag.md) / [progress/advanced-rag.md](./progress/advanced-rag.md), the dated log in [progress/memory.md](./progress/memory.md), and the full day-by-day narrative in [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

**Naive RAG (Phase 1) — condensed; full reasoning in [progress/naive-rag.md](./progress/naive-rag.md):**

| Component | Choice | Why |
|---|---|---|
| RAG paradigm | Modular RAG (built as Naive → Advanced → Modular) | Each stage becomes a swappable module measured against a real baseline, not guessed |
| Models | `qwen2.5:7b` chat + `nomic-embed-text` embeddings, both via native Ollama | Fit consumer hardware; model-agnostic pipeline means either can change freely |
| Corpus & loading | DwarfStar (`ds4`) docs + 2 arXiv papers, `.txt`/`.md`/`.pdf` via `pdfplumber` | Real, on-theme, public files — makes retrieval quality meaningful to test |
| Chunking | Recursive character split, hand-rolled, 2000/200 char | No framework dependency; `chunk_document` interface kept stable for later swaps |
| Vector store | LanceDB, embedded, cosine metric set explicitly | Local-first (ADR-0001); explicit metric avoids relying on a normalization coincidence |
| Generation setup | Strict-grounding system prompt, temperature 0.0, top-5 context | Makes faithfulness measurable; refuses out-of-scope questions instead of guessing |
| Testing & eval | pytest hitting the real local stack; hand-rolled LLM-as-judge, not RAGAS | `ragas` 0.4.3 has a broken dependency chain (deprecated transitive package) — replicated its 3 core metrics directly, ~150 lines, no heavy dependency |

**Advanced RAG (Phase 2) — the current, actively-tuned part of the system:**

| Component | Choice | Why |
|---|---|---|
| Indexing preprocessing | Reverse HyDE — 3 hypothetical questions/chunk, RRF-fused with chunk embeddings | Closes the query/document vocabulary gap from the indexing side; a raw-pooled first attempt regressed every metric until fused by rank (RRF) instead of raw distance. `RRF_K=5`, not the textbook web-search default of 60 — that constant assumes lists of thousands, and in this small single-topic corpus it was starving the single best match out of the pool entirely |
| Query preprocessing | None — query-time HyDE tried, measured negative (answer relevance below baseline, +latency), reverted | One-variable-at-a-time discipline: keep what's measured better than baseline, drop what isn't, regardless of "it's part of the roadmap" |
| Reranker | Cross-encoder, `ms-marco-MiniLM-L-6-v2` (`sentence-transformers`) | Chosen over LLM-based (zero new deps but slower per query) and MMR (free but diversity-, not quality-, focused). Scores `(query, chunk)` pairs jointly — separates real content from garbled PDF-extraction noise that bi-encoder distance couldn't (4.97 vs. 3.58 score; raw distance barely told them apart) |
| Reranker model swap | `BAAI/bge-reranker-base` → `ms-marco-MiniLM-L-6-v2` | The larger model measured well but cost 20.19s/query, and its ~1.1GB of weights risked contending with Ollama for the ~1.6GB of VRAM actually free on this 6GB GPU (torch was CPU-only anyway — never touched the GPU at all). MiniLM (~90MB) is ~6x cheaper per candidate and sidesteps the VRAM question entirely |
| Candidate pool size | `CANDIDATE_K=40` | Swept 5–80 across both reranker models, twice. With MiniLM, quality is byte-identical from 40 through 80 — no reason to go higher; latency at 40 (8.87s) is barely above 30 (8.85s) since MiniLM's extra candidates are nearly free |
| Latency, overall | 6.87s → 20.19s → 8.87s | Reranking is inherently the dominant cost (a cross-encoder can't precompute anything, unlike bi-encoder search) — fixed by changing the model doing the work, not by tuning `CANDIDATE_K` alone, which plateaued |

## Project structure

```
RagStar/
├── src/
│   ├── loader/         # .txt/.md/.pdf → raw text
│   ├── chunker/        # raw text → Chunk objects (recursive character split)
│   ├── embedder/       # text → vectors (Ollama /v1/embeddings)
│   ├── enrichment/      # chunk → hypothetical questions (Reverse HyDE, LLM)
│   ├── retriever/      # query → candidate chunks (LanceDB, RRF-fused chunk+question search)
│   ├── reranker/        # query + candidates → top-K by cross-encoder relevance
│   ├── prompt/          # query + chunks → grounded chat messages
│   ├── generator/       # chat messages → LLM answer
│   ├── ingest.py         # wires loader → chunker → embedder/enrichment → LanceDB
│   ├── query.py          # wires retriever → reranker → prompt → generator; CLI entry point
│   └── eval.py            # LLM-as-judge quality scores + latency/throughput
├── tests/                 # pytest — mirrors src/, 22/22 passing
├── eval/
│   └── qa_pairs.json      # 23 real Q&A pairs (results_*.json, candidate_k_sweep_*.json gitignored)
├── data/
│   ├── knowledge/         # the actual RAG corpus (gitignored)
│   └── other/              # anything else — never read by the loader
├── vector_db/               # LanceDB persisted index (gitignored)
├── progress/                 # detailed per-phase docs, decision log, dev diary
│   └── diario_di_bordo.md     # day-by-day reasoning behind every decision
├── ADR-0001-rag-on-dwarfstar.md
├── ROADMAP.md
└── details.md                # RAG literature notes + DwarfStar architecture reference
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux

# Ollama, running locally:
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

cp .env.example .env   # BASE_URL, CHAT_MODEL, EMBED_MODEL

# Drop your own corpus into data/knowledge/, then:
.venv/Scripts/python src/ingest.py   # also generates Reverse HyDE questions per chunk — slower than a plain embed pass

# Ask a question (first run downloads the reranker model, ~90MB, then it's cached):
.venv/Scripts/python src/query.py "What is retrieval-augmented generation?"

# Run the test suite:
.venv/Scripts/python -m pytest

# Run the eval harness:
.venv/Scripts/python src/eval.py
```

## What's next

**Phase 2 — Advanced RAG: done.** Naive RAG's baseline exposed a concrete, measured weakness: context relevance of 0.4975 — the retriever pulled back chunks only loosely related to the question about half the time. Reverse HyDE (indexing-time) + RRF fusion + cross-encoder reranking (post-retrieval) fixed it — every quality metric now beats baseline. Query-time preprocessing (HyDE) was tried and measured negative, so it was reverted rather than kept for roadmap completeness. The first working reranker tripled latency (6.87s → 20.19s); rather than accept that, checked the hardware directly and swapped to a smaller, better-suited cross-encoder, landing at 8.87s — nearly the original speed, with the quality gain intact. Full reasoning: [progress/advanced-rag.md](./progress/advanced-rag.md) and [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

**Phase 3 — Modular RAG.** Now that Advanced RAG's individual techniques are proven, the pipeline gets restructured into composable, swappable modules (retriever, reranker, router, memory) that can be recombined per use case, rather than a fixed linear chain. This is where multi-source retrieval and adaptive/iterative retrieval patterns come in.

**Phase 4 — DwarfStar integration.** The entire pipeline is already built against the model-agnostic OpenAI-compatible interface specifically so this step requires zero code changes: swap the `BASE_URL` env var to point at [`ds4-server`](https://github.com/antirez/ds4) once qualifying hardware is available, and re-run the same eval harness to compare a local llama.cpp-class engine against a purpose-built one. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for the reasoning behind keeping this decoupled from day one.

Full plan: [ROADMAP.md](./ROADMAP.md).

## Docs

- [ROADMAP.md](./ROADMAP.md) — full phase-by-phase build plan (Naive → Advanced → Modular → DwarfStar)
- [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) — why this is built model-agnostic instead of on top of DwarfStar directly
- [details.md](./details.md) — RAG architecture literature notes + DwarfStar technical reference
- [progress/](./progress/) — per-phase detailed docs (`naive-rag.md`, `advanced-rag.md`, `modular-rag.md`), the decision-table log (`memory.md`), and the day-by-day reasoning diary (`diario_di_bordo.md`)
