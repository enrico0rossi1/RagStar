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

| Stage | | Status |
|---|---|---|
| Document loading (`.txt` / `.md` / `.pdf`) | Component 1 | ✅ |
| Chunking | Component 2 | ✅ |
| Embedding | Component 3 | ✅ |
| Vector store | Component 4 | ✅ |
| Indexing-time preprocessing (Reverse HyDE) | Phase 2.1 | ✅ 3 hypothetical questions/chunk, `src/enrichment/` |
| **Indexing pipeline** (`ingest.py`) | | ✅ 157 chunks + 459 questions indexed, verified with a real search |
| Retrieval (RRF-fused chunk + question search) | Component 5 | ✅ |
| Reranking (cross-encoder) | Phase 2.3 | ✅ `BAAI/bge-reranker-base`, `src/reranker/` |
| Prompt assembly | Component 6 | ✅ |
| Generation | Component 7 | ✅ |
| **Querying pipeline** (`query.py`) | | ✅ verified end-to-end, including correctly refusing off-topic questions instead of guessing |
| Evaluation harness | Component 8 | ✅ hand-rolled LLM-as-judge (`src/eval.py`) — see baseline below |
| Tests | | ✅ 22/22 passing (`pytest`) |

**Naive RAG baseline** (2026-08-19) **→ Advanced RAG, final** (2026-08-21), 23 real Q&A pairs against the actual corpus:

| Metric | Naive RAG | Advanced RAG |
|---|---|---|
| Faithfulness | 0.85 | **0.97** |
| Answer relevance | 0.86 | **0.94** |
| Context relevance | 0.4975 | **0.615** |
| Rejection accuracy (out-of-scope questions) | 1.0 | 1.0 |
| Avg latency / query | 6.87s (p95 10.72s) | 20.19s (p95 21.17s) |
| Generation speed | 4.74 tok/s | 1.73 tok/s* |

\* completion tokens ÷ full pipeline latency (retrieval + rerank + generation), not pure LLM decode speed — the drop is reranking overhead, not the model getting slower.

Naive RAG's context relevance landed almost exactly where this project's own planning docs predicted the weak point would be (0.5-0.7) — Advanced RAG (Reverse HyDE + RRF fusion + cross-encoder reranking) moved every quality metric past baseline, at a real, deliberately-accepted latency cost (roughly tripled — a `CANDIDATE_K` sweep from 5 to 50 confirmed there's no way to claw that back without trading away quality; see the diary for the full sweep).

Full phase checklist and running notes: [progress/memory.md](./progress/memory.md). Full decision reasoning, day by day: [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

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
  User query ──► Retriever (RRF-fused chunk + question search, 30 candidates)
             ──► Reranker (cross-encoder, BAAI/bge-reranker-base, → top-5)
             ──► Prompt assembly ──► Generator (qwen2.5:7b) ──► Answer + sources
```

## Key decisions so far

Every choice below was deliberate, not default-because-nobody-looked — reasoning behind each is in [progress/naive-rag.md](./progress/naive-rag.md) / [progress/advanced-rag.md](./progress/advanced-rag.md), the dated log in [progress/memory.md](./progress/memory.md), and the full day-by-day narrative in [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

| Component | Choice | Why |
|---|---|---|
| RAG paradigm | Modular RAG (built as Naive → Advanced → Modular) | Each stage becomes a swappable module measured against a real baseline, not guessed |
| Chat model | `qwen2.5:7b` via native Ollama (Windows) | Fits consumer hardware; model-agnostic pipeline means this can change freely |
| Embedding model | `nomic-embed-text` | Same OpenAI-compatible client as chat, zero extra infra, 768-dim, normalized output |
| Document formats | `.txt`, `.md`, `.pdf` (`pdfplumber`) | Covers the real corpus; `pdfplumber` chosen over `pypdf` for better multi-column/table handling |
| Corpus | DwarfStar (`ds4`) docs + 2 arXiv RAG papers | Real public files, on-theme, topically coherent enough to make retrieval quality meaningful to test |
| Chunking | Recursive character split, hand-rolled, 2000/200 char (~512/50 tok) | Respects paragraph/sentence structure; no framework dependency; interface (`chunk_document`) kept stable so the implementation can swap later without touching the rest of the pipeline |
| Vector store | LanceDB, embedded (no server) | Local-first per ADR-0001; full rebuild per `ingest.py` run — simplest way to guarantee no duplicate chunks at this corpus size |
| Similarity metric | Cosine, set explicitly | LanceDB defaults to squared-L2; set cosine explicitly rather than relying on the coincidence that normalized vectors happen to rank the same either way |
| Final context size | Top-5 chunks sent to the prompt | Unchanged since Naive RAG — the candidate pool that feeds the reranker (below) is what grew, not this |
| Data layout | `data/knowledge/` (indexed) vs. `data/other/` (ignored by the loader) | Directory-based separation of what's actually part of the RAG corpus, gitignored entirely |
| System prompt | Strict grounding (answer only from context, else "I don't know") | Makes faithfulness measurable, and verified in practice — off-topic questions get correctly refused instead of a confident wrong answer |
| Temperature | `0.0` | Reproducible eval runs. Doesn't guarantee correctness on its own — that's retrieval quality + strict grounding; this just removes sampling randomness as a variable |
| Testing | pytest, hits the real local stack (not mocked) | Consistent with self-checks used throughout `src/`; 22/22 passing |
| Eval framework | Hand-rolled LLM-as-judge, not RAGAS | `ragas` 0.4.3 has a broken dependency chain (hard-imports Google Vertex AI support removed from `langchain-community`, itself now deprecated/sunset). Replicated its 3 core metrics directly — ~150 lines, no heavy dependency |
| Indexing preprocessing | Reverse HyDE — 3 hypothetical questions/chunk, RRF-fused with chunk embeddings | Closes the query/document vocabulary gap from the indexing side; a raw-pooled first attempt regressed every metric until fused by rank (RRF) instead of raw distance |
| Query preprocessing | None — query-time HyDE tried, measured negative (answer relevance below baseline, +latency), reverted | One-variable-at-a-time discipline: keep what's measured better than baseline, drop what isn't, regardless of "it's part of the roadmap" |
| Reranker | Cross-encoder, `BAAI/bge-reranker-base` (`sentence-transformers`) | Chosen over LLM-based (zero new deps but slower) and MMR (free but diversity-, not quality-, focused). Scores `(query, chunk)` pairs jointly — correctly separates real content from garbled PDF-extraction noise that bi-encoder distance couldn't |
| Candidate pool size | `CANDIDATE_K=30` | Swept 5–50 against the eval harness. Quality peaks at 30 (beats the naive default of 20 on every metric); no smaller K gave a good latency trade, so kept the quality-optimal point |

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

# Ask a question (first run downloads the reranker model, ~280MB, then it's cached):
.venv/Scripts/python src/query.py "What is retrieval-augmented generation?"

# Run the test suite:
.venv/Scripts/python -m pytest

# Run the eval harness:
.venv/Scripts/python src/eval.py
```

## What's next

**Phase 2 — Advanced RAG: done.** Naive RAG's baseline exposed a concrete, measured weakness: context relevance of 0.4975 — the retriever pulled back chunks only loosely related to the question about half the time. Reverse HyDE (indexing-time) + RRF fusion + cross-encoder reranking (post-retrieval) fixed it — every quality metric now beats baseline. Query-time preprocessing (HyDE) was tried and measured negative, so it was reverted rather than kept for roadmap completeness. The cost: latency roughly tripled (6.87s → 20.19s), a deliberately accepted tradeoff after a `CANDIDATE_K` sweep confirmed there's no way to claw it back without giving up the quality gain. Full reasoning: [progress/advanced-rag.md](./progress/advanced-rag.md) and [progress/diario_di_bordo.md](./progress/diario_di_bordo.md).

**Phase 3 — Modular RAG.** Now that Advanced RAG's individual techniques are proven, the pipeline gets restructured into composable, swappable modules (retriever, reranker, router, memory) that can be recombined per use case, rather than a fixed linear chain. This is where multi-source retrieval and adaptive/iterative retrieval patterns come in.

**Phase 4 — DwarfStar integration.** The entire pipeline is already built against the model-agnostic OpenAI-compatible interface specifically so this step requires zero code changes: swap the `BASE_URL` env var to point at [`ds4-server`](https://github.com/antirez/ds4) once qualifying hardware is available, and re-run the same eval harness to compare a local llama.cpp-class engine against a purpose-built one. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for the reasoning behind keeping this decoupled from day one.

Full plan: [ROADMAP.md](./ROADMAP.md).

## Docs

- [ROADMAP.md](./ROADMAP.md) — full phase-by-phase build plan (Naive → Advanced → Modular → DwarfStar)
- [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) — why this is built model-agnostic instead of on top of DwarfStar directly
- [details.md](./details.md) — RAG architecture literature notes + DwarfStar technical reference
- [progress/](./progress/) — per-phase detailed docs (`naive-rag.md`, `advanced-rag.md`, `modular-rag.md`), the decision-table log (`memory.md`), and the day-by-day reasoning diary (`diario_di_bordo.md`)
