# RagStar

A local-first RAG (Retrieval-Augmented Generation) system, built from scratch as a learning project: **Naive RAG → Advanced RAG → Modular RAG**, each phase measured against the last with a real evaluation harness instead of "it seems better."

Everything runs on a local Ollama backend today, built against the standard OpenAI-compatible chat-completions API so it can later point at [DwarfStar (`ds4`)](https://github.com/antirez/ds4) — antirez's local inference engine — by swapping one env var, with zero code changes. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for why that boundary exists.

## Highlights

- **Full RAG pipeline built from first principles** — loading, chunking, embedding, vector search, prompt engineering, and generation, each as an independently tested module, not assembled from a framework's black box.
- **Real evaluation, not vibes** — a hand-rolled LLM-as-judge harness (faithfulness, context relevance, answer relevance, latency, throughput) run against a hand-written, corpus-grounded Q&A set. Every architectural decision downstream is measured against this baseline.
- **Diagnosed and worked around a real upstream bug**, not just an integration — traced a broken `ragas` install to an unpinned dependency and a since-deprecated transitive package, confirmed it against the project's own open GitHub issue, and made a documented build-vs-buy call instead of patching around it blindly.
- **Engineering discipline throughout**: a full pytest suite (20/20 passing) hitting the real local stack, an architecture decision record (ADR) justifying a key design boundary, and a running, dated decision log for every non-trivial choice — chunking strategy, similarity metric, prompt design, and why.
- **Verified failure modes, not just the happy path** — the system correctly refuses to answer out-of-scope questions ("I don't know based on the provided documents") instead of hallucinating, confirmed with a real automated test.

## Status

**Phase 1 — Naive RAG: complete.** Full pipeline works end-to-end, baseline eval recorded, phase gate satisfied. Next: Phase 2 (Advanced RAG).

| Stage | | Status |
|---|---|---|
| Document loading (`.txt` / `.md` / `.pdf`) | Component 1 | ✅ |
| Chunking | Component 2 | ✅ |
| Embedding | Component 3 | ✅ |
| Vector store | Component 4 | ✅ |
| **Indexing pipeline** (`ingest.py`) | | ✅ 157 chunks indexed, verified with a real search |
| Retrieval (top-K search) | Component 5 | ✅ |
| Prompt assembly | Component 6 | ✅ |
| Generation | Component 7 | ✅ |
| **Querying pipeline** (`query.py`) | | ✅ verified end-to-end, including correctly refusing off-topic questions instead of guessing |
| Evaluation harness | Component 8 | ✅ hand-rolled LLM-as-judge (`src/eval.py`) — see baseline below |
| Tests | | ✅ 20/20 passing (`pytest`) |

**Baseline eval** (2026-08-19, 23 real Q&A pairs against the actual corpus):

| Metric | Score |
|---|---|
| Faithfulness | 0.85 |
| Answer relevance | 0.86 |
| Context relevance | 0.4975 |
| Rejection accuracy (out-of-scope questions) | 1.0 |
| Avg latency / query | 6.87s (p95 10.72s) |
| Generation speed | 4.74 tok/s |

Context relevance landed almost exactly where this project's own planning docs predicted naive RAG's weak point would be (0.5-0.7) — that's the number Advanced RAG's reranking is aimed at.

Full phase checklist and running notes: [progress/memory.md](./progress/memory.md).

## Pipeline

```
INDEXING (offline, ✅ done)
  data/knowledge/ (.txt/.md/.pdf)
      │
      ▼
  Loader ──► Chunker ──► Embedder ──► LanceDB (vector_db/)


QUERYING (online, ✅ done)
  User query ──► Retriever (embed + cosine search, top-5) ──► Prompt assembly ──► Generator (qwen2.5:7b) ──► Answer + sources
```

## Key decisions so far

Every choice below was deliberate, not default-because-nobody-looked — reasoning behind each is in [progress/naive-rag.md](./progress/naive-rag.md) and the dated log in [progress/memory.md](./progress/memory.md).

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
| Top-K | 5 | Standard starting point; tuned later against eval scores |
| Data layout | `data/knowledge/` (indexed) vs. `data/other/` (ignored by the loader) | Directory-based separation of what's actually part of the RAG corpus, gitignored entirely |
| System prompt | Strict grounding (answer only from context, else "I don't know") | Makes faithfulness measurable, and verified in practice — off-topic questions get correctly refused instead of a confident wrong answer |
| Temperature | `0.0` | Reproducible eval runs. Doesn't guarantee correctness on its own — that's retrieval quality + strict grounding; this just removes sampling randomness as a variable |
| Testing | pytest, hits the real local stack (not mocked) | Consistent with self-checks used throughout `src/`; 20/20 passing |
| Eval framework | Hand-rolled LLM-as-judge, not RAGAS | `ragas` 0.4.3 has a broken dependency chain (hard-imports Google Vertex AI support removed from `langchain-community`, itself now deprecated/sunset). Replicated its 3 core metrics directly — ~150 lines, no heavy dependency |

## Project structure

```
RagStar/
├── src/
│   ├── loader/         # .txt/.md/.pdf → raw text
│   ├── chunker/        # raw text → Chunk objects (recursive character split)
│   ├── embedder/       # text → vectors (Ollama /v1/embeddings)
│   ├── retriever/      # query → top-K similar chunks (LanceDB, cosine)
│   ├── prompt/          # query + chunks → grounded chat messages
│   ├── generator/       # chat messages → LLM answer
│   ├── ingest.py         # wires loader → chunker → embedder → LanceDB
│   ├── query.py          # wires retriever → prompt → generator; CLI entry point
│   └── eval.py            # LLM-as-judge quality scores + latency/throughput
├── tests/                 # pytest — mirrors src/, 20/20 passing
├── eval/
│   └── qa_pairs.json      # 23 real Q&A pairs (results_*.json gitignored)
├── data/
│   ├── knowledge/         # the actual RAG corpus (gitignored)
│   └── other/              # anything else — never read by the loader
├── vector_db/               # LanceDB persisted index (gitignored)
├── progress/                 # detailed per-phase docs, decision log, running notes
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
.venv/Scripts/python src/ingest.py

# Ask a question:
.venv/Scripts/python src/query.py "What is retrieval-augmented generation?"

# Run the test suite:
.venv/Scripts/python -m pytest

# Run the eval harness:
.venv/Scripts/python src/eval.py
```

## What's next

**Phase 2 — Advanced RAG.** Naive RAG's baseline exposed a concrete, measured weakness: context relevance of 0.4975 — the retriever pulls back chunks that are only loosely related to the question about half the time. The fix under evaluation is **reranking**: a cross-encoder re-scores the top-K candidates after retrieval, trading a small latency cost for precision. Query rewriting/expansion is the second lever, for cases where the raw question is a poor match for how the corpus is phrased. Every change here gets re-run through the same eval harness — the goal is a documented before/after, not an assumption that "reranking helps."

**Phase 3 — Modular RAG.** Once Advanced RAG's individual techniques are proven, the pipeline gets restructured into composable, swappable modules (retriever, reranker, router, memory) that can be recombined per use case, rather than a fixed linear chain. This is where multi-source retrieval and adaptive/iterative retrieval patterns come in.

**Phase 4 — DwarfStar integration.** The entire pipeline is already built against the model-agnostic OpenAI-compatible interface specifically so this step requires zero code changes: swap the `BASE_URL` env var to point at [`ds4-server`](https://github.com/antirez/ds4) once qualifying hardware is available, and re-run the same eval harness to compare a local llama.cpp-class engine against a purpose-built one. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for the reasoning behind keeping this decoupled from day one.

Full plan: [ROADMAP.md](./ROADMAP.md).

## Docs

- [ROADMAP.md](./ROADMAP.md) — full phase-by-phase build plan (Naive → Advanced → Modular → DwarfStar)
- [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) — why this is built model-agnostic instead of on top of DwarfStar directly
- [details.md](./details.md) — RAG architecture literature notes + DwarfStar technical reference
- [progress/](./progress/) — per-phase detailed docs (`naive-rag.md`, `advanced-rag.md`, `modular-rag.md`) and the running decision log (`memory.md`)
