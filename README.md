# RagStar

A local-first RAG (Retrieval-Augmented Generation) system, built from scratch as a learning project: **Naive RAG → Advanced RAG → Modular RAG**, each phase measured against the last with a real evaluation harness instead of "it seems better."

Everything runs on a local Ollama backend today, built against the standard OpenAI-compatible chat-completions API so it can later point at [DwarfStar (`ds4`)](https://github.com/antirez/ds4) — antirez's local inference engine — by swapping one env var, with zero code changes. See [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) for why that boundary exists.

## Status

**Phase 1 — Naive RAG.** Indexing pipeline is done end-to-end; querying pipeline is next.

| Stage | | Status |
|---|---|---|
| Document loading (`.txt` / `.md` / `.pdf`) | Component 1 | ✅ |
| Chunking | Component 2 | ✅ |
| Embedding | Component 3 | ✅ |
| Vector store | Component 4 | ✅ |
| **Indexing pipeline** (`ingest.py`) | | ✅ 157 chunks indexed, verified with a real search |
| Retrieval (top-K search) | Component 5 | ⬜ |
| Prompt assembly | Component 6 | ⬜ |
| Generation | Component 7 | ⬜ |
| **Querying pipeline** (`query.py`) | | ⬜ next up |
| Evaluation harness (RAGAS + latency/throughput) | Component 8 | ⬜ |

Full phase checklist and running notes: [progress/memory.md](./progress/memory.md).

## Pipeline

```
INDEXING (offline, ✅ done)
  data/knowledge/ (.txt/.md/.pdf)
      │
      ▼
  Loader ──► Chunker ──► Embedder ──► LanceDB (vector_db/)


QUERYING (online, ⬜ next)
  User query ──► Embedder ──► LanceDB search (top-K) ──► Prompt assembly ──► LLM ──► Answer
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

## Project structure

```
RagStar/
├── src/
│   ├── loader/       # .txt/.md/.pdf → raw text
│   ├── chunker/       # raw text → Chunk objects
│   ├── embedder/       # text → vectors (Ollama /v1/embeddings)
│   ├── ingest.py       # wires loader → chunker → embedder → LanceDB
│   └── query.py        # (next) embed query → search → prompt → generate
├── data/
│   ├── knowledge/       # the actual RAG corpus (gitignored)
│   └── other/            # anything else — never read by the loader
├── vector_db/            # LanceDB persisted index (gitignored)
├── progress/              # detailed per-phase docs, decision log, running notes
├── ADR-0001-rag-on-dwarfstar.md
├── ROADMAP.md
└── details.md            # RAG literature notes + DwarfStar architecture reference
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
```

`query.py` and `eval.py` land as the querying pipeline is built.

## Docs

- [ROADMAP.md](./ROADMAP.md) — full phase-by-phase build plan (Naive → Advanced → Modular → DwarfStar)
- [ADR-0001](./ADR-0001-rag-on-dwarfstar.md) — why this is built model-agnostic instead of on top of DwarfStar directly
- [details.md](./details.md) — RAG architecture literature notes + DwarfStar technical reference
- [progress/](./progress/) — per-phase detailed docs (`naive-rag.md`, `advanced-rag.md`, `modular-rag.md`) and the running decision log (`memory.md`)
