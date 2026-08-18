# RagStar — Build Roadmap

**RAG paradigm: Modular RAG** ✓  
**Inference backend (today):** Ollama (any OpenAI-compatible local model)  
**Inference backend (later):** DwarfStar `ds4-server` — swap `BASE_URL` env var, zero code changes  
**Reference:** Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (arXiv:2312.10997)  
**ADR:** [ADR-0001](../ADR-0001-rag-on-dwarfstar.md)

> This file lives in `progress/`. The original is also at the repo root for quick reference.

---

## Why Modular RAG

Modular RAG treats each pipeline stage as a composable, swappable module (retriever, reranker, memory, router, generator). You build Naive RAG first as a baseline, then promote it to Advanced RAG, then unlock modular patterns one by one. This is the right architecture for a learning project: each module you add produces a measurable change in eval metrics, so you always know what you gained.

```
[Naive RAG] → [Advanced RAG] → [Modular RAG]
     ↓               ↓                ↓
  baseline       pre/post-         composable
                 retrieval         patterns +
                 tuning            new modules
```

---

## Phase 0 — Stack Setup

**Goal:** a single Python script that can call a local LLM and a local embedding model.

### 0.1 Runtime
- Python 3.11+
- `openai` SDK (works against Ollama's OpenAI-compatible endpoint unchanged)
- `OLLAMA_BASE_URL=http://localhost:11434/v1` — the only env var that changes when you switch to ds4

### 0.2 Chat model (Ollama)
Pull any of these — pick one that fits your VRAM:

| Model | Size | Notes |
|-------|------|-------|
| `mistral:7b` | ~4 GB | Fast, strong reasoning |
| `llama3:8b` | ~5 GB | Good instruction following |
| `qwen2.5:7b` | ~5 GB | Strong on retrieval-style tasks |
| `phi3:mini` | ~2 GB | CPU-friendly |

**Chosen model:** `qwen2.5:7b` (2026-08-18, native Windows Ollama)

### 0.3 Embedding model (separate from chat LLM — always)
DwarfStar has no embeddings endpoint; this service stays separate even after migration.

| Option | How to run | Dims | Notes |
|--------|-----------|------|-------|
| `nomic-embed-text` | `ollama pull nomic-embed-text` | 768 | Local, zero extra infra |
| `mxbai-embed-large` | `ollama pull mxbai-embed-large` | 1024 | Higher quality, slower |
| `all-MiniLM-L6-v2` | `pip install sentence-transformers` | 384 | Tiny, fast, no Ollama needed |
| `BGE-small-en` | `pip install sentence-transformers` | 384 | Good English retrieval |

**Chosen embedding model:** `nomic-embed-text` (2026-08-18, via Ollama, verified in [src/embedder/embedder.py](../src/embedder/embedder.py))

### 0.4 Verification gate
Before Phase 1: one script that calls chat + embeddings and prints both outputs. No moving forward until this is green.

---

## Phase 1 — Naive RAG (Baseline, Required)

**Goal:** working end-to-end pipeline before optimizing anything. Every later phase is measured against this baseline.

**Detailed doc:** [naive-rag.md](./naive-rag.md)

---

## Phase 2 — Advanced RAG (Pre- and Post-Retrieval)

**Goal:** improve retrieval quality and answer quality without changing the core pipeline shape.

**Detailed doc:** [advanced-rag.md](./advanced-rag.md)

---

## Phase 3 — Modular RAG (The Chosen Paradigm)

**Goal:** compose the pipeline from swappable, independently-testable modules. Add one module at a time; measure impact on eval metrics.

**Detailed doc:** [modular-rag.md](./modular-rag.md)

---

## Phase 4 — Retrieval Source Expansion

Start with plain text. Add other source types only when you have real data that needs them.

| Source type | Approach | Add when |
|------------|----------|----------|
| Unstructured text | ✓ Covered in Phase 1 | Always |
| PDFs with tables | Table extraction → Text-2-SQL or structured parsing | You have tabular data |
| SQLite / structured DB | Text-2-SQL (LLM generates SQL from query) | You have relational data |
| Knowledge graph | Entity-relation triplets, graph traversal | Structured domain with clear entities |
| LLM-generated content | Synthetic documents for corpus gaps | Retrieval consistently misses key facts |

**Chosen expansion path:** TBD

---

## Phase 5 — Embedding Fine-tuning (Defer)

Skip until retrieval quality plateaus on off-the-shelf embeddings. Revisit with a new ADR.

---

## Phase 6 — Evaluation Harness

Build this in parallel with Phase 1. Don't skip it — without numbers you can't tell if Phase 2/3 helped.

| Framework | What it evaluates | Metrics | Local? |
|-----------|------------------|---------|--------|
| **RAGAS** | Retrieval + Generation | Context relevance, Faithfulness, Answer relevance | Yes (uses your LLM) |
| **TruLens** | Retrieval + Generation | Same three + custom | Yes |
| **ARES** | Retrieval + Generation | Same three | Needs labeled data |
| **RGB** | Both | Accuracy, EM (noise, rejection, integration, counterfactual) | Yes |
| **CRUD-RAG** | Broader tasks | BLEU, ROUGE-L, BertScore | Yes |

**Chosen eval framework:** TBD

---

## Phase 7 — DwarfStar Integration (Deferred)

When you have access to qualifying hardware (96GB+ unified memory Mac, multi-GPU rig):

1. `export BASE_URL=http://localhost:8080/v1` → point at `ds4-server`
2. Pull DeepSeek V4 Flash or GLM 5.2 via ds4
3. Run the Phase 6 eval harness against ds4 backend — same pipeline, new numbers
4. Compare results to Ollama baseline

**Nothing else changes.** The embedding service stays separate (ds4 has no embeddings endpoint).

---

## Decision Log

Fill this in as you make each choice:

| Phase | Decision | Chosen | Date |
|-------|----------|--------|------|
| 0.2 | Chat model | qwen2.5:7b | 2026-08-18 |
| 0.3 | Embedding model | nomic-embed-text | 2026-08-18 |
| 1.1 | Document formats | .txt, .pdf, .md | 2026-08-18 |
| 1.1 | Chunking strategy | — | |
| 1.1 | Chunk size / overlap | — | |
| 1.2 | Vector store | — | |
| 1.3 | Similarity metric / top-K | — | |
| 1.5 | Eval framework | — | |
| 2.1 | Metadata fields | — | |
| 2.1 | Reverse HyDE? | — | |
| 2.1 | Small2Big? | — | |
| 2.2 | Query optimization strategy | — | |
| 2.3 | Reranker | — | |
| 2.3 | Context compression | — | |
| 3.3 | Iterative retrieval? | — | |
| 3.3 | Recursive retrieval? | — | |
| 3.3 | Adaptive retrieval pattern | — | |
| 3.4 | Hybrid retrieval? | — | |
| 3.5 | Routing? | — | |
| 4 | Source types to add | — | |

---

## Build Order Summary

```
Phase 0  Stack setup (Ollama + embedding model)
   │
Phase 1  Naive RAG + eval harness  ← establish baseline numbers
   │
Phase 2  Advanced RAG (pre/post-retrieval)  ← improve numbers
   │
Phase 3  Modular RAG  ← compose, swap, iterate
   │
Phase 4  Expand sources (if needed)
   │
Phase 5  Fine-tune embeddings (if needed, with its own ADR)
   │
Phase 7  Swap BASE_URL → ds4-server  ← zero code changes
```

Each phase gate: run the eval harness, record numbers in the Decision Log, then proceed.
