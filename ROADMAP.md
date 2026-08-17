# RagStar — Build Roadmap

**RAG paradigm: Modular RAG** ✓  
**Inference backend (today):** Ollama (any OpenAI-compatible local model)  
**Inference backend (later):** DwarfStar `ds4-server` — swap `BASE_URL` env var, zero code changes  
**Reference:** Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (arXiv:2312.10997)  
**ADR:** [ADR-0001](./ADR-0001-rag-on-dwarfstar.md)

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

**Chosen model:** TBD

### 0.3 Embedding model (separate from chat LLM — always)
DwarfStar has no embeddings endpoint; this service stays separate even after migration.

| Option | How to run | Dims | Notes |
|--------|-----------|------|-------|
| `nomic-embed-text` | `ollama pull nomic-embed-text` | 768 | Local, zero extra infra |
| `mxbai-embed-large` | `ollama pull mxbai-embed-large` | 1024 | Higher quality, slower |
| `all-MiniLM-L6-v2` | `pip install sentence-transformers` | 384 | Tiny, fast, no Ollama needed |
| `BGE-small-en` | `pip install sentence-transformers` | 384 | Good English retrieval |

**Chosen embedding model:** TBD

### 0.4 Verification gate
Before Phase 1: one script that calls chat + embeddings and prints both outputs. No moving forward until this is green.

---

## Phase 1 — Naive RAG (Baseline, Required)

**Goal:** working end-to-end pipeline before optimizing anything. Every later phase is measured against this baseline.

```
[Document] → chunk → embed → vector store
[Query]    → embed → similarity search → top-K chunks → prompt → LLM → answer
```

### 1.1 Document ingestion & chunking

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| Fixed-size tokens | Split every N tokens (100 / 256 / 512) | Simple; splits mid-sentence |
| Recursive character split | Split on `\n\n`, `\n`, ` ` in order | Respects structure; good default |
| Sliding window | Fixed chunks with overlap (e.g. 256 tok, 50 overlap) | Reduces boundary loss; larger index |
| Semantic segmentation | Split on topic shifts detected by embeddings | Best quality; slowest |

**Chosen chunking strategy:** TBD  
**Chunk size:** TBD  
**Overlap:** TBD

### 1.2 Vector store (embedded, local-first)

| Store | Install | Notes |
|-------|---------|-------|
| **LanceDB** | `pip install lancedb` | Embedded, columnar, fast, no server |
| **sqlite-vec** | `pip install sqlite-vec` | Embedded in SQLite, minimal deps |
| **Chroma** | `pip install chromadb` | Embedded mode available, popular |
| **FAISS** | `pip install faiss-cpu` | In-memory; no persistence without extra code |

Avoid a separate always-on vector DB service unless you have multi-user access as a real requirement (you don't yet).

**Chosen vector store:** TBD

### 1.3 Similarity search

| Option | Notes |
|--------|-------|
| Cosine similarity | Standard; scale-invariant |
| Dot product | Faster; assumes normalized vectors |
| L2 (Euclidean) | Less common for text |

**Metric:** TBD  
**Top-K:** TBD (start with 5)

### 1.4 Prompt template

```
System: You are a helpful assistant. Answer using only the provided context.
        If the answer is not in the context, say "I don't know."

Context:
{retrieved_chunks}

Question: {query}
```

Adjust system prompt after you see first outputs.

### 1.5 Naive RAG eval harness (mandatory before Phase 2)

Build a small labeled Q&A set from your own documents (~20-50 pairs). Measure:

| Metric | What it tests |
|--------|--------------|
| Context relevance | Are retrieved chunks actually about the query? |
| Faithfulness | Does the answer contradict the retrieved context? |
| Answer relevance | Does the answer address the question? |

**Chosen eval framework:** TBD (see Phase 6 for options)

---

## Phase 2 — Advanced RAG (Pre- and Post-Retrieval)

**Goal:** improve retrieval quality and answer quality without changing the core pipeline shape.

### 2.1 Indexing Optimization

#### Metadata attachment
Attach metadata to every chunk at index time so you can filter at retrieval time.

| Metadata field | Use |
|---------------|-----|
| filename, page number | Source attribution |
| section header | Structural filtering |
| timestamp | Freshness-aware retrieval |
| chunk summary (LLM-generated) | Richer search signal |

**Chosen fields:** TBD

#### Hypothetical questions (Reverse HyDE)
For each chunk, have the LLM generate 2-3 questions this chunk could answer. Index the questions alongside the chunk. Retrieval matches query → question, which is semantically closer than query → raw chunk.

**Use it?** TBD

#### Hierarchical index (Small2Big)
Index at sentence level; at retrieval time, expand to the parent paragraph. Increases recall without widening context.

**Use it?** TBD

#### Knowledge Graph index
Index entity-relation triplets instead of (or in addition to) flat chunks. Heavier to build; better for structured domains.

**Use it?** TBD — defer unless your corpus is highly structured

### 2.2 Query Optimization (Pre-Retrieval)

Pick **one** as your primary strategy; combine later if needed.

| Technique | How it works | When to use |
|-----------|-------------|-------------|
| **Query rewriting (RRR)** | LLM rewrites the raw query to be more retrieval-friendly | Queries are conversational / ambiguous |
| **HyDE** | LLM generates a hypothetical answer; embed that answer; retrieve on it | Queries that are short or underspecified |
| **Step-back prompting** | Abstract the query one level up (e.g. "What is X?" → "What domain does X belong to?") | Queries that require background knowledge |
| **Multi-query expansion** | Generate N paraphrases; retrieve for each; union + deduplicate | Queries with multiple valid phrasings |
| **Sub-query decomposition** | Break complex query into sub-questions; answer each, synthesize | Multi-hop / complex reasoning queries |

**Chosen query strategy:** TBD

### 2.3 Post-Retrieval Processing

#### Reranking
Run after top-K retrieval to re-score and re-order chunks before passing to the LLM.

| Reranker | How | Cost |
|----------|-----|------|
| MMR (Maximum Marginal Relevance) | Rule-based: balance relevance + diversity | Free, deterministic |
| Cross-encoder (BGE-reranker-base) | Small BERT model scores each (query, chunk) pair | ~200MB model, local |
| Cohere Rerank | API call | Cloud, paid |
| LLM-based | Ask the chat LLM to score/sort chunks | Slow, expensive per query |

**Chosen reranker:** TBD

#### Context compression
Reduce the number of tokens sent to the LLM by filtering/compressing retrieved chunks.

| Method | How | Notes |
|--------|-----|-------|
| LLMLingua | Small LM removes low-info tokens | Good compression, local |
| LLM self-critique | LLM reads chunks, drops irrelevant ones before answering | Simple; costs an extra LLM call |
| RECOMP | Contrastive learning to extract relevant sentences | Best quality; needs fine-tuning |

**Chosen compression:** TBD

---

## Phase 3 — Modular RAG (The Chosen Paradigm)

**Goal:** compose the pipeline from swappable, independently-testable modules. Add one module at a time; measure impact on eval metrics.

### 3.1 Core module map

```
               ┌─────────┐
Query ─────────► Router  ├──────────────────────────────────┐
               └────┬────┘                                  │
                    │ route to source(s)                     │
          ┌─────────▼─────────┐                             │
          │   Search module   │  ← sparse + dense + web     │
          └─────────┬─────────┘                             │
                    │ retrieved docs                         │
          ┌─────────▼─────────┐                             │
          │  Memory module    │  ← conversation history     │
          └─────────┬─────────┘                             │
                    │ augmented context                      │
          ┌─────────▼─────────┐                             │
          │  Predict module   │  ← synthetic gap-fill       │
          └─────────┬─────────┘                             │
                    │                                       │
          ┌─────────▼─────────┐     ┌────────────────────-─┘
          │    Generator      │◄────┤  Task Adapter module
          └───────────────────┘     └──────────────────────
```

### 3.2 Module build order

Build in this order — each one is independently testable:

| Order | Module | What it does | Add when |
|-------|--------|-------------|----------|
| 1 | **Search module** | Wraps retrieval; handles multi-source, hybrid search | Always — this is the core |
| 2 | **Reranker module** | Sits between search and generator | Retrieval quality plateaus |
| 3 | **Memory module** | Stores conversation turns; injects relevant history | You add multi-turn conversation |
| 4 | **Router module** | Directs query to right index / source | You have >1 data source |
| 5 | **Predict module** | LLM fills corpus gaps with generated context | You hit "I don't know" too often on real queries |
| 6 | **Task Adapter** | Per-task prompt/output customization | You have multiple downstream tasks |

### 3.3 Augmentation patterns (retrieval orchestration)

These are the Modular RAG "patterns" — ways the modules sequence.

#### Iterative retrieval (ITER-RETGEN)
```
Retrieve → Generate → Retrieve again (on generated text) → Generate final
```
Use when: first retrieval misses context that only becomes clear after a partial answer.

**Use it?** TBD

#### Recursive retrieval
```
Query → sub-questions (CoT) → retrieve per sub-question → synthesize
```
IRCoT: chain-of-thought steps guide each retrieval.  
ToC: clarification trees for ambiguous queries.

**Use it?** TBD

#### Adaptive retrieval (decide whether to retrieve at all)

| Method | How | Notes |
|--------|-----|-------|
| **FLARE** | Retrieve only when next-token probability < threshold | No retrieval on high-confidence spans |
| **Self-RAG** | LLM emits reflection tokens (`[Retrieve]`, `[Critique]`) to self-direct | Needs a fine-tuned model or prompt engineering |
| **Agent-based** | LLM uses retrieval as a tool via tool-calling | Natural fit for `ds4-agent` later |

**Chosen adaptive pattern:** TBD

### 3.4 Hybrid retrieval

Combine sparse (keyword) + dense (semantic) retrieval for complementary coverage.

| Component | Library | Notes |
|-----------|---------|-------|
| Sparse (BM25) | `rank_bm25` | Classic keyword matching; zero embedding cost |
| Dense (vector) | Your vector store from Phase 1 | Semantic similarity |
| Fusion | **Reciprocal Rank Fusion (RRF)** | Score-agnostic; just uses rank positions |
| RAG-Fusion | Multi-query + RRF | Combines query expansion + hybrid scoring |

**Use hybrid retrieval?** TBD  
**Fusion method:** TBD

### 3.5 Routing

| Type | How | When to add |
|------|-----|-------------|
| Metadata router | Keyword extraction → filter by metadata field | You have tagged data sources |
| Semantic router | Embed query, pick source by similarity to source descriptor | You have heterogeneous sources |
| Hybrid | Both | Complex multi-source setups |

**Add routing?** TBD

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

**Retrieval granularity options:**

| Granularity | Best for |
|-------------|---------|
| Token | Precise span extraction |
| Sentence | Short factual answers |
| Chunk (default) | General QA |
| Document | Broad topic queries |
| Entity / triplet | Knowledge graph retrieval |
| Sub-graph | Multi-hop reasoning over KG |

**Chosen granularity:** TBD (start with chunk)

---

## Phase 5 — Embedding Fine-tuning (Defer)

Skip until retrieval quality plateaus on off-the-shelf embeddings. Revisit with a new ADR.

When you do get here, options:
- **LSR** (LM-supervised Retriever): use LLM feedback to supervise retriever training
- **LLM-Embedder**: dual-signal (hard labels + soft rewards)
- **REPLUG**: KL divergence between retriever distribution and generator preference
- **PROMPTAGATOR**: few-shot query generation for domain adaptation
- **Adapter-based** (UPRISE, AAR, PRCA): lightweight adapters on top of frozen embedder

---

## Phase 6 — Evaluation Harness

Build this in parallel with Phase 1. Don't skip it — without numbers you can't tell if Phase 2/3 helped.

### 6.1 Retrieval metrics

| Metric | What it measures |
|--------|----------------|
| Hit Rate | Is the answer chunk in top-K? |
| MRR (Mean Reciprocal Rank) | How high is the first correct chunk? |
| NDCG | Ranked quality of all top-K results |

### 6.2 Generation metrics

| Metric | What it measures |
|--------|----------------|
| Faithfulness | Does the answer contradict the retrieved context? |
| Context relevance | Are retrieved chunks actually relevant to the query? |
| Answer relevance | Does the answer address the question asked? |

### 6.3 Robustness tests (mandatory before calling any phase "done")

| Test | What you're checking |
|------|---------------------|
| Noise robustness | Inject question-related-but-wrong docs; does answer degrade? |
| Negative rejection | Remove answer from corpus; does the model say "I don't know"? |
| Information integration | Answer requires merging facts from 2+ docs; does it work? |
| Counterfactual robustness | Inject a doc with a deliberate factual error; does model trust it? |

### 6.4 Eval frameworks

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

Optional later milestone: replace Python orchestration with `ds4-agent` tool-calling, where retrieval is a native tool. Document that decision in ADR-0002.

---

## Decision Log

Fill this in as you make each choice:

| Phase | Decision | Chosen | Date |
|-------|----------|--------|------|
| 0.2 | Chat model | — | |
| 0.3 | Embedding model | — | |
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
