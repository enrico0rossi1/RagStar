# Naive RAG — Detailed Architecture & Decision Guide

**Status:** Not started  
**Gate:** Complete eval harness and record baseline scores before moving to [advanced-rag.md](./advanced-rag.md)  
**Progress:** [memory.md](./memory.md)

---

## What Naive RAG Is (and Isn't)

Naive RAG is the simplest possible implementation of Retrieval-Augmented Generation. It answers queries by:
1. Finding relevant text chunks from a corpus
2. Pasting them into a prompt
3. Asking the LLM to answer based on that context

**What it does NOT do** (these come in Advanced/Modular RAG):
- No query rewriting or expansion
- No reranking of retrieved chunks
- No iterative or adaptive retrieval
- No compression of retrieved context
- No routing between data sources
- No conversation memory

This is the baseline. Everything in later phases is measured against it.

---

## Pipeline Diagram

```
INDEXING (offline, done once):
  Documents
      │
      ▼
  [Loader] ──► raw text
      │
      ▼
  [Chunker] ──► text chunks [ ][ ][ ][ ]
      │
      ▼
  [Embedding model] ──► vectors [■][■][■][■]
      │
      ▼
  [Vector Store] ──► persisted index


QUERYING (online, per request):
  User query
      │
      ▼
  [Embedding model] ──► query vector
      │
      ▼
  [Vector Store] ──► similarity search ──► top-K chunks
      │
      ▼
  [Prompt assembler] ──► system prompt + context + query
      │
      ▼
  [LLM (Ollama)] ──► answer
```

---

## Component 1 — Document Loading & Preprocessing

### What it does
Reads source files from disk and produces clean raw text. This is boring but consequential — garbage in, garbage out applies at the loading stage.

### Supported formats

| Format | Library | Notes |
|--------|---------|-------|
| Plain text (`.txt`) | built-in `open()` | Zero dependencies |
| Markdown (`.md`) | built-in `open()` or `mistune` | Strip markdown syntax or keep it |
| PDF | `pypdf` or `pdfplumber` | `pypdf` for simple text; `pdfplumber` if you have tables/columns |
| HTML | `beautifulsoup4` | Strip tags, extract body text |
| DOCX | `python-docx` | If you have Word documents |

**Recommendation for start:** plain text + markdown only. Add PDF when you have real PDFs that matter.

### Preprocessing steps
Apply in this order:

1. **Encoding normalization** — decode to UTF-8, strip null bytes
2. **Whitespace normalization** — collapse multiple blank lines to one, strip trailing spaces
3. **Header/footer removal** — remove page numbers, repeated headers (especially from PDFs)
4. **Deduplication** — exact-match dedup before chunking; avoids duplicate chunks in the index

### Decision
- **Which formats to support first:** `.txt`, `.pdf`, `.md` (2026-08-18)
- **PDF library:** `pdfplumber` (2026-08-19) — better multi-column/table layout handling than `pypdf`
- **Strip markdown syntax or keep it:** TBD (keeping it is fine; embeddings handle it)

---

## Component 2 — Chunking

### Why chunking matters
The chunk is the retrieval unit. Too large: retrieved chunks contain irrelevant filler that confuses the LLM. Too small: retrieved chunks lack enough context to be useful. There is no universal right answer — it depends on your documents and your queries.

### Chunking strategies

#### Option A: Fixed-size token chunking
Split every N tokens, no regard for sentence or paragraph boundaries.

```
"The quick brown fox jumps over the lazy dog. The fox..." 
→ chunk 1: "The quick brown fox jumps over"  (N=6 tokens)
→ chunk 2: "the lazy dog. The fox..."
```

| Chunk size | Use case | Risk |
|-----------|----------|------|
| 100–200 tokens | Short factual snippets | Loses context; many chunks |
| 256–512 tokens | General QA (most common starting point) | May split mid-sentence |
| 512–1024 tokens | Long-form documents | May include too much noise |

**Start with 512 tokens.** Adjust after seeing retrieval quality in eval.

**Pros:** trivial to implement, deterministic, predictable index size  
**Cons:** cuts mid-sentence; splits that land in the middle of a list/table are noisy

#### Option B: Recursive character split
Split on a priority list of separators: `["\n\n", "\n", ". ", " "]`. Try the biggest separator first; fall back to smaller ones if the chunk is still too large.

```python
separators = ["\n\n", "\n", ". ", " "]
```

**Pros:** respects paragraph and sentence boundaries; works well on prose  
**Cons:** slightly more complex; chunk sizes are variable (not always bad)

**This is the recommended default for most use cases.**

#### Option C: Sliding window with overlap
Fixed-size chunks with N tokens of overlap between consecutive chunks. Overlap prevents losing context at chunk boundaries.

```
chunk 1: tokens 0–511
chunk 2: tokens 462–973   (50-token overlap)
chunk 3: tokens 924–1435
```

| Overlap | Effect |
|---------|--------|
| 0% | No redundancy; clean boundary loss |
| 10% (50 tok on 512) | Mild safety net |
| 20% (100 tok on 512) | Good boundary coverage; 20% larger index |
| >30% | Index bloat without proportional gain |

**Pros:** reduces information loss at boundaries  
**Cons:** larger index; duplicate content in retrieved chunks (minor issue, manageable)

#### Option D: Sentence-based splitting
Split on sentence boundaries detected by a tokenizer (NLTK `sent_tokenize` or spaCy). Then group N sentences per chunk.

**Pros:** cleanest semantic units  
**Cons:** requires NLTK/spaCy; very short sentences produce tiny chunks; works poorly on bullet lists and code

#### Option E: Semantic segmentation
Embed each sentence; split when cosine similarity between adjacent sentences drops below a threshold (topic shift detection).

**Pros:** highest-quality semantic boundaries  
**Cons:** slowest; requires an embedding call per sentence at index time; threshold is a hyperparameter to tune

**Defer to Advanced RAG** — too expensive to tune at the naive stage.

### Decision
- **Chunking strategy:** Recursive character split (2026-08-19) — hand-rolled, no framework. See [src/chunker/chunker.py](../src/chunker/chunker.py). Interface `chunk_document(doc) -> list[Chunk]` kept stable so the implementation can be swapped later without touching the rest of the pipeline.
- **Chunk size:** 2000 chars ≈ 512 tokens (2026-08-19) — character-based approximation (~4 chars/token), no tokenizer wired up yet
- **Overlap:** 200 chars ≈ 50 tokens / 10% (2026-08-19)

---

## Component 3 — Embedding Model

### What it does
Converts text chunks (and later, queries) into dense vectors. The quality of these vectors is the single biggest factor in retrieval quality.

### Key properties to care about
- **Embedding dimensions:** higher = more expressive, more memory, slower search
- **Max input tokens:** chunks longer than this get silently truncated — chunk size must stay below this limit
- **MTEB score:** standard benchmark for retrieval quality (higher = better)

### Options

| Model | Run via | Dims | Max tokens | MTEB (retrieval) | Notes |
|-------|---------|------|-----------|-----------------|-------|
| `nomic-embed-text` | `ollama pull nomic-embed-text` | 768 | 8192 | ~62 | Best local option via Ollama; long context |
| `mxbai-embed-large` | `ollama pull mxbai-embed-large` | 1024 | 512 | ~65 | High quality; 512-token limit matters |
| `all-MiniLM-L6-v2` | `pip install sentence-transformers` | 384 | 256 | ~57 | Tiny and fast; great for CPU |
| `BGE-small-en-v1.5` | `pip install sentence-transformers` | 384 | 512 | ~62 | Fast, punches above its size |
| `BGE-base-en-v1.5` | `pip install sentence-transformers` | 768 | 512 | ~64 | Good balance; popular choice |
| `BGE-large-en-v1.5` | `pip install sentence-transformers` | 1024 | 512 | ~65 | Best of BGE family; ~1.3GB |

### How to call them

**Via Ollama (nomic, mxbai):**
```python
import openai
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.embeddings.create(model="nomic-embed-text", input="your text")
vector = response.data[0].embedding
```

**Via sentence-transformers (BGE, MiniLM):**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vector = model.encode("your text")
```

### Important constraint
**The embedding model used at index time must be the same model used at query time.** Mixing models breaks the vector space — retrieval will return garbage. Pick one and commit.

### Decision
- **Embedding model:** `nomic-embed-text` (2026-08-18, via Ollama, same openai client as chat — see [src/embedder/embedder.py](../src/embedder/embedder.py))

---

## Component 4 — Vector Store

### What it does
Persists the embeddings and metadata; executes approximate nearest-neighbor (ANN) search to find the top-K most similar chunks to a query vector.

### Options

#### LanceDB
```
pip install lancedb
```
- Embedded (no server process), columnar storage, Apache Arrow-based
- Supports metadata filtering natively
- Good Python API; active development
- Persists to a local directory
- **Best choice for local-first, no-infra setups**

```python
import lancedb
db = lancedb.connect("./vector_db")
table = db.create_table("chunks", data=[{"vector": vec, "text": chunk, "source": filename}])
results = table.search(query_vec).limit(5).to_list()
```

#### sqlite-vec
```
pip install sqlite-vec
```
- Vector search extension for SQLite
- Zero extra dependencies beyond SQLite (which is everywhere)
- Lower-level API; good if you want to store everything in one SQLite file (vectors + metadata + documents)
- Less active ecosystem than LanceDB

#### Chroma
```
pip install chromadb
```
- Embedded mode available (no server needed for local use)
- Very popular in the RAG/LangChain ecosystem — lots of examples online
- Slightly heavier than LanceDB; stores data in its own format
- Good choice if you want abundant community resources

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("chunks")
collection.add(embeddings=[vec], documents=[chunk], ids=[chunk_id])
results = collection.query(query_embeddings=[query_vec], n_results=5)
```

#### FAISS
```
pip install faiss-cpu
```
- Facebook's library for fast ANN search
- Pure in-memory by default — you must serialize/deserialize the index yourself for persistence
- Fastest for large-scale search, but overkill at learning-project scale
- No metadata storage — you'd need a separate store for the text and source

### Decision
- **Vector store:** LanceDB (2026-08-19) — embedded, no server, per ADR-0001's local-first recommendation. Table `chunks` in `vector_db/`, rebuilt from scratch on every `ingest.py` run (not incremental — fine at this corpus size). See [src/ingest.py](../src/ingest.py).

---

## Component 5 — Similarity Search & Retrieval

### Similarity metrics

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Cosine similarity** | dot(A,B) / (‖A‖ · ‖B‖) | **Default.** Scale-invariant; works regardless of vector magnitude |
| **Dot product** | dot(A,B) | Slightly faster; assumes vectors are normalized (length 1). BGE models are normalized by default |
| **L2 (Euclidean)** | ‖A - B‖ | Less common for text embeddings; measures geometric distance |

**Use cosine unless your embedding model explicitly normalizes vectors** (then dot product is equivalent and marginally faster).

### Top-K selection

K is how many chunks you retrieve and pass to the LLM.

| K | Effect |
|---|--------|
| 3 | Tight context; low noise; might miss the answer if retrieval is imperfect |
| 5 | Good starting point for most use cases |
| 10 | More recall; more noise; higher token cost per query |
| 20+ | Use only with context compression (Advanced RAG) |

**Start with K=5.** Tune after seeing eval results: if faithfulness is high but context relevance is low, increase K. If faithfulness is low (LLM gets confused by noise), decrease K.

### Threshold-based retrieval (alternative to fixed K)
Only return chunks above a minimum similarity score. Prevents returning irrelevant chunks when no good match exists.

**Tradeoff:** harder to tune than K; different embedding models have different score ranges. Easier to start with fixed K.

### Decision
- **Similarity metric:** Cosine, set explicitly (2026-08-19) — `table.search(q).metric("cosine")`. LanceDB's default is squared-L2, not cosine; confirmed `nomic-embed-text` vectors are unit-normalized (norm ≈ 1.0) so L2 and cosine happen to rank identically today (L2² = 2 × cosine_distance), but relying on that silently would break if the embedding model ever changes. Set explicitly so distance values are also directly interpretable (0 = identical, 2 = opposite).
- **Top-K:** 5 (2026-08-19)
- **Threshold-based?** TBD (recommended: no, for now)

---

## Component 6 — Prompt Assembly

### What it does
Takes the retrieved chunks and the user query, formats them into a prompt the LLM will understand, and sends it for generation.

### System prompt options

#### Option A: Strict grounding (recommended for RAG)
```
You are a helpful assistant. Answer the user's question using ONLY the information 
in the provided context. If the answer is not in the context, say exactly: 
"I don't know based on the provided documents."
Do not use prior knowledge.
```
**Best for:** factual QA where hallucination is the main risk. Forces the model to stay in context.

#### Option B: Permissive (context + prior knowledge)
```
You are a helpful assistant. Use the provided context to answer the question. 
You may also use your general knowledge if the context is insufficient.
```
**Best for:** when your corpus has gaps and you'd rather get an imperfect answer than "I don't know."  
**Risk:** harder to evaluate faithfulness; harder to catch hallucinations.

#### Option C: Chain-of-thought grounding
```
You are a helpful assistant. Read the context carefully, then reason step-by-step 
before giving your final answer. Base your answer only on the provided context.
Think through: what does the context say? What is the question asking? 
What is the most accurate answer given the context?
```
**Best for:** complex multi-part questions; slow but more accurate on hard queries.

### Context ordering
How you order the retrieved chunks in the prompt affects LLM attention:

| Order | Effect |
|-------|--------|
| Best-first (highest similarity first) | LLM reads the most relevant chunk first; natural |
| Worst-first, best-last | "Lost in the middle" research suggests LLMs attend better to start/end positions |
| Random | Baseline; avoids ordering bias |

**Start with best-first.** Experiment with best-last if you see the LLM ignoring the top chunk.

### Context window management
If K chunks × chunk_size > model context window − system_prompt − response:

1. **Truncate from the bottom** (drop lowest-scoring chunks first) — simplest
2. **Summarize overflow chunks** (Advanced RAG — defer for now)

For a 7B-8B model with 8K context: 5 × 512-token chunks = ~2560 tokens. With a 512-token response budget, you have ~5000 tokens left for the system prompt. You're fine at K=5, chunk_size=512.

### Prompt template

```
System: {system_prompt}

Context:
---
{chunk_1_text}
[Source: {chunk_1_source}, page {chunk_1_page}]
---
{chunk_2_text}
[Source: {chunk_2_source}, page {chunk_2_page}]
---
... (up to K chunks)

Question: {query}
```

### Decision
- **System prompt style:** Strict grounding (2026-08-19) — see [src/prompt/prompt.py](../src/prompt/prompt.py)
- **Context ordering:** Best-first (2026-08-19) — chunks used in the order `retrieve()` returns them (similarity-ranked)
- **Include source attribution in prompt?** Yes (2026-08-19) — `[Source: filename]` under each chunk, free since the metadata is already there
- **Context window budget:** No truncation logic — 5 chunks × ~512 tok ≈ 2560 tok, under 10% of qwen2.5:7b's 32k window. Revisit if K or chunk size grows enough to matter.

---

## Component 7 — Generation (LLM)

### Model options (Ollama)

| Model | Pull command | VRAM | Strengths | Context window |
|-------|-------------|------|-----------|---------------|
| `phi3:mini` | `ollama pull phi3:mini` | ~2 GB | CPU-friendly; fast | 4K |
| `mistral:7b` | `ollama pull mistral:7b` | ~4 GB | Balanced; fast on GPU | 8K |
| `llama3:8b` | `ollama pull llama3:8b` | ~5 GB | Strong instruction following | 8K |
| `qwen2.5:7b` | `ollama pull qwen2.5:7b` | ~5 GB | Strong on structured tasks | 128K |
| `gemma2:9b` | `ollama pull gemma2:9b` | ~6 GB | Google; good reasoning | 8K |

**The RAG pipeline is model-agnostic.** Try a small model first, observe quality, scale up if needed. A good retrieval strategy matters more than model size for factual QA.

### Temperature
| Value | Effect |
|-------|--------|
| 0.0 | Deterministic; same query always returns same answer. Best for factual RAG. |
| 0.1–0.3 | Slight variation; more natural tone |
| 0.7–1.0 | Creative; not appropriate for grounded QA |

**Use temperature=0.0 for factual RAG.** Increases reproducibility and makes eval results comparable.

### Max tokens for response
- Set a ceiling (e.g. 512 tokens) to prevent the model from rambling
- For summarization tasks, increase this

### OpenAI SDK call (works with Ollama unchanged)
```python
import os, openai

client = openai.OpenAI(
    base_url=os.environ["BASE_URL"],   # http://localhost:11434/v1
    api_key="ollama"
)

response = client.chat.completions.create(
    model=os.environ["CHAT_MODEL"],
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ],
    temperature=0.0,
    max_tokens=512
)
answer = response.choices[0].message.content
```

### Decision
- **Chat model:** `qwen2.5:7b` (2026-08-18, native Windows Ollama)
- **Temperature:** `0.0` (2026-08-19) — reproducible eval runs. Does NOT guarantee correctness on its own; groundedness comes from retrieval quality + the strict-grounding prompt, this just removes sampling randomness as a separate variable.
- **Max response tokens:** `512` (2026-08-19)

---

## Component 8 — Evaluation Harness

### Why this is mandatory before Phase 2
Without baseline numbers, you cannot tell if anything in Advanced RAG actually helped. "It seems better" is not a metric. Build the eval harness in parallel with the pipeline — run it after Phase 1 is working.

### Step 1: Build a labeled Q&A set
Create 20–50 question-answer pairs from your own corpus. Each pair needs:
- `question`: a natural query
- `ground_truth_answer`: the correct answer
- `ground_truth_chunks`: which chunk(s) contain the answer (optional but enables retrieval metrics)

This doesn't need to be large — 20 good pairs reveal most failure modes.

### Step 2: Run retrieval + generation for each question

For each question in the set:
1. Retrieve top-K chunks
2. Generate an answer
3. Record: `question`, `retrieved_chunks`, `generated_answer`, `ground_truth_answer`

### Step 3: Score with a framework

#### RAGAS (recommended starting point)
```
pip install ragas
```
RAGAS uses your local LLM to score the outputs — no external API key needed.

**Three core metrics:**

| Metric | What it measures | How RAGAS scores it |
|--------|----------------|---------------------|
| **Context relevance** | Are the retrieved chunks relevant to the question? | LLM judges what fraction of each chunk is relevant |
| **Faithfulness** | Does the generated answer contradict the retrieved context? | LLM checks each answer claim against context |
| **Answer relevance** | Does the generated answer actually address the question? | LLM checks directness and completeness |

All three produce a score in [0, 1]. Higher is better.

**Typical baseline scores on a well-functioning naive RAG:**
- Context relevance: 0.5–0.7 (low is the main naive RAG weakness)
- Faithfulness: 0.7–0.9 (strict grounding prompt helps)
- Answer relevance: 0.7–0.9

#### TruLens (alternative)
```
pip install trulens-eval
```
Similar metrics to RAGAS; slightly different scoring methodology; good dashboard UI.

#### RGB (robustness-focused)
Manual test framework from the survey paper. Not a library — you construct the test cases yourself. Covers four abilities the other frameworks don't:

| Test | How to build it |
|------|----------------|
| **Noise robustness** | Add 3–5 question-related-but-wrong documents to the corpus; check if answer quality degrades |
| **Negative rejection** | Remove the correct answer chunk; check if model says "I don't know" |
| **Information integration** | Create questions that require combining facts from 2+ chunks |
| **Counterfactual robustness** | Add a document with a deliberate factual error; check if model repeats it |

### Step 3a: Performance metrics (speed, not quality)

RAGAS/RGB tell you if answers are *good*; they say nothing about whether the pipeline is *fast enough to use*. Track both from the same eval run, in `eval.py`:

| Metric | How to measure |
|--------|----------------|
| End-to-end latency per query | `time.perf_counter()` around retrieval + generation |
| Generation speed (tokens/sec) | `response.usage.completion_tokens / generation_time` |

Record avg/p50/p95 across the Q&A set. A pipeline that's accurate but too slow to use, or fast but wrong, are both failures — report them together. This becomes the baseline you compare against when switching Ollama → `ds4-server` (Phase 7), and when deciding if an Advanced RAG technique's quality gain (e.g. reranking) is worth its added latency.

### Retrieval-only metrics (if you have ground_truth_chunks)

| Metric | What it measures |
|--------|----------------|
| **Hit Rate** | Is any ground-truth chunk in the top-K results? |
| **MRR** | Mean Reciprocal Rank — how high is the first correct chunk? |
| **NDCG** | Normalized Discounted Cumulative Gain — weighted ranking quality |

### What to record
After running the eval, record these numbers in [memory.md](./memory.md):

```
Naive RAG baseline (YYYY-MM-DD):
  Context relevance:  X.XX
  Faithfulness:       X.XX
  Answer relevance:   X.XX
  Hit Rate @ K=5:     X.XX  (if measured)
```

These numbers are your Phase 2 target to beat.

### Decision
- **Eval framework:** Hand-rolled LLM-as-judge, not RAGAS (2026-08-19) — verified, not assumed: `ragas` 0.4.3 declares `langchain-community` with no version constraint, so it always pulls the latest (0.4.2), which dropped a module ragas still hard-imports. Confirmed as a known, open upstream bug ([vibrantlabsai/ragas#2753](https://github.com/vibrantlabsai/ragas/issues/2753), 3 pending fix PRs). Forking considered and rejected — per ADR-0001's own stance against forking a fast-moving upstream for a fix already in progress there. Replicated the three core metrics (faithfulness, context relevance, answer relevance) directly with our own `generate()`. See [src/eval.py](../src/eval.py). Known limitation: self-judging (same local model grades its own answers) — treat scores as directional, not absolute. Revisit real RAGAS once the upstream fix merges.
- **Q&A set size:** 23 pairs (2026-08-19) — 20 answerable (hand-written from the real corpus content) + 3 out-of-scope for negative rejection. See [eval/qa_pairs.json](../eval/qa_pairs.json).

**Baseline results (2026-08-19):**

| Metric | Score |
|---|---|
| Faithfulness | 0.85 |
| Answer relevance | 0.86 |
| Context relevance | 0.4975 |
| Rejection accuracy | 1.0 |
| Avg latency/query | 6.87s (p95 10.72s) |
| Generation speed | 4.74 tok/s |

Context relevance landed almost exactly on this doc's own prediction of naive RAG's weak point (0.5-0.7) — that's the number Advanced RAG's reranking should move.

---

## Component 9 — Python Project Structure

Keep it flat. No frameworks (LangChain, LlamaIndex) yet — you're learning the concepts, not the framework API.

```
RagStar/
├── progress/               ← this directory
│   ├── ROADMAP.md
│   ├── naive-rag.md
│   ├── advanced-rag.md
│   ├── modular-rag.md
│   └── memory.md
├── ADR-0001-rag-on-dwarfstar.md
├── ROADMAP.md              ← root copy for quick access
│
├── src/
│   ├── ingest.py           ← load docs → chunk → embed → store
│   ├── query.py            ← query → embed → search → prompt → generate
│   └── eval.py             ← run labeled Q&A set → score with RAGAS
│
├── data/                    ← gitignored (not committed)
│   ├── knowledge/          ← the actual RAG corpus; only this gets loaded/indexed
│   └── other/              ← anything else (scratch downloads, notes) — never read by the loader
│
├── vector_db/              ← LanceDB or Chroma persisted index (gitignore this)
│
├── eval/
│   └── qa_pairs.json       ← your labeled Q&A set
│
├── .env                    ← BASE_URL, EMBED_MODEL, CHAT_MODEL (gitignore)
└── requirements.txt
```

### Environment variables (`.env`)
```
BASE_URL=http://localhost:11434/v1
CHAT_MODEL=mistral:7b
EMBED_MODEL=nomic-embed-text
```

These are the only things that change when you switch from Ollama to ds4-server.

### Entry point design

**`ingest.py`** — run once per corpus update:
```
python ingest.py --input data/documents/
```
Loads, chunks, embeds, stores. Idempotent (re-running should not duplicate chunks).

**`query.py`** — interactive:
```
python query.py "What is the capital of France?"
```
Retrieves, assembles prompt, generates, prints answer + source chunks.

**`eval.py`** — run after any pipeline change:
```
python eval.py --qa eval/qa_pairs.json
```
Runs all Q&A pairs, prints RAGAS scores, saves results to `eval/results_TIMESTAMP.json`.

---

## Decision Summary Table

| Component | Decision | Status |
|-----------|----------|--------|
| Document formats to support | .txt, .pdf, .md | Done |
| Chunking strategy | Recursive character split | Done |
| Chunk size (tokens) | 2000 chars (~512 tok) | Done |
| Chunk overlap (tokens) | 200 chars (~50 tok) | Done |
| Embedding model | nomic-embed-text | Done |
| Vector store | LanceDB | Done |
| Similarity metric | Cosine, explicit | Done |
| Top-K | 5 | Done |
| System prompt style | Strict grounding | Done |
| Context ordering | Best-first | Done |
| Chat model | qwen2.5:7b | Done |
| Temperature | 0.0 | Done |
| Max response tokens | 512 | Done |
| Eval framework | Hand-rolled LLM-as-judge | Done |
| Q&A set size | 23 (20 answerable + 3 out-of-scope) | Done |

---

## Phase Gate — ✅ SATISFIED (2026-08-19)

Before moving to [advanced-rag.md](./advanced-rag.md):

- [x] `ingest.py` runs end-to-end without errors
- [x] `query.py` returns a grounded answer for a test question
- [x] `eval.py` produces scores for all 3 quality metrics (hand-rolled, not RAGAS — see decision above)
- [x] Negative rejection tested — 3/3 out-of-scope questions correctly refused (rejection_accuracy 1.0); noise robustness / info integration / counterfactual robustness not yet separately tested
- [x] Baseline scores recorded in [memory.md](./memory.md)
- [x] All decisions in the table above filled in
