# DwarfStar-RAG — Reference Notes
*Compiled for later project use. Two parts: (1) RAG architecture literature, (2) DwarfStar / ds4 technical architecture. Part 3 sketches how they connect.*

---

## Part 1 — RAG Architecture: Papers & Concepts

### 1.1 The three RAG paradigms (mental model)

Most of the literature frames RAG's evolution in three stages — useful as a checklist when designing a pipeline:

| Stage | What it means | Weakness it fixes |
|---|---|---|
| **Naive RAG** | Chunk documents → embed → store in a vector DB → embed the query → top-k similarity search → stuff chunks into the prompt → generate | Baseline; established but has low precision/recall, no query understanding, no reranking |
| **Advanced RAG** | Adds pre-retrieval steps (query rewriting, HyDE, query routing) and post-retrieval steps (reranking, compression, filtering) around the naive core | Retrieval mismatch, irrelevant chunks, "lost in the middle" |
| **Modular RAG** | Treats retrieval, augmentation, and generation as swappable modules that can be reordered, run iteratively, or looped (retrieve → generate → retrieve again) | Rigid fixed pipeline; can't handle multi-hop or adaptive retrieval |

Source: Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv:2312.10997 — this taxonomy is the most commonly cited framing in later papers.

### 1.2 Foundational papers (pre-LLM-era RAG, still the conceptual base)

| Paper | arXiv | Contribution |
|---|---|---|
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020) | [2005.11401](https://arxiv.org/abs/2005.11401) | Coined "RAG." Combines a dense retriever (query encoder + document index, MIPS search) with a seq2seq generator; retriever and generator are trained jointly end-to-end. This is the architecture the term "RAG" originally names. |
| **REALM: Retrieval-Augmented Language Model Pre-Training** (Guu et al., 2020) | [2002.08909](https://arxiv.org/abs/2002.08909) | Pretrains the retriever *jointly* with the language model rather than bolting retrieval on at inference time. Established that retrieval can be a first-class part of pretraining, not just an inference-time trick. |
| **Dense Passage Retrieval for Open-Domain QA** (Karpukhin et al., 2020) | 2004.04906 | DPR — the dual-encoder (query encoder + passage encoder, dot-product similarity) that most dense retrievers are still descended from. Showed dense retrieval beats BM25/TF-IDF for open-domain QA. |
| **Fusion-in-Decoder (FiD)** (Izacard & Grave, 2020) | [2007.01282](https://arxiv.org/abs/2007.01282) | Encodes each retrieved passage *independently*, then fuses all of them in the decoder's cross-attention. Lets a system scale to dozens of retrieved passages without quadratic attention cost — the architecture behind why "retrieve many, let the decoder pick" works. |

### 1.3 Survey papers (start here for a full literature map)

| Paper | arXiv | Focus |
|---|---|---|
| **Retrieval-Augmented Generation for Large Language Models: A Survey** (Gao et al., 2024) | [2312.10997](https://arxiv.org/abs/2312.10997) | The most-cited modern RAG survey. Naive/Advanced/Modular taxonomy; breaks the architecture into retrieval, generation, and augmentation techniques. Best single starting point. |
| **A Survey on Retrieval-Augmented Text Generation for LLMs** (Huang & Huang, 2024) | [2404.10981](https://arxiv.org/abs/2404.10981) | Organizes RAG into four pipeline stages — pre-retrieval, retrieval, post-retrieval, generation — with an evaluation-methods section. Good complement to Gao et al. for pipeline-stage thinking. |
| **RAG and RAU: A Survey on Retrieval-Augmented Language Models in NLP** (Hu & Lu, 2024) | [2404.19543](https://arxiv.org/abs/2404.19543) | Covers both generation (RAG) and understanding (RAU) tasks; useful if the project ever needs classification/extraction, not just Q&A. |
| **Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers** (2025/26) | [2506.00054](https://arxiv.org/abs/2506.00054) | Newer survey, more focused on robustness/failure modes — useful for a "what can go wrong" checklist. |

### 1.4 Advanced techniques worth knowing by name

| Technique / Paper | arXiv | What it does |
|---|---|---|
| **HyDE** — Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022) | [2212.10496](https://arxiv.org/abs/2212.10496) | Instead of embedding the raw query, first asks an LLM to write a *hypothetical* answer document, then embeds *that* for similarity search. Closes the vocabulary gap between short queries and long documents. Known weakness: hurts precise numerical/entity queries (see benchmark note below). |
| **RAPTOR** — Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024) | [2401.18059](https://arxiv.org/abs/2401.18059) | Recursively clusters and summarizes chunks into a tree, so retrieval can pull a fine-grained passage *or* a high-level summary depending on the question. Strong on questions that need holistic/whole-document understanding, not just one paragraph. |
| **Self-RAG** (Asai et al., 2023) | [2310.11511](https://arxiv.org/abs/2310.11511) | Trains the model to emit special "reflection tokens" that decide *whether* to retrieve, judge passage relevance, and critique its own output for support/completeness — retrieval becomes a learned, on-demand decision instead of a fixed step. |
| **CRAG — Corrective Retrieval Augmented Generation** (Yan et al., 2024) | [2401.15884](https://arxiv.org/abs/2401.15884) | Adds a lightweight retrieval evaluator that grades retrieved documents as correct / ambiguous / incorrect, and falls back to web search when local retrieval is bad. Directly relevant to a "grounded, don't hallucinate" objective. |
| **GraphRAG** (Edge et al., Microsoft, 2024) | [2404.16130](https://arxiv.org/abs/2404.16130) | Builds a knowledge graph + community summaries from the corpus, then retrieves/answers over that structure. Strong for multi-hop and "summarize across the whole corpus" queries that plain chunk-retrieval can't answer. |
| **LightRAG** (Guo et al., 2024) | [2410.05779](https://arxiv.org/abs/2410.05779) | A lighter-weight graph-based RAG aimed at lower indexing cost than GraphRAG — relevant if the project ever wants graph structure without full GraphRAG overhead. |
| **HippoRAG** (Jiménez Gutiérrez et al., 2024) | [2405.14831](https://arxiv.org/abs/2405.14831) | Neurobiologically-inspired long-term memory scheme for LLMs; frames RAG as continual, non-parametric memory rather than one-shot document QA. |

**Practical benchmark note:** a 2026 text-and-table benchmark (T²-RAGBench, arXiv:2604.01733) found that a two-stage hybrid-retrieval + neural-reranking pipeline beat every single-stage method tested, that plain BM25 can beat dense retrieval on some domains, and that HyDE/query-expansion gave little or no benefit for precise numerical queries. Worth remembering before assuming any one "advanced" technique is automatically better — test on the actual corpus.

### 1.5 Generic RAG pipeline anatomy (for implementation reference)

**A. Indexing (offline, done once per corpus update)**
1. Load documents (PDF/HTML/markdown/DB rows).
2. Chunk — fixed-size, sentence-window, semantic/recursive, or parent-child chunking (small chunk for retrieval precision, larger parent chunk sent to the model).
3. Embed each chunk with a dense encoder (and optionally compute sparse/BM25 representations for hybrid search).
4. Store in a vector index (HNSW, IVF, DiskANN-style, or a managed vector DB) plus metadata (source, position, timestamp).

**B. Retrieval (online, per query)**
1. Optionally rewrite/expand the query (HyDE, multi-query, decomposition for multi-hop).
2. Embed the query; run dense search, sparse/BM25 search, or both (hybrid, merged via Reciprocal Rank Fusion).
3. Rerank the candidate set with a cross-encoder or LLM-based reranker (precision boost — bi-encoder retrieval is high-recall/low-precision, reranking fixes precision).
4. Optionally filter/compress context (drop irrelevant passages, summarize long ones) to fight "lost in the middle" degradation on long contexts.

**C. Augmentation + Generation**
1. Build the final prompt: system instructions + (deduplicated, ordered) retrieved context + user question.
2. Generate with the LLM, ideally instructed to cite/ground claims in the provided passages.
3. Optionally self-check (Self-RAG-style critique, or a corrective-retrieval fallback if confidence is low).

**D. Evaluation**
- Retrieval metrics: Recall@k, MRR, nDCG.
- Generation metrics: faithfulness/groundedness (does the answer only use retrieved facts?), answer relevance, context precision/recall — the RAGAs framework is the most commonly used off-the-shelf toolkit for these.

---

## Part 2 — DwarfStar (ds4) Technical Architecture

Source: [github.com/antirez/ds4](https://github.com/antirez/ds4) (README, fetched August 2026 — beta software, changes fast; re-check before relying on exact flag names). Author: Salvatore Sanfilippo (antirez), creator of Redis. MIT licensed.

### 2.1 What it is

A native inference engine (mostly C, Objective-C for Metal, CUDA for NVIDIA, plus a ROCm branch) built specifically for **DeepSeek V4 Flash** (with PRO and GLM 5.2 support), not a general-purpose GGUF runner. The project's stance is deliberately narrow: one model family at a time, validated against official logits, rather than broad model coverage like Ollama/llama.cpp.

Four things are meant to work together out of the box: (A) the inference engine + HTTP API, (B) GGUF files specifically tuned for this engine, (C) coding-agent-level testing/validation, (D) purpose-built agents.

### 2.2 Backends & hardware targets

| Backend | Target hardware | Notes |
|---|---|---|
| **Metal** | macOS, primary target, from 96GB RAM MacBooks up | Fastest path when the model fits in GPU-addressable memory |
| **CUDA** | NVIDIA, special care for DGX Spark (GB10 chip, 128GB unified memory) | `make cuda-spark` vs `make cuda-generic` |
| **ROCm** | AMD Strix Halo (e.g., Framework Desktop) | Maintained on a separate branch, rebased by the community |
| **CPU** | Reference/debug only | Explicitly *not* a production target; macOS CPU path can even crash the kernel due to an OS-level VM bug |

### 2.3 Quantization & "bigger model, smaller box"

- Uses an **asymmetric 2-bit quantization**: only the routed MoE experts are quantized (up/gate at IQ2_XXS, down at Q2_K) since they dominate model size; shared experts, projections, and routing stay at higher precision to protect quality.
- `download_model.sh` fetches pre-built GGUFs: `q2-imatrix` (96/128GB machines), `q2-q4-imatrix`, `q4-imatrix` (≥256GB), `pro-q2-imatrix` (512GB), plus split files for distributed PRO Q4 runs.
- **SSD streaming** (Metal only): when the full model doesn't fit in RAM, non-routed weights stay resident while routed MoE experts are cached in RAM and streamed from the GGUF on cache miss. Turns "does it fit in RAM" from a hard yes/no into a continuous speed/RAM tradeoff. Cache budget is auto-sized (80% of Metal's recommended working set, minus non-routed weights) or can be set manually (`--ssd-streaming-cache-experts`).

### 2.4 Disk-first KV cache — the core mechanism

This is the piece most relevant to a RAG project.

- **Why it exists:** chat/completion APIs are stateless — clients resend the whole conversation every request. `ds4-server` keeps exactly **one live session in RAM**; when a new, unrelated session replaces it, the old one is lost *unless* it was written to the on-disk cache. Disk cache is the resume mechanism across session switches and server restarts; RAM cache only covers the currently active session.
- **Cache key:** SHA1 hash of the **exact rendered byte-prefix** of the prompt (not the token IDs, the rendered text bytes). Files are stored as `<sha1>.kv` under `--kv-disk-dir`.
- **What's stored:** the literal cached prefix text plus the full DS4 session payload — checkpoint token IDs, next-token logits (so a resumed session can sample immediately without one extra decode step), and per-layer KV/compressor/indexer state. Written with plain read/write I/O (not mmap) to avoid extra VM mappings on a process that already maps the whole model.
- **Reuse logic:** a fast exact-prefix check first, then a byte-comparison fallback (handles cases where a generated token's decoded text later comes back as multiple re-tokenized tokens from the client). Only the *new suffix* after the cached bytes gets tokenized/prefilled.
- **Save triggers:** `cold` (first long prompt stabilizes, before generation), `continued` (every ~10k tokens by default, at aligned chunk boundaries), `evict` (before an unrelated request replaces the live session), `shutdown` (clean server exit). Cold saves intentionally trim a small tail and align to a chunk boundary to avoid BPE retokenization mismatches.
- **Tool-call replay:** tool calls are also cached — an unguessable tool-call ID maps to the *exact* sampled tool-call text (DSML format), so a client's JSON tool-call history can be replayed byte-for-byte instead of being re-rendered slightly differently (which would break the prefix match). Canonical re-rendering from JSON is only a fallback path.
- **Relevant flags:** `--kv-disk-dir`, `--kv-disk-space-mb`, `--kv-cache-min-tokens`, `--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`, `--kv-cache-boundary-trim-tokens`, `--kv-cache-boundary-align-tokens`, `--kv-cache-reject-different-quant` (disables 2-bit/4-bit cross-reuse), `--tool-memory-max-ids`, `--disable-exact-dsml-tool-replay`.
- **Practical implication for RAG:** if the rendered context block (system prompt + retrieved chunks) is byte-identical across two requests, the second request skips prefill on that whole block. If the retrieved chunk set or its order changes even slightly, the hash changes and it's a full cache miss — this is a hard requirement, not a fuzzy "similar enough" match.

### 2.5 Server / API

Start with: `./ds4-server --ctx <N> --kv-disk-dir <path> --kv-disk-space-mb <N>`

**Endpoints:**
- `GET /v1/models`, `GET /v1/models/deepseek-v4-flash`, `GET /v1/models/deepseek-v4-pro` (aliases — both report whatever GGUF was actually loaded with `-m`)
- `POST /v1/chat/completions` — standard OpenAI-style params (`messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stream`, `tools`, `tool_choice`)
- `POST /v1/responses` — OpenAI Responses-style API, preferred by Codex CLI
- `POST /v1/completions`
- `POST /v1/messages` — Anthropic-compatible, used by Claude-Code-style clients (`system`, `messages`, `tools`, `tool_choice`, `max_tokens`, thinking controls)

All chat/Responses/Anthropic endpoints support SSE streaming, including live streaming of "thinking"/reasoning tokens separately from final text, and progressive tool-call argument streaming.

**Concurrency model (important limitation):** request parsing/sockets are multi-threaded, but inference itself is serialized through a single graph worker — **no batching of concurrent requests**. This is a single-session engine, not a multi-tenant server.

**Other server details:**
- `--cors` for browser clients; `--host 0.0.0.0` to expose beyond localhost.
- Default sampling: `temperature=1, top_p=1, min_p=0.05` (relative-probability filtering, not nucleus). Thinking mode uses fixed sampling and ignores client sampling params, mirroring DeepSeek's own API behavior.
- Tool-call syntax (DSML tags, JSON punctuation) is sampled at `temperature=0` for parseability, while argument *payload* text (file contents, string values) uses normal sampling — keeps structure reliable without flattening long generated text into repetition.
- `--power N` (default 100) throttles GPU usage percentage by inserting sleeps between layers/tokens — for heat/battery/fan-noise management, doesn't change output.

### 2.6 Native agent & sessions

`ds4-agent` runs inference in-process (no socket/API boundary) so the session *is* the on-disk KV cache — no separate "is my cache in sync" bookkeeping. Sessions live in `~/.ds4/kvcache`; `/save`, `/list`, `/switch <sha>`, `/del <sha>`, `/strip <sha>` (keep transcript, drop the heavy KV payload) manage them. Alpha-quality per the project's own status notes.

### 2.7 Distributed inference

Splits transformer layers across machines (`--layers N:M` ranges) with one `coordinator` and one or more `worker` roles, activations sent over plain TCP (no encryption/auth — trusted-network only). Two distinct effects:
- **Capacity:** run a model too big for one machine (e.g., full PRO Q4 split across two 512GB Mac Studios).
- **Prefill speedup:** pipelined across machines, meaningful gains on large prompts (measured up to ~1.85x on a 64k-token prompt across two MacBooks over Thunderbolt 5).
- **Generation is *not* sped up** — it's strictly autoregressive, so distributed generation is measurably *slower* than single-process (a measured ~19% loss in one benchmark) due to the added cross-machine hop per token.

### 2.8 Steering

`dir-steering/` implements single-vector activation steering, based on the "Refusal in Language Models Is Mediated by a Single Direction" line of work. A single direction vector can push behavior (verbosity, topic refusal, willingness to answer certain question types) up or down at inference time, without fine-tuning. Test vectors are captured from short/long-context continuations against the official DeepSeek V4 Flash API.

### 2.9 Reference speed numbers (Metal, `--ctx 32768`, greedy, from project README)

| Machine | Quant | Prompt | Prefill | Generation |
|---|---|---|---|---|
| MacBook Pro M3 Max, 128GB | q2 | short | 58.5 t/s | 26.7 t/s |
| MacBook Pro M3 Max, 128GB | q2 | 11.7k tok | 250.1 t/s | 21.5 t/s |
| MacBook Pro M5 Max, 128GB | q2 | 11.7k tok | 463.4 t/s | 25.9 t/s |
| Mac Studio M3 Ultra, 512GB | q4 | 12k tok | 448.8 t/s | 26.6 t/s |
| DGX Spark GB10, 128GB | q2 | 7k tok | 343.8 t/s | 13.75 t/s |

(Treat as a rough reference — beta software, numbers move fast with each release.)

### 2.10 Status caveats to remember

- Explicitly **beta**; "a few days old" as of the README's own framing, with distributed inference and SSD streaming called out as recent/still-settling features.
- No encryption/auth on the distributed protocol — trusted network only.
- Single live in-memory session, no request batching — not a multi-user production server as-is.
- Developed with heavy AI coding assistance (disclosed openly by the author) — worth knowing if that matters for how you frame using/contributing to it.

---

## Part 3 — Synthesis Notes for DwarfStar-RAG

**Core idea:** exploit §2.4 (disk-first, prefix-hash-keyed KV cache) for the *indexing → retrieval* half of the pipeline in §1.5. Because the cache key is an exact byte-hash of the rendered prefix, the entire value of this approach depends on making the "system prompt + retrieved context" block **byte-identical** across requests that should share a cache entry — i.e., canonical, deduplicated, consistently-ordered chunk retrieval (see earlier deck: "canonical context builder" stage).

**What's a genuinely novel angle vs. what's just "using the feature":**
- Reusing the disk cache for repeat queries on stable context = using the feature as designed.
- *Pre-warming* the cache offline (walk the corpus, populate `--kv-disk-dir` before any user query arrives) = a more original contribution, not documented as a built-in workflow.
- A small tool to inspect/report on cache hit rate, staleness, and prefix coverage (the `KVC`/`DSV4` on-disk format in §2.4 is fully inspectable) = a genuinely useful, publishable side-utility.
- Applying `dir-steering` (§2.8) toward "answer only from retrieved context, cite sources" = an experiment worth *measuring*, not an established capability — no evidence yet it works well for grounding specifically.

**Known constraints to design around:**
- No concurrent/batched requests server-side (§2.5) — one session at a time.
- Cache hits require exact byte-identical prefixes — different retrieved chunk sets (the normal case across a broad corpus with varied questions) won't hit the cache; the win is narrowest for repeated/follow-up questions on the *same* narrow document.
- Quantization is 2-bit on the routed experts — validate output quality on the actual target task before assuming parity with a full-precision frontier model, especially for numerically precise or compliance-sensitive extraction.

**Suggested next research/build steps:**
1. Stand up `ds4-server` locally, confirm `--kv-disk-dir` cache hits are actually observed (hexdump/inspect `.kv` files as the README suggests) with a real small corpus.
2. Build the canonical-context-builder layer (dedupe + fixed chunk ordering) — this is the load-bearing piece, not the model or the cache itself.
3. Measure: cache-hit rate and cold-vs-warm latency, on a corpus/question distribution representative of the intended use.
4. Experiment with a grounding steering vector; measure hallucination/citation accuracy with vs. without it, don't assume it works.
5. Only then consider pre-warming, multi-corpus cache management, or agentic (tool-calling) retrieval as extensions.

---

*Compiled August 2026. DwarfStar/ds4 is beta software under active daily development — re-verify flag names, endpoint behavior, and file formats against the live repo before implementation.*
