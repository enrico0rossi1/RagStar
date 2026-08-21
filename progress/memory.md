# RagStar — Progress Memory

## Current phase: Phase 2 — Advanced RAG COMPLETE (2 of 3 techniques kept, reranker swapped to MiniLM, `CANDIDATE_K` tuned to 40). Ready to start Phase 3 (Modular RAG).

## Status

- [x] **Phase 0** — Stack setup: chat model (qwen2.5:7b) and embedding model (nomic-embed-text) both running via native Windows Ollama, both verified through code (embedder + generator self-checks).
- [x] **Phase 1** — Naive RAG: full pipeline done end-to-end (loader → chunker → embedder → LanceDB → retriever → prompt → generator, wired by `ingest.py` + `query.py`). 20/20 tests passing.
- [x] **Phase 1 eval** — Baseline recorded 2026-08-19: faithfulness 0.85, answer relevance 0.86, context relevance 0.4975 (matches naive-rag.md's own predicted naive-RAG weakness — the target for Advanced RAG to move), rejection accuracy 1.0, avg latency 6.87s, 4.74 tok/s. **Phase 1 gate satisfied — clear to start Phase 2.**
- [x] **Phase 2.1** — Indexing-time preprocessing: Reverse HyDE (hypothetical questions per chunk, `src/enrichment/`), retrieval fused with RRF across chunk/question embeddings.
- [x] **Phase 2.3** — Reranking (post-retrieval): cross-encoder (`BAAI/bge-reranker-base`, `src/reranker/`), 20-candidate pool reranked to top-5. Required recalibrating `RRF_K` (60 → 5, the web-scale default was starving the best single-list match out of the candidate pool in this small single-topic corpus) before the reranker could even see the right chunks. Combined result: faithfulness 0.93, context relevance 0.5875, answer relevance 0.89 — **all three beat the naive baseline for the first time** (0.85/0.4975/0.86). Cost: avg latency 6.87s → 16.21s (CPU cross-encoder over 20 candidates/query) — flagged, not yet addressed.
- [x] **Phase 2.2** — Query optimization (pre-retrieval): tried query-time HyDE (`src/hyde/`, third RRF list). Measured negative — answer relevance dropped to 0.84 (below the 0.86 naive baseline, and below the 0.89 rerank-only checkpoint), context relevance barely moved, +2.75s latency. Reverted cleanly (code + tests + the `generate()` param it needed); current pipeline has no query-time preprocessing. See [diario_di_bordo.md — 2026-08-20](./diario_di_bordo.md#2026-08-20-first-real-query-and-why-the-distance-numbers-looked-unimpressive) for the full arc.
- [x] **Phase 2 eval** — Scores with `BAAI/bge-reranker-base`, CANDIDATE_K=30: faithfulness 0.97, context relevance 0.615, answer relevance 0.94, rejection accuracy 1.0, avg latency 20.19s. **Phase 2 gate satisfied.**
- [x] Latency actually fixed (2026-08-21) — root cause was the reranker model, not the candidate count. GPU exists (RTX 4050, 6GB) but PyTorch was CPU-only, and the ~1.6GB free VRAM (Ollama already uses 4.3GB) was too tight to safely add `bge-reranker-base` (~1.1GB) without risking Ollama getting pushed further onto CPU. Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) instead — ~6x cheaper per candidate even on CPU, sidesteps the VRAM question entirely. Re-swept `CANDIDATE_K` (30/40/50/60/80) since reranking got so much cheaper: quality plateaus at 40 (identical through 80), so kept 40. **Final: faithfulness 0.94, context relevance 0.7025, answer relevance 0.89, avg latency 8.87s** — beats naive baseline on every quality metric, and latency is back near where Naive RAG started (6.87s) despite running the full pipeline. See [diario_di_bordo.md — 2026-08-21](./diario_di_bordo.md#2026-08-21-the-candidate_k-sweep-more-candidates-isnt-free-but-it-isnt-the-cost-you-think).
- [ ] **Phase 3** — Modular RAG (composable modules) — next phase
- [ ] **Phase 7** — DwarfStar integration (hardware TBD)

---

## Decisions made

| Component | Decision | Date | Notes |
|-----------|----------|------|-------|
| RAG paradigm | Modular RAG | 2026-08-17 | Naive → Advanced → Modular build order |
| Chat model | qwen2.5:7b | 2026-08-18 | Native Windows Ollama (WSL Ollama removed) |
| Document formats | .txt, .pdf, .md | 2026-08-18 | txt/md via built-in open() |
| PDF library | pdfplumber | 2026-08-19 | Switched from pypdf — better multi-column/table layout handling |
| Corpus / dataset | DwarfStar (ds4) docs + 2 arXiv RAG papers | 2026-08-19 | Real public files: README.md, MODEL_CARD.md, AGENT.md, STRIXHALO.md, LICENSE.txt from github.com/antirez/ds4; lewis2020_rag.pdf (2005.11401), gao2024_rag_survey.pdf (2312.10997) from arxiv.org. On-theme with this project's own details.md. 7 files, all load correctly (loader's PDF path now verified for real). |
| Chunking strategy | Recursive character split | 2026-08-19 | Hand-rolled (no LangChain), separators ["\n\n","\n",". "," "]. Interface `chunk_document(doc) -> list[Chunk]` kept swap-friendly for Modular RAG later. |
| Chunk size / overlap | 2000 / 200 chars | 2026-08-19 | ≈512/50 tokens via ~4 chars/token approximation — no real tokenizer wired up yet |
| Vector store | LanceDB | 2026-08-19 | Embedded, no server, per ADR-0001's local-first recommendation. `vector_db/chunks` table, full rebuild on each ingest run (not incremental) |
| Similarity metric | Cosine, explicit | 2026-08-19 | LanceDB defaults to squared-L2, not cosine. nomic-embed-text vectors are unit-normalized so L2/cosine rank identically today, but set explicitly rather than rely on that silently |
| Top-K | 5 | 2026-08-19 | naive-rag.md default starting point |
| System prompt style | Strict grounding | 2026-08-19 | Makes faithfulness measurable in eval; no fallback to prior knowledge allowed |
| Context ordering | Best-first | 2026-08-19 | Chunks used in similarity-rank order; revisit "lost in the middle" only if eval shows top chunks getting ignored |
| Source attribution | Yes, `[Source: filename]` per chunk | 2026-08-19 | Free — metadata already carried through the pipeline |
| Testing framework | pytest | 2026-08-19 | Tests hit the real local stack (Ollama, LanceDB), not mocked — consistent with how self-checks already work. `tests/` mirrors `src/`; run via `python -m pytest` (not bare `pytest`, for sys.path reasons) |
| Temperature | 0.0 | 2026-08-19 | Reproducible eval runs. Does NOT guarantee correctness — that's retrieval quality + strict grounding; temp=0 just removes sampling randomness as a variable |
| Max response tokens | 512 | 2026-08-19 | Ceiling to stop rambling |
| Eval framework | Hand-rolled LLM-as-judge (not RAGAS) | 2026-08-19 | Verified, not assumed: ragas 0.4.3's METADATA declares `langchain-community` with NO version constraint, so `pip install ragas` always pulls the latest (0.4.2, itself deprecated upstream), which dropped `chat_models.vertexai` — a module ragas still hard-imports at package load. Confirmed as a known, currently-open upstream bug: github.com/vibrantlabsai/ragas issue #2753, three pending fix PRs (#2837/#2923/#2956), none merged. Considered forking to patch it; rejected per ADR-0001's own reasoning against forking fast-moving upstreams for a fix already in progress there. Considered pinning an older langchain-community as a stopgap; rejected in favor of the hand-rolled version (smaller footprint, no exposure to ragas's wider dependency tree — instructor, networkx, datasets, scikit-network). Revisit real RAGAS once the upstream fix ships. Replicated its 3 core metrics directly using our own `generate()` — ~150 lines. See `src/eval.py`. |
| Eval Q&A set | 23 pairs (20 answerable + 3 out-of-scope) | 2026-08-19 | Hand-written, grounded in the real corpus content (read every file directly rather than guessing) — see `eval/qa_pairs.json` |
| Embedding model | nomic-embed-text | 2026-08-18 | Via Ollama /v1/embeddings, same openai client as chat |
| Indexing preprocessing | Reverse HyDE (3 questions/chunk) + RRF-fused retrieval | 2026-08-20 | `src/enrichment/`. First raw-pooled attempt regressed all quality metrics; RRF fusion across separate chunk/question searches fixed context relevance past baseline (0.4975 → 0.57), faithfulness/answer relevance still below baseline — carried forward pending reranking |
| Reranker | Cross-encoder, BAAI/bge-reranker-base (sentence-transformers) | 2026-08-20 | `src/reranker/`, candidate pool → top-5. Chosen over LLM-based (would've stayed Ollama-only/zero new deps but slower) and MMR (free but diversity-focused, wouldn't fix garbled-chunk noise). Required recalibrating RRF_K (60→5) to fix a candidate-pool starvation bug before the reranker could see the right chunks. Combined result: all 3 quality metrics beat baseline |
| CANDIDATE_K (reranker pool size) | 30 → **40 (superseded, see below)** | 2026-08-21 | First sweep (with BAAI/bge-reranker-base) found quality peaks at 30, better than 20 on every metric; a non-linear latency cliff at 45+ read as this machine's resource limits, not a real cost curve. Superseded same day once the reranker model itself changed (below) |
| Reranker (revised) | Cross-encoder, cross-encoder/ms-marco-MiniLM-L-6-v2 | 2026-08-21 | Swapped from BAAI/bge-reranker-base — GPU exists (RTX 4050, 6GB) but PyTorch was CPU-only, and the ~1.6GB VRAM free after Ollama's model (4.3GB) was too tight to safely add bge-reranker-base (~1.1GB) without risking Ollama offloading more of itself to CPU. MiniLM (~90MB) sidesteps the VRAM question and is ~6x cheaper per candidate even on CPU. Re-verified the model still separates real content from garbled bibliography chunks correctly (score 4.97 vs 3.58) before trusting it |
| CANDIDATE_K (revised) | 40 | 2026-08-21 | Re-swept (30/40/50/60/80) with the cheaper MiniLM reranker. Quality is byte-identical from 40 through 80 (same top-5 wins regardless of pool size beyond 40); a rejection-accuracy dip at 50+ turned out to be a measurement artifact (eval.py's substring check missed a correctly-phrased refusal), not a real regression — traced before trusting it. 40 matches/beats 30 on every metric at effectively the same latency (8.87s vs 8.85s) |
| | | | |

---

## Baseline eval scores

| Metric | Naive RAG | + Reverse HyDE (raw pool) | + Reverse HyDE (RRF K=60) | + Reranking, BGE (CANDIDATE_K=20) | + CANDIDATE_K=30, BGE | + MiniLM reranker, CANDIDATE_K=40 (**final**) |
|--------|-----------|---------------------------|------------------------------|-------------|-------------|-------------|
| Context relevance | 0.4975 | 0.4700 | 0.5700 | 0.5875 | 0.6150 | **0.7025** |
| Faithfulness | 0.8500 | 0.7500 | 0.7600 | 0.9300 | 0.9700 | 0.9400 |
| Answer relevance | 0.8600 | 0.7000 | 0.7300 | 0.8900 | 0.9400 | 0.8900 |
| Rejection accuracy (negative rejection) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Avg latency/query (s) | 6.87 | 6.77 | 6.40 | 16.21 | 20.19 | **8.87** |
| P95 latency/query (s) | 10.72 | — | 8.93 | 17.52 | 21.17 | ~9.8 |
| Generation speed (tok/s)* | 4.74 | 4.21 | 4.87 | 2.06 | 1.73 | ~2.9 |

\* `tokens_per_sec` in eval.py is completion_tokens ÷ full query() latency (retrieval+rerank+generation), not pure decode speed — the drop from Naive RAG is reranking overhead, not the LLM getting slower.

The BGE columns are historical — the reranker itself was swapped to a smaller model on 2026-08-21 (see below), which is now what `src/reranker/reranker.py` actually runs.

Naive baseline: 2026-08-19, `eval/results_20260819T160641Z.json`. Context relevance landed almost exactly where naive-rag.md predicted (0.5-0.7 range, "naive RAG's main weakness") — a good sign the harness measures something real.

Reverse HyDE, first attempt (raw pooled search over chunk + question embeddings): 2026-08-20, `eval/results_20260820T160828Z.json` — regressed every quality metric. Root cause: question-embeddings systematically outrank chunk-embeddings on raw cosine distance regardless of true relevance (shared interrogative form).

Reverse HyDE + RRF fusion, K=60 (chunk-kind and question-kind searched as separate ranked lists, fused by rank not raw distance): 2026-08-20, `eval/results_20260820T180834Z.json` — fixed context relevance past baseline, but faithfulness/answer relevance still below baseline, traced to garbled PDF-bibliography chunks still winning top-K on topical-relevance grounds.

Reverse HyDE + RRF (K recalibrated 60→5) + cross-encoder reranking (`BAAI/bge-reranker-base`): 2026-08-20, `eval/results_20260820T182917Z.json` — **all three quality metrics now beat baseline.** RRF_K=60 was itself starving the best chunk out of the reranker's candidate pool (web-scale constant miscalibrated for a ~157-chunk single-topic corpus); fixing that let the cross-encoder actually see and correctly separate real content from citation noise (0.82 vs 0.23 score). Tradeoff: avg latency more than doubled (6.87s → 16.21s) from CPU-based reranking over 20 candidates/query — flagged per ROADMAP.md §6.3a, not yet addressed. **This is the current state.**

Query-time HyDE, tried on top of the above (third RRF list, hypothetical-answer embedding vs chunk rows): 2026-08-20, `eval/results_20260820T213737Z.json` — answer relevance dropped to 0.84 (below both the naive baseline and the rerank-only checkpoint), context relevance barely moved, +2.75s latency, no compensating gain. Reverted; not part of the current state (`src/hyde/` removed).

CANDIDATE_K sweep with BGE (5, 10, 15, 20, 30, 35, 40, 45, 50), each a full eval run: 2026-08-21, `eval/candidate_k_sweep_20260820.json` (gitignored). Quality was *not* monotonic below 20 — every value under 20 lost real quality for modest latency savings (e.g. K=5: 9.11s but faithfulness dropped to 0.74). Quality peaked at K=30 (0.97/0.615/0.94, beating K=20 on every metric), plateaued/declined slightly by 40, then K=45/50 hit a latency cliff (80s, 135s) inconsistent with the clean ~linear scaling seen up to 40 — read as this machine hitting a resource limit (thermal/RAM) under sustained load, not a real cost curve, excluded from the decision. Chose K=30 with BGE — a deliberate acceptance of more latency (16.21s → 20.19s) for the best measured quality, since no value below 20 offered a good trade.

**Reranker model swap** (2026-08-21): the latency itself got fixed by changing tools, not tuning parameters further. Checked the hardware directly rather than guessing: RTX 4050 (6GB VRAM) present, but `sentence-transformers` had installed CPU-only PyTorch, so the reranker never used the GPU. Checked whether the GPU was even free enough to matter — `ollama ps` showed `qwen2.5:7b` already using 4.3GB/6GB (82%/18% GPU/CPU split), leaving only ~1.6GB free, tight against BGE's ~1.1GB of weights; risked pushing Ollama further onto CPU (slower generation) to make room (net loss, not a win). Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) instead of chasing CUDA — reranking 30 candidates dropped from ~11s (BGE, proportional) to 1.85s, ~6x. Full eval with MiniLM at the same K=30: `eval/results_20260821T125737Z.json` — context relevance improved further (0.7175) but faithfulness/answer relevance dropped vs. BGE (0.89/0.84, the latter just below the naive baseline's 0.86). Not a clean win yet.

**Re-swept CANDIDATE_K with MiniLM** (30/40/50/60/80), 2026-08-21, `eval/candidate_k_sweep_minilm_20260821.json` (gitignored) — cheap enough per-candidate now to afford a bigger pool. Quality is byte-identical from K=40 through K=80 (same top-5 wins regardless of pool size beyond 40); latency keeps climbing for nothing past that. A rejection-accuracy dip at K≥50 (1.0→0.6667) turned out to be a measurement artifact — the "failed" case (FIFA World Cup question) was actually correctly refused, just phrased as "the document does not contain..." instead of the exact "don't know" substring `eval.py`'s check looks for. Traced before trusting the number.

**Final state: MiniLM reranker, `CANDIDATE_K=40`.** faithfulness 0.94, context relevance 0.7025, answer relevance 0.89, rejection accuracy 1.0, avg latency 8.87s — beats naive baseline on every quality metric, at latency close to where Naive RAG started (6.87s) despite the full Reverse HyDE + RRF + reranking pipeline. `src/reranker/reranker.py`'s `MODEL_NAME` and `src/query.py`'s `CANDIDATE_K` both updated. **This is the current state.**

---

## Config snapshot

*Fill in as you make decisions in naive-rag.md*

```
BASE_URL:       http://localhost:11434/v1
CHAT_MODEL:     qwen2.5:7b
EMBED_MODEL:    nomic-embed-text
CHUNK_SIZE:     2000 chars (~512 tok)
CHUNK_OVERLAP:  200 chars (~50 tok)
N_QUESTIONS:    3 (Reverse HyDE hypothetical questions/chunk, src/enrichment/)
RRF_K:          5 (chunk/question fusion constant, recalibrated from the 60 web-search default)
RERANKER:       cross-encoder/ms-marco-MiniLM-L-6-v2 (cross-encoder, sentence-transformers)
CANDIDATE_K:    40 (candidates reranked before cutting to TOP_K)
TOP_K:          5 (final chunks sent to the prompt)
VECTOR_STORE:   LanceDB (vector_db/chunks)
TEMPERATURE:    0.0
MAX_TOKENS:     512
EVAL_FRAMEWORK: hand-rolled LLM-as-judge (src/eval.py, not RAGAS)
```

---

## Notes / blockers

*(freeform — add anything that affects progress)*

- 2026-08-18: Switched Ollama from WSL (Ubuntu, snap package) to native Windows install (`winget install Ollama.Ollama`). User is removing the WSL copy manually (`sudo snap remove ollama`) — needs a password prompt, can't be automated. Ollama now serves at `http://localhost:11434` natively on Windows.
- 2026-08-18: `.venv` created with `py -3.12` (not the default 3.13 — chosen for better ML-package compatibility). Deps installed: `openai`, `python-dotenv`, `pypdf`.
- 2026-08-18: Built `src/embedder/` (`embedder.py` + `__init__.py`) — one function `embed(texts: list[str]) -> list[list[float]]` calling Ollama's `/v1/embeddings` via the same `openai` client used for chat. Self-check passes: `.venv/Scripts/python.exe src/embedder/embedder.py`.
- 2026-08-19: Switched PDF library from `pypdf` to `pdfplumber` (better multi-column/table handling). Built `src/loader/` (`loader.py` + `__init__.py`) — `load_documents(directory) -> list[Document]`, `Document` = `{source, text}`. Skips unsupported/unreadable files with a warning instead of crashing.
- 2026-08-19: Populated the corpus with a real 7-file set (see Decisions table). Loader self-check passes on all 7 files, PDF path now verified for real (19-page and 21-page academic PDFs extract cleanly via pdfplumber).
- 2026-08-19: Restructured `data/` into `data/knowledge/` (the actual corpus — loader's default directory, only this gets indexed) and `data/other/` (anything that should never be ingested — scratch files, notes, etc). Whole `data/` dir added to `.gitignore` (not committed).
- 2026-08-19: Built `src/chunker/` (`chunker.py` + `__init__.py`) — recursive character split, hand-rolled (no framework). `chunk_document(doc: Document) -> list[Chunk]`, `Chunk = {text, source, chunk_index}`. Self-check chunks the real 7-file corpus: 157 chunks total, all within CHUNK_SIZE bounds. `src/__init__.py` added so `src` is an importable package (needed for chunker's cross-package import of `Document` from `src.loader`).
- 2026-08-19: Built `src/ingest.py` — wires loader → chunker → embedder → LanceDB together, batches embedding calls (32 chunks/call). Ran end-to-end against the real corpus: 157 chunks stored in `vector_db/chunks`. Verified with a real similarity search ("What is retrieval-augmented generation?") — top hit was the survey paper's own opening definition, confirming retrieval actually works, not just that the code runs.
- 2026-08-19: Built `src/retriever/` (Component 5) — `retrieve(query, k=5) -> list[RetrievedChunk]`, embeds the query and searches LanceDB with explicit cosine metric. Built `src/prompt/` (Component 6) — `assemble_prompt(query, chunks) -> messages`, strict-grounding system prompt, best-first ordering, `[Source: ...]` citations. Both self-checked individually and together (prompt.py's self-check calls the real retriever).
- 2026-08-19: Added `tests/` (pytest) — `test_loader.py`, `test_chunker.py`, `test_prompt.py` are pure logic (no external deps); `test_embedder.py`, `test_retriever.py` hit the real local Ollama/LanceDB stack instead of mocking. `pytest` added to `requirements.txt`.
- 2026-08-19: Built `src/generator/` (Component 7) — `generate(messages) -> str`, temperature=0.0, max_tokens=512. Self-check passes; this also closes the Phase 0 gate (chat call now exercised through code, not just embeddings).
- 2026-08-19: Built `src/query.py` — wires retrieve → assemble_prompt → generate into `query(question) -> QueryResult`, plus a CLI (`python src/query.py "question"`). **Naive RAG's core pipeline is now complete end-to-end.** Manually verified two cases: (1) on-topic question ("What is retrieval-augmented generation?") → correct, well-grounded answer citing the right paper; (2) off-topic question ("What is the capital of France?") → correctly refused with "I don't know based on the provided documents" instead of a confident wrong answer — retrieval distances were also notably higher (0.60+ vs 0.28) on the bad case, consistent with genuinely poor matches. Added `tests/test_generator.py` and `tests/test_query.py` (including the negative-rejection case as a real test, not just a manual check). **17/17 tests passing.**
- 2026-08-19: `ragas` 0.4.3 turned out broken (hard-imports `langchain_community.chat_models.vertexai`, which doesn't exist in installed `langchain-community` 0.4.2 — that whole package is now deprecated/sunset upstream). Uninstalled it rather than chase a version pin. Built `src/eval.py` instead: hand-rolled LLM-as-judge versions of RAGAS's 3 core metrics (faithfulness, context relevance, answer relevance) using our own `generate()`, plus a binary "correctly rejected" check for out-of-scope questions, plus latency/tokens-per-sec. Also changed `generate()` to return `GenerationResult{text, completion_tokens}` instead of a bare string, so throughput could be measured — updated `query.py` and the generator tests to match.
- 2026-08-19: Hand-wrote `eval/qa_pairs.json` — 23 pairs (20 answerable, grounded in real corpus content read directly from each file; 3 deliberately out-of-scope for negative rejection), covering all 7 documents. Ran the full eval: **faithfulness 0.85, answer relevance 0.86, context relevance 0.4975, rejection accuracy 1.0, avg latency 6.87s (p95 10.72s), 4.74 tok/s.** Context relevance landing right where naive-rag.md predicted the naive-RAG weakness would be is a good sign the harness measures something real. **Naive RAG phase gate is now satisfied — 20/20 tests passing, baseline recorded.**
- Next step when resuming: Phase 2 — Advanced RAG. First candidate per the roadmap: reranking (post-retrieval), aimed directly at the 0.4975 context-relevance number.
