# RagStar — Progress Memory

## Current phase: Phase 1 — Naive RAG COMPLETE. Ready to start Phase 2 (Advanced RAG).

## Status

- [x] **Phase 0** — Stack setup: chat model (qwen2.5:7b) and embedding model (nomic-embed-text) both running via native Windows Ollama, both verified through code (embedder + generator self-checks).
- [x] **Phase 1** — Naive RAG: full pipeline done end-to-end (loader → chunker → embedder → LanceDB → retriever → prompt → generator, wired by `ingest.py` + `query.py`). 20/20 tests passing.
- [x] **Phase 1 eval** — Baseline recorded 2026-08-19: faithfulness 0.85, answer relevance 0.86, context relevance 0.4975 (matches naive-rag.md's own predicted naive-RAG weakness — the target for Advanced RAG to move), rejection accuracy 1.0, avg latency 6.87s, 4.74 tok/s. **Phase 1 gate satisfied — clear to start Phase 2.**
- [ ] **Phase 1 eval** — Baseline scores recorded (gate before Phase 2)
- [ ] **Phase 2** — Advanced RAG (pre/post-retrieval)
- [ ] **Phase 2 eval** — Scores vs. baseline recorded
- [ ] **Phase 3** — Modular RAG (composable modules)
- [ ] **Phase 3 eval** — Scores vs. Phase 2 recorded
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
| | | | |

---

## Baseline eval scores

*Fill in after Phase 1 eval run (`python eval.py`)*

| Metric | Naive RAG | Advanced RAG | Modular RAG |
|--------|-----------|-------------|-------------|
| Context relevance | 0.4975 | — | — |
| Faithfulness | 0.8500 | — | — |
| Answer relevance | 0.8600 | — | — |
| Rejection accuracy (negative rejection) | 1.0000 | — | — |
| Avg latency/query (s) | 6.87 | — | — |
| P95 latency/query (s) | 10.72 | — | — |
| Generation speed (tok/s) | 4.74 | — | — |

Baseline run: 2026-08-19, 23 Q&A pairs (20 answerable + 3 out-of-scope), `eval/results_20260819T160641Z.json`. Context relevance landed almost exactly where naive-rag.md predicted (0.5-0.7 range, "naive RAG's main weakness") — a good sign the harness measures something real. This is the number Advanced RAG's reranking should move.

---

## Config snapshot

*Fill in as you make decisions in naive-rag.md*

```
BASE_URL:      http://localhost:11434/v1
CHAT_MODEL:    qwen2.5:7b
EMBED_MODEL:   nomic-embed-text
CHUNK_SIZE:    2000 chars (~512 tok)
CHUNK_OVERLAP: 200 chars (~50 tok)
TOP_K:         5
VECTOR_STORE:  LanceDB (vector_db/chunks)
TEMPERATURE:   0.0
MAX_TOKENS:    512
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
