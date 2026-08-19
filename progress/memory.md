# RagStar — Progress Memory

## Current phase: Phase 1 — Naive RAG (indexing done, querying not started)

## Status

- [~] **Phase 0** — Stack setup: chat model (qwen2.5:7b) and embedding model (nomic-embed-text) both running via native Windows Ollama, embedder verified with a self-check script. Still missing: a script that also exercises the chat call directly (Phase 0 gate technically wants both chat + embeddings green in one script).
- [~] **Phase 1** — Naive RAG: indexing half done end-to-end (loader → chunker → embedder → LanceDB, verified with a real search). Querying half (retrieval/prompt assembly/generation/`query.py`) not started.
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
| Embedding model | nomic-embed-text | 2026-08-18 | Via Ollama /v1/embeddings, same openai client as chat |
| | | | |

---

## Baseline eval scores

*Fill in after Phase 1 eval run (`python eval.py`)*

| Metric | Naive RAG | Advanced RAG | Modular RAG |
|--------|-----------|-------------|-------------|
| Context relevance | — | — | — |
| Faithfulness | — | — | — |
| Answer relevance | — | — | — |
| Hit Rate @ K=? | — | — | — |
| MRR | — | — | — |
| Avg latency/query (s) | — | — | — |
| Generation speed (tok/s) | — | — | — |

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
EVAL_FRAMEWORK: TBD
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
- 2026-08-19: Added `tests/` (pytest) — `test_loader.py`, `test_chunker.py`, `test_prompt.py` are pure logic (no external deps); `test_embedder.py`, `test_retriever.py` hit the real local Ollama/LanceDB stack instead of mocking. 14 tests, all passing. `pytest` added to `requirements.txt`.
- Next step when resuming: Component 7 (generation — the actual chat call) and `query.py` to wire retrieval → prompt → generation together, then the eval harness (Component 8).
