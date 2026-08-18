# RagStar — Progress Memory

## Current phase: Phase 0 — Stack Setup

## Status

- [~] **Phase 0** — Stack setup: chat model (qwen2.5:7b) and embedding model (nomic-embed-text) both running via native Windows Ollama, embedder verified with a self-check script. Still missing: a script that also exercises the chat call (Phase 0 gate needs both chat + embeddings green).
- [~] **Phase 1** — Naive RAG pipeline end-to-end: document formats decided (.txt/.pdf/.md), embedder built (`src/embedder/`). Loader/chunker/vector store/prompt/generation not started yet.
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
| Document formats | .txt, .pdf, .md | 2026-08-18 | pypdf for PDF text extraction; txt/md via built-in open() |
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

---

## Config snapshot

*Fill in as you make decisions in naive-rag.md*

```
BASE_URL:      http://localhost:11434/v1
CHAT_MODEL:    qwen2.5:7b
EMBED_MODEL:   TBD
CHUNK_SIZE:    TBD
CHUNK_OVERLAP: TBD
TOP_K:         TBD
VECTOR_STORE:  TBD
EVAL_FRAMEWORK: TBD
```

---

## Notes / blockers

*(freeform — add anything that affects progress)*

- 2026-08-18: Switched Ollama from WSL (Ubuntu, snap package) to native Windows install (`winget install Ollama.Ollama`). User is removing the WSL copy manually (`sudo snap remove ollama`) — needs a password prompt, can't be automated. Ollama now serves at `http://localhost:11434` natively on Windows.
- 2026-08-18: `.venv` created with `py -3.12` (not the default 3.13 — chosen for better ML-package compatibility). Deps installed: `openai`, `python-dotenv`, `pypdf`.
- 2026-08-18: Built `src/embedder/` (`embedder.py` + `__init__.py`) — one function `embed(texts: list[str]) -> list[list[float]]` calling Ollama's `/v1/embeddings` via the same `openai` client used for chat. Self-check passes: `.venv/Scripts/python.exe src/embedder/embedder.py`.
- Next step when resuming: build the document loader (Component 1 in naive-rag.md) for .txt/.pdf/.md, then the chunker (Component 2), before wiring loader → chunker → embedder together.
