# RagStar — Progress Memory

## Current phase: Phase 0 — Stack Setup

## Status

- [ ] **Phase 0** — Stack setup (Ollama + embedding model working)
- [ ] **Phase 1** — Naive RAG pipeline end-to-end
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
CHAT_MODEL:    TBD
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
