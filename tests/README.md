# Tests

Run from the repo root: `.venv/Scripts/python.exe -m pytest` (use `-m pytest`, not the bare `pytest` command, so the repo root is on `sys.path` and `from src...` imports resolve).

`test_loader.py` and `test_chunker.py` and `test_prompt.py` are pure logic — no external dependencies. `test_embedder.py` and `test_retriever.py` hit the real local stack instead of mocking it (consistent with how every self-check in `src/` already works): they need Ollama running locally, and `test_retriever.py` additionally needs `python src/ingest.py` to have been run first so `vector_db/chunks` exists.
