# Diario di bordo

A real day-by-day dev diary — the reasoning behind every decision, in the order it actually happened, in first person. This is where the *why* lives now; the per-component docs ([naive-rag.md](./naive-rag.md), [advanced-rag.md](./advanced-rag.md), [modular-rag.md](./modular-rag.md)) just point back here. For the flat "what was chosen" tables, see [memory.md](./memory.md).

---

## 2026-08-17 — Picking the paradigm, and saying no to a fork

Started today with a plan that felt obvious and turned out to be wrong: build a RAG engine following the Gao et al. survey architecture, then build a middleware layer on top of DwarfStar (`ds4`), possibly forking the repo to get there.

Before committing to that I forced myself to be honest about what DwarfStar actually is right now: a narrow, beta-quality inference engine for exactly two model families (DeepSeek V4, GLM 5.2), no embeddings endpoint, no plugin system — the maintainer's own suggested customization path is "fork it and let an AI agent modify the source." There's no retrieval logic in it at all. And "smaller machines" is relative — the benchmark hardware is a $12k 512GB Mac Studio. My actual target machine is a normal consumer box. DeepSeek V4 Flash alone needs 96–128GB even at 2-bit quant. It will not run here, full stop, no configuration changes that.

So I wrote [ADR-0001](../ADR-0001-rag-on-dwarfstar.md) and decided: build the RAG engine model-agnostic, against the standard OpenAI-compatible chat-completions interface, and treat DwarfStar as a swappable backend behind that boundary — not a foundation to build on top of. No fork. The two riskiest pieces of this project (retrieval quality, and a beta single-author engine) stay decoupled, so a bad week on one doesn't sink the other. I can prototype and iterate against a small local model today (Ollama), and the day I get access to qualifying hardware I just swap `BASE_URL` — zero code changes.

Also settled the overall shape: **Modular RAG** as the destination, built in three stages — Naive → Advanced → Modular — so every module I add produces a measurable eval delta instead of "it seems better."

---

## 2026-08-18 — Stack setup: models, formats, and swapping Ollama off WSL

Goal for today: one script that can call a local LLM and a local embedding model, nothing fancier yet.

**Chat model.** Went with `qwen2.5:7b` — fits consumer hardware, strong on retrieval-style/structured tasks, and since the pipeline is model-agnostic by design (per yesterday's ADR) this can change freely later without touching any RAG logic.

**Embedding model.** `nomic-embed-text`, pulled via Ollama. Kept it deliberately separate from the chat model even though they're both served the same way — DwarfStar has no embeddings endpoint, so this service was always going to need to stay independent, may as well design it that way from day one. 768 dims, normalized output, zero extra infra since it rides the same OpenAI-compatible client as chat.

**Document formats.** `.txt`, `.pdf`, `.md` — covers the real corpus I'm planning to use (DwarfStar docs + RAG papers), no point supporting formats I don't have.

Hit a real environment snag: Ollama was running inside WSL (Ubuntu, snap package), which is an annoying extra hop for a Windows-native project. Switched to `winget install Ollama.Ollama` for a native Windows install, now serving at `localhost:11434` directly. Had to leave the WSL copy for the user to remove manually (`sudo snap remove ollama` needs a password prompt I can't automate).

Created `.venv` with `py -3.12` specifically, not the default 3.13 — better compatibility with the ML-adjacent packages I'll need. Installed `openai`, `python-dotenv`, `pypdf` (pypdf didn't last — see tomorrow). Built `src/embedder/embedder.py`: one function, `embed(texts) -> list[list[float]]`, calling Ollama's `/v1/embeddings` through the same client as chat. Self-check passes. Phase 0 gate: half green (embeddings verified through code; chat verification came later once the generator existed).

---

## 2026-08-19 — The big one: chunking through eval, baseline recorded

This was the day the whole naive pipeline came together. Writing it roughly in build order since that's also the decision order.

**PDF library.** Swapped `pypdf` for `pdfplumber` before even finishing the loader — better multi-column/table handling, and the corpus I was about to add has real academic PDFs with two-column layouts. Built `src/loader/` (`load_documents(directory) -> list[Document]`), skips unreadable files with a warning instead of crashing the whole ingest.

**Corpus.** Populated `data/knowledge/` with something real instead of toy files: the DwarfStar (`ds4`) docs (`README.md`, `MODEL_CARD.md`, `AGENT.md`, `STRIXHALO.md`, `LICENSE.txt`) plus two arXiv RAG papers (Lewis 2020, Gao 2024 survey) — on-theme with the project's own `details.md`, and public/real so retrieval quality would mean something when I got to eval. Loader self-check passed on all 7 files; this was also the first real test of the pdfplumber path (19- and 21-page PDFs extracted cleanly). Restructured `data/` into `data/knowledge/` (the only thing the loader ever reads) vs `data/other/` (scratch, never touched) — whole `data/` gitignored.

**Chunking.** Went with recursive character split — hand-rolled, no LangChain — over fixed-size token chunking (respects paragraph/sentence structure, which fixed-size doesn't) and over semantic segmentation (too expensive to tune this early, deferred to Advanced RAG). Separators `["\n\n", "\n", ". ", " "]`, tried biggest first. Size: 2000 chars ≈ 512 tokens (rough 4-chars/token approximation, no real tokenizer wired up), 200 char / 10% overlap as a boundary safety net without bloating the index. Kept the interface `chunk_document(doc) -> list[Chunk]` deliberately stable so the implementation can be swapped later — that's the whole point of building toward Modular RAG. `src/__init__.py` added so cross-package imports work. Self-check on the real corpus: 157 chunks, all within bounds.

**Vector store.** LanceDB — embedded, no server process, which is exactly what ADR-0001's local-first stance calls for. Considered sqlite-vec (less active ecosystem) and Chroma (heavier, its own storage format) but LanceDB's Python API and Arrow-based columnar storage won. Full rebuild on every `ingest.py` run rather than incremental upsert logic — simplest way to guarantee no duplicate chunks at this corpus size, and I don't have a reason yet to optimize for incremental re-indexing. Built `src/ingest.py` wiring loader → chunker → embedder → LanceDB, batching 32 chunks per embedding call. Ran it for real: 157 chunks stored. Verified with an actual similarity search, not just "the code ran" — asked "What is retrieval-augmented generation?" and the top hit was the survey paper's own opening definition. That's the kind of check that actually tells you something.

**Similarity metric.** Cosine, set *explicitly*. LanceDB's default is squared-L2, not cosine, and I confirmed `nomic-embed-text` vectors come out unit-normalized — meaning L2 and cosine happen to rank identically today (L2² = 2 × cosine_distance for unit vectors). I could have left the default alone and gotten the same ranking. Decided not to rely on that coincidence silently; if the embedding model ever changes and stops normalizing, an implicit L2 default would quietly break ranking with no error anywhere. Set cosine explicitly so the distance values are also directly interpretable (0 = identical, 2 = opposite) — which turned out to matter later (see 2026-08-20).

**Top-K.** 5 — the standard starting point across the literature, meant to be tuned against eval scores once I had any. Built `src/retriever/retriever.py` (`retrieve(query, k=5)`) and `src/prompt/prompt.py` (`assemble_prompt`) together, self-checked jointly (prompt.py's self-check calls the real retriever, not a stub).

**System prompt.** Strict grounding — answer only from context, else say "I don't know based on the provided documents," no falling back to prior knowledge. Chose this over the permissive option specifically because it makes faithfulness *measurable*: a permissive prompt makes it much harder to tell a grounded answer from a lucky guess. Context ordering: best-first (similarity rank order) — "lost in the middle" research suggests worst-first/best-last can help, but that's a real hypothesis to test against eval data, not something to guess at up front. Source attribution (`[Source: filename]` per chunk) — free, since the metadata was already flowing through the pipeline, no reason not to.

**Generation.** `qwen2.5:7b`, temperature 0.0 for reproducible eval runs (doesn't guarantee correctness on its own — that's retrieval quality + the strict prompt; temp=0 just removes sampling randomness as a separate variable I'd otherwise have to control for), max 512 response tokens as a ceiling against rambling. Built `src/generator/generator.py`, self-check passed — this closed out the rest of the Phase 0 gate (chat now verified through code, not just embeddings).

**Wiring it end to end.** Built `src/query.py` — `query(question) -> QueryResult` plus a CLI. Manually checked two cases that actually matter: an on-topic question got a correct, well-grounded, well-cited answer; an off-topic one ("What is the capital of France?") got correctly refused instead of a confident wrong answer — and its retrieval distances were visibly worse (0.60+ vs ~0.28), which was reassuring: the distance number was tracking something real even before I understood its scale properly (see tomorrow). Added `tests/test_generator.py` and `tests/test_query.py`, including the negative-rejection case as an actual test, not just something I eyeballed once. 17/17 passing at this point. **Naive RAG's core pipeline was done, end to end.**

**Eval framework.** Reached for RAGAS first, since it's the standard tool. It was broken: `ragas` 0.4.3 hard-imports `langchain_community.chat_models.vertexai`, which doesn't exist in the `langchain-community` version it itself pulls (0.4.2) with no version constraint — that whole module was dropped when Vertex AI support was removed from a now-deprecated package. Didn't just assume this and move on — checked upstream, found it's a known, currently open bug (`vibrantlabsai/ragas#2753`, three pending fix PRs, none merged). Considered forking to patch it and rejected that immediately — it's exactly the situation ADR-0001 already argued against, forking a fast-moving upstream for a fix that's already in progress there. Considered pinning an older `langchain-community` as a stopgap and rejected that too — smaller footprint and no exposure to ragas's wider dependency tree (instructor, networkx, datasets, scikit-network) mattered more than avoiding ~150 lines of my own code. Built `src/eval.py`: hand-rolled LLM-as-judge versions of RAGAS's three core metrics (faithfulness, context relevance, answer relevance) using the same `generate()` already built, plus a binary correctly-rejected check and latency/throughput timing. Had to change `generate()` to return a `GenerationResult{text, completion_tokens}` instead of a bare string so throughput could actually be measured — small ripple into `query.py` and its tests.

**Eval set.** Hand-wrote 23 Q&A pairs in `eval/qa_pairs.json` — 20 answerable (read every one of the 7 corpus files myself rather than guessing at plausible questions) plus 3 deliberately out-of-scope, for negative rejection. Ran the full eval:

```
Faithfulness:        0.85
Answer relevance:     0.86
Context relevance:    0.4975
Rejection accuracy:   1.0
Avg latency/query:    6.87s (p95 10.72s)
Generation speed:     4.74 tok/s
```

Context relevance landing almost exactly in the 0.5–0.7 range I'd predicted in naive-rag.md as "naive RAG's known weak point" was a genuinely good sign — it meant the harness was measuring something real, not just producing numbers. That's the number Phase 2's reranking is aimed at. **Phase 1 gate satisfied: 20/20 tests passing, baseline recorded.** Long day.

---

## 2026-08-20 — First real query, and why the distance numbers looked "unimpressive"

Ran the pipeline for real for the first time outside of self-checks and eval — asked it "what are the differences between advanced rag and modular rag?" against a corpus that has an entire survey paper about exactly this. Got a correct, well-grounded answer. But the retrieval distances were only 0.31–0.38, which looked low-confidence at a glance for a question the corpus is *entirely* about.

Worked through why, and it's not a bug:

1. **The distance scale itself.** Cosine metric here means `distance = 1 − cosine_similarity`, ranging 0 (identical) to 2 (opposite) — this is exactly the interpretability I set the metric explicitly for back on 08-19. 0.31–0.38 is similarity ≈ 0.62–0.69, which is actually a solid match. I was implicitly expecting near-0 distances for "obviously relevant," but real (non-duplicate) text pairs essentially never get there — that was my wrong intuition, not a broken retriever.

2. **Query/document form mismatch.** My query is a short question; the source is long declarative survey prose explaining the same thing in different words. Embeddings put semantically related text close, but a question and its differently-phrased answer still sit further apart in vector space than two passages phrased the same way would. This is the textbook query–document vocabulary gap, and it's literally what HyDE exists to fix (embed a hypothetical *answer* instead of the raw query — noted in `details.md` §1.4).

This connects directly back to yesterday's baseline: context relevance of 0.4975 is this exact phenomenon showing up in the eval numbers, not a separate issue. It's real confirmation that Phase 2 (reranking, query rewriting/HyDE) is aimed at something actually observed in practice, not just a theoretical weakness from a survey paper.

Also did some housekeeping today: started this diary, and moved the per-step "why" reasoning out of `naive-rag.md`'s component sections and into here, organized by the day each decision actually got made, instead of scattered across a spec document. `naive-rag.md` keeps the *what* (chosen values, code pointers); this file keeps the *why* and the order things actually happened in.

**Later the same day — starting Phase 2, and a real negative result.** Picked indexing-time preprocessing as the first Advanced RAG piece to build, specifically Reverse HyDE (hypothetical questions per chunk, indexed alongside the chunk itself) over metadata attachment (nothing downstream filters on metadata yet, so more fields wouldn't move retrieval quality on their own), Small2Big (solves a chunk-size problem I haven't diagnosed), or a KG index (roadmap's own advice: defer unless the corpus is highly structured, which mine isn't). Reverse HyDE goes straight at the vocabulary-gap problem diagnosed this morning, from the indexing side instead of the query side.

Built `src/enrichment/enrichment.py` (`generate_questions(chunk_text, n=3)`, one LLM call per chunk, best-effort parsing) and wired it into `ingest.py`: every chunk now gets stored twice — once under its own embedding (`kind="chunk"`), once under each hypothetical question's embedding (`kind="question"`), all pointing back to the same chunk text. `retriever.py` had to start deduping, since a chunk can now surface via either row. Ran it for real: 157 chunks → 459 generated questions → 616 rows.

First measurement was a genuine regression, not an improvement — worth recording exactly as it happened rather than smoothing over it:

```
                    Baseline   Reverse HyDE (raw pool)
Faithfulness         0.85          0.75
Context relevance    0.4975        0.47
Answer relevance     0.86          0.70
```

Every quality metric got worse, including the one this was supposed to fix. Traced it with a concrete example: "What license are the DeepSeek-V4 model weights released under?" — answered correctly in the baseline — came back "I don't know" after Reverse HyDE, because the real answer-bearing chunk didn't even make the top-5 anymore; every slot was filled by `kind="question"` hits instead. Root cause: hypothetical questions are phrased as questions, so they sit close to *any* user query in embedding space just by sharing interrogative form — regardless of whether the underlying chunk is actually relevant. Pooling chunk-embeddings and question-embeddings into one raw-distance ranking let that syntactic similarity systematically beat genuine semantic similarity. Confirmed further with the original self-check query ("What is retrieval-augmented generation?"): the top hit became a garbled, PDF-extraction-mangled bibliography chunk, matched via a hypothetical question the model invented from citation text it couldn't really parse.

Talked through two ways forward — revert and call it a documented negative result, or fix the actual mechanism (raw-distance comparison across two different embedding populations) and re-measure once before deciding. Went with the fix: rewrote `retrieve()` to search chunk-kind and question-kind rows as two **separate** ranked lists, then combine with Reciprocal Rank Fusion (RRF) — rank position within each list decides the fused score, so raw distances from the two populations never get compared directly. This is the same fix the roadmap already names for sparse+dense hybrid search (Phase 3.4); it turned out to be exactly the right tool for chunk-vs-question fusion too, just needed pulling forward.

Second measurement:

```
                    Baseline   Reverse HyDE (raw pool)   Reverse HyDE + RRF
Faithfulness         0.85          0.75                     0.76
Context relevance    0.4975        0.47                     0.57
Answer relevance     0.86          0.70                     0.73
Avg latency           6.87s         6.77s                    6.40s
```

Mixed, and worth being honest about which part actually worked: context relevance — the metric this technique specifically targets — beat baseline for the first time (0.57 vs 0.4975). RRF genuinely fixed the ranking-domination bug. But faithfulness and answer relevance are still below baseline, just less badly than the broken version. Traced the remaining cause to the same garbled-bibliography chunks from before: they still generate plausible-sounding, on-theme hypothetical questions even though the underlying text is unusable PDF-extraction noise from two-column citation lists, so they still win top-K slots on topical-relevance grounds while actively hurting what the LLM can do with them.

Decision: don't chase this further in isolation. Locked in the RRF-fused version as the current state rather than reverting — it's a real, targeted, understood fix (not a guess), and it beat baseline on the metric it was built for. But it's not "done" — moving on to reranking next, since a cross-encoder scoring query-chunk relevance directly is likely to catch exactly the garbled-chunk problem that's still dragging faithfulness/answer relevance down, for free, without more speculative tuning of this component in isolation. Will re-run the full eval again once reranking is in and see the combined picture rather than declaring this piece finished on its own.

**Still later — reranking, one more root-cause fix, and a real win.** Built `src/reranker/reranker.py`: a cross-encoder (`BAAI/bge-reranker-base`, via `sentence-transformers`) that scores `(query, chunk)` pairs jointly instead of comparing two independently-computed embeddings — the standard purpose-built tool the roadmap names for this. Picked it over an LLM-based reranker (would've kept the project Ollama-only, zero new deps, but slower/costlier per query) and over MMR (free, but optimizes for result diversity, not for filtering low-quality content — wouldn't touch the actual garbled-chunk problem). First non-Ollama model in the project; pulled in PyTorch via `sentence-transformers`, a real dependency-weight jump, worth it for a purpose-built tool over hand-rolling one. Wired into `query.py`: retrieve a 20-candidate pool (`CANDIDATE_K`), rerank down to the final `TOP_K=5` before prompt assembly.

First self-check was a repeat of the same symptom from this morning: reranking still put the garbled bibliography chunk (42) on top, *not* the clean definition chunk (0). Before assuming the reranker was the problem, checked what candidate pool it was actually given — and chunk 0 (the single best raw match, distance 0.2794) **wasn't even in the top-20 candidates** RRF fusion handed it. Traced to a miscalibrated constant, not a new bug: `RRF_K=60` is standard for web-scale search, where lists run to thousands of results and rank 1 vs rank 60 is a huge gap. This corpus is ~157 chunks, single-topic (every chunk is *about* RAG at some level), so almost everything shows up in both the chunk-list and question-list within a generous window — RRF's "credit for appearing in both lists" ends up rewarding two mediocre placements over the one genuinely strong single-list match, and pushed the best chunk out of the fused top-20 entirely. Dropped `RRF_K` to 5 so rank position matters much more steeply. Chunk 0 came right back into the pool (5th of 20) — and once the reranker actually got to see it, it scored it 0.82 against the bibliography chunk's 0.23, a far cleaner separation than raw cosine distance ever gave (0.2794 vs 0.2168 — barely distinguishable by distance alone).

Full eval, all three techniques together (Reverse HyDE + RRF(5) + reranking):

```
                    Baseline   HyDE+RRF(60)   HyDE+RRF(5)+rerank
Faithfulness         0.85         0.76             0.93
Context relevance    0.4975       0.57             0.5875
Answer relevance     0.86         0.73             0.89
Avg latency           6.87s        6.40s            16.21s
```

First time in this whole Phase 2 arc that every quality metric beat the naive baseline at once. But it came with a real cost, not a free lunch: CPU cross-encoder reranking over 20 candidates per query more than doubled end-to-end latency (6.87s → 16.21s). Not ignoring that — flagged per the roadmap's own §6.3a principle (fast-but-wrong and accurate-but-slow are both failures) as the next thing to weigh deliberately rather than silently accept, once query-time preprocessing (the one remaining Phase 2 bucket) is also in and there's a full picture to decide against.

The bigger lesson from today, across three separate root-cause chases (raw-pool domination → RRF miscalibration → candidate-pool starvation): every regression traced back to something genuinely explainable and fixable, not to the underlying techniques being wrong. Measuring after every single change — instead of judging "reranking" or "Reverse HyDE" as a verdict after one run — is the only reason any of these fixes were findable at all.

**Still later — query-time HyDE, the one that didn't pan out, and Phase 2 closes.** Last remaining Phase 2 bucket: query-time preprocessing. Picked query-time HyDE specifically (over query rewriting, step-back prompting, multi-query expansion, sub-query decomposition) because it's the direct query-side mirror of what already worked on the indexing side: generate a short hypothetical passage that would answer the query, embed *that* instead of relying only on the raw interrogative query, since declarative prose sits closer to declarative prose in embedding space than a question does — the same vocabulary-gap logic as Reverse HyDE, just closed from the other direction. Built `src/hyde/hyde.py` (`generate_hypothetical_answer`, capped at 150 tokens since it only gets embedded, never read), added a small `max_tokens` override to `generator.generate()` to support that cap, and wired a third search list into `retriever.py`'s fusion (hyde-vector vs chunk rows, RRF-fused alongside the existing chunk/question lists).

Self-check looked promising in isolation — the hypothetical passage put the real definition chunk at distance 0.1984, tighter than the raw query ever got (0.2794). But the full eval didn't back it up:

```
                    Rerank checkpoint   + query-time HyDE
Faithfulness              0.93               0.92
Context relevance         0.5875             0.59
Answer relevance          0.89               0.84
Avg latency               16.21s             18.96s
```

Context relevance barely moved (noise-level), faithfulness held, but answer relevance dropped a real amount — 0.84, below even the naive baseline's 0.86 — for ~2.75s more latency per query. Read this as: the reranker is already doing the actual work of separating good candidates from noise, regardless of which list surfaced them. A third, noisier candidate source (a *generated* passage, which can be subtly wrong even while being declarative-prose-shaped) had nothing left to contribute once a strong reranker sits downstream of it — it just gave the reranker more chances to be fooled, for a real latency tax.

Reverted cleanly rather than keeping it as "we tried the whole roadmap": removed the third list from `retriever.py`, deleted `src/hyde/` and its test, reverted the `max_tokens` parameter on `generate()` since nothing else used it. Didn't re-run eval after reverting — the code is now byte-for-byte the same as what already produced the reranking-checkpoint numbers above, so re-measuring an unchanged state wouldn't be new information, just a repeat run.

**Phase 2 status: closed, with one deliberately-not-taken bucket.** Final state — Reverse HyDE (indexing) + RRF fusion + cross-encoder reranking (post-retrieval). Query-time preprocessing was tried, measured, and correctly rejected rather than kept on the strength of "it's part of the roadmap." Final numbers, naive baseline → current:

```
                    Naive baseline   Current (Reverse HyDE + RRF + rerank)
Faithfulness           0.85                    0.93
Context relevance      0.4975                  0.5875
Answer relevance       0.86                    0.89
Rejection accuracy     1.0                     1.0
Avg latency            6.87s                   16.21s
```

All three quality metrics beat baseline. The open item, deliberately deferred rather than dropped: latency more than doubled, and nothing has addressed that yet. That's the next real decision, not query-time preprocessing — which, for this corpus and this reranker, turned out not to be worth having.

---

## 2026-08-21 — The CANDIDATE_K sweep: more candidates isn't free, but it isn't the cost you think

Came back to the latency question with a real proposal from the user: cache the "most important" chunks based on how often they get retrieved, so repeat usage gets faster over time. Worth explaining carefully why that specific mechanism doesn't map onto where the cost actually lives, rather than just building it and finding out later: profiled the pipeline (warm process, matching how `eval.py` actually runs) and found retrieval costs ~2.2–2.4s, reranking ~7.2–7.4s, generation ~5.2–5.4s. Reranking is the dominant added cost from yesterday, and it's inherent to the (query, chunk) *pair* — a cross-encoder has to jointly encode the query and the chunk together, so no amount of knowing "this chunk is popular" tells it anything about a *new* question being asked right now. A chunk-frequency cache would have nothing to actually skip. Explained the bi-encoder/cross-encoder distinction in full since it came up naturally — the whole reason reranking works better than raw retrieval is that a cross-encoder lets query and chunk tokens attend to each other directly, which is also exactly why it can't be precomputed.

What actually controls reranking cost is `CANDIDATE_K` (how many candidates get reranked, currently 20) — linear in candidate count, since each one is a fresh forward pass. Ran a sweep at 5/10/15/20 first:

```
K     faithfulness  ctx_rel  ans_rel  avg_lat
5     0.74          0.5475   0.73     9.11s
10    0.82          0.5025   0.79     11.93s
15    0.77          0.51     0.74     13.56s
20    0.92          0.575    0.89     15.44s
```

No clean tradeoff curve — K=20 (current) won on *every* quality metric, and K=15 scored worse than K=10 on all three, which shouldn't happen if more candidates strictly helped (most likely eval noise: one run per K, 23 questions, self-judging). Every K below 20 lost real quality for savings that didn't come cheap. Reported this plainly rather than picking a "good enough" smaller K just because that was the original ask — the data didn't support it.

User's read of the data was sharper than mine: the big jump was specifically 5→10, and asked to push higher (30/35/40/45/50) rather than accept "smaller K always costs quality." Right call — extending the sweep:

```
K     faithfulness  ctx_rel  ans_rel  avg_lat    p95_lat
30    0.97          0.615    0.94     20.19s     21.17s
35    0.97          0.6025   0.94     20.90s     22.82s
40    0.93          0.6025   0.89     22.15s     24.87s
45    0.93          0.5625   0.88     80.28s     144.85s
50    0.93          0.5625   0.93     135.31s    140.30s
```

Two real findings, not one. First: quality *peaks* at K=30, beating K=20 on every single metric — not a tradeoff, a strict improvement, and it costs *more* time, not less. That inverted the whole premise of the exercise: there's no K where you save time without a real quality cost, because the quality-optimal point sits above the current default, not below it. Second: K=45 and K=50 aren't a continuation of the same curve — latency scaled cleanly and near-linearly from 5 through 40 (~0.4s/extra candidate), then 4-6x'd suddenly, and p95 was *double* the average at K=45 (144.85s vs 80.28s avg), which is the signature of individual queries varying wildly within one run, not a stable cost. Read that as this machine hitting a real resource limit (thermal throttling or memory pressure, after 40+ minutes of sustained CPU load across the whole sweep) rather than the reranker genuinely needing 6x longer for 12% more candidates — flagged it as unreliable rather than reporting "K=50 costs 135s" as a fact, and it played no part in the final decision.

Landed on **CANDIDATE_K=30** as final. Not a fix for the original complaint — latency went *up* another ~4s (16.21s → 20.19s), the opposite of what the whole exercise set out to do. But it's an honest outcome: once quality was actually measured across the range, 20 turned out to be a mediocre point on the curve, not a reasonable middle ground, and there was no version of "reduce latency via this knob" that didn't cost more quality than it was worth. Chose to keep the quality-optimal point rather than manufacture a latency win the data didn't support.

**Phase 2 final numbers, naive baseline → current:**

```
                    Naive baseline   Current (Reverse HyDE + RRF + rerank, K=30)
Faithfulness           0.85                    0.97
Context relevance      0.4975                  0.615
Answer relevance       0.86                    0.94
Rejection accuracy     1.0                     1.0
Avg latency            6.87s                   20.19s
```

Every quality metric improved substantially over the session; latency roughly tripled. That's the honest, measured tradeoff this project's whole discipline was built to surface rather than paper over — and it's where Phase 2 closes.

---

**Later the same day — actually fixing the latency, not just accepting it.** Came back to the 20.19s problem with a real proposal from the user: cache the "most important" chunks by retrieval frequency, so repeat usage gets faster over time. Worth explaining why that specific mechanism doesn't map onto the actual cost before building it: profiled the warm pipeline (retrieve ~2.2–2.4s, rerank ~7.2–7.4s, generate ~5.2–5.4s) and reranking is dominant. A cross-encoder has to jointly encode the query and chunk *together* — no amount of "this chunk is popular" tells it anything about a brand-new question, so a chunk-frequency cache has nothing to skip. Walked through the bi-encoder/cross-encoder distinction in full since it explains both why reranking helps (query and chunk tokens attend to each other directly) and why it's slow (nothing about that can be precomputed).

Checked the actual hardware next instead of guessing: an RTX 4050 with 6GB VRAM is present, but `sentence-transformers` had pulled the CPU-only PyTorch build, so the reranker never touched the GPU at all. Also checked whether Ollama itself uses the GPU — it does (`ollama ps` showed 82%/18% GPU/CPU split for `qwen2.5:7b`), and it's already using 4.3 of the 6GB, leaving only ~1.6GB free. That matters: `bge-reranker-base`'s ~1.1GB of weights would be tight against that headroom, and if it didn't fit cleanly, Ollama would silently push more of its own layers to CPU to make room — trading a reranking speedup for a generation slowdown, not a net win. Explained this whole chain to the user before touching anything, since "put it on the GPU" sounded simple but the actual risk (two models fighting over one memory pool) wasn't obvious from the outside.

Chose the safer fix first: swap `bge-reranker-base` (278M params, ~1.1GB) for `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, ~90MB) — comfortably clear of the VRAM risk without touching the GPU question at all, and inherently faster per-candidate even on CPU. Self-check still separated the real definition chunk from the garbled bibliography chunk cleanly (scores 4.97 vs 3.58), so the ranking behavior held. Measured: reranking 30 candidates dropped from ~11s (BGE, proportional) to 1.85s — about 6x.

Full eval came back mixed, not a clean win — worth reporting exactly as measured:

```
                    BGE (K=30)   MiniLM (K=30)
Faithfulness           0.97         0.89
Context relevance      0.615        0.7175
Answer relevance       0.94         0.84
Avg latency            20.19s       8.85s
```

Latency more than halved and context relevance actually improved, but faithfulness and answer relevance both dropped — answer relevance (0.84) landed slightly *below* even the naive baseline's 0.86. Not a strict win yet.

Since MiniLM is so much cheaper per candidate (~0.06s vs BGE's ~0.4s), there was real headroom to spend on a bigger pool without approaching BGE's cost — swept 30/40/50/60/80. Quality turned out to be byte-identical from K=40 through K=80 (the same top-5 chunks win the rerank regardless of pool size beyond that point), while latency kept climbing for nothing. One real scare in the middle of that: rejection accuracy dropped to 0.6667 at K≥50. Traced it before trusting the number — the "failed" case was the FIFA World Cup question, and the model's actual answer was *"The document provided does not contain any information about the 2022 FIFA World Cup winner"* — a completely correct refusal, just phrased differently than the exact substring `eval.py`'s rejection check looks for (`"don't know" in answer.lower()`). Not a real regression, a measurement blind spot in the harness. Good reminder not to let a single crude metric override actually reading the answer.

**Landed on `CANDIDATE_K=40` with the MiniLM reranker.** Matches or beats the K=30/MiniLM numbers on every quality metric (faithfulness 0.94, context relevance 0.7025, answer relevance 0.89), at essentially the same latency as K=30 (8.87s vs 8.85s — the extra 10 candidates cost almost nothing with a model this cheap).

**Final numbers, the whole day's arc:**

```
                    Naive baseline   BGE (K=30)   MiniLM (K=40, final)
Faithfulness           0.85            0.97           0.94
Context relevance      0.4975          0.615          0.7025
Answer relevance       0.86            0.94           0.89
Avg latency            6.87s           20.19s         8.87s
```

This is the actual win the whole exercise was looking for: every quality metric still clears the naive baseline, context relevance is the best it's been all session, and latency is back down near where Naive RAG started — despite running the full Reverse HyDE + RRF + reranking pipeline. Swapping the *tool* (a lighter, better-suited model) beat every attempt to tune the existing tool's parameters, which is worth remembering next time a knob-turning session stalls out on diminishing returns.
