# ADR-0001: RAG Engine on Top of DwarfStar (ds4)

**Status:** Proposed
**Date:** 2026-08-17
**Deciders:** Enrico

## Context

The goal is a fully local RAG stack: private/on-prem data, no cloud API dependency, backed by DwarfStar (`ds4`), antirez's local inference engine. The initial proposal was: (1) build the RAG engine following the architecture in the Gao et al. survey (arXiv:2312.10997), (2) build a middleware layer connecting it to DwarfStar, possibly forking the DwarfStar repo.

Before endorsing that plan, it's worth being blunt about what DwarfStar actually is, because the pitch ("run bigger models on smaller machines") oversells it slightly, and the "fork the repo" instinct is probably solving a problem that doesn't exist yet.

**What DwarfStar actually is**, as of the current release: a narrow, single-purpose inference engine built on llama.cpp/GGML, currently supporting exactly two model families — DeepSeek V4 (Flash/PRO) and GLM 5.2. It is explicitly **beta quality**, described by the author as "fast changing," with instability expected between releases. It ships an OpenAI/Anthropic-compatible HTTP server (`ds4-server`: `/v1/chat/completions`, `/v1/messages`, `/v1/completions`, `/v1/models`) plus a CLI and a `ds4-agent` with tool calling. It does **not** ship an embeddings endpoint, does not have a plugin/extension system, and the maintainer's suggested customization path is literally "fork it and let an AI coding agent modify the source for you." There is no retrieval or RAG functionality anywhere in it.

**"Smaller machines" is relative, not small.** DwarfStar's target hardware is 96GB+ unified-memory Macs, multi-GPU CUDA/ROCm rigs, or SSD-streaming setups for MoE experts. The benchmark antirez himself cites is a $12k Mac Studio M3 Ultra (512GB) running DeepSeek V4 PRO at 2-bit quantization, getting ~150 t/s prefill and ~10-13 t/s decode. That's "smaller than a datacenter," not "smaller than a laptop." If the actual target machine for this project is a normal dev box or a single consumer GPU, DwarfStar's supported models won't run on it at all — the model choice is dictated by the engine, not the other way around.

## Decision

Build the RAG engine as a **model-agnostic component that talks to any OpenAI-compatible chat-completions endpoint**, and treat DwarfStar as a swappable inference backend behind that interface — not as a foundation the RAG engine is built on top of. Do not fork DwarfStar as part of this roadmap. Only consider forking later, and only if a concrete capability gap shows up that can't be solved at the application layer.

## Options Considered

### Option A: RAG engine + DwarfStar middleware/fork (the original proposal)
| Dimension | Assessment |
|-----------|------------|
| Complexity | High — you're maintaining a fork of a beta, fast-moving engine on top of your own RAG code |
| Cost | High ongoing cost: every upstream ds4 release is a rebase/merge risk |
| Scalability | Fine for inference; irrelevant to the fork decision |
| Team familiarity | Depends entirely on comfort reading llama.cpp/GGML-derived C/C++ |

**Pros:** total control if you eventually need engine-level hooks (e.g. custom KV-cache handling for retrieved context, native retrieval-as-a-tool in `ds4-agent`).
**Cons:** solves a problem you don't have yet. ds4 already exposes an OpenAI-compatible API — RAG (retrieval + prompt assembly) is normally an application-layer concern that sits *in front of* a chat-completions API, not something that requires touching the inference engine's source. Forking a project the author himself calls beta and "fast changing" means signing up for continuous rebase pain for no proven benefit.

### Option B: RAG engine built against the standard chat-completions interface, DwarfStar plugged in as one backend among several
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low-medium — standard RAG engineering, zero engine-internals work |
| Cost | Low — no fork to maintain; upstream ds4 updates are a drop-in `git pull` + binary swap |
| Scalability | Backend-agnostic: can swap ds4 for vLLM, llama.cpp-server, Ollama, or a cloud API without touching RAG code |
| Team familiarity | Matches ordinary RAG/backend engineering skills |

**Pros:** decouples the two highest-risk pieces of the project (retrieval quality tuning, and a beta narrow-scope engine) so a bad week for either doesn't sink the other. Lets you prototype and iterate on retrieval quality against a stable backend (even a cheap cloud model) before your local hardware/DwarfStar setup is fully working.
**Cons:** if you eventually need something ds4 doesn't expose over HTTP (streaming partial retrieval into KV cache, native embeddings), you'll need a workaround or, eventually, Option A anyway — but only then, with a concrete justification instead of a hunch.

## Trade-off Analysis

The real trade-off isn't "RAG engine vs. middleware" — it's about where you put your integration risk. DwarfStar is young, single-model-family-scoped, and explicitly unstable. Baking a fork of it into your RAG architecture from day one means your RAG project inherits DwarfStar's instability as a *load-bearing* dependency. Building against the OpenAI-compatible surface it already exposes means DwarfStar is a replaceable component — which is the correct posture toward any beta software maintained largely by one person, however good that person is.

Two things the original plan glossed over, both harder than the middleware question:

1. **Embeddings.** RAG needs a retrieval embedding model, and ds4 doesn't provide one. This is a separate service you have to stand up regardless of what you do with DwarfStar — e.g. a small local embedding model served via llama.cpp's own embedding mode, or a lightweight local library. Don't let "connect to DwarfStar" absorb this — it's an independent workstream.
2. **Latency budget.** ~10-13 tok/s decode on a $12k machine is slow for a chat product, and RAG makes prompts *longer* (retrieved chunks add to prefill, which is comparatively fast, but a long, badly-curated context still costs you). This argues for tight top-k retrieval, reranking, and context compression (all "Advanced RAG" techniques in the survey) rather than naively stuffing in large amounts of retrieved text — on this hardware, sloppy retrieval is not just lower-quality, it's slow.

## Consequences

- Easier to swap inference backends later (including away from DwarfStar entirely if it stalls or a better local engine appears).
- Retrieval quality work can start immediately, without waiting on DwarfStar setup or hardware procurement.
- You still need to separately decide on: embedding model/server, vector store (favor an embedded, local-first store — e.g. LanceDB or sqlite-vec — over standing up a separate vector DB service, consistent with the "keep everything local/simple" goal), and a chunking/indexing strategy.
- You are explicitly deferring, not ruling out, a DwarfStar fork. Revisit only when a concrete, named capability gap appears (e.g., you want retrieval invoked as a native tool inside `ds4-agent`'s loop rather than orchestrated by your own code).
- Hardware reality has to be settled explicitly and early: DwarfStar's two supported model families are large MoE models that need serious memory (96GB+, realistically 512GB for the PRO-tier model at usable quality). If that's not the target machine, the "run on DwarfStar" plan needs a different, smaller model or a different engine.

## Action Items

1. [ ] Confirm target hardware and whether it actually meets DwarfStar's supported-model requirements (or pick DeepSeek V4 Flash / GLM 5.2 explicitly at a quantization level that fits).
2. [ ] Build indexing pipeline: chunking strategy + a separate local embedding model/service (not DwarfStar).
3. [ ] Pick a local-first vector store (avoid an extra always-on service unless multi-user access is a real requirement).
4. [ ] Build the RAG engine (retrieval → optional rerank/compression → prompt assembly) against the standard OpenAI chat-completions interface, prototyping against any compatible backend.
5. [ ] Stand up `ds4-server` and point the RAG engine's backend URL at it — no fork required for this step.
6. [ ] Build a retrieval + generation eval harness (the survey's evaluation frameworks, e.g. RAGAS-style metrics, are a reasonable starting point) before calling any of this "done."
7. [ ] Only after 1-6: if a specific ds4-internals limitation blocks something you actually need, scope a fork as its own follow-up ADR.

## Addendum (2026-08-17): Hardware reality check

Follow-up context: this is explicitly a **personal learning project**, not a product — the goal is to learn RAG hands-on and have it naturally track DwarfStar's progress over time, not to ship something stable. That reframes priorities (fine to accept instability, fine to explore internals eventually) but it doesn't change the core recommendation, and one new fact makes it non-negotiable rather than just prudent.

**Confirmed target hardware: a normal laptop/desktop, consumer GPU or CPU-only.**

This matters because of exact numbers, not vibes: DeepSeek V4 Flash is a 284B-parameter MoE model (13B active). Even at aggressive 2-bit quantization, DwarfStar's own docs recommend **96-128GB** of unified memory or equivalent VRAM just for the Flash tier; the PRO tier (1.6T total params) needs 512GB. There is no configuration, SSD-streaming mode included, that gets either supported model running on a normal consumer machine. This isn't "it'll be slow" — it's "it will not run."

It's also not a gap DwarfStar's own roadmap is trying to close. The improvements antirez is exploring (distributed inference across multiple high-memory machines, tensor/pipeline parallelism, RDMA between Macs) are about making *bigger* models feasible across a cluster of already-expensive boxes — not about shrinking requirements down toward consumer hardware. If anything the trend among the models DwarfStar targets is upward (V4 PRO is 1.6T, larger than Flash). So "wait for DwarfStar to get better and it'll eventually run on my laptop" is not a bet the current trajectory supports.

**What this means for "build it so it evolves with DwarfStar":** the recommendation from the main ADR body already gets you exactly what you're after, and now it's the only viable path rather than just the tidier one. Build the RAG engine against the standard OpenAI-compatible chat-completions interface, and develop/test it today against any small local model that actually fits current hardware (a quantized 3B-8B model via Ollama or llama.cpp's own server — same API shape as `ds4-server`). The RAG logic — chunking, embeddings, retrieval, reranking, evaluation — is 100% transferable and is the part that's actually the learning objective. The day you get access to qualifying hardware (rented cloud GPU/Mac, a future upgrade, a friend's machine), you point the same RAG engine's base URL at `ds4-server` and it works, unmodified. Nothing about DwarfStar's future improvements requires you to have started coupled to it now — the API boundary means you inherit those improvements automatically whenever you do gain access to compatible hardware.
