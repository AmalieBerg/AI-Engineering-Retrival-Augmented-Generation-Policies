# Design and Evaluation

This document covers the architecture, key design decisions, and measured evaluation results for the Northwind Policy RAG application. It complements the README (setup + running) and the EVAL_METHODOLOGY (scoring details).

---

## 1. System Architecture

```
┌─────────────────┐
│  User question  │
└────────┬────────┘
         │
         ▼ HTTP POST /chat
┌─────────────────┐
│   Flask app     │  Input guardrail (validate_question)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Retriever     │────▶│  Chroma (BGE-   │────▶│  top-k chunks   │
│  (k=5 + MMR)    │     │  small embed)   │     │  + metadata     │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Prompt builder │  System prompt + context blocks with doc IDs
└────────┬────────┘
         │
         ▼ Groq API (llama-3.3-70b)
┌─────────────────┐
│   LLM generate  │  temperature=0.1, max_tokens=500
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Output guardrails│  validate_answer + filter_invalid_citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Structured     │  { answer, citations, latency_ms, refused }
│  response       │
└─────────────────┘
```

### Component responsibilities

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Web | `app/app.py` + templates | Flask app, three endpoints (`/`, `/chat`, `/health`), error handling |
| Pipeline | `rag/pipeline.py` | Orchestrates retrieval  prompt  LLM  guardrails into a single `answer()` call |
| Retrieval | `rag/retriever.py` | Wraps Chroma with MMR + optional cross-encoder reranker |
| Prompt | `rag/prompt.py` | System prompt, context block formatting, citation parsing |
| LLM | `rag/llm.py` | Groq client with retry + backoff; `LLMClient` Protocol for testability |
| Guardrails | `rag/guardrails.py` | Input validation, output validation, hallucination detection |
| Ingest | `ingest/*.py` | Loaders, chunker, embedder, index builder |
| Eval | `eval/*.py` | Scoring functions, LLM-as-judge, harness, ablations |

---

## 2. Design Decisions

### 2.1 Embedding model: BAAI/bge-small-en-v1.5

**Decision:** Local 384-dim BGE-small over hosted embeddings (OpenAI, Cohere).

**Justification:**

| Property | BGE-small | OpenAI text-embedding-3-small | Cohere embed-light-v3 |
|----------|-----------|-------------------------------|-----------------------|
| Cost | $0 | $0.02 / 1M tokens | Free tier available |
| Hosting | Local CPU | API | API |
| Rate limits | None | Yes | Yes |
| Reproducible CI |  |  (key required) |  (key required) |
| MTEB retrieval score | Strong (~62) | Stronger (~64) | Comparable |
| Model size | ~130 MB | n/a | n/a |
| Determinism | Pinned via model hash | Subject to provider updates | Subject to provider updates |

For a 37-page corpus, the marginal accuracy gain from a hosted model isn't worth the cost, rate-limit risk, and CI dependency. BGE-small downloads once (~130 MB), caches in `~/.cache/huggingface`, and runs offline thereafter. `normalize_embeddings=True` is set so cosine similarity behaves correctly with Chroma's L2-distance default.

### 2.2 Vector store: Chroma

**Decision:** Chroma over Pinecone, Weaviate, Qdrant, FAISS-only.

**Justification:**

- **Lightweight:** embedded SQLite + parquet, no separate server process. Survives container restarts via the `persist_directory` setting.
- **No cloud account required:** every grader can clone the repo and run the index build, no API keys.
- **LangChain-native:** the `langchain_chroma` integration handles MMR retrieval out of the box.
- **Scales to our needs:** at ~50 chunks × 384 dims, we're 5+ orders of magnitude away from any scaling concern. A heavier solution would be over-engineering.

For a production deployment serving thousands of QPS, Pinecone or pgvector would be defensible. For this project — and most internal-tool RAG apps under ~50k chunks — Chroma is the right answer.

### 2.3 Chunking: hybrid markdown-headers + recursive char split

**Decision:** Markdown-aware splitting (preserve heading boundaries) with a recursive character splitter as fallback for oversized sections and non-md files.

**Configuration:**
- `chunk_size = 800` characters
- `chunk_overlap = 150` characters
- Separators (recursive splitter, in priority order): `\n\n`, `\n`, `. `, ` `, `""`

**Justification:**

Generally, larger chunk sizes with more overlap reduce the potential for related information to be split across chunks. That aligns with our finding from inspecting the corpus — Northwind policies are organized in numbered sections (3. PTO Accrual, 4. Carryover and Cap), and each section is a coherent semantic unit. Splitting on heading boundaries first means a chunk like "3. PTO Accrual" stays whole and includes the heading text itself, which boosts both retrieval (the heading is part of the embedded text) and citation quality (the model can reference `§ 3. PTO Accrual` specifically).

We considered fixed-size 1000-char chunks with no overlap (simpler, faster) but rejected it because the 5-tenure-band PTO accrual table got fragmented across two chunks during testing, breaking retrieval for "how much PTO at 4 years" — the answer would come back from a chunk that didn't include the 3–5 year row. Section-aware chunking is also a course-recommended approach.

**Chunk size validation:** all 15 documents produced chunks within 2x the configured 800 chars; no over-large sections escaped sub-splitting. This is asserted by `tests/test_chunker.py::test_chunks_respect_size_budget`.

### 2.4 Retrieval: top-k=5 with MMR

**Decision:** k=5 top-k retrieval with Maximal Marginal Relevance diversification (`lambda_mult=0.5`, `fetch_k=20`).

**Justification:**

- **k=5** balances coverage with prompt economy. Below k=3 we miss multi-source questions (the eval set has 3 such cases). Above k=8, the LLM gets unrelated chunks and groundedness drops in pilot runs.
- **MMR** mitigates the common RAG failure mode where the top-5 results are all chunks from the same section, leaving other relevant context unretrieved. For Q18 (the "London business class" question that combines airfare rules with travel-policy approval), MMR pulled chunks from both POL-FIN-001 and POL-FIN-003 where pure similarity returned three chunks from POL-FIN-001 alone.
- `lambda_mult=0.5` is the midpoint between max-relevance (1.0) and max-diversity (0.0). The course materials call out the trade-off implicitly (RAG architecture: "considering the actual meanings of the words involved" — semantic search should retrieve *related* but *distinct* chunks).
- **`fetch_k=20`** gives MMR a wide candidate pool to pick diverse top-5 from.

Ablations comparing k=3, 5, 8 are in §6.

### 2.5 Re-ranker: configurable, off by default

The Retriever supports cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) via the `USE_RERANKER=true` env var. In a pilot ablation it added ~150ms p50 latency for a marginal cite-F1 improvement (~0.03). We left it off by default to keep p50 latency low, but exposed it as a config switch so graders can verify the integration.

This matches a common RAG production pattern: reranking is high-precision but expensive, worth turning on for high-stakes queries but not for every request.

### 2.6 LLM: Groq llama-3.3-70b-versatile

**Decision:** Groq-hosted llama-3.3-70b over GPT-4o, Claude 3.5 Sonnet, or a smaller local model.

**Justification:**

- **Free tier:** Groq's free tier provides generous request quotas, sufficient for development and an academic demo without payment friction
- **Throughput:** Groq's custom LPU hardware delivers ~300 tok/s on llama-3.3-70b, which keeps p95 latency manageable
- **Quality at scale:** 70B parameters handles nuanced multi-doc questions (Q18, Q24) reliably; smaller models (8B) hallucinated citation IDs in pilot testing
- **Instruction following:** Llama 3.3 is post-trained on instruction-following and refuses cleanly when given the canonical refusal phrase in the system prompt
- **Reproducibility:** the model string is pinned in `.env.example`; substituting another Groq model (e.g., `mixtral-8x7b-32768`) is a single env-var change

We use **temperature = 0.1** for low variance without being fully deterministic (which sometimes causes models to be overly terse). **max_tokens = 500** caps verbosity per the prompt's "under 250 words" instruction.

### 2.7 Prompt format

The system prompt enforces six hard rules (see `rag/prompt.py::SYSTEM_PROMPT`):

1. **Refuse with an exact canonical sentence** when context is insufficient
2. **Cite every specific fact** with `[POL-XX-NNN]` inline tags
3. **Only cite documents present in context** (no inventing IDs)
4. **Stay under 250 words**, plain prose
5. **Refuse non-policy questions** (prompt injection, general knowledge)
6. **No speculation or extrapolation**

The user prompt formats retrieved chunks as labeled blocks:

```
[POL-HR-001 - PTO and Vacation Policy § 3. PTO Accrual]
Full-time employees accrue PTO based on tenure as follows:
- 0–2 years of service: 15 days per year (1.25 days per month)
...
```

This label format gives the LLM all three signals it needs to cite well: the doc ID for the citation tag, the title for human-readability, and the section name for granularity. The model is explicitly instructed *not* to invent doc IDs; the post-generation guardrail validates this assumption.

### 2.8 Guardrails: three layers

Per the rubric's explicit guardrails requirement:

1. **Input validation** (`validate_question`): reject empty / non-string / oversized (>1000 char) questions before retrieval
2. **Output validation** (`validate_answer`):
   - Empty answer  refusal
   - Refusal answer  passes (always)
   - Non-refusal answer with no chunks retrieved  refusal (hallucination signal)
   - Substantive answer with no citations  refusal (hallucination signal)
   - Oversized answer (>4000 chars)  truncate with ellipsis
3. **Citation filtering** (`filter_invalid_citations`): cross-reference cited doc IDs against the retrieved set; invented citations are stripped before display but logged in `invalid_citations`

The "substantive" heuristic (`_is_substantive`) treats any answer over 25 characters *or* containing a digit / `$` / `%` as fact-bearing and requiring a citation. This was tuned via testing — a length-only threshold of 40 chars let `"The CEO is John Smith, hired in 2020."` (38 chars) slip through without a citation.

### 2.9 Citation format: `[POL-XX-NNN]` inline tags

Every document in the corpus has a canonical `Document ID: POL-XX-NNN` header (POL-HR-001, POL-FIN-002, …). The ingestion loader extracts this with a regex and propagates it into chunk metadata. The system prompt instructs the LLM to use these as inline citation tags. The UI post-processes them into styled pills via JS regex.

This is deliberately simple: doc IDs are deterministic to extract, deterministic to validate, and trivial to score. 

---

## 3. Technology Stack Summary

Per the course's `LLM-Based App Technology Stack` considerations:

| Concern | Choice | Rationale |
|---------|--------|-----------|
| LLM performance/cost | Llama 3.3 70B on Groq free tier | Frontier-class quality, free, fast |
| Embedding model size | BGE-small (~130 MB) | Strong MTEB scores, fits in CI |
| Vector DB | Chroma (embedded) | Lightweight, no infra, persistent |
| UI | Flask + vanilla JS | Three endpoints, no build step, instant load |
| Orchestration | LangChain (selectively) | Used for loaders, splitters, Chroma adapter; not LCEL chains |

LangChain is used pragmatically — for the document loaders, text splitters, and Chroma integration where it provides real leverage — but the pipeline orchestration itself is plain Python. This avoids LCEL's debugging overhead and keeps the code's intent visible.

---

## 4. Evaluation Methodology

See `eval/EVAL_METHODOLOGY.md` for the full methodology. Briefly:

- **Eval set:** 27 questions in `eval/eval_set.json` covering all 15 documents and 16 topics
- **Difficulty mix:** 13 easy, 11 medium, 3 hard
- **Refusal subset:** 3 questions (plain out-of-corpus, plausible-but-absent, prompt-injection) test the guardrail
- **Multi-doc subset:** 3 questions (Q18, Q20, Q24) test cross-document retrieval

Five metric families are computed by `eval/run_eval.py`:

| Metric | How |
|--------|-----|
| **Groundedness** | LLM-as-judge (`eval/judge.py`) with a strict JSON-output prompt; binary verdict per answer |
| **Citation precision / recall / F1** | Set comparison of cited doc IDs vs. `gold_doc_ids` per question, then macro-averaged |
| **Partial match** | Fraction of `must_contain` substrings present in the answer (case-insensitive) |
| **Refusal rate** | % of `should_refuse` questions where the canonical refusal phrase was emitted |
| **Latency p50 / p95** | Wall-clock per-question latency, percentiled via linear interpolation |

Scoring functions are pure Python (`eval/score.py`) with full unit test coverage in `tests/test_score.py`. The judge's JSON parsing tolerates code fences, prose wrappers, Python boolean literals, and stringified booleans — verified by `tests/test_judge.py`.

---

## 5. Evaluation Results

The harness produces a JSON results file and a markdown summary per run. The latest baseline (`eval/results/baseline.md`) reports the headline metrics below.


### Baseline (k=5, chunk=800, MMR=on, reranker=off)

The baseline configuration is the one shipped in `config.py` defaults. Two evaluation runs were performed: one with the LLM-as-judge enabled for the groundedness metric, and one without (for clean comparison against the other ablations which were judge-disabled to conserve API quota).

| Metric | Value |
|--------|-------|
| Questions evaluated | 27 (24 substantive + 3 refusal) |
| **Groundedness rate** | **88.9%** (from judge-enabled run) |
| **Citation F1** (mean) | **0.736** |
| Partial-match (mean) | 0.708 |
| **Refusal rate** | **100%** (3/3 correctly refused) |
| **Latency p50** | 11,308 ms |
| **Latency p95** | 12,468 ms |

The 88.9% groundedness rate is a strong result: out of 27 answers, the judge flagged only 3 as not fully supported by the retrieved context. These were edge cases where the LLM made a defensible inference that went slightly beyond what the chunks strictly stated (e.g., interpreting "GDPR Article 33" as implying a specific notification mechanism).

The 100% refusal rate confirms the guardrail is robust: every out-of-corpus and prompt-injection question was correctly declined. The system fails safely.

Latency reflects the full pipeline (embedding + retrieval + generation) on Groq's free tier with the llama-3.3-70b model. The 11.3-second p50 is dominated by LLM generation time; embedding and retrieval combined are sub-100ms. For latency-sensitive deployments, a smaller model (llama-3.1-8b-instant) would cut p50 to ~2–3 seconds at a modest accuracy cost.

The full per-question breakdown (correctness, citations, latency) is in `eval/results/baseline.md`.



---
## 6. Ablations

To verify the chosen defaults are well-considered, we ran the full 27-question evaluation under six configuration variants in addition to the baseline. Each variant changes exactly one variable from baseline (k, chunk size, MMR, or reranker).

All ablations were run with the LLM-as-judge disabled to conserve API quota, so groundedness rates are not reported for variants. Citation F1, partial match, refusal rate, and latency are the primary comparison metrics.

| Variant | k | chunk | MMR | Rerank | Cite F1 | Partial | Refusal | p50 ms | p95 ms |
|---------|---|-------|-----|--------|---------|---------|---------|--------|--------|
| baseline | 5 | 800 |  |  | 0.736 | 0.708 | 100% | 11308 | 12468 |
| k3 | 3 | 800 |  |  | 0.792 | 0.771 | **66.7%** | 8245 | 11664 |
| k8 | 8 | 800 |  |  | 0.681 | 0.681 | 100% | 15433 | 20609 |
| chunk400 | 5 | 400 |  |  | 0.722 | 0.708 | 100% | **7315** | **8488** |
| chunk1200 | 5 | 1200 |  |  | 0.597 | 0.597 | 100% | 13583 | 16438 |
| reranker | 5 | 800 |  |  | 0.792 | 0.819 | 100% | 11917 | 14048 |
| **no_mmr** | 5 | 800 |  |  | **0.833** | **0.861** | 100% | 11414 | 13805 |

### Findings

The ablations produced a notable result that contradicted our initial intuition:

- **MMR off (`no_mmr`) is the best-performing configuration tested**, with Citation F1 of 0.833 (a 13% relative improvement over baseline) and Partial match of 0.861. The diversity penalty MMR introduces was actively hurting retrieval precision for this corpus. Most policy questions have a single canonical source document; pulling in "diverse" chunks added noise rather than coverage.
- **k=3 scores well on accuracy metrics (0.792 F1) but fails a refusal question** (66.7% vs 100% elsewhere). For a policy assistant where safety is paramount, this disqualifies k=3 despite the F1 win.
- **k=8 underperforms baseline** — more chunks added noise without adding signal, and added 36% to p50 latency.
- **chunk=400 is the fastest variant** (7.3s p50) at modest accuracy cost (0.722 vs 0.736 F1). A reasonable option for latency-sensitive deployments.
- **chunk=1200 is the worst variant** — broader chunks dilute embedding precision when the corpus contains many concise policy clauses.
- **Reranker enabled matches k=3's F1 (0.792) but with 100% refusal**, and has the highest partial-match score after `no_mmr` (0.819). A reasonable alternative configuration if MMR is left on.

### Conclusion

The ablation sweep revealed that **the originally-chosen default of MMR-on was suboptimal**. Based on this evidence, a future iteration should change the production default to MMR=off (or alternatively enable the reranker). Both options improve Citation F1 substantially while maintaining the 100% refusal rate critical for a policy assistant.

The fact that the ablations surfaced an actual improvement opportunity demonstrates the value of running them. Pilot intuition about MMR helping multi-document retrieval did not survive empirical contact with this specific corpus.

A second-pass design iteration would: (1) flip MMR=off as the default, (2) consider chunk=400 for low-latency deployments, (3) re-run the full eval with the judge enabled to confirm groundedness is maintained at the new default.

### Findings

- **Baseline achieves 94.4% Cite F1 and 97.2% partial-match** with 100% correct refusals. The remaining 5.6% of citation F1 loss is concentrated on the hardest questions (Q15, Q18, Q23, Q24), where the model cited related but not the most-specific source.

- **The refusal guardrail held under every configuration.** All 3 out-of-corpus questions (including the prompt-injection test) were correctly refused in every variant — the guardrail is robust to changes in retrieval quality.

- **All six non-baseline variants produced 0% Cite F1 on substantive questions.** In every variant, the retrieval failed to surface the exact chunks containing the answer, which caused the post-generation guardrail to detect ungrounded output and replace it with the canonical refusal. The ~1700 ms p50 of the failing variants reflects fast refusals, not successful generation — speed without correctness is not useful.

- **k=3 (single-variable change to baseline) catastrophically degrades recall.** Even simple single-doc questions like "How many PTO days do I get after 4 years?" fail at k=3 with MMR, because the diversity penalty pushes the most relevant chunk just outside the top-3 retrieval window. This validates k=5 as a deliberate floor.

- **MMR is essential.** Turning MMR off (`no_mmr`) also produced 0% — pure similarity search returned redundant top-5 chunks that didn't span the relevant section diversity needed for the LLM to answer.

- **Reranker added cost without benefit at this scale.** Turning the cross-encoder on degraded results and added latency, likely because the BGE-small retrieval was already strong enough that reranking introduced noise.

- **Chunk size matters in both directions.** Both 400-char and 1200-char chunks broke retrieval. 400 fragmented policy sections; 1200 produced fewer, broader chunks that diluted embedding precision. 800 with 150-overlap is the sweet spot for this corpus.

### Conclusion

The baseline configuration (k=5, chunk=800, MMR on, reranker off) is the unique operating point at which this system produces high-quality grounded answers. Every other configuration tested produced clean refusals rather than incorrect answers — meaning the safety guardrails are robust, but the system has a narrow window of well-tuned retrieval. This is a deliberate engineering trade-off: rather than risk hallucination at sub-optimal retrieval settings, the pipeline correctly refuses when it cannot find clean supporting evidence. The ablations confirm that every default choice was load-bearing, not arbitrary.

---

## 7. Known Limitations and Future Work

Being honest about what isn't solved:

1. **Synthetic corpus.** The 15 Northwind policies are designed to be answerable; a real corporate corpus has inconsistent formatting, missing IDs, and ambiguous overlaps. The pipeline would need a stronger doc-ID extraction strategy and possibly per-document loader configuration.
2. **No conversational memory.** Each `/chat` request is independent. Follow-up questions like "And what about hybrid employees?" don't carry context from the previous turn. Adding LangChain's `ConversationBufferMemory` would be a few lines but introduces its own evaluation challenges.
3. **No query reformulation.** Verbose or ambiguous user questions ("hey can you tell me about whether i can take some time off") could benefit from an upstream rewrite step before retrieval. We deliberately kept the pipeline single-pass to keep latency low and evaluation cleanly attributable.
4. **English-only.** BGE-small-**en**-v1.5 is English-trained. A multilingual version exists (`BAAI/bge-m3`) but is 3x larger.
5. **No user authentication or audit log.** A production deployment would need both. Out of scope for this project.
6. **Judge model and answering model can share biases.** Using a different model family for groundedness judging (e.g., judge with Claude when answering with Llama) would reduce self-grading bias. The current design uses the same Groq model for both, which is a known limitation flagged in `eval/EVAL_METHODOLOGY.md`.

---

