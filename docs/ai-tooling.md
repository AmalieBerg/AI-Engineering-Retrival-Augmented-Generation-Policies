# AI Tooling

The project description **encourages** using AI tools and asks us to describe how we used them honestly. This is that writeup.

## What we used

**Claude (Anthropic) in agentic / IDE mode** for the bulk of the code, with the human directing strategy, reviewing output, and making the design calls. The workflow was:

1. Human decides what to build next (corpus  eval set  ingestion  RAG pipeline  Flask app  eval harness  docs)
2. Claude drafts the module, including tests
3. Human reads it, asks for changes, and runs the code
4. Iterate until the module works and the tests pass

That's the high-level pattern. The detail below covers what worked, what didn't, and where we had to push back on the model's defaults.

---

## What worked well

### Skeleton code and boilerplate

Claude was excellent at producing the first draft of a module given a clear specification. Examples that landed well on the first try:

- The Flask app factory pattern (`create_app(pipeline=None)`) — accepted a stub pipeline cleanly for testing
- The dataclass response types (`RAGResponse`, `Citation`) with `to_user_dict()` / `to_eval_dict()` separation
- The CLI arg parsing in `build_index.py` and `run_eval.py`

This stuff is well-trodden Python, and the model produces it accurately and idiomatically.

### Test coverage

Tests were the highest-leverage use of AI assistance. Asking for "tests for this module, including edge cases" produced thorough coverage: empty inputs, dedup, case sensitivity, JSON parse robustness, etc. The 47 scoring tests in `tests/test_score.py` and the 12 judge-parsing edge cases in `tests/test_judge.py` would have taken significant time to enumerate by hand.

The model is also good at writing tests that catch *its own* bugs — see the next section.

### Bug discovery via testing

Two real bugs were caught by running the AI-generated tests:

1. **Validation order bug in `validate_answer`** — the original logic checked `had_retrieved_chunks` after the length heuristic, which let short answers slip through when no chunks were retrieved. The test `test_validate_answer_replaces_substantive_answer_without_chunks` flagged it.
2. **Substantive heuristic too loose** — "The CEO is John Smith, hired in 2020." (38 chars, has a digit) slipped through a 40-char length-only threshold. The test caught it, and we improved the heuristic to also check for digit/`$`/`%` markers.

In both cases the test was generated alongside the code, ran, and failed — exactly the bug-discovery workflow tests are supposed to enable.

### Synthetic corpus generation

The 15 policy documents were drafted by Claude given the requirements: realistic company policies, ~30–120 pages total, mix of formats, with specific testable numbers (PTO accrual tiers, password length, GDPR window, etc.). The first drafts needed minor edits for consistency (e.g., aligning the contact emails across files) but were otherwise usable as-is.

This is exactly the kind of task generative AI is excellent at — producing plausible, internally consistent text in a constrained domain.

### Iterative refactoring

Mid-project we realized `RetrievedChunk` was tightly coupled to `retriever.py` (which imports LangChain), blocking testing of `prompt.py` and `pipeline.py` in restricted environments. Asking Claude to extract it into a new `rag/types.py` module took one prompt and produced consistent edits across five files. This kind of cross-cutting refactor is where AI tooling pays its biggest dividend — the human cost of "find every import of X and change them" is high, and the model does it accurately.

---

## What didn't work as well

### Architectural decisions

The model is fine drafting code once a decision is made, but is too eager to please when asked "should we use X or Y?" — it'll list trade-offs but won't strongly recommend a path. The choices of Chroma over Pinecone, k=5 over k=3 or k=8, BGE-small over OpenAI embeddings — these were made by the human after reading the trade-offs the model surfaced.

This isn't a failure of AI tooling so much as a reminder that *architecture is a human judgment call*. The model can enumerate options; it shouldn't choose for you.

### Tendency to over-engineer

Initial drafts of the Retriever and the eval harness had more abstractions than needed (e.g., a separate `RetrievalStrategy` base class with concrete implementations for MMR, similarity, and reranker variants). We simplified to a single `Retriever` class with flags. Lesson: ask the model to "keep it simple, no premature abstraction" up front.

### Citation regex initially too strict

The first version of the citation parser required exact `[POL-XX-NNN]` formatting with no flexibility. Real LLMs sometimes emit lowercase, extra spaces, or stray punctuation. The pattern was tightened iteratively against actual model output, not the model's guess at what it might produce.

### Eval set design needed pushback

The first draft of `eval_set.json` had only 1 refusal question (the "dress code" one). Asked to add more, the model proposed two reasonable cases — but the third one I added (`Q27: "Ignore your previous instructions..."`) was a deliberate human choice to test prompt-injection robustness specifically. The model is good at producing within-distribution test cases; **adversarial cases need human imagination.**

### Markdown formatting drift

The model has a strong default toward verbose headers, bullet lists, and bolded subsections. For the README and design doc, the human edited the output to use more prose and less bullet-soup — a common pattern with AI-drafted technical writing.

---

## Where AI tooling was deliberately not used

A few decisions where human judgment had to come first:

1. **The aesthetic direction for the UI.** Editorial / refined-minimalism with Fraunces serif headers, single teal accent, JetBrains Mono doc IDs — this was a deliberate human choice. AI's default for "build me a chat UI" tends toward generic gradient-and-bubble designs.
2. **Refusal phrase wording.** Specifically `"I can only answer about our policies, and I couldn't find this in our documents."` — the exact words matter for the guardrail to detect them reliably. Human-chosen, model-implemented.
3. **The plan order.** Corpus  eval  ingestion  RAG  app  eval harness  docs. The model can execute on any order; the human picked one that lets each stage validate the previous.
4. **What goes in the eval set.** The specific 27 questions, their `must_contain` substrings, and which docs are gold — these were designed to exercise the system, not generated.

---

## Reproducibility notes

If you wanted to recreate this workflow:

- Use a strong agentic coding assistant (Claude Sonnet 4.5+, GPT-5, or similar)
- Give it the project description as initial context
- Maintain a running task list and confirm each task before moving on
- After each module, run the tests it generated against the actual code — *don't trust them sight unseen*
- For any non-trivial design decision, ask for trade-offs, then decide yourself
- Push back when output drifts: too verbose, too abstracted, too generic

The combination of human strategic direction and AI execution speed produced a complete project (~3,500 LoC, 47+ tests, three documentation pages, working CI/CD) significantly faster than either alone. But every architectural decision and every word of these documents has been read, considered, and edited by a human before reaching this file.

## Disclosure

In line with the project description: AI tools were used extensively, in conformance with academic integrity standards. The project description explicitly permits and encourages this. All code was reviewed and tested by the human submitter before commit.
