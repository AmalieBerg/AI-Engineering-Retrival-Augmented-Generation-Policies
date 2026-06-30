# Northwind Policy RAG

A retrieval-augmented generation (RAG) application that answers questions about company policies. Built as the AI Engineering capstone project.

The corpus describes a fictional company called Northwind Technologies. Questions like *"How many PTO days do I get after 4 years?"* or *"What's the home office stipend?"* are answered with cited sources drawn from a 15-document policy library.

[![CI](https://github.com/AmalieBerg/RAG-Policies/actions/workflows/ci.yml/badge.svg)](https://github.com/AmalieBerg/RAG-Policies/actions/workflows/ci.yml)

## Architecture

```
┌─────────────────┐
│  User question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Flask /chat    │────▶│   Retriever     │────▶│  Chroma vector  │
│  endpoint       │     │  (top-k + MMR)  │     │  DB (BGE embed) │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  Prompt builder │
         │              │  + guardrails   │
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│   Groq llama-3.3-70b      Answer with  │
│                            citations    │
└─────────────────────────────────────────┘
```

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/AmalieBerg/RAGp-Policies.git
cd rag-policies
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. Get a free key at <https://console.groq.com>.

### 3. Build the vector index

```bash
python -m ingest.build_index --rebuild
```

This loads all 15 corpus documents, chunks them (markdown headings + recursive splitter), embeds with BGE-small (runs locally; downloads ~130 MB on first run), and stores them in `./chroma_db/`. Takes ~30 seconds on a laptop.

### 4. Start the app

```bash
python -m app.app
```

Then open <http://localhost:8000> in your browser.

## Endpoints

- `GET /` — Web chat UI
- `POST /chat` — JSON: `{"question": "..."}`  `{"answer", "citations", "latency_ms"}`
- `GET /health` — Returns `{"status": "ok"}`

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many PTO days do I accrue after 4 years?"}'
```

## Running tests

```bash
pytest -q                          # All non-integration tests
pytest -q tests/test_loaders.py    # Just the loader tests
pytest -m "not slow"               # Skip slow tests
```

## Running the evaluation

```bash
python -m eval.run_eval --output eval/results/run-$(date +%Y%m%d).json
```

The evaluation reports groundedness, citation accuracy, partial-match, refusal rate, and p50/p95 latency across 27 questions. See [`eval/EVAL_METHODOLOGY.md`](eval/EVAL_METHODOLOGY.md) for full details.

## Project structure

```
rag-policies/
├── app/                        # Flask web app
├── ingest/                     # Loaders, chunker, embedder, index builder
├── rag/                        # Retriever, prompt, generator, guardrails
├── corpus/                     # 15 policy documents (md, html, pdf, txt)
│   ├── md/   html/   pdf/   txt/
├── eval/                       # Eval set, harness, methodology
│   ├── eval_set.json
│   ├── EVAL_METHODOLOGY.md
│   └── judge_prompt.txt
├── tests/                      # Pytest suite
├── scripts/
│   └── verify_corpus.py        # Standalone corpus sanity check (no langchain needed)
├── docs/
│   └── corpus-manifest.md      # Corpus inventory and design notes
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── requirements.txt
├── config.py                   # Single source of truth for settings
├── .env.example
└── README.md
```

## Configuration

All settings live in `config.py` and are overridable via environment variables. See `.env.example` for the full list. Common ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | (required) | API key for the LLM |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters of overlap between adjacent chunks |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved per query |
| `USE_MMR` | `true` | Use Maximal Marginal Relevance for diversity |
| `USE_RERANKER` | `false` | Enable cross-encoder reranking |

## Design choices

See [`docs/design-and-evaluation.md`](docs/design-and-evaluation.md) for detailed justifications of:

- Why BGE-small for embeddings (free, local, strong MTEB scores, no rate limits)
- Why Chroma for the vector store (lightweight, embedded, no infra)
- Why hybrid markdown-headers + recursive chunking
- Why Groq llama-3.3-70b for generation (fast, capable, free tier)
- Chunk size and `k` ablation results

## AI tooling used

See [`docs/ai-tooling.md`](docs/ai-tooling.md) for how AI tools were used during development.

## License

Source code: MIT. Corpus documents: original work, free to use and modify.
