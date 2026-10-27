# AI Agent Development Guide

Guidelines for AI agents working on this production-grade Agentic RAG system.

## Project Overview

A production-ready RAG platform for arXiv CS.AI papers, built with:

- **FastAPI** — REST API with dependency injection
- **LangGraph** — Agentic workflows (guardrails, grading, query rewrite)
- **OpenSearch** — Hybrid retrieval (BM25 + vector + RRF)
- **PostgreSQL** — Paper metadata and content
- **OpenAI** — Direct LLM API (Chat Completions)
- **Langfuse** — RAG pipeline observability
- **Apache Airflow** — Ingestion orchestration

Read [docs/ARTICLE.md](docs/ARTICLE.md) for architectural reasoning before making structural changes.

---

## Code Organization: Domain Structure, Not Technical Layers

**Organize code by business capability (domain), not by technical role.**

### Do this (domain-oriented)

Group everything a capability needs in one place:

```
src/
├── api/                          # Thin HTTP adapters only
├── agents/                       # LangGraph agent bounded contexts
│   ├── fusionsearch/             # Agentic RAG graph, nodes, tools, schemas
│   ├── knowledgerouter/          # Multi-source routing agent
│   └── texttosql/                # SQL agent
└── domain/                       # Shared infrastructure & business domains
    ├── opensearch/               # Index config, query builder, client
    ├── llm/                      # LLMClient protocol + factory
    ├── jinaai/                   # Embeddings + reranker
    ├── indexing/                 # Chunking, contextualization
    ├── arxiv_ingestion/          # Fetch, parse, index pipeline
    ├── paper/                    # Persistence models + repository
    ├── agent_fault_tolerance/    # Retry, timeout, error-handler policies
    └── ...
```

Each domain folder typically contains its own `client.py`, `factory.py`, `config.py`, `schemas.py`, and `exceptions.py` as needed. Colocate logic with the capability it serves.

### Do not do this (technical-layer layout)

Avoid splitting the same feature across horizontal layers:

```
# ❌ Anti-pattern — do not introduce or extend this structure
src/
├── controllers/
├── services/
├── repositories/
├── models/
└── schemas/
```

A new retrieval feature belongs under `domain/opensearch/` (or the relevant agent under `agents/`), not spread across `services/` and `repositories/`.

### Where new code goes

| What you're adding | Location |
|--------------------|----------|
| New HTTP endpoint | `src/api/` — delegate to domain or agent services |
| New LangGraph agent or node | `src/agents/<agent-name>/` |
| New external integration | `src/domain/<integration>/` with `make_*` factory |
| Cross-cutting policy (retry, timeout) | `src/domain/agent_fault_tolerance/` |
| Agent-specific schemas/prompts | Inside the agent's folder, not a global `schemas/` tree |
| Unit tests | Mirror source layout under `tests/unit/domain/` or `tests/api/` |

The `api/` layer should stay thin: validate input, call a service, map the response. Business logic lives in `domain/` and `agents/`.

---

## Architectural Patterns

### Factory functions + dependency injection

- Use `make_*` factory functions to construct services (e.g. `make_opensearch_client`, `make_agentic_rag_service`).
- Wire dependencies in `src/main.py` lifespan → `app.state`.
- Expose FastAPI dependencies via typed aliases in `src/dependencies.py` (e.g. `AgenticRAGDep`).

### LLM abstraction

- Domain code depends on `LLMClient` (`src/domain/llm/protocol.py`), not a provider SDK.
- LLM access is via LangChain `BaseChatModel` from `make_chat_model()` in `src/platform/llm/`.
- Shared RAG prompt logic lives in `src/domain/llm/rag.py`.

### Configuration

- Pydantic Settings with nested env vars and `__` delimiter (e.g. `OPENSEARCH__HOST`, `CHUNKING__CHUNK_SIZE`).
- Settings classes are frozen — config does not mutate at runtime.
- Invalid config must fail at startup, not on first request.

### LangGraph agents

- One folder per agent under `src/agents/`.
- Typical files: `graph.py`, `state.py`, `config.py`, `factory.py`, `schemas.py`, `prompts.py`, `nodes/`.
- Fault-tolerance policies come from `src/domain/agent_fault_tolerance/`.

---

## Code Style

- **Python 3.12** — `requires-python = ">=3.12,<3.13"`
- **Type hints** on all function signatures
- **Pydantic v2** models for request/response validation and agent state
- **Imports at the top of the file** — never inside functions (except where circular imports are unavoidable and already established)
- **Ruff** for linting/formatting — line length 130
- Match existing naming: `snake_case` files, `make_*` factories, domain-specific exception classes
- Keep changes minimal and scoped — reuse existing abstractions before adding new ones

---

## Testing

```bash
make test          # or: uv run pytest
make test-cov      # with coverage
make lint          # ruff + mypy
```

- Unit tests mirror domain layout: `tests/unit/domain/<domain>/`
- API tests live in `tests/api/routers/`
- Use existing fixtures in `tests/conftest.py` and domain-specific `conftest.py` files; retrieval mocks shared by agent and rerank tests live in `tests/unit/domain/conftest.py`
- Do not declare `pytest_plugins` in a non-root `conftest.py`: pytest aborts collection of the whole suite
- `tests/integration/pdf_parser/` runs the real Docling pipeline on `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` (~2 min on CPU, downloads Docling models on the first run) and is part of `make test`; run it alone with `uv run pytest tests/integration/pdf_parser`
- Add tests only when they cover meaningful behavior — avoid trivial assertions

The `evals/` directory is part of the main project. Evals use the root virtualenv and `.env`, and can import from `src` like tests. Run evals from the repo root (`make eval-quick`, `make rag-eval`).

---

## Essential Commands

```bash
cp .env.example .env    # configure before first run
uv sync                 # install dependencies
make start              # docker compose up --build -d
make health             # verify services
make test               # run pytest
make format             # ruff format
make lint               # ruff check + mypy
```

Key env vars: `JINA_API_KEY` (required for hybrid search), `OPENAI_API_KEY`, `LANGFUSE__*` (optional tracing).

---

## When Making Changes

1. Read the existing implementation in the relevant domain or agent folder first.
2. Place new code in the correct **domain** or **agent** folder — do not create technical-layer directories.
3. Follow the `make_*` factory + FastAPI DI pattern.
4. Keep `api/` handlers thin.
5. Use `LLMClient` protocol for any new LLM usage.
6. Run `make lint` and `make test` before finishing.
7. Update tests in the mirrored path under `tests/`.

---

## Key References

| Resource | Path |
|----------|------|
| Architecture deep dive | [docs/ARTICLE.md](docs/ARTICLE.md) |
| Onboarding & service access | [docs/onboarding.md](docs/onboarding.md) |
| OpenSearch retrieval layer | [docs/opensearch-search.md](docs/opensearch-search.md) |
| Evals framework | [evals/README.md](evals/README.md) |
| Interactive API docs | http://localhost:8000/docs (when stack is running) |
