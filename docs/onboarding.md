# Onboarding Guide

This guide gets you set up and productive in the codebase: how to run the stack, where the services live, how to inspect them, and what breaks first. The design philosophy is deliberate: **keyword search first, vectors second, LLM last, observability and agents on top**. That ordering mirrors how strong production retrieval systems are actually built.

For the architectural reasoning behind each layer, read [Anatomy of a Production-Grade Agentic RAG System](ARTICLE.md).

---

## 1. What You're Building

You're building a production-grade RAG system for arXiv CS.AI papers with agentic extensions. The API is a FastAPI server that exposes RAG endpoints. The UI is a Telegram bot that routes to the API. 
---

## 2. Tooling Stack

### Runtime & Dev Tools

| Tool | Role |
|------|------|
| **Python 3.12** | Application runtime |
| **UV** | Dependency management (`uv sync`, `uv run`) |
| **Docker Compose** | Multi-service orchestration (`compose.yml`) |
| **Ruff + MyPy + Pytest** | Lint, type-check, test (`make lint`, `make test`) |

### Infrastructure Services (Docker)

| Service | Port | Purpose |
|---------|------|---------|
| **FastAPI (`api`)** | 8000 | REST API |
| **PostgreSQL** | 5412 | Paper metadata + parsed content |
| **OpenSearch** | 9200 | BM25 + vector search |
| **OpenSearch Dashboards** | 5601 | Search UI / debugging |
| **Apache Airflow** | 8080 | Scheduled ingestion DAG |

**Langfuse Cloud** — hosted tracing dashboard at [cloud.langfuse.com](https://cloud.langfuse.com); no local Docker services required.

### Application Libraries (notable)

- **FastAPI + Pydantic Settings** — API + typed config via `.env`
- **SQLAlchemy** — PostgreSQL ORM
- **Docling** — Scientific PDF parsing; `parse_pdf(pdf_path, markdown_output_dir=...)` can also export `{pdf_stem}.md` plus `{pdf_stem}_assets/` (`figure_NNN.png`, `table_NNN.csv`) to inspect what was extracted (not used by ingestion). Exporting the same PDF to the same folder replaces the previous `.md` and `figure_*.png`/`table_*.csv`; a figure without an image shows as `_<caption> (image unavailable)_`
- **opensearch-py** — Search client
- **Jina AI** — 1024-dim embeddings and cross-encoder reranking
- **Langfuse SDK** — RAG tracing
- **LangGraph + LangChain** — Agent orchestration

---

## 3. Getting In — Setup & Access

### Prerequisites

- Docker Desktop (8GB+ RAM, ~20GB disk)
- Python 3.12 (`requires-python = ">=3.12,<3.13"`)
- [UV](https://docs.astral.sh/uv/getting-started/installation/)

### Bootstrap

```bash
git clone <repo-url>
cd production-agentic-rag-course

cp .env.example .env
# Edit .env — at minimum set JINA_API_KEY, optionally Langfuse keys

uv sync
docker compose up --build -d
```

### Run the API locally

The `api` service in `compose.yml` is commented out — the API runs on the host, not in Docker. From the project root:


```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API listens on http://localhost:8000.

Wait 2–3 minutes for health checks, then verify:

```bash
curl http://localhost:8000/api/v1/health
make health   # checks API, OpenSearch, Airflow
```

Set `OPENAI_API_KEY` in `.env` before running generation (see [OpenAI](#openai-llm)).

### Service URLs

| URL | What |
|-----|------|
| http://localhost:8000/docs | Swagger API |
| http://localhost:8080 | Airflow web UI |
| http://localhost:5601 | OpenSearch Dashboards |
| https://cloud.langfuse.com | Langfuse Cloud tracing dashboard |

### Accessing PostgreSQL, OpenSearch, Airflow & Langfuse

Papers are persisted in three layers: **PostgreSQL** (metadata + parsed text), **PDF cache** (`./data/arxiv_pdfs/` inside the Airflow container), and **OpenSearch** (chunked + embedded documents for search). **Langfuse** records end-to-end RAG traces for debugging and performance analysis.

#### PostgreSQL

Primary store for paper metadata and parsed PDF content (`papers` table — see `src/domain/paper/model.py`).

| Setting | Value |
|---------|-------|
| Host | `localhost` |
| Port | `5412` |
| Database | `rag_db` |
| Username | `rag_user` |
| Password | `rag_password` |

**Connection string** (for local API / GUI tools):

```
postgresql+psycopg2://rag_user:rag_password@localhost:5412/rag_db
```

**CLI access:**

```bash
docker exec -it rag-postgres psql -U rag_user -d rag_db
```

**Useful queries:**

```sql
-- Paper counts
SELECT COUNT(*) FROM papers;

-- Recent metadata
SELECT arxiv_id, title, pdf_processed, created_at
FROM papers
ORDER BY created_at DESC
LIMIT 10;

-- Full record for one paper
SELECT * FROM papers WHERE arxiv_id = '2606.05522v1';

-- Parsed content (populated after PDF processing succeeds)
SELECT arxiv_id, title, left(raw_text, 500) AS preview, sections
FROM papers
WHERE raw_text IS NOT NULL
LIMIT 5;
```

**GUI:** connect with DBeaver, TablePlus, or any PostgreSQL client using the settings above.

#### OpenSearch

Search index for BM25 and hybrid retrieval. Chunk documents live in the **`arxiv-papers-chunks`** index (`{OPENSEARCH__INDEX_NAME}-{OPENSEARCH__CHUNK_INDEX_SUFFIX}`).

| Setting | Value |
|---------|-------|
| API | http://localhost:9200 |
| Dashboards UI | http://localhost:5601 |
| Security | Disabled for local dev (`DISABLE_SECURITY_PLUGIN=true`) |

**CLI checks:**

```bash
# Cluster health
curl http://localhost:9200/_cluster/health?pretty

# Indexed chunk count
curl http://localhost:9200/arxiv-papers-chunks/_count

# Sample chunks
curl -X POST "http://localhost:9200/arxiv-papers-chunks/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{"size": 3, "query": {"match_all": {}}, "_source": { "excludes": ["embedding"] }}'

# Chunks for a specific paper
curl -X POST "http://localhost:9200/arxiv-papers-chunks/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{"query": {"term": {"arxiv_id": "2606.05522v1"}}}'
  

# Chunks for a specific paper in the devtools
GET arxiv-papers-chunks/_search
  {
    "query": {
      "term": { "arxiv_id": "2607.14408v1" }
    },
    "size": 10,
    "sort": [{ "chunk_index": "asc" }],
    "_source": ["arxiv_id", "title", "chunk_index", "chunk_text"]
  }
```

**Dashboards:** open http://localhost:5601 → **Dev Tools** → run the same JSON queries. Use **Stack Management → Index Management** to inspect `arxiv-papers-chunks`.
[Discover dashboard](http://localhost:5601/app/data-explorer/discover#?_a=(discover:(columns:!(authors,chunk_text),isDirty:!f,sort:!()),metadata:(indexPattern:'220ceb30-838f-11f1-b01d-e7092c63ec12',view:discover))&_q=(filters:!(),query:(language:kuery,query:''))&_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now-15w,to:now))) 

#### Airflow

Orchestrates the ingestion DAG (`arxiv_paper_ingestion`) that fetches papers, parses PDFs, writes to PostgreSQL, and indexes chunks in OpenSearch.

| Setting | Value |
|---------|-------|
| URL | http://localhost:8080 |
| Username | `admin` |
| Password | `admin` |

Credentials are created by `airflow/entrypoint.sh` on first startup. If login fails (e.g. after a DB volume reset), reset the password:

```bash
docker exec -it rag-airflow airflow users reset-password --username admin --password admin
```

**Common tasks:**

- Toggle the `arxiv_paper_ingestion` DAG on, then click **Trigger DAG** to run ingestion manually.
- Open a task instance → **Log** to debug fetch, parse, or indexing failures.
- PDFs are persisted on the host at `./data/arxiv_pdfs/` (mounted into the container at `/opt/airflow/data/arxiv_pdfs/`):

```bash
ls ./data/arxiv_pdfs/
# or from inside the container:
docker exec rag-airflow ls /opt/airflow/data/arxiv_pdfs/
```

**Health check:**

```bash
curl http://localhost:8080/health
```

#### Langfuse

[Langfuse Cloud](https://cloud.langfuse.com) for RAG pipeline tracing. Traces are sent from the RAG API via the Langfuse Python SDK — there is no self-hosted Langfuse stack in `compose.yml`.

**Setup:**

1. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com) (or [us.cloud.langfuse.com](https://us.cloud.langfuse.com) for US region)
2. Create a project (e.g. `Agentic RAG`)
3. Go to **Settings → API Keys** and create a key pair

**SDK configuration** (required for traces to appear):

Add these to `.env` using the **double-underscore** prefix (`LANGFUSE__*`) that `src/config.py` reads:

```bash
LANGFUSE__ENABLED=true
LANGFUSE__HOST=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com for US
LANGFUSE__PUBLIC_KEY=pk-lf-...
LANGFUSE__SECRET_KEY=sk-lf-...
LANGFUSE__DEBUG=true   # optional — verbose SDK logging in API output
```

Restart the API after updating `.env`:

```bash
uv run python src/main.py
```

**Generate traces** — send a request that runs the full pipeline:

```bash
# Standard RAG
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 3}'

# Agentic RAG
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What are attention mechanisms?", "top_k": 3}'
```

**View traces in the dashboard:**

1. Open [cloud.langfuse.com](https://cloud.langfuse.com) and sign in
2. Select your project
3. Go to **Tracing → Traces** in the left sidebar
4. Click a trace to open the detail view (spans, inputs/outputs, latency, token usage)

Traces appear within a few seconds of each request (`LANGFUSE__FLUSH_INTERVAL` defaults to 1s).

**What to look for by endpoint:**

| Endpoint | Top-level trace | Child spans |
|----------|-----------------|-------------|
| `/api/v1/ask` | RAG request | `query_embedding`, `search_retrieval`, `prompt_construction`, `llm_generation` |
| `/api/v1/ask-agentic` | `agentic_rag_request` | `guardrail_validation`, `document_retrieval_initiation`, `document_grading`, `query_rewriting`, `answer_generation` |

**Useful dashboard filters:**

- **Time range** — narrow to the last hour after a test run
- **Latency** — inspect end-to-end request duration per span
- **User ID** — standard RAG requests use `api_user`
- **Search** — filter by query text in trace input/metadata

**If traces are missing:**

- Confirm `LANGFUSE__PUBLIC_KEY` and `LANGFUSE__SECRET_KEY` are set (double underscore)
- Verify `LANGFUSE__HOST` matches your Langfuse Cloud region
- Restart the API after changing `.env`
- Send a fresh query after restarting the API
- Enable `LANGFUSE__DEBUG=true` and check API logs for SDK errors

#### OpenAI (LLM)

All generation and agent calls use the [OpenAI API](https://platform.openai.com/docs/api-reference) directly via `langchain-openai`.

**Application configuration:**

```bash
OPENAI_API_KEY=your_openai_api_key_here
AGENT_MODEL=gpt-5.6-luna
LLM_FALLBACK_MODELS=gpt-5.6-luna,gpt-4o-mini
LLM_REQUEST_TIMEOUT=300
REASONING_EFFORT=low
```

Optional: `OPENAI_BASE_URL` to point at an OpenAI-compatible proxy (default `https://api.openai.com/v1`).

**Code layout:**

| Module | Role |
|--------|------|
| `src/platform/llm/` | `make_chat_model()` — builds LangChain `ChatOpenAI` with fallbacks |

**Health check:** `/api/v1/health` reports OpenAI status under the `openai` service key.

### Project Layout

```
src/
├── main.py                    # FastAPI lifespan, service wiring
├── config.py                  # Pydantic settings (nested env: ARXIV__, OPENSEARCH__, etc.)
├── dependencies.py            # FastAPI dependency annotations
├── api/                       # HTTP endpoints
└── domain/                    # Business logic (factory pattern throughout)
    ├── agents/                # LangGraph agents
    │   ├── fusionsearch/      # Agentic RAG graph + nodes
    │   ├── texttosql/         # SQL agent
    │   ├── knowledgerouter/   # Multi-source router
    │   └── fault_tolerance/   # Retry, timeout, error-handler policies
    ├── arxiv/ + arxiv_ingestion/   # Fetching and ingestion orchestration
    ├── indexing/              # Chunking, contextualization, hybrid indexing
    ├── opensearch/            # Index config, query builder, search client
    ├── jinaai/                # Embeddings + reranker clients
    ├── platform/llm/   # OpenAI ChatOpenAI factory
    ├── langfuse/              # Tracing
    ├── db/ + paper/           # Persistence and repositories
    └── pdf_parser/            # Docling wrapper + optional Markdown export (markdown_export.py)
airflow/dags/                  # Ingestion pipeline
evals/                         # Langfuse trace + Ragas retrieval evaluation
compose.yml                    # Full stack definition
```

**Architectural pattern:** Factory functions (`make_*`) + FastAPI dependency injection + lifespan initialization in `main.py`. Services are attached to `app.state` and injected via `src/dependencies.py`.

---

## 4. End-to-End Data Flow

```mermaid
flowchart LR
    A[Airflow DAG] --> B[arXiv API]
    B --> C[PDF Download]
    C --> D[Docling Parse]
    D --> E[(PostgreSQL)]
    E --> F[Chunk + Embed]
    F --> G[(OpenSearch)]

    H[User Query] --> I{Cache?}
    I -->|Hit| J[Response]
    I -->|Miss| K[Hybrid Search]
    K --> G
    K --> L[OpenAI LLM]
    L --> M[Cache Store]
    M --> N[Langfuse Trace]
    N --> J
```

---

## 5. API Reference (Complete)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health |
| `/api/v1/hybrid-search/` | POST | BM25 / vector / hybrid search |
| `/api/v1/ask` | POST | Standard RAG |
| `/api/v1/stream` | POST | Streaming RAG (SSE) |
| `/api/v1/ask-agentic` | POST | Agentic RAG |
| `/api/v1/ask-sql` | POST | Text-to-SQL agent over PostgreSQL |
| `/api/v1/ask-router` | POST | Knowledge router across retrieval agents |
| `/api/v1/feedback` | POST | User feedback on answers |

Full interactive docs: http://localhost:8000/docs

---

## 6. Configuration Cheat Sheet

| Variable | Required | Purpose |
|----------|----------|---------|
| `JINA_API_KEY` | Yes | Hybrid/vector search and reranking |
| `LANGFUSE__PUBLIC_KEY` / `SECRET_KEY` | Optional | Tracing (create in Langfuse Cloud → Settings → API Keys) |
| `LANGFUSE__HOST` | Optional | Langfuse Cloud URL (`https://cloud.langfuse.com` or `https://us.cloud.langfuse.com`) |
| `OPENAI_API_KEY` | Yes | OpenAI API key for all LLM calls |
| `LLM_FALLBACK_MODELS` | Optional | Comma-separated fallback models (default `gpt-5.6-luna,gpt-4o-mini`) |
| `LLM_REQUEST_TIMEOUT` | Optional | Generation timeout in seconds (default `300`) |
| `AGENT_MODEL` | Optional | Model used by agent graphs (default `gpt-5.6-luna`) |
| `REASONING_EFFORT` | Optional | Reasoning effort for GPT-5.x models (default `low`) |
| `CHUNKING__*` | Optional | Chunk size, overlap, and contextualization tuning |
| `REDIS__TTL_HOURS` | Optional | Cache expiration (default `6`) |

All defaults work for local dev except external API keys. See [.env.example](../.env.example) for the full list.

---

## 7. Recommended Onboarding Path

0. **Read first:** [Anatomy of a Production-Grade Agentic RAG System](ARTICLE.md) for the architectural reasoning
1. **Day 1:** `docker compose up`, then explore `compose.yml` and the `src/main.py` lifespan
2. **Day 2:** Trace ingestion: Airflow DAG → `MetadataFetcher` → PostgreSQL. Trigger the DAG manually
3. **Day 3:** Search layer: read `OpenSearchClient.search_unified()`, test all three modes via API
4. **Day 4:** RAG path: follow `ask.py` end-to-end, test streaming
5. **Day 5:** Ops layer: explore Langfuse traces and latency breakdowns
6. **Day 6:** Agents: step through the LangGraph nodes, test `/ask-agentic` with guardrail and rewrite scenarios

### Key Files to Read in Order

```
compose.yml → src/config.py → src/main.py → src/dependencies.py
→ src/domain/arxiv_ingestion/ (ingestion)
→ src/domain/indexing/text_chunker.py (chunking)
→ src/domain/opensearch/client.py (search)
→ src/domain/jinaai/jina_reranker_client.py (reranking)
→ src/platform/llm/factory.py (LLM client factory)
→ src/platform/llm/chat_model.py (ChatOpenAI factory)
→ src/api/ask.py (RAG)
→ src/platform/langfuse/client.py (tracing)
→ src/agents/fusionsearch/agentic_rag.py (agentic RAG graph)
→ src/domain/agents/knowledgerouter/graph.py (multi-source routing)
```

---

## 8. Essential Commands

### Makefile (recommended)

```bash
make help      # List all commands
make start     # docker compose up --build -d
make stop      # docker compose down
make health    # Check all services
make test      # Run pytest
make lint      # Ruff + MyPy
make clean     # docker compose down -v + prune
```

### Direct Commands

```bash
docker compose ps
docker compose logs -f api
uv run pytest
```

---

## 9. Common Issues

| Symptom | Fix |
|---------|-----|
| Services won't start | Wait 2–3 min; check `docker compose logs <service>` |
| Hybrid search returns BM25 only | Verify `JINA_API_KEY`; check API logs for embedding errors |
| OpenAI 401 / auth errors | Verify `OPENAI_API_KEY` in `.env` and restart the API |
| OpenAI connection / timeout errors | Check network; increase `LLM_REQUEST_TIMEOUT` if needed |
| Empty search results | Run Airflow DAG to ingest + index; check `curl localhost:9200/arxiv-papers-chunks/_count` |
| Langfuse empty | Verify `LANGFUSE__*` keys from [Langfuse Cloud](https://cloud.langfuse.com); confirm `LANGFUSE__HOST` matches your region; restart API; send a new query; enable `LANGFUSE__DEBUG=true` |
| Port conflicts | Stop services on 8000, 8080, 5412, 9200 |
| Docling fails with `CUDA error: no kernel image is available` | The GPU is too old for the installed torch build (e.g. compute capability 6.1). Run on CPU: `CUDA_VISIBLE_DEVICES="" <command>` or `DOCLING_DEVICE=cpu` (works for the API, `PDFParserService` and Airflow); `DoclingParser(..., accelerator_device="cpu")` only applies when you build the parser yourself |

Full reset:

```bash
docker compose down -v && docker compose up --build -d
```

---

## 10. External Resources

### In-repo deep dives

- [Anatomy of a Production-Grade Agentic RAG System](ARTICLE.md) — architecture, trade-offs, and failure design
- [How OpenSearch Powers Retrieval](opensearch-search.md) — index mapping, query construction, RRF fusion

### Blog series

| Topic | Post |
|-------|------|
| Infrastructure | [The Infrastructure That Powers RAG Systems](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) |
| Data ingestion | [Bringing Your RAG System to Life](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life) |
| BM25 search | [The Search Foundation Every RAG System Needs](https://jamwithai.substack.com/p/the-search-foundation-every-rag-system) |
| Chunking & hybrid search | [Chunking Strategies and Hybrid RAG](https://jamwithai.substack.com/p/chunking-strategies-and-hybrid-rag) |
| Full RAG pipeline | [The Complete RAG System](https://jamwithai.substack.com/p/the-complete-rag-system) |
| Monitoring & caching | [Production-Ready RAG: Monitoring and Caching](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and) |
| Agentic RAG | [Agentic RAG with LangGraph](https://jamwithai.substack.com/p/agentic-rag-with-langgraph-and-telegram) |
