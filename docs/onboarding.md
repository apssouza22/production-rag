# arXiv Paper Curator — Senior Engineer Onboarding Guide

This guide covers the **7-week, production-oriented RAG course** that builds an academic research assistant end-to-end. The design philosophy is deliberate: **keyword search first, vectors second, LLM last, observability and agents on top**. That ordering mirrors how strong production retrieval systems are actually built.

---

## 1. What You're Building

**System:** arXiv Paper Curator — fetches CS.AI papers from arXiv, parses PDFs, indexes chunks in OpenSearch, and answers research questions via RAG (with optional agentic reasoning).

**Evolution by week:**

| Week | Layer Added | Core Capability |
|------|-------------|-----------------|
| 1 | Infrastructure | Docker stack, FastAPI, health checks |
| 2 | Ingestion | arXiv → PDF → PostgreSQL |
| 3 | Keyword retrieval | BM25 via OpenSearch |
| 4 | Semantic retrieval | Chunking + embeddings + hybrid (RRF) |
| 5 | Generation | Ollama LLM |
| 6 | Production ops | Langfuse tracing + Redis cache |
| 7 | Intelligence | LangGraph agentic RAG |

---

## 2. Tooling Stack

### Runtime & Dev Tools

| Tool | Role |
|------|------|
| **Python 3.12** | Application runtime |
| **UV** | Dependency management (`uv sync`, `uv run`) |
| **Docker Compose** | Multi-service orchestration (`compose.yml`) |
| **Ruff + MyPy + Pytest** | Lint, type-check, test (`make lint`, `make test`) |
| **Jupyter** | Weekly hands-on notebooks |

### Infrastructure Services (Docker)

| Service | Port | Purpose |
|---------|------|---------|
| **FastAPI (`api`)** | 8000 | REST API |
| **PostgreSQL** | 5412 | Paper metadata + parsed content |
| **OpenSearch** | 9200 | BM25 + vector search |
| **OpenSearch Dashboards** | 5601 | Search UI / debugging |
| **Apache Airflow** | 8080 | Scheduled ingestion DAG |
| **Ollama** | 11434 | Local LLM inference |
| **Bifrost** | 8090 | Optional LLM gateway (OpenAI-compatible API → Ollama) |
| **Redis** | 6379 | RAG response cache |

**Langfuse Cloud** (Week 6+) — hosted tracing dashboard at [cloud.langfuse.com](https://cloud.langfuse.com); no local Docker services required.

### Application Libraries (notable)

- **FastAPI + Pydantic Settings** — API + typed config via `.env`
- **SQLAlchemy** — PostgreSQL ORM
- **Docling** — Scientific PDF parsing
- **opensearch-py** — Search client
- **Jina AI** — 1024-dim embeddings (Week 4+)
- **Langfuse SDK** — RAG tracing (Week 6+)
- **LangGraph + LangChain** — Agent orchestration (Week 7)

---

## 3. Getting In — Setup & Access

### Prerequisites

- Docker Desktop (8GB+ RAM, ~20GB disk)
- Python 3.12+
- [UV](https://docs.astral.sh/uv/getting-started/installation/)

### Bootstrap

```bash
git clone <repo-url>
cd production-agentic-rag-course

cp .env.example .env
# Edit .env — at minimum set JINA_API_KEY (Week 4+), optionally Langfuse keys

uv sync
docker compose up --build -d
```

### Run the API locally

The `api` service in `compose.yml` is commented out — the API runs on the host, not in Docker. From the project root:

```bash
uv run python src/main.py
```

Or with auto-reload during development:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API listens on http://localhost:8000.

Wait 2–3 minutes for health checks, then verify:

```bash
curl http://localhost:8000/api/v1/health
make health   # checks API, OpenSearch, Airflow, Ollama, Bifrost
```

Before Week 5, pull an Ollama model (see [Ollama](#ollama)):

```bash
docker exec rag-ollama ollama pull llama3.2:1b
```

### Service URLs

| URL | What |
|-----|------|
| http://localhost:8000/docs | Swagger API |
| http://localhost:8080 | Airflow web UI |
| http://localhost:5601 | OpenSearch Dashboards |
| https://cloud.langfuse.com | Langfuse Cloud tracing dashboard (Week 6+) |
| http://localhost:11434 | Ollama API |
| http://localhost:8090 | Bifrost gateway (Web UI + `/v1` API) |

### Accessing PostgreSQL, OpenSearch, Airflow & Langfuse

Papers are persisted in three layers: **PostgreSQL** (metadata + parsed text), **PDF cache** (`./data/arxiv_pdfs/` inside the Airflow container), and **OpenSearch** (chunked + embedded documents for search). **Langfuse** (Week 6+) records end-to-end RAG traces for debugging and performance analysis.

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
  -d '{"size": 3, "query": {"match_all": {}}}'

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
- PDFs are cached at `./data/arxiv_pdfs/` inside the container (`/opt/airflow/data/arxiv_pdfs/`):

```bash
docker exec rag-airflow ls ./data/arxiv_pdfs/
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

**Generate traces** — send a request that runs the full pipeline (cache hits return early and produce minimal tracing):

```bash
# Standard RAG (Week 5–6)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 3}'

# Agentic RAG (Week 7)
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
- **Latency** — compare cache miss (~15–20s) vs cache hit (~100ms)
- **User ID** — standard RAG requests use `api_user`
- **Search** — filter by query text in trace input/metadata

**If traces are missing:**

- Confirm `LANGFUSE__PUBLIC_KEY` and `LANGFUSE__SECRET_KEY` are set (double underscore)
- Verify `LANGFUSE__HOST` matches your Langfuse Cloud region
- Restart the API after changing `.env`
- Send a non-cached query (identical queries may hit Redis cache and produce minimal tracing)
- Enable `LANGFUSE__DEBUG=true` and check API logs for SDK errors

#### Ollama

Local LLM inference runs in the **`rag-ollama`** container (`ollama/ollama:0.11.2`). Models are **not** bundled with the image — you pull them after the stack is up. Week 1 health checks only verify the Ollama service is running; a model is required from **Week 5** onward (RAG generation, agentic flows).

| Setting | Value |
|---------|-------|
| API | http://localhost:11434 |
| Container | `rag-ollama` |
| Default model | `llama3.2:1b` (`OLLAMA_MODEL` in `.env`) |
| Timeout | 300s (`OLLAMA_TIMEOUT`) |

**Recommended models** (pull only what you need — sizes are approximate):

| Model | Size | Speed | Use case |
|-------|------|-------|----------|
| `llama3.2:1b` | ~1.3 GB | Fastest | **Course default** — dev, tests |
| `llama3.2:3b` | ~2.0 GB | Balanced | Better answers, still reasonable on laptop |
| `llama3.1:8b` | ~4.7 GB | Slower | Higher quality when you have RAM/VRAM |
| `qwen2.5:7b` | ~4.7 GB | Slower | Alternative multilingual model |

API requests can override the default via the `model` field on `/api/v1/ask` and `/api/v1/ask-agentic`.

**Install the default model:**

```bash
docker exec rag-ollama ollama pull llama3.2:1b
```

**Alternative — HTTP API** (no `docker exec`; useful if Ollama runs outside Docker):

```bash
curl -X POST http://localhost:11434/api/pull -d '{"name":"llama3.2:1b"}'
```

**Verify installation:**

```bash
# List downloaded models
docker exec rag-ollama ollama list

# Quick generation test
docker exec rag-ollama ollama run llama3.2:1b "What is machine learning in one sentence?"

# See which model is loaded in memory (empty until first request)
docker exec rag-ollama ollama ps
```

**Application configuration** — when the API runs on the host (default), point it at the published port:

```bash
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=300
```

To route through Bifrost instead, see [Bifrost](#bifrost-optional-llm-gateway).

If the API service in `compose.yml` is enabled instead, use `OLLAMA_HOST=http://ollama:11434` (container DNS).

**Pull additional models** (optional):

```bash
docker exec rag-ollama ollama pull llama3.2:3b
docker exec rag-ollama ollama pull llama3.1:8b
docker exec rag-ollama ollama pull qwen2.5:7b
```

Models persist in the `ollama_data` Docker volume across restarts.

#### Bifrost (optional LLM gateway)

[Bifrost](https://docs.getbifrost.ai/) is an optional OpenAI-compatible gateway in front of Ollama. The RAG API can talk to Ollama **directly** or **through Bifrost** — switch with `LLM_PROVIDER` in `.env`.

| Setting | Value |
|---------|-------|
| Web UI + API | http://localhost:8090 |
| Container | `rag-bifrost` |
| LangChain endpoint | `http://localhost:8090/langchain` |
| Config file | `bifrost/config.json` |
| Data volume | `./bifrost/` (SQLite config + logs) |

**When to use Bifrost:**

- Unified OpenAI-compatible API for LangChain clients
- Request logging, metrics, and provider management via the Web UI
- A stepping stone toward multi-provider routing in production

**Start Bifrost:**

```bash
docker compose up -d bifrost ollama
```

Bifrost depends on Ollama being healthy. Port **8090** is used because Airflow already occupies **8080**.

**Verify Bifrost:**

```bash
curl http://localhost:8090/health
curl http://localhost:8090/v1/models

curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello, Bifrost!"}]
  }'
```

**Application configuration** — host API talking to Bifrost on the published port:

```bash
LLM_PROVIDER=bifrost
BIFROST_HOST=http://localhost:8090
BIFROST_API_KEY=dummy-key
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=300
```

If the API runs inside Docker Compose instead, use `BIFROST_HOST=http://bifrost:8080`.

**Switch back to direct Ollama:**

```bash
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

**Code layout:**

| Module | Role |
|--------|------|
| `src/domain/llm/` | `LLMClient` protocol + `make_llm_client()` factory |
| `src/domain/ollama/` | Direct Ollama client (`ChatOllama`, native `/api/generate`) |
| `src/domain/bifrost/` | Bifrost client (`ChatOpenAI` via `/langchain`) |

Shared RAG prompt logic lives in `src/domain/llm/rag.py`; both clients implement the same interface.

**Bifrost provider config** (`bifrost/config.json`) points at the Ollama container and allows private Docker network access:

```json
{
  "providers": {
    "ollama": {
      "network_config": { "allow_private_network": true },
      "keys": [{
        "name": "ollama-docker",
        "value": "",
        "models": ["*"],
        "weight": 1.0,
        "ollama_key_config": { "url": "http://ollama:11434" }
      }]
    }
  }
}
```

After editing `config.json`, restart Bifrost. If the SQLite config DB was already seeded, remove `bifrost/config.db` and restart so the file changes are picked up.

**Health check:** `/api/v1/health` reports the active provider (`ollama` or `bifrost`) based on `LLM_PROVIDER`.

### Clone a Specific Week's Snapshot

Each week has a tagged release on the upstream repo (`week1.0` … `week7.0`). Useful if you want incremental diffs rather than the full final codebase.

```bash
git clone --branch week3.0 https://github.com/jamwithai/arxiv-paper-curator
cd arxiv-paper-curator
uv sync
docker compose down -v
docker compose up --build -d
```

### Project Layout

```
src/
├── main.py              # FastAPI lifespan, service wiring
├── config.py            # Pydantic settings (nested env: ARXIV__, OPENSEARCH__, etc.)
├── api/             # HTTP endpoints
├── domain/            # Business logic (factory pattern throughout)
│   ├── llm/           # LLMClient protocol + provider factory
│   ├── ollama/        # Direct Ollama client
│   └── bifrost/       # Bifrost gateway client
├── repositories/        # DB access
├── models/ + schemas/   # SQLAlchemy + Pydantic
airflow/dags/            # Ingestion pipeline
notebooks/week{N}/       # Guided implementation per week
compose.yml              # Full stack definition
```

**Architectural pattern:** Factory functions (`make_*`) + FastAPI dependency injection + lifespan initialization in `main.py`. Services are attached to `app.state` and injected via `src/dependencies.py`.

---

## 4. End-to-End Data Flow (Final State)

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
    K --> L[Ollama LLM]
    L --> M[Cache Store]
    M --> N[Langfuse Trace]
    N --> J
```

---

## Week 1: Infrastructure Foundation

### Objective

Stand up the full container stack and verify inter-service connectivity before writing domain logic.

### What Gets Built

- `compose.yml` — 13+ services on `rag-network`
- FastAPI app with lifespan hooks
- Health endpoint at `/api/v1/health`
- PostgreSQL schema bootstrap
- OpenSearch cluster (single-node, security disabled for dev)
- Ollama container (models pulled separately)
- Bifrost gateway container (optional; routes to Ollama on port 8090)
- Airflow with LocalExecutor

### Key Implementation Details

**Service bootstrap** happens in `src/main.py` lifespan:

- Database connection via `make_database()`
- OpenSearch client with auto index setup on startup
- All downstream services initialized: arXiv, PDF parser, embeddings, LLM client (Ollama or Bifrost), Langfuse, cache

**Configuration** is centralized in `src/config.py` using Pydantic Settings with `__` nested delimiter (e.g. `OPENSEARCH__HOST`, `CHUNKING__CHUNK_SIZE`).

### Senior-Engineer Notes

- OpenSearch starts with `DISABLE_SECURITY_PLUGIN=true` — fine for local dev, not for prod.
- Ollama has no models pre-installed; pull before Week 5 — see [Ollama](#ollama) above (`docker exec rag-ollama ollama pull llama3.2:1b`).
- Airflow mounts `./src` into the container so DAG tasks can import application services directly.

### Verification

```bash
uv run jupyter notebook notebooks/week1/week1_setup.ipynb
make status && make health
```

**Notebook:** [notebooks/week1/week1_setup.ipynb](../notebooks/week1/week1_setup.ipynb)  
**Blog:** [The Infrastructure That Powers RAG Systems](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag)

---

## Week 2: Data Ingestion Pipeline

### Objective

Automated pipeline: arXiv metadata → PDF download → structured parse → PostgreSQL storage.

### Architecture

```
MetadataFetcher (orchestrator)
  ├── ArxivClient      — rate-limited HTTP + Atom XML parsing
  ├── PDFParserService — Docling wrapper
  └── PaperRepository  — SQLAlchemy upsert
```

### Key Components

**`ArxivClient`** (`src/domain/arxiv/client.py`):

- 3-second rate limit between requests (arXiv ToS compliance)
- Concurrent downloads (default 5) with semaphore control
- PDF cache at `./data/arxiv_pdfs`
- Retry with exponential backoff

**`MetadataFetcher`** (`src/domain/arxiv_ingestion/service.py`):

- Async orchestration: fetch → download → parse → store
- Graceful degradation: metadata stored even if PDF parse fails
- Configurable concurrency for downloads vs parsing

**Airflow DAG** (`airflow/dags/arxiv_paper_ingestion.py`):

```
setup → fetch → index_hybrid → report → cleanup
```

Schedule: Mon–Fri 06:00 UTC. In Week 2 the focus is fetch + PostgreSQL; indexing is extended in later weeks.

### Data Model

Papers stored in PostgreSQL with: arxiv_id, title, authors, abstract, full_text, sections (JSON), pdf_path, categories, published_date. Upsert on arxiv_id prevents duplicates.

### Operational Characteristics

- ~20 papers/min (rate-limited)
- PDF parse: 2–5s/paper, 80–90% success on academic PDFs
- Docling limits: max 30 pages, 20MB (`PDF_PARSER__*` settings)

### Verification

```bash
docker compose up --build -d   # Required — new deps (docling)
uv run jupyter notebook notebooks/week2/week2_arxiv_integration.ipynb
# Trigger DAG manually in Airflow UI
```

**Notebook:** [notebooks/week2/week2_arxiv_integration.ipynb](../notebooks/week2/week2_arxiv_integration.ipynb)  
**Blog:** [Building Data Ingestion Pipelines for RAG](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life)

---

## Week 3: BM25 Keyword Search

### Objective

Index papers in OpenSearch and expose BM25 search — the **retrieval foundation** before vectors.

### Why This Matters

BM25 handles exact terms (paper IDs, acronyms like "BERT", "GPT"), is fast (~50ms), interpretable, and requires no external API. Most production RAG systems use keyword search as baseline or as one leg of hybrid retrieval.

### Key Components

**OpenSearch client** (`src/domain/opensearch/client.py`):

- Factory pattern via `make_opensearch_client()`
- Index auto-created on startup via `setup_indices()`
- Health check against cluster status green/yellow

**Query builder** (`src/domain/opensearch/query_builder.py`):

- Multi-field search with boosting: title (3x), abstract (2x), content (1x)
- Category filters, date ranges, fuzzy matching
- Highlighting for matched terms

### Index Design

Index name pattern: `{OPENSEARCH__INDEX_NAME}-{OPENSEARCH__CHUNK_INDEX_SUFFIX}` → `arxiv-papers-chunks` (evolved in Week 4 to chunk-level docs).

Week 3 indexes whole papers; Week 4 migrates to chunk-level indexing on the same index schema.

### Airflow Integration

The DAG's `index_papers_hybrid` task (in `src/domain/arxiv_ingestion/indexing.py`) pushes PostgreSQL content into OpenSearch after fetch.

### Verification

```bash
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer", "use_hybrid": false, "size": 5}'
```

Check OpenSearch Dashboards at `:5601` for index stats and query debugging.

**Notebook:** [notebooks/week3/week3_opensearch.ipynb](../notebooks/week3/week3_opensearch.ipynb)  
**Blog:** [The Search Foundation Every RAG System Needs](https://jamwithai.substack.com/p/the-search-foundation-every-rag-system)

---

## Week 4: Chunking & Hybrid Search

### Objective

Break papers into retrieval-sized chunks, generate embeddings, and combine BM25 + vector search via **Reciprocal Rank Fusion (RRF)**.

### Chunking Strategy

**`TextChunker`** (`src/domain/indexing/text_chunker.py`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_size` | 600 words | Target chunk size |
| `overlap_size` | 100 words | Boundary context preservation |
| `min_chunk_size` | 100 words | Drop tiny fragments |
| `section_based` | true | Respect Docling-parsed section boundaries |

Strategy:

- Sections 100–800 words → single chunk (+ title/abstract prefix)
- Small sections → merged with neighbors
- Large sections → word-based sliding window with overlap
- No sections → fallback paragraph chunking

### Embeddings

**Jina AI** (`src/domain/jinaai/jina_client.py`):

- Model: `jina-embeddings-v3`, 1024 dimensions
- Requires `JINA_API_KEY` in `.env`
- Graceful fallback to BM25-only if key missing or API fails

### Hybrid Search

**Single index** (`arxiv-papers-chunks`) supports three modes via `search_unified()`:

| Mode | Latency | Mechanism |
|------|---------|-----------|
| BM25 | ~50ms | OpenSearch `match` queries |
| Vector | ~100ms | kNN on `embedding` field |
| Hybrid | ~2–4s | RRF fusion of both rank lists |

RRF is implemented manually (OpenSearch 2.19 compatibility) — BM25 and vector queries run separately, ranks merged with reciprocal rank formula.

**API:** `POST /api/v1/hybrid-search/`

```json
{
  "query": "attention mechanisms in transformers",
  "use_hybrid": true,
  "size": 10,
  "categories": ["cs.AI"]
}
```

### Index Mapping

Key fields in `index_config_hybrid.py`: `chunk_text`, `chunk_id`, `section_name`, `embedding` (knn_vector, 1024d), `arxiv_id`, `title`, `paper_categories`, `published_date`.

### Senior-Engineer Notes

- Hybrid latency is dominated by Jina embedding API call (~2s), not OpenSearch.
- `use_hybrid: true` with a failed embedding silently falls back to BM25 — check logs.
- Chunking quality directly impacts RAG answer quality; this is the highest-leverage tuning point.

**Notebook:** [notebooks/week4/week4_hybrid_search.ipynb](../notebooks/week4/week4_hybrid_search.ipynb)  
**Blog:** [The Chunking Strategy That Makes Hybrid Search Work](https://jamwithai.substack.com/p/chunking-strategies-and-hybrid-rag)

---

## Week 5: Complete RAG Pipeline

### Objective

Connect retrieval to generation: query → search → prompt → Ollama → answer with citations.

### Architecture

```
POST /api/v1/ask  (or /stream)
  → cache check (added Week 6)
  → embed query (if hybrid)
  → search_unified(top_k chunks)
  → build prompt with chunk context
  → LLM generate (Ollama direct or via Bifrost)
  → return answer + arXiv PDF sources
```

### Key Components

**RAG router** (`src/api/ask.py`):

- Two routers: `ask_router` (full response) and `stream_router` (SSE tokens)
- `_prepare_chunks_and_sources()` — retrieval + source URL extraction
- Prompt optimized: ~80% token reduction vs naive approach (title/abstract metadata stripped from chunks sent to LLM)

**LLM client** (`src/domain/llm/factory.py`):

- `make_llm_client()` selects the backend from `LLM_PROVIDER` (`ollama` or `bifrost`)
- Direct Ollama: `src/domain/ollama/client.py` — `ChatOllama`, native `/api/generate`
- Bifrost gateway: `src/domain/bifrost/client.py` — `ChatOpenAI` via `/langchain` (see [Bifrost](#bifrost-optional-llm-gateway))
- Default model: `llama3.2:1b` (configurable via `OLLAMA_MODEL` or per-request `model` param)
- Model must be pulled into Ollama first — see [Ollama](#ollama)
- System prompt in `src/domain/ollama/prompts/rag_system.txt`
- 300-word response cap for focused answers

### Performance Profile

| Config | Latency |
|--------|---------|
| top_k=1, BM25 only | ~2.4s |
| top_k=3, hybrid | ~15–20s |
| top_k=5, hybrid | ~25–30s |
| Streaming first token | ~2–3s |

### API Examples

```bash
# Standard RAG
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 3, "use_hybrid": true}'

# Streaming
curl -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain attention mechanism", "top_k": 2}' --no-buffer
```

**Notebook:** [notebooks/week5/week5_complete_rag_system.ipynb](../notebooks/week5/week5_complete_rag_system.ipynb)  
**Blog:** [The Complete RAG System](https://jamwithai.substack.com/p/the-complete-rag-system)

---

## Week 6: Monitoring & Caching

### Objective

Production observability (Langfuse) and response caching (Redis) for repeated queries.

### Langfuse Integration

**`LangfuseTracer`** (`src/domain/langfuse/client.py`) wraps the Langfuse v3 SDK; **`RAGTracer`** (`src/domain/langfuse/tracer.py`) adds RAG-specific span helpers on top.

- Traces full request lifecycle: embedding → search → prompt → generation
- Spans for each stage with timing, metadata, and token usage
- Integrated into `ask.py` via context managers; agentic nodes in Week 7 add per-node spans
- Week 7 also uses `CallbackHandler` for LangGraph/LangChain auto-tracing

Langfuse Cloud integration via the Langfuse v3 Python SDK. Dashboard at [cloud.langfuse.com](https://cloud.langfuse.com).

See [Accessing Langfuse](#langfuse) above for account setup, API key configuration, and how to browse traces in the UI.

### Redis Cache

**`CacheClient`** (`src/domain/cache/client.py`):

- **Exact-match** caching: SHA-256 hash of `(query, model, top_k, use_hybrid, categories)`
- TTL: 6 hours default (`REDIS__TTL_HOURS`)
- Graceful degradation: cache failures don't block RAG pipeline

### Performance Impact

| Scenario | Latency |
|----------|---------|
| Cache miss | 15–20s |
| Cache hit | 50–100ms (150–400x faster) |
| Tracing overhead | <2% |

### Verification

```bash
# First request (cache miss)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 3}'

# Second identical request (cache hit ~100ms)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 3}'
```

**Notebook:** [notebooks/week6/week6_cache_testing.ipynb](../notebooks/week6/week6_cache_testing.ipynb)  
**Blog:** [Production-ready RAG: Monitoring & Caching](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and)

---

## Week 7: Agentic RAG

### Objective

Replace fixed retrieve-then-generate with an **adaptive agent** that decides when to retrieve, grades relevance, and rewrites queries.

### Part A: Agentic RAG (LangGraph)

**Traditional RAG (Week 5–6):**

```
Query → Always Retrieve → Generate
```

**Agentic RAG (Week 7):**

```
Query → Guardrail → Retrieve → Grade → Generate
                  ↘ Out of scope    ↘ Rewrite → Retry
```

**Graph** (`src/domain/agents/agentic_rag.py`):

| Node | Responsibility |
|------|----------------|
| `guardrail` | Domain check (CS/AI/ML scope), score vs threshold |
| `out_of_scope` | Polite rejection for off-topic queries |
| `retrieve` | Decision: retrieve or respond directly |
| `tool_retrieve` | ToolNode wrapping OpenSearch hybrid search |
| `grade_documents` | LLM relevance scoring on retrieved chunks |
| `rewrite_query` | Refine vague queries for better retrieval |
| `generate_answer` | Final answer synthesis with sources |

Uses LangGraph's `context_schema=Context` for dependency injection — nodes receive `Runtime[Context]` with OpenSearch, LLM, and embeddings clients.

**API:** `POST /api/v1/ask-agentic`

Response includes `reasoning_steps[]` and `retrieval_attempts` for transparency.

```bash
# Simple question (should respond directly, no retrieval)
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?", "top_k": 3, "use_hybrid": true}'

# Research question (should retrieve papers)
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What are attention mechanisms?", "top_k": 3, "use_hybrid": true}'
```

### Comparison

| Aspect | Standard RAG | Agentic RAG |
|--------|-------------|-------------|
| Simple math question | 15–20s (still retrieves) | 2–5s (direct answer) |
| Vague query | Single attempt, poor results | Query rewrite + retry |
| Off-topic query | Hallucinates from random papers | Guardrail rejection |
| Transparency | Black box | Reasoning steps exposed |

**Notebook:** [notebooks/week7/week7_agentic_rag.ipynb](../notebooks/week7/week7_agentic_rag.ipynb)  
**Blog:** [Agentic RAG with LangGraph](https://jamwithai.substack.com/p/agentic-rag-with-langgraph-and-telegram)

---

## 5. API Reference (Complete)

| Endpoint | Method | Week | Description |
|----------|--------|------|-------------|
| `/api/v1/health` | GET | 1 | Service health |
| `/api/v1/hybrid-search/` | POST | 4 | BM25 / vector / hybrid search |
| `/api/v1/ask` | POST | 5 | Standard RAG |
| `/api/v1/stream` | POST | 5 | Streaming RAG (SSE) |
| `/api/v1/ask-agentic` | POST | 7 | Agentic RAG |
| `/api/v1/feedback` | POST | 7 | User feedback on answers |

Full interactive docs: http://localhost:8000/docs

---

## 6. Configuration Cheat Sheet

| Variable | Required When | Purpose |
|----------|---------------|---------|
| `JINA_API_KEY` | Week 4+ | Hybrid/vector search |
| `LANGFUSE__PUBLIC_KEY` / `SECRET_KEY` | Week 6 | Tracing (create in Langfuse Cloud → Settings → API Keys) |
| `LANGFUSE__HOST` | Week 6 | Langfuse Cloud URL (`https://cloud.langfuse.com` or `https://us.cloud.langfuse.com`) |
| `LLM_PROVIDER` | Week 5+ | `ollama` (direct) or `bifrost` (gateway) |
| `OLLAMA_HOST` | Week 5+ (`LLM_PROVIDER=ollama`) | Ollama API URL (`http://localhost:11434` on host; `http://ollama:11434` in compose) |
| `BIFROST_HOST` | Week 5+ (`LLM_PROVIDER=bifrost`) | Bifrost URL (`http://localhost:8090` on host; `http://bifrost:8080` in compose) |
| `BIFROST_API_KEY` | Week 5+ (`LLM_PROVIDER=bifrost`) | API key for Bifrost (use `dummy-key` for local dev) |
| `OLLAMA_MODEL` | Week 5+ | Default LLM (default `llama3.2:1b`; see [Ollama](#ollama)) |
| `OLLAMA_TIMEOUT` | Week 5+ | Generation timeout in seconds (default `300`) |
| `CHUNKING__*` | Week 4+ | Chunk size/overlap tuning |
| `REDIS__TTL_HOURS` | Week 6+ | Cache expiration |

All defaults work for local dev except external API keys. See [.env.example](../.env.example) for the full list.

---

## 7. Recommended Onboarding Path

1. **Day 1:** `docker compose up`, run Week 1 notebook, explore `compose.yml` and `src/main.py` lifespan
2. **Day 2:** Trace ingestion: Airflow DAG → `MetadataFetcher` → PostgreSQL. Trigger DAG manually
3. **Day 3:** Search layer: read `OpenSearchClient.search_unified()`, test all three modes via API
4. **Day 4:** RAG path: follow `ask.py` end-to-end, test streaming
5. **Day 5:** Ops layer: verify cache hit/miss, explore Langfuse traces
6. **Day 6:** Agent: step through LangGraph nodes, test `/ask-agentic` with guardrail/rewrite scenarios

### Key Files to Read in Order

```
compose.yml → src/config.py → src/main.py → src/dependencies.py
→ src/domain/arxiv_ingestion/ (ingestion)
→ src/domain/opensearch/client.py (search)
→ src/domain/indexing/text_chunker.py (chunking)
→ src/domain/llm/factory.py (LLM provider selection)
→ src/domain/ollama/client.py (direct Ollama)
→ src/domain/bifrost/client.py (Bifrost gateway)
→ src/api/ask.py (RAG)
→ src/domain/langfuse/tracer.py (tracing)
→ src/domain/cache/client.py (caching)
→ src/domain/agents/agentic_rag.py (agent)
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
uv run jupyter notebook notebooks/week1/week1_setup.ipynb
```

---

## 9. Common Issues

| Symptom | Fix |
|---------|-----|
| Services won't start | Wait 2–3 min; check `docker compose logs <service>` |
| Hybrid search returns BM25 only | Verify `JINA_API_KEY`; check API logs for embedding errors |
| Ollama 404 / model not found | Pull model: `docker exec rag-ollama ollama pull llama3.2:1b`; verify with `docker exec rag-ollama ollama list` |
| Bifrost 502 / private IP blocked | Ensure `allow_private_network: true` in `bifrost/config.json`; restart Bifrost |
| Bifrost connection refused | `docker compose up -d bifrost`; verify with `curl http://localhost:8090/health` |
| Wrong LLM backend in use | Check `LLM_PROVIDER` in `.env` and restart the API |
| Ollama slow / OOM | Use `llama3.2:1b`; reduce `top_k`; check `docker exec rag-ollama ollama ps` for loaded model size |
| Empty search results | Run Airflow DAG to ingest + index; check `curl localhost:9200/arxiv-papers-chunks/_count` |
| Langfuse empty | Verify `LANGFUSE__*` keys from [Langfuse Cloud](https://cloud.langfuse.com); confirm `LANGFUSE__HOST` matches your region; restart API; send a non-cached query; enable `LANGFUSE__DEBUG=true` |
| Port conflicts | Stop services on 8000, 8080, 8090, 5412, 9200 |

Full reset:

```bash
docker compose down -v && docker compose up --build -d
```

---

## 10. External Resources

| Week | Blog Post | Notebook |
|------|-----------|----------|
| 1 | [Infrastructure](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) | [week1_setup.ipynb](../notebooks/week1/week1_setup.ipynb) |
| 2 | [Data Ingestion](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life) | [week2_arxiv_integration.ipynb](../notebooks/week2/week2_arxiv_integration.ipynb) |
| 3 | [BM25 Search](https://jamwithai.substack.com/p/the-search-foundation-every-rag-system) | [week3_opensearch.ipynb](../notebooks/week3/week3_opensearch.ipynb) |
| 4 | [Chunking & Hybrid](https://jamwithai.substack.com/p/chunking-strategies-and-hybrid-rag) | [week4_hybrid_search.ipynb](../notebooks/week4/week4_hybrid_search.ipynb) |
| 5 | [Complete RAG](https://jamwithai.substack.com/p/the-complete-rag-system) | [week5_complete_rag_system.ipynb](../notebooks/week5/week5_complete_rag_system.ipynb) |
| 6 | [Monitoring & Caching](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and) | [week6_cache_testing.ipynb](../notebooks/week6/week6_cache_testing.ipynb) |
| 7 | [Agentic RAG](https://jamwithai.substack.com/p/agentic-rag-with-langgraph-and-telegram) | [week7_agentic_rag.ipynb](../notebooks/week7/week7_agentic_rag.ipynb) |
