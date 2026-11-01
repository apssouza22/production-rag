# Evals

Evaluation frameworks for the API, integrated with the main project:

- **example_eval** — minimal, dependency-free example using the shared eval infrastructure
- **live_req_evals** — scores Langfuse traces with LLM-as-judge metrics
- **rag_evals** — Ragas retrieval evaluation using contexts from `rerank_search_service.search`
- **experiments** — self-contained experiment: ingests local PDFs, generates its own question set, then scores retrieval with Ragas

Evals share the root virtualenv and `.env` file. Import from `src` the same way tests do (e.g. `from src.domain.rerank.factory import make_rerank_search_service`), and use the `evals.*` package namespace for eval code (e.g. `from evals.live_req_evals.config import get_settings`).

## Minimal example

`example_eval` demonstrates an inline dataset, a deterministic exact-match metric, the shared evaluator, and unified report generation:

```bash
uv run python -m evals.example_eval.main
```

It runs without API keys or external services.

## Prerequisites

1. The main RAG API is running with Langfuse tracing enabled (`LANGFUSE__*` in the root `.env`)
2. You have sent at least one query that produced an unscored trace in [Langfuse Cloud](https://cloud.langfuse.com)
3. An OpenAI API key for the judge LLM

## Setup

From the repo root:

```bash
cp .env.example .env
uv sync
```

Configure eval-related variables in the root `.env` (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (same project as the RAG API) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` or `https://us.cloud.langfuse.com` |
| `EVALUATION_LLM` | Judge model (e.g. `gpt-4o-mini`) |
| `EVALUATION_API_KEY` | API key for the judge LLM (falls back to `OPENAI_API_KEY`) |
| `EVALUATION_BASE_URL` | OpenAI-compatible endpoint (default: `https://api.openai.com/v1`) |
| `EVALUATION_SLEEP_TIME` | Seconds to wait between traces (default: `10`) |

## Usage

From the repo root:

```bash
make eval-quick      # run with defaults
make eval            # interactive mode
make eval-no-report  # skip JSON report
```

Or directly:

```bash
uv run python -m evals.live_req_evals.main --quick
uv run python -m evals.live_req_evals.main --interactive
uv run python -m evals.live_req_evals.main --no-report
```

## Metrics

Metric prompts live in `evals/live_req_evals/metrics/prompts/`. Add a new `.md` file and it is discovered automatically:

- `relevancy`
- `helpfulness`
- `conciseness`
- `hallucination`
- `toxicity`

Reports are written to `evals/reports/`.

## Ragas Retrieval Evaluation (`rag_evals`)

Evaluates retrieval and response quality with Ragas metrics. For each query in the dataset, contexts are fetched from the running RAG API's `/api/v1/rerank-search` endpoint, which delegates to `rerank_search_service.search`.

### Prerequisites

1. The main RAG API stack is running (`make start` from the repo root)
2. OpenSearch is indexed with papers
3. An OpenAI API key for Ragas judge metrics

### Metrics

| Metric | Reference required | Description |
|--------|-------------------|-------------|
| `context_relevance` | No | Whether retrieved contexts are pertinent to the query |
| `context_utilization` | No | Context precision without reference (uses generated response) |
| `faithfulness` | No | Whether the response is supported by retrieved contexts |
| `context_precision` | Yes | Whether relevant chunks are ranked above irrelevant ones |
| `context_recall` | Yes | How much of the reference answer is covered by retrieved contexts |
| `factual_correctness` | Yes | Whether response claims match the reference answer |

### Configuration

| Variable | Description |
|----------|-------------|
| `RAG_API_BASE_URL` | RAG API base URL (default: `http://localhost:8000`) |
| `RAG_EVAL_DATASET_PATH` | JSON file with queries (default: `evals/rag_evals/queries.json`) |
| `RAG_EVAL_TOP_K` | Documents to retrieve per query (default: `5`) |
| `RAG_EVAL_USE_HYBRID` | Use hybrid search (default: `true`) |
| `RAG_EVAL_RERANK_ENABLED` | Enable Jina reranking (default: `true`) |
| `RAG_EVAL_GENERATE_RESPONSES` | Generate answers when omitted from dataset (default: `true`) |

Dataset format:

```json
[
  {
    "user_input": "What is RAG?",
    "reference": "Retrieval augmented generation combines search with LLM generation."
  },
  {
    "user_input": "How does hybrid search work?",
    "reference": "Hybrid search combines BM25 keyword search with vector similarity.",
    "response": "Optional pre-generated answer for response-based metrics."
  }
]
```

`reference` is required. `response` is optional and generated from retrieved contexts when omitted.

### Usage

```bash
make rag-eval
make rag-eval-no-report
uv run python -m evals.rag_evals.main --quick
```

Reports use the unified schema described below.

### Langfuse Experiments (`rag_evals` + Langfuse SDK)

Runs the same retrieval + Ragas pipeline through [Langfuse experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk). Each dataset item becomes a traced task execution with Ragas scores attached in Langfuse for comparison across runs.

**Prerequisites:** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in the root `.env` (same project as production traces).

| Variable | Description |
|----------|-------------|
| `RAG_EVAL_EXPERIMENT_NAME` | Experiment name in Langfuse (default: `rag-retrieval-eval`) |
| `RAG_EVAL_EXPERIMENT_DESCRIPTION` | Optional experiment description |
| `RAG_EVAL_LANGFUSE_DATASET` | Optional hosted Langfuse dataset name; when set, skips the local JSON file |
| `RAG_EVAL_MAX_CONCURRENCY` | Concurrent items processed by the experiment runner (default: `3`) |

**Local dataset** (default): items are built from `RAG_EVAL_DATASET_PATH` with `input` = query and `expected_output` = reference.

**Hosted Langfuse dataset**: create a dataset in Langfuse where each item has:

- `input`: the question string (or `{"user_input": "..."}`)
- `expected_output`: the reference answer
- optional `metadata.response`: pre-generated answer to skip response generation

```bash
make rag-eval-experiment
make rag-eval-experiment-no-report
uv run python -m evals.rag_evals.main --langfuse-experiment --quick
uv run python -m evals.rag_evals.main --langfuse-experiment --dataset-name my-rag-eval-set
```

Local summaries use the unified report schema. When using a hosted dataset, Langfuse also provides a dataset run URL for side-by-side comparison in the UI.

## Local-PDF Experiment (`experiments`)

A single command that takes the PDFs in `evals/experiments/arxiv_pdfs/` from raw files to Ragas scores. Unlike `rag_evals`, it does not need the API running — it builds the retrieval stack in-process and brings its own corpus and question set.

1. Counts chunks in OpenSearch for the experiment's ingestion partition. If any exist, ingestion is skipped.
2. Otherwise runs `src.ingest.run_pipeline` with `skip_download=True`, entering the pipeline at the local-PDF loader and continuing through `parse_pdfs_to_markdown` → `persist_papers` → `index_papers_hybrid`.
3. Generates question/reference pairs from the indexed chunks with the judge model and caches them to `evals/experiments/generated_queries.json`.
4. For each question, retrieves contexts with `rerank_search_service.search` and generates an answer from those contexts.
5. Scores every sample with the Ragas metrics from `rag_evals` and writes a unified report.

Everything is scoped to a dedicated ingestion partition (default `evals-experiments`), which the CLI exports as `OPENSEARCH__INGESTION_NAME` so indexing and retrieval filter on the same value. Experiment chunks therefore stay out of your production or dev partitions.

### Prerequisites

1. OpenSearch and PostgreSQL are running (`make start` from the repo root)
2. `JINA_API_KEY` for embeddings and reranking
3. An OpenAI API key for question generation, answer generation, and the Ragas judges
4. Network access to the arXiv API — paper metadata is fetched by filename stem (e.g. `2609.21059v1.pdf`)

### Configuration

Settings load from [`evals/experiments/data/settings/experiment-config.json`](../evals/experiments/data/settings/experiment-config.json) by default. The file has four sections:

| Section | Purpose |
|---------|---------|
| `pdf_parser` | Docling PDF parser limits (`max_pages`, `do_ocr`, …) — applied as `PDF_PARSER__*` env vars before ingestion |
| `chunking` | Chunk size, overlap, contextualization — applied as `CHUNKING__*` before indexing |
| `query` | Retrieval knobs for `rerank_search_service` (`use_hybrid`, `rerank_enabled`, `rerank_model`, `rerank_candidate_multiplier`) |
| `experiment` | Judge model, ingestion partition, PDF directory, query cache paths, question counts, `experiment_top_k` |

Merge order for the `experiment` section: defaults → JSON → environment variables. API keys (`OPENAI_API_KEY`, `EVALUATION_API_KEY`) stay in the root `.env`, not in the JSON file.

Pass a different file with `--settings` (path relative to the repo root unless absolute):

```bash
uv run python -m evals.experiments.main --settings evals/experiments/data/settings/my-variant.json
make experiment-eval EXPERIMENT_SETTINGS=evals/experiments/data/settings/my-variant.json
```

Environment variables still override individual fields when set:

| Variable | Description |
|----------|-------------|
| `EXPERIMENT_INGESTION_NAME` | OpenSearch partition for the experiment (default: `evals-experiments`) |
| `EXPERIMENT_PDFS_DIR` | Directory of PDFs to ingest (default: `evals/experiments/arxiv_pdfs`) |
| `EXPERIMENT_QUERIES_PATH` | Cache file for generated questions (default: `evals/experiments/generated_queries.json`) |
| `EXPERIMENT_QUESTIONS_PER_PAPER` | Questions generated per paper (default: `4`) |
| `EXPERIMENT_CHUNKS_PER_PAPER` | Chunks shown to the generator per paper (default: `12`) |
| `EXPERIMENT_TOP_K` | Contexts retrieved per query (default: `5`) |

### Usage

```bash
make experiment-eval         # skips ingestion and reuses cached questions when possible
make experiment-eval-fresh   # --force-ingest --regenerate-queries

uv run python -m evals.experiments.main --ingestion-name my-experiment
uv run python -m evals.experiments.main --no-report
```

Because the same model writes both the questions and the answers, absolute scores skew high. Use them to compare retrieval configurations across runs rather than as a quality bar.

## How live_req_evals works

1. Fetches unscored Langfuse traces from the last 24 hours
2. Extracts `query` and `answer` from each trace (supports RAG API and LangGraph trace formats)
3. Scores each trace against every metric using the configured judge LLM
4. Pushes scores back to Langfuse
5. Optionally writes a unified JSON report

## Unified evaluator

`evals.core.evaluator.Evaluator` is the only metric execution loop. It accepts any shared `Dataset` and `Metric` implementations and returns an `EvaluationResponse` without knowing about report formats or providers.

The trace and RAG runners configure datasets, metrics, and integrations, then pass the evaluator response to the shared report builder. Langfuse experiments keep their SDK orchestration and adapt `ExperimentResult` into the same `EvaluationResponse` before report generation.

## Unified reports

All execution paths emit the same versioned schema:

- `schema_version`, `timestamp`, and `run_type`
- evaluation model and dataset source/location
- total, evaluated, successful, and failed item counts
- per-metric success/failure counts and average scores
- normalized item previews, metadata, and metric results
- runner-specific details under `metadata`

Files are written to `evals/reports/evaluation_report_<run_type>_<timestamp>.json`.

## Tests

From the repo root:

```bash
uv run pytest tests/evals/ -q
```

Evals tests are also included when running `make test`.
