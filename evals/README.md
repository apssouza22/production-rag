# Evals

Evaluation frameworks for the API, integrated with the main project:

- **custom_evals** — scores Langfuse traces with LLM-as-judge metrics
- **rag_evals** — Ragas retrieval evaluation using contexts from `rerank_search_service.search`

Evals share the root virtualenv and `.env` file. Import from `src` the same way tests do (e.g. `from src.domain.rerank.factory import make_rerank_search_service`), and use the `evals.*` package namespace for eval code (e.g. `from evals.custom_evals.config import get_settings`).

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
uv run python -m evals.custom_evals.main --quick
uv run python -m evals.custom_evals.main --interactive
uv run python -m evals.custom_evals.main --no-report
```

## Metrics

Metric prompts live in `evals/custom_evals/metrics/prompts/`. Add a new `.md` file and it is discovered automatically:

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

## How custom_evals works

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
