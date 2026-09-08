# Custom Evals

Standalone evaluation frameworks for the API:

- **custom_evals** — scores Langfuse traces with LLM-as-judge metrics
- **rag_evals** — Ragas retrieval evaluation using contexts from `rerank_search_service.search`

Both packages share the same `pyproject.toml`, virtualenv, and `.env` file.

## Prerequisites

1. The main RAG API is running with Langfuse tracing enabled (`LANGFUSE__*` in the root `.env`)
2. You have sent at least one query that produced an unscored trace in [Langfuse Cloud](https://cloud.langfuse.com)
3. An OpenAI API key (or Bifrost gateway) for the judge LLM

## Setup

```bash
cd evals
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Description |
|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (same project as the RAG API) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` or `https://us.cloud.langfuse.com` |
| `EVALUATION_LLM` | Judge model (e.g. `gpt-4o-mini`; use `openai/gpt-4o-mini` with Bifrost) |
| `EVALUATION_API_KEY` | API key for the judge LLM (falls back to `OPENAI_API_KEY`) |
| `EVALUATION_BASE_URL` | OpenAI-compatible endpoint (default: `https://api.openai.com/v1`) |
| `EVALUATION_SLEEP_TIME` | Seconds to wait between traces (default: `10`) |

Optional Bifrost routing (when the main stack uses `LLM_PROVIDER=bifrost`):

```bash
BIFROST_ENABLED=true
BIFROST_HOST=http://localhost:8090
BIFROST_API_KEY=sk-bf-agent-1-dev   # virtual key from bifrost/config.json
EVALUATION_LLM=openai/gpt-4o-mini   # optional; unprefixed names also work
```

With `enforce_auth_on_inference: true` in Bifrost, `dummy-key` is rejected — use a configured virtual key.

Install dependencies:

```bash
uv sync
```

## Usage

```bash
make eval-quick      # run with defaults
make eval            # interactive mode
make eval-no-report  # skip JSON report
```

Or directly:

```bash
uv run python -m custom_evals.main --quick
uv run python -m custom_evals.main --interactive
uv run python -m custom_evals.main --no-report
uv run custom-eval --quick
```

## Metrics

Metric prompts live in `custom_evals/metrics/prompts/`. Add a new `.md` file and it is discovered automatically:

- `relevancy`
- `helpfulness`
- `conciseness`
- `hallucination`
- `toxicity`

Reports are written to `reports/`.

## Ragas Retrieval Evaluation (`rag_evals`)

Evaluates retrieval and response quality with Ragas metrics. For each query in the dataset, contexts are fetched from the running RAG API's `/api/v1/rerank-search` endpoint, which delegates to `rerank_search_service.search`.

### Prerequisites

1. The main RAG API stack is running (`make start` from the repo root)
2. OpenSearch is indexed with papers
3. An OpenAI API key (or Bifrost gateway) for Ragas judge metrics

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
| `RAG_EVAL_DATASET_PATH` | JSON file with queries (default: `rag_evals/queries.json`) |
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
uv run python -m rag_evals.main --quick
uv run rag-eval --quick
```

Reports are written to `reports/ragas_evaluation_report_<timestamp>.json`.

## How custom_evals works

1. Fetches unscored Langfuse traces from the last 24 hours
2. Extracts `query` and `answer` from each trace (supports RAG API and LangGraph trace formats)
3. Scores each trace against every metric using the configured judge LLM
4. Pushes scores back to Langfuse
5. Optionally writes a JSON summary report to `reports/evaluation_report_<timestamp>.json`

## Tests

```bash
uv run pytest tests/ -q
```
