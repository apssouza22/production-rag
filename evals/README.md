# RAG Evals

Standalone evaluation framework that scores Langfuse traces from the RAG API using LLM-as-judge metrics.

This project is independent from the main app: it has its own `pyproject.toml`, virtualenv, and `.env` file.

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
uv run python -m rag_evals.main --quick
uv run python -m rag_evals.main --interactive
uv run python -m rag_evals.main --no-report
uv run rag-eval --quick
```

## Metrics

Metric prompts live in `rag_evals/metrics/prompts/`. Add a new `.md` file and it is discovered automatically:

- `relevancy`
- `helpfulness`
- `conciseness`
- `hallucination`
- `toxicity`

Reports are written to `reports/`.

## How it works

1. Fetches unscored Langfuse traces from the last 24 hours
2. Extracts `query` and `answer` from each trace (supports RAG API and LangGraph trace formats)
3. Scores each trace against every metric using the configured judge LLM
4. Pushes scores back to Langfuse
5. Optionally writes a JSON summary report to `reports/evaluation_report_<timestamp>.json`

## Tests

```bash
uv run pytest tests/ -q
```
