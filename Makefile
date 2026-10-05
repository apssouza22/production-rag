.PHONY: help start stop restart status logs health setup format lint test test-cov clean \
	eval eval-quick eval-no-report rag-eval rag-eval-no-report

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Service management
start: ## Start all services
	docker compose up --build -d

stop: ## Stop all services
	docker compose down

restart: ## Restart all services
	docker compose restart

status: ## Show service status
	docker compose ps

logs: ## Show service logs
	docker compose logs -f

# Health checks
health: ## Check all services health
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@curl -s http://localhost:9200/_cluster/health | jq . || echo "OpenSearch not responding"
	@curl -s http://localhost:8080/health || echo "Airflow not responding"

# Development
setup: ## Install Python dependencies
	uv sync

format: ## Format code
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

# Evaluations (uses root .env)
eval: ## Run Langfuse trace evaluation (interactive)
	uv run python -m evals.custom_evals.main --interactive

eval-quick: ## Run Langfuse trace evaluation with defaults
	uv run python -m evals.custom_evals.main --quick

eval-no-report: ## Run Langfuse trace evaluation without JSON report
	uv run python -m evals.custom_evals.main --no-report

rag-eval: ## Run Ragas retrieval evaluation
	uv run python -m evals.rag_evals.main --quick

rag-eval-no-report: ## Run Ragas retrieval evaluation without JSON report
	uv run python -m evals.rag_evals.main --quick --no-report

rag-eval-experiment: ## Run Ragas eval as a Langfuse experiment
	uv run python -m evals.rag_evals.main --langfuse-experiment --quick

rag-eval-experiment-no-report: ## Run Langfuse experiment without local JSON report
	uv run python -m evals.rag_evals.main --langfuse-experiment --quick --no-report

# Cleanup
clean: ## Clean up everything
	docker compose down -v
	docker system prune -f