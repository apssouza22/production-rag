.PHONY: help setup eval eval-quick eval-no-report

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	uv sync

eval: ## Run evaluation with interactive mode
	uv run python -m rag_evals.main --interactive

eval-quick: ## Run evaluation with default settings
	uv run python -m rag_evals.main --quick

eval-no-report: ## Run evaluation without generating a JSON report
	uv run python -m rag_evals.main --no-report
