# One-command ergonomics. `make help` lists everything.
.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install deps (incl. dev tools + hooks)
	uv sync
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

test: ## Run the offline unit tests (no workspace needed)
	uv run pytest

lint: ## Lint + format check
	uv run ruff check .
	uv run ruff format --check .

eval: ## Run the eval gate locally (needs workspace auth; spends judge tokens)
	uv run python evals/run_agent_eval.py

validate: ## Validate the bundle against the workspace
	databricks bundle validate

deploy: ## Deploy the bundle (dev target)
	databricks bundle deploy

gate: ## Run the eval gate as a Databricks job
	databricks bundle run $$(basename $$(pwd))_eval_gate

.PHONY: help install test lint eval validate deploy gate
