# Development tasks.
#
# Everything runs through the project virtual environment, so a task behaves
# the same whether or not a shell has it activated.

VENV := .venv
PY   := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
PY   := $(VENV)/Scripts/python
endif

SRC   := src/nl2sql
TESTS := tests
ALL   := src tests scripts migrations

.DEFAULT_GOAL := help
.PHONY: help venv install install-local clean lint format format-check style type \
        test test-unit test-security test-integration test-api test-evaluation \
        test-live coverage check migrate downgrade migration serve ask schema \
        config eval seed docker-build docker-up docker-down hooks

help: ## Show the available tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
venv: ## Create the virtual environment
	python -m venv $(VENV)

install: venv ## Install the project with development dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

install-local: venv ## Also install the local Hugging Face model extra
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,local]"

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +

hooks: ## Install the pre-commit hooks
	$(PY) -m pre_commit install

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint: ## Run the linter
	$(PY) -m ruff check $(ALL)

format: ## Format the code and fix what can be fixed
	$(PY) -m ruff check --fix $(ALL)
	$(PY) -m ruff format $(ALL)

format-check: ## Check formatting without changing anything
	$(PY) -m ruff format --check $(ALL)

style: ## Check the house style rules a linter does not cover
	$(PY) scripts/check_style.py

type: ## Run the type checker
	$(PY) -m mypy $(SRC)

test: ## Run every offline test
	$(PY) -m pytest

test-unit: ## Run the unit tests
	$(PY) -m pytest $(TESTS)/unit -q

test-security: ## Run the security tests
	$(PY) -m pytest $(TESTS)/security $(TESTS)/sql_validation -q

test-integration: ## Run the integration tests
	$(PY) -m pytest $(TESTS)/integration -q

test-api: ## Run the API tests
	$(PY) -m pytest $(TESTS)/api -q

test-evaluation: ## Run the evaluation framework tests
	$(PY) -m pytest $(TESTS)/evaluation -q

test-live: ## Run the tests that need real Azure resources
	$(PY) -m pytest $(TESTS)/live -q -m "live_azure_sql or live_azure_openai"

coverage: ## Run the tests with a coverage report
	$(PY) -m pytest --cov=$(SRC) --cov-report=term-missing --cov-report=html

check: style lint format-check type test ## Everything CI runs

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
migrate: ## Apply migrations up to head
	$(PY) -m alembic upgrade head

downgrade: ## Revert the most recent migration
	$(PY) -m alembic downgrade -1

migration: ## Create a migration from the models. Usage: make migration m="what changed"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

seed: ## Seed the demonstration schema. Usage: make seed url="sqlite:///./data/demo.db"
	$(PY) scripts/seed_demo_database.py --url "$(url)"

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
serve: ## Run the API with reload
	$(PY) -m nl2sql.cli serve --reload

ask: ## Ask one question. Usage: make ask q="Which facilities used the most energy?"
	$(PY) -m nl2sql.cli ask "$(q)"

schema: ## Show the tables the assistant can see
	$(PY) -m nl2sql.cli schema

config: ## Validate the configuration and report what it enables
	$(PY) -m nl2sql.cli check-config

eval: ## Run an evaluation dataset. Usage: make eval d=evaluation/datasets/sustainability_demo.jsonl
	$(PY) scripts/run_evaluation.py --dataset "$(d)"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the container image
	docker build -f docker/Dockerfile -t nl2sql-assistant:local .

docker-up: ## Start the local stack
	docker compose up --build

docker-down: ## Stop the local stack and remove its volumes
	docker compose down -v
