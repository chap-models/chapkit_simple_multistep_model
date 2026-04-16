.PHONY: help build run run-ghcr test lint check clean

GHCR_IMAGE  ?= ghcr.io/chap-models/chapkit_simple_multistep_model:latest

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run        docker compose up (foreground, builds if needed)"
	@echo "  build      docker compose build --no-cache"
	@echo "  run-ghcr   docker compose -f compose.ghcr.yml up (prebuilt GHCR image)"
	@echo "  test       pytest end-to-end (in-process via FastAPI TestClient)"
	@echo "  lint       ruff format + lint --fix (modifies files)"
	@echo "  check      ruff format check + lint without fixing (for CI)"
	@echo "  clean      remove caches, build artifacts, and CLI outputs (keeps .venv and uv.lock)"

build:
	@docker compose build --no-cache

run:
	@docker compose up --build

run-ghcr:
	@docker compose -f compose.ghcr.yml up

test:
	@uv run pytest -v

lint:
	@uv run ruff format .
	@uv run ruff check --fix .

check:
	@uv run ruff format --check .
	@uv run ruff check .

clean:
	@rm -rf data target dist build .pytest_cache .ruff_cache .mypy_cache .pyright *.egg-info
	@find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -not -path "./.venv/*" -delete
	@rm -f model.pickle predictions.csv
	@echo "cleaned"

.DEFAULT_GOAL := help
