.PHONY: help build run run-ghcr test lint check

GHCR_IMAGE  ?= ghcr.io/chap-models/chapkit_simple_multistep_template:latest
BASE_URL    ?= http://localhost:8000

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run        docker compose up (foreground, builds if needed)"
	@echo "  build      docker compose build --no-cache"
	@echo "  run-ghcr   docker compose -f compose.ghcr.yml up (prebuilt GHCR image)"
	@echo "  test       Run 'chapkit test' against $(BASE_URL)"
	@echo "  lint       Run ruff format + lint --fix (modifies files)"
	@echo "  check      Run ruff format check + lint without fixing (for CI)"

build:
	@docker compose build --no-cache

run:
	@docker compose up --build

run-ghcr:
	@docker compose -f compose.ghcr.yml up

test:
	@uv run chapkit test --url $(BASE_URL) --timeout 180 --verbose

lint:
	@uv run ruff format .
	@uv run ruff check --fix .

check:
	@uv run ruff format --check .
	@uv run ruff check .

.DEFAULT_GOAL := help
