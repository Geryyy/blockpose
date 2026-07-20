# Canonical entry points. All of them assume you are INSIDE the container
# (devcontainer or `docker compose run --rm core`).

.PHONY: help image shell python-sync pytest lint format check

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-14s %s\n", $$1, $$2}'

# ---- host-side (need docker, nothing else) ----------------------------------
image:            ## build the core dev image
	docker compose build core

shell:            ## dev shell in the core container
	docker compose run --rm core

# ---- container-side ----------------------------------------------------------
python-sync:      ## create/update the uv venv from the lockfile
	uv sync --all-extras

pytest: python-sync ## python tests
	uv run pytest

lint:             ## ruff check (no changes)
	uv run ruff check python tests

format:           ## apply ruff
	uv run ruff check --fix python tests && uv run ruff format python tests

check: pytest lint ## full local gate
