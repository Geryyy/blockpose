# Canonical entry points — the agent and CI use ONLY these targets.
# All of them assume you are INSIDE the container (devcontainer or
# `docker compose run --rm core`).

.PHONY: help image image-ros2 shell shell-ros2 configure build test \
        python-sync pytest lint format check clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-14s %s\n", $$1, $$2}'

# ---- host-side (need docker, nothing else) ----------------------------------
image:            ## build the core dev image
	docker compose build core

image-ros2: image ## build the ROS 2 image (extends core)
	docker compose build ros2

shell:            ## dev shell in the core container
	docker compose run --rm core

shell-ros2:       ## dev shell in the ROS 2 container
	docker compose run --rm ros2

# ---- container-side ----------------------------------------------------------
configure:        ## cmake configure (dev preset)
	cmake --preset dev

build: configure  ## build everything
	cmake --build --preset dev

test: build       ## C++ unit tests incl. env_check
	ctest --preset dev

python-sync:      ## create/update the uv venv from the lockfile
	uv sync --all-extras

pytest: python-sync ## python tests (env + bindings + tools)
	uv run pytest

lint:             ## ruff + clang-format check (no changes)
	uv run ruff check python tests
	@git ls-files '*.cpp' '*.hpp' | xargs -r clang-format --dry-run --Werror

format:           ## apply ruff + clang-format
	uv run ruff check --fix python tests && uv run ruff format python tests
	@git ls-files '*.cpp' '*.hpp' | xargs -r clang-format -i

check: test pytest lint ## full local gate == CI

clean:
	rm -rf build build-release build-asan
