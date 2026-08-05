# Canonical entry points. All of them assume you are INSIDE the container
# (devcontainer or `docker compose run --rm core`).

.PHONY: help image shell python-sync pytest lint format check ship-check

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

# ship-check is deliberately NOT a dependency of `check`. It exercises the
# ROS-facing surface (cpp/CMakeLists.txt -> installed blockpose_core ->
# cpp/tests/consumer), which pytest+lint says nothing about: a Python-only
# change can pass `check` while silently breaking the thing the ROS node
# links against (see the missing-Eigen incident this target exists to catch).
# It is not folded into `check` anyway because it is a different kind of
# cost: a from-scratch CMake configure + build of the whole core plus a
# consumer binary is tens of seconds to minutes, versus pytest+lint's
# sub-second-to-seconds turnaround, and it needs a C++ toolchain + Eigen3
# that a Python-only contributor's environment has no other reason to have.
# Run it before shipping anything the ROS node depends on, and in CI.
ship-check:        ## build+install the core and run the ROS-facing consumer against it (see tools/check_cpp_consumer.sh)
	tools/check_cpp_consumer.sh
