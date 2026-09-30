# Development tasks (CPU only; nothing here runs on or downloads to the pod).
#   make dev dev-bpy     create .venv-dev (Python 3.12: CPU pipeline deps + pytest + ruff) and .venv-bpy (bpy 5.2.2)
#   make bundle          rebuild dist/setup.sh (the single file for the pod) from setup.sh + src/
#   make check           bundle up to date + lint + tests
#   make check-identical Phase 1 acceptance: src/, dist/setup.sh and --code-only output == setup.sh 2.0.3
UV ?= uv
PY ?= .venv-dev/bin/python
BPY ?= .venv-bpy/bin/python
SHELLCHECK ?= shellcheck

.PHONY: help dev dev-bpy bundle check-bundle check-identical test lint ruff shellcheck check

help:
	@sed -n '1,6p' Makefile

dev:
	$(UV) venv .venv-dev --python 3.12
	$(UV) pip install --python $(PY) --only-binary :all: -r requirements-dev.lock

dev-bpy:
	$(UV) venv .venv-bpy --python 3.13
	$(UV) pip install --python $(BPY) --only-binary :all: -r requirements-bpy.lock

bundle:
	python3 tools/bundle.py

check-bundle:
	python3 tools/bundle.py --check

check-identical:
	python3 tools/check_identical.py --strict

test:
	$(PY) -m pytest -q tests

ruff:
	$(PY) -m ruff check src tools tests

shellcheck:
	SHELLCHECK=$(SHELLCHECK) bash tools/shellcheck_all.sh

lint: ruff shellcheck

check: check-bundle lint test
