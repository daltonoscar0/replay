.PHONY: help install lint test test-fast demo demo-assets eval-table clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## sync the uv environment (add EXTRA=yolo for the detector)
	uv sync --extra yolo

lint: ## ruff
	uv run ruff check src tests scripts

test: ## full test suite
	uv run pytest

test-fast: ## skip slow and network tests
	uv run pytest -m "not slow and not network"

demo-assets: ## regenerate the bundled synthetic clip and its ground truth
	uv run python scripts/make_demo_clip.py --out demo/assets --seed 0 --duration 30

demo: ## end-to-end offline demo
	uv run replay demo

eval-table: ## regenerate the README accuracy table from a real evaluation
	uv run replay eval --dataset demo --write-readme --readme README.md
	@echo "accuracy section in README.md regenerated"

clean:
	rm -rf out .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
