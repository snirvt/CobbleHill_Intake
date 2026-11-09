PYTEST := uv run pytest
ARGS ?=

.PHONY: help test test-llm test-buckets test-all

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

test:  ## Fast tests, no LLM (skips @integration)
	$(PYTEST) tests/ -m "not integration" -q $(ARGS)

test-llm:  ## LLM integration tests only (needs ollama / vLLM up)
	$(PYTEST) tests/ -m integration -v $(ARGS)

test-buckets:  ## Diagnosis-bucket prompt tests only (LLM)
	$(PYTEST) tests/integration/test_diagnosis_buckets.py -m integration -v $(ARGS)

test-all:  ## Every test, fast + LLM
	$(PYTEST) tests/ -v $(ARGS)
