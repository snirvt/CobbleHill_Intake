# CLAUDE.md

## Interaction Rules

**ALWAYS ask clarifying questions before implementing.** Never assume intent — confirm scope, edge cases, naming, and design tradeoffs with the user first. Prefer a short question over a wrong implementation. When multiple valid approaches exist, present them as options and let the user choose.

If a task is large, propose a plan and get approval before writing code.

---

## Project Overview

A Python application that classifies and categorizes documents from various file types (PDFs, images/screenshots, etc.). Accepts a single file path or a folder path containing multiple files. Runs as both a CLI tool and a FastAPI server.

---

## Architecture Principles

### Dependency Injection — Non-Negotiable

All major components communicate through **abstract interfaces** (Python `Protocol` or `ABC`). Concrete implementations are injected, never hardcoded.

Core abstractions:
- **`ContentExtractor` (Protocol)** — extracts raw content from a file. One implementation per file type (PDF, image/screenshot, plaintext, etc.). New file types = new extractor, nothing else changes.
- **`LLMProvider` (Protocol)** — sends prompts, returns responses. Implementations for local models (e.g. Ollama, llama.cpp) and vendor APIs (Anthropic, OpenAI, etc.). Must be swappable without touching classification logic.
- **`Classifier` (Protocol)** — takes extracted content and returns classification results. A classifier is **not limited to a single call**. It can be a multi-step pipeline combining multiple LLM calls, code logic, validation, and post-processing. For example: one LLM call to extract key features → code logic to normalize/filter → a second LLM call to assign a category → code logic to validate confidence thresholds. Each classifier encapsulates its own internal workflow. The protocol only defines the outer contract (content in → classification result out); the steps inside are an implementation detail.
- **`FileRouter`** — resolves a file path to the correct `ContentExtractor` based on file type/extension/mime.

When creating any new class or function, always accept dependencies as constructor/function parameters. Never instantiate dependencies internally.

### Modularity

- Adding a new file type = add one `ContentExtractor` implementation + register it.
- Adding a new LLM provider = add one `LLMProvider` implementation + register it.
- Adding a new classification strategy = add one `Classifier` implementation.
- No feature addition should require modifying existing working code (Open/Closed Principle).

### Async First

- Use `async/await` for all I/O: file reading, LLM calls, API endpoints.
- Use `asyncio.gather` or `asyncio.TaskGroup` for batch processing (folder of files).
- Sync wrappers are acceptable only at CLI entry points (`asyncio.run`).
- Use `aiofiles` for async file I/O.
- Use `httpx.AsyncClient` (not `requests`) for HTTP calls to LLM vendor APIs.

---

## Project Structure

```
project-root/
├── CLAUDE.md
├── pyproject.toml
├── config/
│   └── settings.py          # All configurable values (paths, model names, thresholds, API keys, etc.)
├── src/
│   └── classifier/
│       ├── __init__.py
│       ├── main.py           # Wiring / DI container / app factory
│       ├── cli.py            # CLI entry point (argparse or typer)
│       ├── api.py            # FastAPI app and routes
│       ├── models.py         # Pydantic models for inputs, outputs, classification results
│       ├── protocols.py      # All Protocol/ABC definitions
│       ├── extractors/       # ContentExtractor implementations
│       │   ├── __init__.py
│       │   ├── pdf.py
│       │   ├── image.py
│       │   └── plaintext.py
│       ├── providers/        # LLMProvider implementations
│       │   ├── __init__.py
│       │   ├── stub.py       # Deterministic constants — for dev on personal laptop
│       │   ├── local.py      # Ollama / local model
│       │   └── anthropic.py  # Claude API (or other vendors)
│       ├── classifiers/      # Classifier implementations
│       │   ├── __init__.py
│       │   └── llm_classifier.py
│       ├── routing.py        # FileRouter — maps file types to extractors
│       └── pipeline.py       # Orchestrates: route → extract → classify
├── tests/
│   ├── conftest.py           # Shared fixtures, mock factories
│   ├── unit/
│   │   ├── test_extractors.py
│   │   ├── test_providers.py
│   │   ├── test_classifiers.py
│   │   ├── test_routing.py
│   │   └── test_pipeline.py
│   └── integration/
│       ├── test_cli.py
│       └── test_api.py
└── fixtures/                 # Sample test files (small PDF, PNG, TXT)
    ├── sample.pdf
    ├── sample.png
    └── sample.txt
```

---

## Config — `config/settings.py`

All hard-coded values, thresholds, model names, file-type mappings, API endpoints, timeouts, and paths live here. Use **Pydantic `BaseSettings`** with env variable support.

```python
# Pattern — not the literal file, just the shape:
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    llm_provider: str = "local"           # "local" | "anthropic" | "openai" | "stub"
    stub_mode: bool = False               # True = all LLM calls return deterministic constants
    local_model_url: str = "http://localhost:11434"
    supported_extensions: dict[str, str]   # {".pdf": "pdf", ".png": "image", ...}
    max_concurrent_files: int = 10
    classification_categories: list[str]   # the target categories
    ...
```

**Never scatter magic strings or config values in business logic.** Always import from `settings`.

### Stub Mode (Sensitive Data / Personal Laptop)

Data is sensitive and must not leave the company network. When developing on a personal laptop:

- Set `llm_provider=stub` (or `stub_mode=True`) — every `LLMProvider` call returns a **fixed, deterministic constant response** defined in the stub implementation. No real model is invoked, no data is sent anywhere.
- The `StubProvider` lives in `src/classifier/providers/stub.py`. It implements the same `LLMProvider` protocol and returns predefined classification results (e.g. `{"category": "STUB_CATEGORY", "confidence": 0.99}`).
- Stub responses should be realistic in **shape** (same Pydantic models, same fields) but obviously fake in **content** so they're never mistaken for real results.
- Use stub fixtures in `fixtures/` for development — never copy real sensitive files to a personal machine.
- Final validation with real data and real models happens **only on the company computer**.
- The DI container in `main.py` selects `StubProvider` when stub mode is active — no other code needs to know or care.

---

## Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Run tests — always do this after any code change
pytest tests/ -v

# Run a single test file
pytest tests/unit/test_extractors.py -v

# Run with coverage
pytest tests/ --cov=src/classifier --cov-report=term-missing

# Type checking
mypy src/

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Run CLI
python -m classifier.cli --input path/to/file_or_folder

# Run API server
uvicorn classifier.api:app --reload
```

---

## Testing Rules — Strictly Enforced

1. **Every public function and class gets a test.** No exceptions.
2. Use `pytest` with `pytest-asyncio` for async tests.
3. **All external dependencies must be mocked in unit tests** — LLM calls, file I/O, HTTP requests. Use `unittest.mock.AsyncMock` for async mocks.
4. Fixtures go in `conftest.py`. Create factory fixtures for mock extractors, mock providers, etc.
5. Use `tmp_path` (pytest built-in) for any test that touches the filesystem.
6. Tests must be fast. If a test needs a real LLM call, mark it `@pytest.mark.integration` and skip by default.
7. **After writing or modifying any code, immediately write or update the corresponding tests. Do not defer testing to later.**
8. Aim for clear test names: `test_pdf_extractor_returns_text_from_valid_pdf`, not `test_pdf_1`.

---

## Code Style

- Python 3.11+.
- Type hints on every function signature and return type — no exceptions.
- Pydantic models for all data structures that cross boundaries (API input/output, config, classification results).
- Docstrings on every public class and function (one-liner is fine if intent is obvious).
- No bare `except`. Always catch specific exceptions.
- Prefer `pathlib.Path` over `os.path` string manipulation.
- Use `logging` (stdlib) — never `print()` for runtime output.

---

## LLM Call Conventions

- All LLM interactions go through an `LLMProvider` implementation. No direct HTTP calls to LLM APIs from business logic.
- Prompt templates live in their own module or as class attributes on the `Classifier` — not inline strings buried in logic.
- Always handle LLM failures gracefully: timeouts, rate limits, malformed responses. Return a structured error, never crash.
- Log every LLM call (prompt summary + latency) at DEBUG level.
- **Multi-step classifiers:** A classifier may chain several LLM calls with code logic in between. Each intermediate step should be its own testable method (e.g. `_extract_features()`, `_normalize()`, `_assign_category()`, `_validate()`). This keeps each step independently mockable and testable even though the classifier orchestrates them as a pipeline. If the chain can fail at any step, return partial results or a structured error that identifies which step failed.

---

## Workflow for New Features

1. **Ask the user** what the feature does, its edge cases, and where it fits.
2. Define or extend the relevant Protocol if needed.
3. Write the implementation.
4. Write tests (unit first, integration if applicable).
5. Register the new component in `main.py` / DI wiring.
6. Run the full test suite and confirm everything passes.
7. Update this CLAUDE.md if the feature introduces a new pattern or convention.

---

## Common Pitfalls — Avoid These

- Do NOT put LLM provider selection logic inside a classifier. That's the DI container's job.
- Do NOT read files synchronously when an async path exists.
- Do NOT add a new config value anywhere other than `config/settings.py`.
- Do NOT write a test that depends on a real external service without the `@pytest.mark.integration` marker.
- Do NOT create god classes. If a class does more than one thing, split it.
- Do NOT commit real/sensitive data files to the repo. Test fixtures must use synthetic data only.
- Do NOT bypass stub mode by hardcoding a real provider. If `stub_mode` is on, every LLM call must return the constant.