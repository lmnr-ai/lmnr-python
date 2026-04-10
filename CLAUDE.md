# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Laminar Python SDK - tracing and evaluation framework for LLM applications. Provides automatic instrumentation for LLM providers (OpenAI, Anthropic, Groq, etc.), manual tracing via decorators, and an evaluation framework for testing LLM outputs.

## Commands

```bash
# Install dependencies with all extras
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_initialize.py

# Run specific test
uv run pytest tests/test_initialize.py::test_laminar_initialize_url_parsing -v

# Lint
uv run flake8 src/

# Format
uv run autopep8 --in-place --aggressive src/lmnr/**/*.py

# CLI commands
lmnr eval <file.py>           # Run evaluations
lmnr dev <file.py> --function <name>  # Interactive debugger
lmnr datasets pull <id>       # Pull dataset
```

## Architecture

### Core SDK (`src/lmnr/sdk/`)
- `laminar.py` - Main `Laminar` class with `initialize()` entry point and span management
- `decorators.py` - `@observe()` decorator for manual instrumentation
- `evaluations.py` - `evaluate()` function and `Evaluation` class for running tests
- `client/` - HTTP clients (`LaminarClient`, `AsyncLaminarClient`) for API interactions

### OpenTelemetry Layer (`src/lmnr/opentelemetry_lib/`)
- `tracing/instruments.py` - `Instruments` enum listing all auto-instrumentable providers
- `tracing/__init__.py` - `TracerWrapper` singleton managing OTEL setup
- `opentelemetry/instrumentation/` - Provider-specific instrumentors (anthropic/, openai/, groq/, etc.)
- Each provider instrumentor has: `__init__.py` (instrumentor class), `utils.py` (parsing), `span_utils.py` (span creation)

### CLI (`src/lmnr/cli/`)
- `evals.py` - `lmnr eval` command
- `dev.py` - `lmnr dev` interactive debugger

## Instrumentation Notes

### Adding new sandbox/tool instrumentors
- Use the Daytona instrumentor (`opentelemetry/instrumentation/daytona/`) as the reference template for sandbox-style instrumentors. It covers sync/async wrapping, log emission, and background log streaming patterns.
- Registration requires three changes: add to `Instruments` enum in `tracing/instruments.py`, add an initializer class in `tracing/_instrument_initializers.py`, and add the mapping entry in `INSTRUMENTATION_INITIALIZERS`.

### Modal SDK specifics
- Modal uses the `synchronicity` library to auto-generate sync/async variants from a single `_Sandbox` class. `wrapt.wrap_function_wrapper` works on the public `modal.sandbox.Sandbox` class despite this indirection — no need to register separate async wrapper specs.
- `Sandbox.exec()` returns a `ContainerProcess` with lazy `stdout`/`stderr` iterators. To capture logs, wrap the returned process object rather than trying to read output inside the wrapper.

## Key Patterns

- **Singleton**: `Laminar` and `TracerWrapper` are singletons initialized once at startup
- **Provider instrumentors** extend OpenTelemetry's base instrumentor and are dynamically initialized based on `Instruments` enum
- **Tests use VCR.py** to record/replay HTTP responses in `tests/cassettes/`
- **Tests use InMemorySpanExporter** configured in `tests/conftest.py` for span assertions

## Environment Variables

```
LMNR_PROJECT_API_KEY  # API key (can also pass to initialize())
LMNR_BASE_URL         # API base URL (default: https://api.lmnr.ai)
```
