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

## Key Patterns

- **Singleton**: `Laminar` and `TracerWrapper` are singletons initialized once at startup
- **Provider instrumentors** extend OpenTelemetry's base instrumentor and are dynamically initialized based on `Instruments` enum
- **Tests use VCR.py** to record/replay HTTP responses in `tests/cassettes/`
- **Tests use InMemorySpanExporter** configured in `tests/conftest.py` for span assertions

## Dev dependencies

- All entries in the `[dependency-groups].dev` section of `pyproject.toml` MUST be pinned to a specific version with `==X.Y.Z`. Do NOT use unbounded specifiers (`>=`, `^`, `~`, `<`, ranges, or bare package names). Pinning keeps the test matrix deterministic across developers and CI. When adding a new dev dep, look up the current release on https://pypi.org and pin to that exact version; bumps then go through a normal PR.

## examples/hermes-plugin

- Standalone pip package that bridges [Hermes Agent](https://github.com/nousresearch/hermes-agent) plugin hooks to Laminar spans. Ships via the `hermes_agent.plugins` entry-point group (`lmnr-hermes = "lmnr_hermes:register"`), so `pip install -e examples/hermes-plugin` plus `hermes plugins enable lmnr-hermes` is enough — no pip publish needed.
- The `examples/hermes-plugin` directory is a uv workspace member; adding a new example dir with `tool.uv.sources.lmnr = { workspace = true }` requires updating `[tool.uv.workspace].members` in the root `pyproject.toml` or uv errors with "not a workspace member".
- Hermes calls hooks on different threads (ThreadPoolExecutor for concurrent tool calls; delegate/subagent workers). The plugin keeps a `session_id → turn span` map and parents tool spans via `parent_span_context=Laminar.get_laminar_span_context_dict(turn_span)` rather than relying on the OTel current context, which is thread-local.
- Laminar writes session_id as the attribute `lmnr.association.properties.session_id` (prefix = `ASSOCIATION_PROPERTIES` constant). Tests asserting session scoping must check this exact key, not `session_id` or `SESSION_ID`.

## Environment Variables

```
LMNR_PROJECT_API_KEY  # API key (can also pass to initialize())
LMNR_BASE_URL         # API base URL (default: https://api.lmnr.ai)
```

## pydantic_ai instrument

- `Instruments.PYDANTIC_AI` is **auto-enabled by default** when `pydantic-ai-slim` (or `pydantic-ai`) is installed. When auto-enabled, the overlapping raw-provider instrumentors — OPENAI, ANTHROPIC, GOOGLE_GENAI, GROQ, MISTRAL, COHERE, BEDROCK — are auto-removed from the default set so the same model call isn't traced twice (pydantic_ai emits its own GenAI spans at the model abstraction layer). The exact set lives in `_PYDANTIC_AI_PROVIDER_CONFLICTS` in `src/lmnr/opentelemetry_lib/tracing/instruments.py`.
- To opt out of the auto-enable, pass `disabled_instruments={Instruments.PYDANTIC_AI}` to `Laminar.initialize`. To keep both pydantic_ai and the raw SDK instrumentors active (accepting duplicate spans), pass an explicit `instruments` set that includes both.
- Installation: `pip install lmnr pydantic-ai-slim>=1.0` — there is no `[pydantic-ai]` extra in `pyproject.toml`.
- The instrumentor does two things:
  1. Calls `Agent.instrument_all(InstrumentationSettings(...))`, which sets a module-level default on pydantic_ai's `Agent` class so subsequent `Agent()` instances pick it up automatically.
  2. Monkey-patches `pydantic_ai.models.instrumented.InstrumentationSettings.__init__` so that *every* construction of `InstrumentationSettings` — including ones triggered by user code calling `Agent.instrument_all(True)` or `Agent(instrument=True)` — defaults to our tracer provider and uses semconv `version=5` when the caller didn't pick one (or picked the legacy `version=1`). User-supplied `tracer_provider=` is respected; a caller-supplied `version >= 2` is also respected unchanged, so callers pinning to an older-but-still-supported semconv contract keep their choice.
- `_instrument` performs no version/installation guards of its own: `BaseInstrumentor.instrument()` already enforces the `instrumentation_dependencies()` constraint (`pydantic-ai-slim >= 1.0.0`), and `PydanticAIInstrumentorInitializer` in `_instrument_initializers.py` only returns this instrumentor when `pydantic-ai-slim` or `pydantic-ai` is installed.
- The test suite runs with `disabled_instruments={Instruments.PYDANTIC_AI}` in `tests/conftest.py` so the raw-provider instrumentor tests still operate on the SDK-level spans they assert on.
