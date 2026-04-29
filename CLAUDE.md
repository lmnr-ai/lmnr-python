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

## Environment Variables

```
LMNR_PROJECT_API_KEY  # API key (can also pass to initialize())
LMNR_BASE_URL         # API base URL (default: https://api.lmnr.ai)
```

## Instrumentation tests (VCR)

- Tests under `tests/test_instrumentations/**` replay via VCR cassettes in `cassettes/<module-name>/`. Sensitive headers are filtered by the root `vcr_config` (`authorization`, `api-key`, `x-api-key`, `x-goog-api-key`).
- The sandbox sets `ANTHROPIC_BASE_URL` / `ANTHROPIC_BEDROCK_BASE_URL` to a local proxy; when recording new anthropic cassettes, **unset those two env vars** first or the cassette will record `http://127.0.0.1:...` and fail to replay against `api.anthropic.com`. Use `unset ANTHROPIC_BASE_URL ANTHROPIC_BEDROCK_BASE_URL` before running `pytest --record-mode=once`.
- Re-record a single module with `rm -rf tests/.../cassettes/<module>/ && pytest <path> --record-mode=once`. VCR's default `record_mode` is `once`, so replays work without flags as long as the cassette exists.

## Anthropic instrumentor

- `messages.parse` (the structured-output helper introduced in `anthropic>=0.59`) is wrapped with the same `_wrap` / `_awrap` that wrap `messages.create` — the resulting span name is `anthropic.chat` and the attribute set is identical, plus `gen_ai.request.structured_output_schema`.
- Schema extraction lives in `span_utils._extract_structured_output_schema`. It handles: (1) `output_format=PydanticModel` via `model_json_schema()`, (2) `output_format=<any TypeAdapter-compatible type>` via `pydantic.TypeAdapter(...).json_schema()`, (3) raw-dict `output_format` (bare schema or `JSONOutputFormatParam`), and (4) `output_config={"format": {"type": "json_schema", "schema": {...}}}` on `messages.create`.
- `anthropic.lib.bedrock._beta_messages.Messages` does **not** expose `parse` (only regular + `beta.messages.messages` do). `_instrument` already swallows `ModuleNotFoundError` / `AttributeError` so the missing bedrock.parse attempt is a silent no-op.

## deepagents instrument

- `Instruments.DEEPAGENTS` is auto-enabled when `deepagents` is installed. When auto-enabled, `LANGCHAIN` and `LANGGRAPH` are auto-removed from the default instrument set (see `_DEEPAGENTS_NOISE_CONFLICTS` in `tracing/instruments.py`) — the LangSmith-style node-level spans they emit add no signal on top of what `LaminarMiddleware` already captures, and clutter the transcript view.
- Root span (`deep_agent`, type DEFAULT) is opened by wrapping the compiled graph's `invoke`/`ainvoke`/`stream`/`astream` at the instance level in `DeepagentsInstrumentor._wrap_graph_methods`, NOT from `AgentMiddleware.before_agent`/`after_agent`. LangGraph runs each middleware hook as its own graph node (a separate task), so OTel context attached in `before_agent` is popped before downstream tool nodes execute and the root span fails to parent subsequent tool spans (symptom: tool spans emit with `parent_span_id=0` on a new trace_id). Wrapping the graph entrypoint keeps the context active for the whole execution.
- `Pregel.invoke` internally calls `self.stream`, so wrapping both without a guard produces two root spans per top-level call. A `contextvars.ContextVar` sentinel (`_root_active` in `deepagents/instrumentor.py`) collapses that to one.
- Subagent cards in the frontend transcript are derived from `lmnr.span.prompt_hash` fingerprinting (see `computeSubagentBoundaries` in `frontend/components/traces/trace-view/store/utils.ts`) — no dedicated subagent span type is needed. A TOOL span named `task` nesting the subagent's LLM/tool spans is enough; the frontend groups them automatically.
- `deepagents` depends on `langchain>=1.0`. `DeepagentsInstrumentorInitializer` returns `None` unless both `deepagents` and `langchain` are installed, so the instrumentor is a silent no-op in environments where only one is present.
- Association props lifecycle: when calling `set_association_props_in_context(span)`, store the returned token on the span as `span._lmnr_assoc_props_token` (see `decorators/__init__.py`, `sdk/laminar.py`). `LaminarSpan.end()` detaches it automatically. Stashing the token on an ad-hoc handle without detaching it leaks an isolated-context attachment per span.
- Stream wrappers (`_wrap_graph_stream` / `_awrap_graph_stream`) must open the span inside the returned generator, not eagerly in the outer function: if the caller never iterates, eager `_SpanHandle.__enter__()` would never be paired with `__exit__`. However, the `_root_active` sentinel must NOT be mutated inside a sync generator body — Python leaks ContextVar changes from sync generators to the caller's context on `yield`, which would make a second concurrent `graph.stream()` call short-circuit and skip instrumentation. The sentinel only needs to be set by `_wrap_graph_invoke` (in its plain-function frame) to collapse `Pregel.invoke → self.stream` into one root span.

## pydantic_ai instrument

- `Instruments.PYDANTIC_AI` is **auto-enabled by default** when `pydantic-ai-slim` (or `pydantic-ai`) is installed. When auto-enabled, the overlapping raw-provider instrumentors — OPENAI, ANTHROPIC, GOOGLE_GENAI, GROQ, MISTRAL, COHERE, BEDROCK — are auto-removed from the default set so the same model call isn't traced twice (pydantic_ai emits its own GenAI spans at the model abstraction layer). The exact set lives in `_PYDANTIC_AI_PROVIDER_CONFLICTS` in `src/lmnr/opentelemetry_lib/tracing/instruments.py`.
- To opt out of the auto-enable, pass `disabled_instruments={Instruments.PYDANTIC_AI}` to `Laminar.initialize`. To keep both pydantic_ai and the raw SDK instrumentors active (accepting duplicate spans), pass an explicit `instruments` set that includes both.
- Installation: `pip install lmnr pydantic-ai-slim>=1.0` — there is no `[pydantic-ai]` extra in `pyproject.toml`.
- The instrumentor does two things:
  1. Calls `Agent.instrument_all(InstrumentationSettings(...))`, which sets a module-level default on pydantic_ai's `Agent` class so subsequent `Agent()` instances pick it up automatically.
  2. Monkey-patches `pydantic_ai.models.instrumented.InstrumentationSettings.__init__` so that *every* construction of `InstrumentationSettings` — including ones triggered by user code calling `Agent.instrument_all(True)` or `Agent(instrument=True)` — defaults to our tracer provider and uses semconv `version=5` when the caller didn't pick one (or picked the legacy `version=1`). User-supplied `tracer_provider=` is respected; a caller-supplied `version >= 2` is also respected unchanged, so callers pinning to an older-but-still-supported semconv contract keep their choice.
- `_instrument` performs no version/installation guards of its own: `BaseInstrumentor.instrument()` already enforces the `instrumentation_dependencies()` constraint (`pydantic-ai-slim >= 1.0.0`), and `PydanticAIInstrumentorInitializer` in `_instrument_initializers.py` only returns this instrumentor when `pydantic-ai-slim` or `pydantic-ai` is installed.
- The test suite runs with `disabled_instruments={Instruments.PYDANTIC_AI}` in `tests/conftest.py` so the raw-provider instrumentor tests still operate on the SDK-level spans they assert on.
