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
- **Deepagents wins over pydantic_ai when both are installed**: `init_instrumentations` normally strips the raw LLM-provider instrumentors (`OPENAI`, `ANTHROPIC`, …) when `pydantic_ai` is auto-enabled (`_PYDANTIC_AI_PROVIDER_CONFLICTS`), but the deepagents instrumentation relies on those raw instrumentors to emit LLM spans underneath each `TOOL` span — without them the `deep_agent` trace shows only root + tool spans and no LLM children. When `_deepagents_installed()` is true, `init_instrumentations` skips the pydantic_ai conflict-removal so the providers stay enabled. Users with both libraries who want pydantic_ai's de-dup back can either block deepagents (`block_instruments={Instruments.DEEPAGENTS}`) or pass an explicit `instruments` set.
- Root span (`deep_agent`, type DEFAULT) is opened by wrapping the compiled graph's `invoke`/`ainvoke`/`stream`/`astream` at the instance level in `DeepagentsInstrumentor._wrap_graph_methods`, NOT from `AgentMiddleware.before_agent`/`after_agent`. LangGraph runs each middleware hook as its own graph node (a separate task), so OTel context attached in `before_agent` is popped before downstream tool nodes execute and the root span fails to parent subsequent tool spans (symptom: tool spans emit with `parent_span_id=0` on a new trace_id). Wrapping the graph entrypoint keeps the context active for the whole execution.
- `Pregel.invoke` internally calls `self.stream`, so wrapping both without a guard produces two root spans per top-level call. A `contextvars.ContextVar` sentinel (`_root_active` in `deepagents/instrumentor.py`) collapses that to one.
- Subagent cards in the frontend transcript are derived from `lmnr.span.prompt_hash` fingerprinting (see `computeSubagentBoundaries` in `frontend/components/traces/trace-view/store/utils.ts`) — no dedicated subagent span type is needed. A TOOL span named `task` nesting the subagent's LLM/tool spans is enough; the frontend groups them automatically.
- `deepagents` depends on `langchain>=1.0`. `DeepagentsInstrumentorInitializer` returns `None` unless both `deepagents` and `langchain` are installed, so the instrumentor is a silent no-op in environments where only one is present.
- Span-lifetime split: the `invoke`/`ainvoke` and tool-call wrappers use `Laminar.start_as_current_span(...)` as a context manager — the span lives for exactly the `with` block. `stream`/`astream` cannot — the wrapper returns before the caller iterates, so a `with start_as_current_span(...)` around the wrapper body would end the span before any chunk is yielded. Use `Laminar.start_span(...)` to mint the span imperatively *inside* the generator, then re-activate it with `Laminar.use_span(span, end_on_exit=True)`; that handles the 3-way context attach (OTel global context, Laminar isolated context, TracerWrapper span stack), lets `@contextmanager`'s built-in `GeneratorExit` handling leave the span status as UNSET on `break`, and ends the span when the generator closes. Do NOT hand-roll a `_SpanHandle`-style wrapper for this — `use_span(..., end_on_exit=True)` is the public API and is what every other lmnr-python instrumentor uses.
- Stream wrappers (`_wrap_graph_stream` / `_awrap_graph_stream`) must open the span inside the returned generator, not eagerly in the outer function: if the caller never iterates, an eagerly-opened span would never be ended. Additionally, the `_root_active` sentinel must NOT be mutated inside a sync generator body — Python leaks ContextVar changes from sync generators to the caller's context on `yield`, which would make a second concurrent `graph.stream()` call short-circuit and skip instrumentation. The sentinel only needs to be set by `_wrap_graph_invoke` (in its plain-function frame) to collapse `Pregel.invoke → self.stream` into one root span.

## Langfuse bridge

- `Instruments.LANGFUSE` is **auto-enabled by default** when `langfuse >= 3.0` is installed. When auto-enabled, the overlapping raw-provider instrumentors — `OPENAI`, `ANTHROPIC`, `GOOGLE_GENAI`, `GROQ`, `MISTRAL`, `COHERE`, `BEDROCK`, `LANGCHAIN` — are auto-removed (see `_LANGFUSE_PROVIDER_CONFLICTS` in `tracing/instruments.py`) because Langfuse's own `langfuse.openai`, `langfuse.langchain`, and `@observe` wrappers already emit GenAI/tool spans. Deepagents still wins over Langfuse (same reasoning as pydantic_ai): deepagents needs the raw providers to emit LLM children under tool spans.
- `_langfuse_installed()` in `tracing/instruments.py` gates on **both** `is_package_installed("langfuse")` and `get_package_version("langfuse") >= 3.0.0`. Bare presence is not enough: the bridge initializer (`LangfuseInstrumentorInitializer`) returns `None` for 2.x, so if `_langfuse_installed()` reported True on 2.x the `_LANGFUSE_PROVIDER_CONFLICTS` auto-removal would still strip the raw-provider instrumentors and leave nothing installed in their place.
- The bridge lives in `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/langfuse/__init__.py` and has two parts:
  1. `LangfuseInstrumentor.instrument(lmnr_tracer_provider, lmnr_span_processor)` attaches Laminar's `SpanProcessor` to every Langfuse-owned `TracerProvider` by iterating `LangfuseResourceManager._instances` and monkey-patching `_initialize_instance` so Langfuse clients created AFTER the bridge installs also get dual-attached. `_handled_providers: set[int]` (keyed by `id()`) prevents double-attach; the Laminar provider itself is skipped.
  2. `LangfuseAttributeTranslator` is a `SpanProcessor` that mutates Langfuse-scoped spans on `on_end` — rewriting `langfuse.observation.model.name` → `gen_ai.request.model`/`gen_ai.response.model`, `langfuse.observation.usage_details` → `gen_ai.usage.{input,output}_tokens`/`llm.usage.total_tokens`, `langfuse.observation.cost_details` → `gen_ai.usage.{input,output,}_cost`, `langfuse.observation.input/output` → `lmnr.span.input/output`, and observation-type mapping (`generation|completion|embedding` → `SPAN_TYPE=LLM`, `tool` → `SPAN_TYPE=TOOL`). Trace-level attrs `session.id` / `user.id` / `langfuse.trace.tags` / `langfuse.trace.metadata.*` map into `lmnr.association.properties.{session_id,user_id,tags,metadata.*}`.
- `LangfuseInstrumentor` does NOT extend `BaseInstrumentor` because `BaseInstrumentor.instrument()` ignores kwargs other than `tracer_provider`/`logger_provider`, and the bridge needs the caller-supplied Laminar `SpanProcessor`. `init_instrumentations` has a special-case branch (`if instrument == Instruments.LANGFUSE: instrumentor.instrument(lmnr_tracer_provider=..., lmnr_span_processor=...)`).
- `LangfuseInstrumentor` idempotency state (`_installed`, `_translator`, `_original_initialize_instance`, `_lmnr_span_processor`) is stored **on the class, not the instance** — both `Laminar.connect_to_langfuse()` and `init_instrumentations` construct a fresh `LangfuseInstrumentor()` each call, so `self.X = v` would create an instance attribute that shadows the class default and every new instance would read the unshadowed `_installed = False`, duplicating the translator processor and layering monkey-patches. Always assign via `type(self).X = v` (or `LangfuseInstrumentor.X = v`) in `instrument`/`uninstrument`/`_patch_resource_manager`/`_unpatch_resource_manager`. Reads via `self._X` are fine because attribute lookup falls through to the class.
- `LangfuseAttributeTranslator.on_end` cannot use `span.set_attribute(...)` to inject the translated attrs: (a) `on_end` is called with an OTel `ReadableSpan`, which has no `set_attribute` method, and (b) even when the object happens to be the recording `Span`, its `end()` has already run so `set_attribute` is a silent no-op. The translator writes directly to `span._attributes[k] = v` — safe because `ReadableSpan._attributes` is the *same dict object* as the recording `Span._attributes`, so the exporter sees the mutation when it serializes the span.
- Processor **order** on the shared `TracerProvider` matters. `SynchronousMultiSpanProcessor` runs `on_end` in insertion order, and when `disable_batch=True` the Laminar exporter is backed by `SimpleSpanProcessor`, which `export()`s synchronously inside its own `on_end`. If the translator were appended after the exporter (the default of `add_span_processor`), the exporter would ship the pre-translation `langfuse.*` shape. The bridge uses `_prepend_span_processor` to reorder the underlying `_span_processors` tuple so the translator is first. Tests previously masked this because `InMemorySpanExporter` stores span *references* and mutations stick — the SimpleSpanProcessor regression test (`test_translator_mutates_before_synchronous_exporter`) exercises the real timing.
- Public API: `Laminar.connect_to_langfuse()` manually installs the bridge — only needed when `Laminar.initialize(instruments={...})` omits `Instruments.LANGFUSE`. Returns `True` on success, `False` if Laminar isn't initialized, langfuse isn't importable, or langfuse is < 3.0. Gates on `_langfuse_installed()` (same helper the auto-enable path uses, which version-gates on >= 3.0) rather than `is_package_installed("langfuse")` alone — on 2.x `instrument()` would attach a useless translator and flip `_installed=True`, blocking any later valid install. Returns `LangfuseInstrumentor._installed` (not a hard-coded `True`) so early-return paths inside `instrument()` surface as False. Idempotent (the class-level `_installed` flag collapses repeat calls to no-ops).
- The test suite blocks `Instruments.LANGFUSE` in `tests/conftest.py` for the same reason it blocks `PYDANTIC_AI`: dev deps include `langfuse==3.14.6`, and without the block the `_LANGFUSE_PROVIDER_CONFLICTS` auto-removal would strip OPENAI/ANTHROPIC/etc. from the default set and break every raw-provider test.
- Trace-level attrs in langfuse 3.x live on the **root** span only — `session.id` / `user.id` are NOT inherited onto children by Langfuse itself. The translator applies them only where they already exist (root span). Tests in `test_langfuse.py` exercise each translation branch against a real `Langfuse()` client + `InMemorySpanExporter`.
- Langfuse emits usage/cost as JSON-encoded strings over OTel (`langfuse.observation.usage_details = '{"input":10,...}'`) because OTel attribute values can't be dicts. The translator `_parse_json`s them; it also falls back gracefully if the value is already a dict (shouldn't happen at on_end, but cheap to handle).
- `LangfuseInstrumentor.uninstrument()` is the inverse of `instrument()` — it must remove the translator (and the Laminar `SpanProcessor`) from every provider it was attached to AND clear every class-level bookkeeping field (`_installed`, `_handled_providers`, `_attached_providers`, `_translator`, `_lmnr_span_processor`, `_lmnr_tracer_provider`). Clearing only `_installed` is a bug: a subsequent `instrument()` call would prepend a second translator onto Laminar's provider (the first was never removed) and `_handled_providers` would still carry stale `id()` values so `_attach_to_existing_langfuse_providers` would short-circuit on Langfuse providers it already saw. The bridge tracks every provider it touches in `_attached_providers` (keyed by `id()`, value = the provider) for this reason; `_remove_span_processor` mirrors `_prepend_span_processor` and walks the `_active_span_processor._span_processors` tuple under its lock to drop the processor.
- The "skip Laminar's own provider" guard in `_attach_to_provider` cannot rely on `TracerWrapper.verify_initialized()` alone: during the auto-install path, `init_instrumentations` is called by `TracerWrapper.__new__` BEFORE `cls.instance = obj` is assigned (see `tracing/__init__.py`), so the guard's `TracerWrapper.verify_initialized()` branch returns False and a Langfuse client that shares the newly-created Laminar provider would double-attach the translator + Laminar span processor. `instrument()` pre-registers `id(lmnr_tracer_provider)` into `_handled_providers` up-front so the short-circuit is id-based and independent of TracerWrapper lifecycle.
- `instrument()` prepends the translator to Laminar's provider *before* running the attach-to-existing + resource-manager-patch phases. If either of those later phases raises (e.g. a `RuntimeError` walking `LangfuseResourceManager._instances` under concurrent modification), the translator would be orphaned on the provider while `_installed` stayed False — a subsequent `instrument()` call (e.g. via `Laminar.connect_to_langfuse()`) would pass the guard and prepend a SECOND translator, double-translating every Langfuse span. The attach + patch phases are therefore wrapped in a `try/except` that calls `uninstrument()` before re-raising. `_installed=True` is set *before* the fallible block so `uninstrument()` (which short-circuits when `_installed=False`) actually runs the cleanup path.
- **Langfuse wins over pydantic_ai** when both are installed (no deepagents). Each conflict set only strips raw providers (`OPENAI`, `ANTHROPIC`, …); without a priority rule between the two top-level enums, both `Instruments.LANGFUSE` and `Instruments.PYDANTIC_AI` would auto-enable and produce duplicate spans — pydantic_ai's model-layer GenAI spans AND Langfuse's `@observe`/`langfuse.openai` wrapper spans for the same call. When `langfuse_active and not deepagents_active`, `init_instrumentations` additionally strips `{Instruments.PYDANTIC_AI}` from the set. Priority ladder is `deepagents > langfuse > pydantic_ai`: a user who installed Langfuse chose it as their trace surface, and pydantic_ai's auto-enable is incidental. Callers who want pydantic_ai alongside Langfuse can pass an explicit `instruments` set.

## pydantic_ai instrument

- `Instruments.PYDANTIC_AI` is **auto-enabled by default** when `pydantic-ai-slim` (or `pydantic-ai`) is installed. When auto-enabled, the overlapping raw-provider instrumentors — OPENAI, ANTHROPIC, GOOGLE_GENAI, GROQ, MISTRAL, COHERE, BEDROCK — are auto-removed from the default set so the same model call isn't traced twice (pydantic_ai emits its own GenAI spans at the model abstraction layer). The exact set lives in `_PYDANTIC_AI_PROVIDER_CONFLICTS` in `src/lmnr/opentelemetry_lib/tracing/instruments.py`.
- To opt out of the auto-enable, pass `disabled_instruments={Instruments.PYDANTIC_AI}` to `Laminar.initialize`. To keep both pydantic_ai and the raw SDK instrumentors active (accepting duplicate spans), pass an explicit `instruments` set that includes both.
- Installation: `pip install lmnr pydantic-ai-slim>=1.0` — there is no `[pydantic-ai]` extra in `pyproject.toml`.
- The instrumentor does two things:
  1. Calls `Agent.instrument_all(InstrumentationSettings(...))`, which sets a module-level default on pydantic_ai's `Agent` class so subsequent `Agent()` instances pick it up automatically.
  2. Monkey-patches `pydantic_ai.models.instrumented.InstrumentationSettings.__init__` so that *every* construction of `InstrumentationSettings` — including ones triggered by user code calling `Agent.instrument_all(True)` or `Agent(instrument=True)` — defaults to our tracer provider and uses semconv `version=5` when the caller didn't pick one (or picked the legacy `version=1`). User-supplied `tracer_provider=` is respected; a caller-supplied `version >= 2` is also respected unchanged, so callers pinning to an older-but-still-supported semconv contract keep their choice.
- `_instrument` performs no version/installation guards of its own: `BaseInstrumentor.instrument()` already enforces the `instrumentation_dependencies()` constraint (`pydantic-ai-slim >= 1.0.0`), and `PydanticAIInstrumentorInitializer` in `_instrument_initializers.py` only returns this instrumentor when `pydantic-ai-slim` or `pydantic-ai` is installed.
- The test suite runs with `disabled_instruments={Instruments.PYDANTIC_AI}` in `tests/conftest.py` so the raw-provider instrumentor tests still operate on the SDK-level spans they assert on.
