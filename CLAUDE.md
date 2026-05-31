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

### Debug mode / replay (`src/lmnr/sdk/debug/`)
- A "debug run" is a normal process started with `LMNR_DEBUG*` env vars; there is no cache server, SSE, orchestrator, or `lmnr dev` command anymore (all removed in the debugger rework). The replay cache lives in-process.
- This is a **cross-language parity surface** with the TS SDK `lmnr-ts/packages/lmnr/src/debug/`. Keep `config.py`, `spine.py`, `replay_cache.py`, `pointer.py`, `replay.py`, `source_trace.py`, `__init__.py` line-comparable. The shared test vectors in `tests/data/debug/*.json` are byte-identical copies of `lmnr-ts/packages/lmnr/test/data/debug/`; change them in lockstep across both repos.
- Parity invariants that MUST match the TS SDK: truthy set `["true","1","yes","on"]`; `replay_enabled = replay_trace_id is not None and cache_until > 0`; pointer key order `[trace_id, session_id, replay_trace_id, cache_until, debugger_url, started_at]`; `CONSOLE_PREFIX = "LMNR_DEBUG_RUN "`; spine = shallowest looping path / tie-break earliest start / fallback shallowest single call; `ReplayCache` truncates `payloads[:cache_until]`.
- The pointer's **`started_at` is captured at `DebugRuntime` construction (SDK init), NOT inside `build_pointer`/`emit_pointer`**. The pointer is emitted at shutdown, so reading the clock in `build_pointer` would stamp `started_at` near process exit rather than run start. `DebugRuntime.__init__` records `self._started_at` and passes it through to `build_pointer(..., started_at=...)`; the `started_at` param defaults to now only for standalone callers (the pointer unit tests). TS mirrors this with `DebugRuntime._startedAt` → `buildPointer({ startedAt })`.
- **Disabled tracing must NOT mask span names in the path while a replay run is active.** `LaminarSpanProcessor.on_start` normally rewrites each span's name to `"_"` in `lmnr.span.path` when `LMNR_DISABLE_TRACING=true` (privacy — the span is never exported anyway). But the debug replay wrappers (`rollout.py` → `span_path_from_span` → `cached_payload_for`) read that SAME in-process `lmnr.span.path` attribute to match the replay cache, which keys on the source trace's REAL dotted paths from SQL. A masked `"_"` path never matches, so replay silently runs live while the occurrence counter still advances. `on_start` therefore keeps the real `span.name` when `_replay_active()` (a lazy import of `replay.replay_enabled()`) is true; the span still isn't exported, so nothing leaks. This is a **Python-only divergence**: the TS processor never masks span names (`processor.ts` always uses `span.name`), so TS replay-while-disabled already works without a guard. Tested in `tests/test_debug_runtime.py` (`test_processor_keeps_real_span_path_for_replay_when_disabled` / `..._masks_span_path_when_disabled_without_replay`).
- **`replay_enabled()` gates on `get_runtime().replay_configured`, NOT on `get_runtime() is not None`.** A debug-no-replay run (`LMNR_DEBUG` set but no `LMNR_DEBUG_REPLAY_TRACE_ID` / `LMNR_DEBUG_CACHE_UNTIL`) still creates a runtime — needed to stamp `rollout.session_id` and emit the run pointer — but replay must stay off there, or every LLM call resolves a span path and advances a per-path occurrence counter against a cache that will never be built. `replay_configured` is `self._config.replay_enabled` (`replay_trace_id is not None and cache_until > 0`). TS mirrors this with `DebugRuntime.replayConfigured` — keep both in lockstep. In TS the getter also stays true during the async cache-load window (`_cache is None` but replay configured) so the counter still advances; Python builds the cache synchronously and has no such window, so its getter is purely for line-parity.
- The per-path **occurrence counter lives on `DebugRuntime` (`_counters`), NOT on `ReplayCache`**. `DebugRuntime.get_cached` advances the counter unconditionally — even while `_cache is None` — then consults the cache only if present. This is for parity with TS, where `init_debug_runtime` fills the cache asynchronously (`set_cache`) after returning, so spine calls in the loading window would otherwise run live without consuming a slot and misalign record-vs-replay after the cache lands. Python builds the cache synchronously and has no such window, but mirrors the design so the two `__init__.py`/`index.ts` stay line-comparable. `ReplayCache.get_cached(span_path, occurrence)` is now a pure lookup with no internal counter.
- Naming: new env/module naming is `debug`; the persisted metadata key stays `rollout.session_id` (intentional — do not rename). The provider wrappers' files are still named `rollout.py` (e.g. `instrumentation/openai/rollout.py`) but now import from `lmnr.sdk.debug.replay` and serve from the in-process `ReplayCache`.
- `init_debug_runtime` is one-shot (the `_initialized` global makes it idempotent), but `Laminar.shutdown()` supports a later `initialize()`, so it MUST call `reset_debug_runtime()` to clear `_runtime`/`_initialized` — otherwise a second `initialize()` resurrects the first run's stale cache / session metadata / spent pointer and never re-reads `LMNR_DEBUG*`. `reset_debug_runtime` is the public reset (tests use it too); TS mirrors this with `resetDebugRuntime` called from its `shutdown()`.
- `source_trace.py` fetches in two phases over `LaminarClient.sql.query`: phase 1 SELECTs `path, span_type, start_time, end_time` (identify it by the `"span_type, start_time"` substring when faking the SQL client in tests); phase 2 SELECTs payloads for the chosen spine path only. Any fetch/build failure degrades to debug-no-replay (warn, never crash the user's program).
- **A missing / null `end_time` maps to `inf`, NOT `0.0`** (`_to_epoch(value, missing_default=inf)` for the end-time column; `start_time` keeps the `0.0` default). The §F overlap guard is `cur.start_time < prev.end_time`; collapsing an unknown end to `0.0` would make that comparison always false and let overlapping spine calls pass the guard, so replay would proceed when it should force a live run. `inf` makes an unknown end conservatively overlap. TS `toEpoch(value, missingDefault)` mirrors this (`Infinity` for `end_time`) — keep both in lockstep.
- **The run pointer's `trace_id` is recorded from two places**, because not every debug run opens a root span. The normal path is `LaminarSpanProcessor.on_start` → `_record_debug_trace_id` when `span.parent is None`. But a run attached via `LMNR_SPAN_CONTEXT` pushes a `NonRecordingSpan` parent onto the context (`_initialize_context_from_env`), so EVERY span the process creates has a non-None parent and the root-span hook never fires — the pointer would emit an empty `trace_id`. `_initialize_context_from_env` therefore also calls `Laminar._record_debug_trace_id_from_env(otel_span_context)` to record the inherited trace id (all spans in the run share it; `record_trace_id` keeps the first). `_init_debug_runtime` runs before `_initialize_context_from_env` in `initialize()`, so the runtime is already registered at attach time. TS mirrors this in `_initializeContextFromEnv` (`getRuntime()?.recordTraceId(...)`). The `on_start` root-span recording MUST run BEFORE the `if is_disabled: return` early-return — when `LMNR_DISABLE_TRACING=true` spans aren't exported, but replay is gated only on `get_runtime() is not None` (independent of disabled tracing), so the pointer would otherwise emit an empty `trace_id` while replay is still active. TS's processor has no disable-tracing gate, so this ordering is Python-only.
- **`_to_epoch` (in `source_trace.py`) parses ClickHouse `DateTime64(9)` strings and must not regress on two fronts.** (1) Parsing: ClickHouse emits nanosecond fractional seconds, but `datetime.fromisoformat` rejects >6 fractional digits before Python 3.11 (the package floor is 3.10), so the fractional part is truncated to microseconds via `re.sub(r"(\.\d{6})\d+", r"\1", ...)` BEFORE parsing — without this, every real timestamp falls through to the lexical fallback on 3.10. (2) Lexical fallback: when parsing still fails, the score must be a *fractional positional* encoding (`score += min(ord(c),255) * weight; weight /= 256` per char, earliest char weighted most) — a flat `sum(ord(c) for c in s)` is position-insensitive and silently reorders the spine (a later timestamp can score lower). Float64 only resolves the first ~6-7 chars; closer strings tie (never reverse), and the SQL `ORDER BY start_time` + stable sort preserve original order on a tie. TS `toEpoch` mirrors the same fallback (its `Date.parse` already handles ns strings, so it needs no parsing fix) — keep both in lockstep.
- Debug tests (`tests/test_debug_*.py`) are isolated unit tests that do NOT need VCR cassettes — they use `monkeypatch` for env vars and a `_FakeSql`/`_FakeClient` double.

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

## pydantic_ai instrument

- `Instruments.PYDANTIC_AI` is **auto-enabled by default** when `pydantic-ai-slim` (or `pydantic-ai`) is installed. When auto-enabled, the overlapping raw-provider instrumentors — OPENAI, ANTHROPIC, GOOGLE_GENAI, GROQ, MISTRAL, COHERE, BEDROCK — are auto-removed from the default set so the same model call isn't traced twice (pydantic_ai emits its own GenAI spans at the model abstraction layer). The exact set lives in `_PYDANTIC_AI_PROVIDER_CONFLICTS` in `src/lmnr/opentelemetry_lib/tracing/instruments.py`.
- To opt out of the auto-enable, pass `disabled_instruments={Instruments.PYDANTIC_AI}` to `Laminar.initialize`. To keep both pydantic_ai and the raw SDK instrumentors active (accepting duplicate spans), pass an explicit `instruments` set that includes both.
- Installation: `pip install lmnr pydantic-ai-slim>=1.0` — there is no `[pydantic-ai]` extra in `pyproject.toml`.
- The instrumentor does two things:
  1. Calls `Agent.instrument_all(InstrumentationSettings(...))`, which sets a module-level default on pydantic_ai's `Agent` class so subsequent `Agent()` instances pick it up automatically.
  2. Monkey-patches `pydantic_ai.models.instrumented.InstrumentationSettings.__init__` so that *every* construction of `InstrumentationSettings` — including ones triggered by user code calling `Agent.instrument_all(True)` or `Agent(instrument=True)` — defaults to our tracer provider and uses semconv `version=5` when the caller didn't pick one (or picked the legacy `version=1`). User-supplied `tracer_provider=` is respected; a caller-supplied `version >= 2` is also respected unchanged, so callers pinning to an older-but-still-supported semconv contract keep their choice.
- `_instrument` performs no version/installation guards of its own: `BaseInstrumentor.instrument()` already enforces the `instrumentation_dependencies()` constraint (`pydantic-ai-slim >= 1.0.0`), and `PydanticAIInstrumentorInitializer` in `_instrument_initializers.py` only returns this instrumentor when `pydantic-ai-slim` or `pydantic-ai` is installed.
- The test suite runs with `disabled_instruments={Instruments.PYDANTIC_AI}` in `tests/conftest.py` so the raw-provider instrumentor tests still operate on the SDK-level spans they assert on.
- `_instrument` also wraps `pydantic_ai.agent.Agent._run_span_end_attributes` to strip the `pydantic_ai.all_messages` attribute (and the v1-equivalent `all_messages_events`) from the parent agent run span. pydantic_ai dumps the full message history of a run on that span as a JSON blob, but the same content is already on the per-step `chat {model}` child spans via `gen_ai.input.messages` / `gen_ai.output.messages`, so keeping it on the parent doubles ingestion + storage cost. The accompanying `logfire.json_schema` entry is updated in lockstep so backends that read it stay consistent. Tests live in `tests/test_pydantic_ai_settings_patch.py` (`test_run_span_end_attributes_strips_duplicate_messages` / `..._preserves_other_attrs`).
