# AGENTS.md

Guidance for coding agents working in this repository. Module-specific knowledge lives in nested `AGENTS.md` files next to the code (each has a `CLAUDE.md` symlink so Claude Code loads it on demand); read the relevant one before editing a module.

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

# Lint / format / typecheck (what CI runs)
uv run ruff check
uv run ruff format          # CI: uv run ruff format --check
uv run basedpyright

# CLI commands
lmnr eval <file.py>           # Run evaluations
lmnr datasets pull <id>       # Pull dataset
```

- CI: `lint.yml` (ruff check + format check), `typecheck.yml` (basedpyright), `run-tests.yml` (pytest).
- The `uv run --with openai==3.26.0 ...` overlay step in `run-tests.yml` is a temporary workaround (litellm caps `openai<3`). Do not copy that pattern for other packages/versions.
- The husky pre-commit hook (`.husky/pre-commit`) runs `ruff check --fix` + `ruff format` on staged Python files and fails the commit if ruff changed anything; re-stage and commit again.

## Tooling

- Python `>=3.11` (`requires-python`).
- ruff is the linter and formatter: 100-column lines, `target-version = "py311"`, `BLE001` ignored on purpose (instrumentation catches bare `Exception` so it never crashes the host app). There is no flake8 config; don't lint with it.
- basedpyright reads `pyrightconfig.json`: `reportAny` / `reportExplicitAny` are off everywhere, and `tests/` also turns off `reportPrivateUsage` / `reportUnusedParameter`.
- Build backend is `uv_build`; `AGENTS.md` / `CLAUDE.md` are excluded from the sdist and wheel in `[tool.uv.build-backend]`.

## Architecture

### Core SDK (`src/lmnr/sdk/`)
- `laminar.py` - Main `Laminar` class with `initialize()` entry point and span management
- `decorators.py` - `@observe()` decorator for manual instrumentation
- `evaluations/` - `evaluate()` function and `Evaluation` class for running tests
- `client/` - HTTP clients (`LaminarClient`, `AsyncLaminarClient`) for API interactions
- `debug/` - debug mode / replay (cross-language parity surface with the TS SDK)

### OpenTelemetry Layer (`src/lmnr/opentelemetry_lib/`)
- `tracing/instruments.py` - `Instruments` enum listing all auto-instrumentable providers
- `tracing/__init__.py` - `init_tracing()` and the module-level tracing singleton (`get_tracer_wrapper` / `is_tracing_initialized` / `flush_tracing` / `shutdown_tracing`); `TracerWrapper` is the plain object holding the OTEL plumbing
- `opentelemetry/instrumentation/` - Provider-specific instrumentors (anthropic/, openai/, groq/, etc.)
- Each provider instrumentor has: `__init__.py` (instrumentor class), `utils.py` (parsing), `span_utils.py` (span creation)

### CLI (`src/lmnr/cli/`)
- `evals.py` - `lmnr eval` command

### Integrations (`src/lmnr/integrations/`)
- `harbor/` - Harbor `JobPlugin`

## Coding practices

- **Annotations:** start modules with `from __future__ import annotations` and write annotations unquoted, including names imported only under `TYPE_CHECKING`. Quote a type only where Python evaluates it at runtime: `cast("X", ...)`, `TypeVar` bounds, base classes, and `TypeAlias` values. Pydantic models resolve their annotations from module globals, so their field types must be importable at runtime.
- **Zero import cycles.** `basedpyright src/lmnr` reports **zero** import cycles; keep it that way. Pyright counts function-local imports as edges, so "lazy" does not hide a cycle from it.
- Inside `src/lmnr`, import `Laminar` from `lmnr.sdk.laminar`, never `from lmnr import Laminar` (the package `__init__` imports everything).
- Initializer registry injection and the `tracing/wrapper.py` leaf: `src/lmnr/opentelemetry_lib/tracing/AGENTS.md`.
- **Logging:** always get loggers via `get_default_logger` (`sdk/log.py`); never hand-roll `logging.getLogger(...) + addHandler`, and keep `propagate=False`. Details: `src/lmnr/sdk/AGENTS.md`.
- **HTTP errors:** never assume an error body is JSON; render non-2xx bodies with `describe_response` (`sdk/utils.py`). Keep sync/async client resources in lockstep. Details: `src/lmnr/sdk/AGENTS.md`.
- **Span attributes:** serialize through `json_dumps` (`sdk/utils.py`). It owns bytes encoding (standard base64) and never raises (returns `"{}"`). Details: `src/lmnr/sdk/AGENTS.md`.
- **GenAI messages:** instrumentors stamp `gen_ai.input.messages` / `gen_ai.output.messages`; app-server prefers them over legacy `gen_ai.prompt.{N}.*`. Details: `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/AGENTS.md`.

## Tests

- pytest with VCR.py cassettes for HTTP and an `InMemorySpanExporter` configured in `tests/conftest.py` for span assertions. Cassette recording, sandbox env pitfalls, spy-test patterns, and known flakes: `tests/AGENTS.md`.

## Dev dependencies

- All entries in the `[dependency-groups].dev` section of `pyproject.toml` MUST be pinned to a specific version with `==X.Y.Z`. Do NOT use unbounded specifiers (`>=`, `^`, `~`, `<`, ranges, or bare package names). Pinning keeps the test matrix deterministic across developers and CI. When adding a new dev dep, look up the current release on https://pypi.org and pin to that exact version; bumps then go through a normal PR.

## Releases

- `python-publish.yml` runs on `release: published`. It builds in a job without `id-token`, then publishes from a separate `pypi`-environment job with Trusted Publishing and PEP 740 attestations, then re-verifies the attestations PyPI serves (`pypi-attestations verify pypi --repository ...`, retried because PyPI's integrity API lags).
- The build checks out `github.repository` at `github.sha` with `persist-credentials: false` and fails unless the release tag points to that commit, the commit is on `main`, and the tag (minus a `.postN` / `-post.N` suffix) equals `pyproject.toml`'s version. So: bump the version on `main` first, then tag that commit.
- Keep the verification steps inline in the workflow. A local composite action would be loaded from the checked-out tree, which is the code being verified.
- Actions in this workflow are pinned to commit SHAs with a `# vX.Y.Z` comment. Bump them by resolving the new tag's SHA (`gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`), not by switching back to a floating tag.
- Exception: an action that publishes GitHub immutable releases may be pinned to its exact version tag (`astral-sh/setup-uv@v10.3.0`), since that tag cannot be moved. Check with `gh api repos/<owner>/<repo>/releases/tags/<tag> --jq .immutable` before doing this; floating major tags like `@v10` are never immutable.

## Environment Variables

```
LMNR_PROJECT_API_KEY  # API key (can also pass to initialize())
LMNR_BASE_URL         # API base URL (default: https://api.lmnr.ai)
```

## Nested AGENTS.md files

- `tests/AGENTS.md` - VCR cassettes, spy-test pattern, debug tests, known flakes
- `src/lmnr/sdk/AGENTS.md` - logging, HTTP error reporting, `json_dumps` / bytes, eval trace type
- `src/lmnr/sdk/debug/AGENTS.md` - debug mode / replay, TS parity
- `src/lmnr/opentelemetry_lib/tracing/AGENTS.md` - import-cycle wiring, tracing singleton + provider reuse, default instrument set, span batching / export, payload size limit
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/AGENTS.md` - instrumentor structure, error status (exception events), GenAI message attributes
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/anthropic/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/claude_agent/AGENTS.md` - settings.json precedence, proxy
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/deepagents/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/google_adk/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/google_genai/AGENTS.md` - native Gemini shape, bytes serialization
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/langfuse/AGENTS.md` - Langfuse bridge
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/litellm/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/microsoft_agent_framework/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/openai/AGENTS.md` - Responses-API streaming + replay, Decisions API
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/openai_agents/AGENTS.md` - `SpanError` handling, usage gaps on old SDKs
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/openrouter/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/pydantic_ai/AGENTS.md`
- `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/temporal/AGENTS.md`
- `src/lmnr/integrations/harbor/AGENTS.md` - Harbor plugin
