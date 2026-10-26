# tests/

## Overview

- **Tests use VCR.py** to record/replay HTTP responses in `tests/cassettes/`
- **Tests use InMemorySpanExporter** configured in `tests/conftest.py` for span assertions

## Instrumentation tests (VCR)

- Tests under `tests/test_instrumentations/**` replay via VCR cassettes in `cassettes/<module-name>/`. Sensitive headers are filtered by the root `vcr_config` (`authorization`, `api-key`, `x-api-key`, `x-goog-api-key`).
- **A module-level `vcr_config` fixture REPLACES the root one, it does not merge.** `test_anthropic/conftest.py` overrode it with only `x-api-key`. The sandbox's `ANTHROPIC_API_KEY` is an OAuth-style token that the SDK sends as `authorization: Bearer sk-ant-...`, so a recording there wrote the live token into the cassette. That override now filters `authorization` too. When you override `vcr_config`, keep every root filter, and `grep -c 'sk-ant\|Bearer'` a new anthropic cassette before committing it.
- The sandbox sets `ANTHROPIC_BASE_URL` / `ANTHROPIC_BEDROCK_BASE_URL` to a local proxy; when recording new anthropic cassettes, **unset those two env vars** first or the cassette will record `http://127.0.0.1:...` and fail to replay against `api.anthropic.com`. Use `unset ANTHROPIC_BASE_URL ANTHROPIC_BEDROCK_BASE_URL` before running `pytest --record-mode=once`. Replays need `ANTHROPIC_BASE_URL=https://api.anthropic.com` for the same reason.
- The sandbox also sets `AWS_BEARER_TOKEN_BEDROCK`. anthropic >= 1 reads it as the Bedrock `api_key` and raises "Cannot specify both api_key and AWS credentials" when AWS keys are passed too, so Bedrock fixtures `monkeypatch.delenv` it.
- The sandbox's `AZURE_OPENAI_*` env makes `test_openai/traces/test_azure.py` miss its cassettes (6 `CannotOverwriteExistingCassetteException` failures on a clean tree too).
- Re-record a single module with `rm -rf tests/.../cassettes/<module>/ && pytest <path> --record-mode=once`. The `pytest-recording` default `record_mode` is `none`, not `once`: replays work without flags as long as the cassette exists, but a missing cassette fails with `CannotOverwriteExistingCassetteException`, and recording a new one requires `--record-mode=once`.
- **Recording a single new test:** pass `-k <test_name> --record-mode=once`, not the whole module. A failed recording still writes a cassette file (e.g. an HTTP 400 from a setup call), so delete the stub before retrying or the next run replays the error.
- **Real API keys for recording:** tests that need a live key read it from the environment (e.g. `Client(api_key=os.getenv("GEMINI_API_KEY", "123"))`), so replay works with the `"123"` placeholder. Keys are never written to cassettes: `x-goog-api-key` is a filtered header and Gemini's `key` query parameter is filtered by `filter_query_parameters`. Check a new cassette with `grep -c AIza` before committing.
- **Provider minimums apply to recorded content:** Gemini rejects cached content under 2048 tokens (`Cached content is too small`), so cache fixtures need a large corpus, not a short string.

## Spy tests on global httpx

- **A leaked claude_agent `proxy-monitor-*` health-check thread (see `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/claude_agent/AGENTS.md`) is what made `test_openai/traces/test_chat.py::test_chat_context_propagation` flake in `scheduled-tests.yml`** (~1 run in 8; it passes on re-run, which is why it read as a transient). The context-propagation tests patch `httpx.Client.send` GLOBALLY, so the stray health GET landed inside their spy and blew `assert_called_once` ("Called 2 times"), then hit VCR under `record_mode=none` and surfaced as a confusing `CannotOverwriteExistingCassetteException` for a request no test made. Failure depends only on whether a 1s tick falls inside a few-millisecond window, so a small local sample will NOT reproduce it — reproduce by asserting no `proxy-monitor-*` thread survives the claude_agent module instead. The 6 spy tests (chat / completions / embeddings, sync + async) now select their request via `single_request_to_path(spy.mock, "/v1/...")` rather than asserting on all global httpx traffic; **keep that pattern for any new spy test** — a global `assert_called_once` is a latent flake, and `call_args` (the LAST call) can silently be someone else's request.

## Real sockets in tests

- **Never `bind()` a hard-coded port in a test; bind port 0 and read `getsockname()[1]`.** Fixed ports like 45300 fall inside the kernel's ephemeral range (32768–60999 on CI's Ubuntu), so any outgoing connection that recently picked that port as its *source* port leaves it in TIME_WAIT for ~60s, and `bind` then fails with `[Errno 98] Address already in use` **even with `SO_REUSEADDR`**: Linux only ignores a TIME_WAIT conflict when the old socket also set `SO_REUSEADDR`, and client sockets don't. This is what made `test_claude_agent/test_proxy_unit.py::test_is_port_open` flake on PR checks (LAM-2421). Reproduce it deterministically by connecting a client socket explicitly bound to the port, closing it client-first, then binding. For a guaranteed-closed port (connect gets refused), keep a socket bound to port 0 *without* calling `listen()`. That reserves the port, so nothing else can take it and a connect can't loop back to itself. `_reserve_port()` in `test_proxy_unit.py` is the pattern.

## Debug tests

- Debug tests (`tests/test_debug_*.py`) are isolated unit tests that do NOT need VCR cassettes — they use `monkeypatch` for env vars and `_FakeSpan`/`_FakeRuntime`/`_RecordingCache`/`_SpyDebugClient` doubles. `replay.py` no longer imports `laminar` (the latch moved into `sdk/debug/__init__.py`).

## Known flakes

- **Every langchain-instrumentation test fails on a clean `main` checkout in the agent sandbox** — the four `test_instrumentations/test_langchain.py::test_langchain_langgraph*` cases plus `test_observe_concurrency.py`, `test_tracing.py` and `test_tracing_concurrency.py`'s `*_langchain` cases. They fail as `assert 0 == N`: the instrumentation emits no spans at all. Deterministic (not a flake), environment-only, and unrelated to whatever you are changing; confirm with `git stash` before chasing one.
- **`test_litellm.py::test_litellm_completion_activates_span_for_otel_callbacks` is flaky at roughly 3-in-6, independent of any batching change** (verified by running the full suite six times on a clean tree and six times with the batch-processor work applied — identical rates). It fails with `KeyError: 'span_id'`, i.e. litellm's success callback never fired. Do not chase it as a regression from an export/batching change; a small sample will mislead you into thinking your branch caused it.
