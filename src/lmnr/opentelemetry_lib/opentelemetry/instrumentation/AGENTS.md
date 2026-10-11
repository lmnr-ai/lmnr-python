# opentelemetry_lib/opentelemetry/instrumentation/

## Structure

- Each provider instrumentor has: `__init__.py` (instrumentor class), `utils.py` (parsing), `span_utils.py` (span creation).
- Newer instrumentors build on `shared/base_instrumentor.py::BaseLaminarInstrumentor` + `WrappedFunctionSpec` (`shared/types.py`), which stamps `lmnr.span.instrumentation_scope.{name,version}`.
- `rollout.py` in `openai/`, `anthropic/`, `google_genai/`, `litellm/` holds the debug-replay wrappers; see `src/lmnr/sdk/debug/AGENTS.md`.
- **Provider instrumentors** extend OpenTelemetry's base instrumentor and are dynamically initialized based on `Instruments` enum

## GenAI message attributes

- **The replay cache keys off `gen_ai.input.messages`, which app-server PREFERS over the legacy `gen_ai.prompt.{N}.*` reconstruction.** In `app-server/src/traces/spans.rs::parse_and_enrich_attributes`, the `gen_ai.input.messages` branch (and `gen_ai.output.messages`) OVERWRITES whatever the `gen_ai.prompt.{N}.*` / `ai.prompt.messages` handlers put on `self.input`/`self.output`. So a provider that emits ONLY the legacy per-index attrs has no hashable input for the cache. Both the chat-completions and LiteLLM-responses paths already stamp `gen_ai.input.messages`; the OpenAI **Responses-API** wrapper historically stamped only `gen_ai.prompt.{N}.*`, so it was invisible to replay until we added `build_genai_input_messages` / `build_genai_output_messages` in `instrumentation/openai/v1/responses_wrappers.py` (stamped from `set_data_attributes`, alongside — not replacing — the legacy attrs that the existing `test_responses.py` asserts on).
