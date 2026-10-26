# opentelemetry_lib/opentelemetry/instrumentation/

## Structure

- Each provider instrumentor has: `__init__.py` (instrumentor class), `utils.py` (parsing), `span_utils.py` (span creation).
- Newer instrumentors build on `shared/base_instrumentor.py::BaseLaminarInstrumentor` + `WrappedFunctionSpec` (`shared/types.py`), which stamps `lmnr.span.instrumentation_scope.{name,version}`.
- `rollout.py` in `openai/`, `anthropic/`, `google_genai/`, `litellm/` holds the debug-replay wrappers; see `src/lmnr/sdk/debug/AGENTS.md`.
- **Provider instrumentors** extend OpenTelemetry's base instrumentor and are dynamically initialized based on `Instruments` enum
- **wrapt 1.x and 2.x are both supported** (`wrapt>=1.14,<3`). In 2.x `FunctionWrapper` derives from `BaseObjectProxy`, not `ObjectProxy`, so never detect a wrapped function with `isinstance(..., wrapt.ObjectProxy)` (tests use `tests/_wrapt_compat.py::is_wrapt_proxy`). Don't import 2.x-only names (`BaseObjectProxy`, ...) in src, and don't subscript proxy classes at runtime (`ObjectProxy[Any]` only under `TYPE_CHECKING`; see `ChatStream`).
- **Uninstrument removes only our own wrapt layer.** OTel's `unwrap` strips the OUTERMOST layer whoever owns it, so when a third-party instrumentor stacks on the same target (traceloop langchain>=0.62 wrapped `Pregel.stream`/`astream`, like our langgraph; LangChain is native now, see `langchain/AGENTS.md`, but other third-party layers can still stack) it peeled theirs and left ours. `BaseLaminarInstrumentor` records each installed wrapper (`_own_wrappers`) and `_remove_own_layer` splices exactly that layer out of the chain, reading the holder's `__dict__` (never `getattr`, which re-binds class proxies). Custom `_uninstrument`s that call `unwrap` directly still have the old behavior.

## Error status

- **`set_status(StatusCode.ERROR)` alone does NOT mark a span as failed in Laminar.** app-server's `from_otel_span` (`app-server/src/traces/spans.rs`) never reads `otel_span.status`; the ONLY thing that sets `status = "error"` is the presence of an event named `exception` (`app-server/src/traces/utils.rs::prepare_span_for_recording`). So any instrumentor reporting a failure must emit an `exception` event — the status code is fine to set alongside it, but the event is the load-bearing half. The frontend error card reads `exception.type` (headline), `exception.message` (body) and `exception.stacktrace` off that event.
- **Emit that event via `span.record_exception(exc, attributes=get_event_attributes_from_context())`, never a hand-rolled `add_event("exception", ...)`.** That is the convention everywhere in this repo (`decorators/__init__.py::_process_exception`, `litellm/`, `openai/`, `google_genai/`, `daytona/`, …); it fills in `exception.stacktrace` / `exception.escaped`, and the context attributes carry `lmnr.event.session_id` / `lmnr.event.user_id` onto the event. When the failure is a payload rather than a raised exception, wrap it in an `Exception(...)` — `openai/v1/responses_wrappers.py` does this with `Exception("Response cancelled")`, `openai_agents/span_data.py` with the `SpanError` details — and override `exception.type` through `attributes` when a better headline than the class name exists.

## GenAI message attributes

- **The replay cache keys off `gen_ai.input.messages`, which app-server PREFERS over the legacy `gen_ai.prompt.{N}.*` reconstruction.** In `app-server/src/traces/spans.rs::parse_and_enrich_attributes`, the `gen_ai.input.messages` branch (and `gen_ai.output.messages`) OVERWRITES whatever the `gen_ai.prompt.{N}.*` / `ai.prompt.messages` handlers put on `self.input`/`self.output`. So a provider that emits ONLY the legacy per-index attrs has no hashable input for the cache. Both the chat-completions and LiteLLM-responses paths already stamp `gen_ai.input.messages`; the OpenAI **Responses-API** wrapper historically stamped only `gen_ai.prompt.{N}.*`, so it was invisible to replay until we added `build_genai_input_messages` / `build_genai_output_messages` in `instrumentation/openai/v1/responses_wrappers.py` (stamped from `set_data_attributes`, alongside — not replacing — the legacy attrs that the existing `test_responses.py` asserts on).

## Reasoning / thinking attributes

- **All providers record reasoning config through `shared/reasoning.py::set_reasoning_attributes(span, params)`.** It reads whichever known shape `params` has (kwargs mapping, pydantic config or plain object). There is no semconv, so each knob keeps its provider name:
  - `gen_ai.request.reasoning_effort`: OpenAI / Groq / LiteLLM `reasoning_effort`, or the Responses API's `reasoning.effort`.
  - `gen_ai.request.thinking_type` / `gen_ai.request.thinking_budget`: Anthropic `thinking.type` / `.budget_tokens`.
  - `gen_ai.request.effort`: Anthropic `output_config.effort`.
  - `gen_ai.request.thinking_budget` / `gen_ai.request.thinking_level`: Gemini `thinking_config.*`, or flat on LangChain's `ChatGoogleGenerativeAI` params.
- `gen_ai.response.reasoning_effort` comes from the Responses API's echoed `reasoning.effort` (OpenAI and LiteLLM responses). That is the effort actually applied, including the default when the request left it unset.
- **The helper must never raise into the caller.**
  - It is `@dont_throw`, and every read goes through `_get` / `_as_str` / `_as_int`, which return None for anything unexpected: SDK sentinels (`NOT_GIVEN`), mocks, wrong types, and `bool` budgets.
  - Call it as its own statement so a failure can't skip the attributes after it.
  - Enums are recorded as `.value` for str enums (google-genai `ThinkingLevel`) and as `.name` for int enums (Vertex proto `ThinkingLevel`).
  - A budget of `0` is recorded, not dropped as falsy.
- A new knob goes into the helper, not into each provider, because LiteLLM and LangChain pass several providers' shapes through the same kwargs.
- Coverage: VCR tests exist for OpenAI, Anthropic, google-genai and LiteLLM. Groq has no recording key, Vertex is gRPC, and the LangChain tests use fakes, so those three are wired but untested.
