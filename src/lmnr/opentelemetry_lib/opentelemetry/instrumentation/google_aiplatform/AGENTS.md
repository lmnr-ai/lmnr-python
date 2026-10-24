# google_aiplatform instrumentation

Instruments the `vertexai` SDK shipped inside `google-cloud-aiplatform` (`vertexai.generative_models` and the legacy `vertexai.language_models`). Ported from OpenLLMetry's `opentelemetry-instrumentation-vertexai`, which we no longer depend on. "vertexai" is the legacy name: `Instruments.VERTEXAI` is an enum ALIAS of `Instruments.GOOGLE_AIPLATFORM` (same value), and the `lmnr[vertexai]` extra is kept as an empty no-op so old install commands still resolve.

## What is wrapped

- **Always wrap the DEFINING class, never the public re-export.** `_GenerativeModel` (base of GA `GenerativeModel` and `preview.GenerativeModel`), `_generative_models.ChatSession` (the preview `ChatSession` subclasses it), `_TextGenerationModel`, and `_ChatSessionBase` (base of GA / preview / code `ChatSession`; `CodeChatSession` overrides but delegates via `super()`). Wrapping the public classes misses preview variants; wrapping both double-wraps and emits two spans per call. `tests/test_instrumentor_targets.py` pins the exact target set.
- `predict_streaming_async` / `send_message_streaming_async` are **async generator functions**, not coroutines: calling them returns an async iterator synchronously. They go through `_wrap_async_generator` (a sync wrapper), not `_awrap`.
- Span names keep OpenLLMetry's `vertexai.*` names so existing dashboards/filters keep matching. `gen_ai.system` is `"gemini"` (OpenLLMetry used `"Google"`).

## Span content

- Messages use the native Gemini `{role, parts}` shape, same as `google_genai/` (see its AGENTS.md); output is a list of candidate dicts. No tiktoken token counting, no image upload, no OpenAI-format conversion.
- **Never read `GenerationResponse.text`** — it raises on function calls and on multi-part / multi-candidate responses. Serialize the candidates instead.
- **A per-call `generation_config` REPLACES the model's default, it is not merged** (that's what the SDK does), so request attributes come from the per-call config when one is passed, else the model's.
- Language-model `ChatSession`: `context` becomes a system message, followed by `_message_history`, then the new user message.
- proto-plus 1.29 renamed `including_default_value_fields` → `always_print_fields_with_no_presence`; `utils._NO_DEFAULTS_KWARG` picks the right one via `inspect.signature` at import. The remaining DeprecationWarnings in the tests come from the SDK's own `to_dict()`, not from us.
- The LLM span is made current during the call (`_use_llm_span`), so spans created inside (HTTP, user callbacks) parent to it. **Streams are lazy** — `generate_content(stream=True)` returns the SDK's unstarted `_generate_content_streaming` generator and `*_streaming_async` are async generator functions — so the real gRPC work happens during iteration, after the wrapped method returned. `_stream` / `_astream` therefore re-enter the span around EVERY `next()` / `anext()`, and deliberately not across `yield` (the consumer's code between chunks is not part of the call, and holding a context token across `yield` breaks detach). Pinned by the `test_llm_span_is_current_*` tests. Errors (including mid-stream ones) go through `record_exception` so app-server marks the span failed; a stream abandoned with `break` still ends its span.

## Tests

- The SDK talks gRPC, so **VCR cannot record it; tests mock instead.** `vertexai.init(project=..., location=..., credentials=AnonymousCredentials())` lets models be constructed offline. Generative models are mocked by patching the private `_GenerativeModel._generate_content` / `_generate_content_streaming` (below the wrapped public methods, so the wrapper still runs); language models via `model._endpoint = MagicMock()` returning `Prediction(...)`, and streaming via `_language_models._streaming_prediction.predict_stream_of_dicts_from_single_dict[_async]`.
- `Part.from_image` defaults unknown bytes to `image/jpeg`; compare against the SDK's own mime type rather than hardcoding one.
