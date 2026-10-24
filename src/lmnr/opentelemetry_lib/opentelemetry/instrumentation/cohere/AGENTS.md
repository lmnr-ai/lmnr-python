# cohere instrumentation

Instruments the `cohere` SDK (v1 `Client` / `AsyncClient`, v2 `ClientV2` / `AsyncClientV2` and `co.v2`). Ported from OpenLLMetry's `opentelemetry-instrumentation-cohere`, which we no longer depend on (the `lmnr[cohere]` extra is kept as an empty no-op so old install commands still resolve). Dropped from the port: the `Config` singleton, the event emitter / `use_legacy_attributes` path, and `exception_logger`.

## What is wrapped

- **Methods are wrapped on the class that DEFINES them**: `BaseCohere` / `AsyncBaseCohere` (v1 `chat`, `chat_stream`, `generate`, `rerank`) and `V2Client` / `AsyncV2Client` (v2 `chat`, `chat_stream`, `embed`, `rerank`). Every client that inherits them (`ClientV2(V2Client, Client)`, Bedrock, SageMaker, OCI, ...) is covered by one wrapper and emits one span; wrapping `ClientV2` too would double-wrap. `tests/test_instrumentor_targets.py` pins the exact target set.
- **v1 `embed` is the exception: it is wrapped on `Client` / `AsyncClient`**, not `BaseCohere`. `Client.embed` overrides the base to split large inputs into batches and calls `BaseCohere.embed` once per batch, so wrapping the base would emit one span per batch.
- With `log_warning_experimental_features=True` (the default), `Client.__init__` rebinds `self.chat` / `self.chat_stream` as instance attributes wrapping the method bound at construction time. So clients created BEFORE `Laminar.initialize()` are not traced, and uninstrumenting does not unwrap clients created while instrumented.
- `generate_stream` is not instrumented (neither was it by OpenLLMetry).
- Async `chat_stream` is an **async generator function** (calling it does no I/O), so it goes through `_wrap_async_generator`, not `_awrap`.

## Span content

- Every SDK method is keyword-only, so wrappers read only `kwargs`. Unset optional args arrive as the SDK's `OMIT` sentinel, which is `...` (Ellipsis); `utils._given` drops it along with `None`.
- Span names keep OpenLLMetry's `cohere.chat` / `cohere.completion` / `cohere.rerank` / `cohere.embed`; `gen_ai.system` is `"cohere"`.
- Messages use the OTel GenAI `{role, parts}` shape. v1: `preamble` → system message, then `chat_history` (`USER`/`CHATBOT`/`SYSTEM`/`TOOL` roles), then the `tool_results` kwarg, then `message`. v1 tool calls have no id and carry `parameters`; v1 tool results become `tool_call_response` parts with `result` (the field the frontend reads) taken from `outputs`. Image `data:` URLs become `blob` parts, other URLs `uri` parts. Rerank keeps OpenLLMetry's layout: one system message per document plus the query as a user message; output is `"Doc {index}, Score: {score}"` text parts. Embedding spans have no output messages.
- Usage prefers actual `tokens` over `billed_units`; v2 `usage.cached_tokens` → `gen_ai.usage.cache_read_input_tokens`.
- **The two stream formats differ.** v1 `chat_stream` is newline-delimited JSON (`event_type` discriminator); its `stream-end` event carries the full `NonStreamedChatResponse`, which wins over the accumulated deltas when present. v2 is SSE (`type` discriminator inside `data`), so `V2ChatStreamAccumulator` rebuilds a `V2ChatResponse`-shaped dict from `content-*`, `tool-plan-delta`, `tool-call-*` (arguments appended across deltas) and `message-end` (finish reason + usage).
- Streams are lazy (the request starts on the first `next()`), so `_stream` / `_astream` re-enter the LLM span around EVERY pull and not across `yield`, same as `google_aiplatform/`. Errors (including mid-stream) go through `record_exception`; a stream abandoned with `break` / `close()` still ends its span with the partial output.
- LangChain's `ChatCohere` is not in `_provider_records_call`'s hand-off map (see `langchain/AGENTS.md`): LangChain suppresses instrumentation for its LLM runs and these wrappers honor `_SUPPRESS_INSTRUMENTATION_KEY`, so there are no duplicate spans.

## Tests

- `tests/test_instrumentations/test_cohere/` uses no cassettes: every client gets `httpx_client=httpx.Client(transport=httpx.MockTransport(...))` (or `AsyncClient`), which serves canned JSON / NDJSON / SSE bodies by URL path. The full SDK still runs, including response and stream parsing.
- Pass v2 messages as SDK models (`cohere.UserChatMessageV2(...)`), not dicts; dicts work at runtime but fail basedpyright.
