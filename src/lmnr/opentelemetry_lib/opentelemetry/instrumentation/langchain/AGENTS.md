# langchain/

Native port of OpenLLMetry's `opentelemetry-instrumentation-langchain` (we no longer depend on it; `lmnr[langchain]` is an empty extra). The traceloop `Config` singleton, metrics, events, tiktoken token counting, image upload and `traceloop.*` attributes were dropped on purpose; don't bring them back.

## Design

- **One `LaminarCallbackHandler` per process**, injected by wrapping `BaseCallbackManager.__init__` (skipped if a `LaminarCallbackHandler` is already in `inheritable_handlers`; child managers inherit it). `BaseInstrumentor` is a singleton whose `__init__` runs on every construction, so `LangchainInstrumentor._handler` is a class attribute created once: a fresh handler per construction would split run state between the installed wrappers and the new spec.
- All state is keyed by LangChain `run_id`; parenting comes from `parent_run_id` → the parent run's stored context, falling back to the current Laminar context for root runs. `run_inline = True` so callbacks run in the caller's task, not an executor.
- **Naming:** a chain run without a known parent is `<name>.workflow` (`gen_ai.operation.name=invoke_agent`); nested chains, LangGraph nodes and tools are `execute_task <name>`. LLM runs are `<name>.chat` / `<name>.completion`, type `LLM`. LangGraph's `thread_id` becomes `gen_ai.conversation.id`.
- **Metadata** goes into `langchain.metadata` (sanitized by `span_utils.sanitize_metadata`: plain data is kept, arbitrary objects are DROPPED rather than `str()`-ed, since a model / client repr routinely contains credentials), NOT into `lmnr.association.properties.*` as upstream did, because association properties propagate to every child span.

## Context handling

- Chain / tool / retriever runs attach their span to BOTH the isolated Laminar context (so `observe` / `Laminar.start_span` inside a tool nest under it) and the global OTel context (third-party OTel instrumentations).
- **LLM runs attach only `_SUPPRESS_INSTRUMENTATION_KEY`, set to `_SuppressWhileRunning(run_id)`, never `True` and never the span.** Async LangChain fires `on_llm_end` in a different task than `on_chat_model_start` (`asyncio.gather` in `agenerate`, the `astream` producer task), so the start's token can't be reset there. A plain `True` (or the span) would stay attached in the caller's context and suppress / mis-parent everything after. The flag's `__bool__` returns `active`, which `_close` sets to False, so it stops suppressing wherever the token ended up. Pinned by `test_chat_model_async*`.
- **Every end / error callback calls `_end_run` in a `finally`.** The callbacks are `@dont_throw`, so an exception while recording output (`set_response_attributes`, `_set_output`) used to skip `_end_run`: the span never ended and an LLM run's flag stayed active, suppressing provider instrumentation for the rest of the context. Pinned by `test_run_ends_when_recording_output_fails`.
- The suppression key only silences provider SDK instrumentations (openai, anthropic, ...): their spans for a LangChain-driven call are dropped, so the LangChain span is the single LLM span (no duplicate `openai.chat`). Cost: no provider-specific attributes or debug replay for those calls. Laminar's own `observe` / `start_span` spans are NOT suppressed by this key.
- Context detaches go through `_RUNTIME_CONTEXT.detach` inside `contextlib.suppress`, since cross-task detaches are expected and `context_api.detach` would log an error for each.

## LangGraph

- The `langgraph/` instrumentor opens no spans: it wraps `Pregel.stream` / `astream` (which `invoke` / `ainvoke` go through) and publishes the graph topology in the `LANGGRAPH_TOPOLOGY` ContextVar (`shared/context_keys.py`) only around each `next()` / `anext()` of the graph iterator, so it never leaks into caller code between steps.
- The handler consumes (and clears) the topology on the first chain start it sees, but only stamps it as `lmnr.association.properties.langgraph.nodes` / `.edges` when that run is a root span; nested graphs therefore don't overwrite the outer graph's topology. This replaced the old `LangGraph.workflow` name hack in `tracing/processor.py`.

## OpenAI trace context

- "Pass context" is on by default: `wrap_openai_call` wraps `BaseChatOpenAI._generate/_agenerate/_stream/_astream/_stream_responses/_astream_responses` (`ChatOpenAI._stream` delegates to the last four via `super()`) and `BaseOpenAI._generate/_agenerate/_stream/_astream` (langchain_openai + langchain_community), finds the LLM run, and injects a W3C `traceparent` into `extra_headers`. Note the flags byte is `03` (sampled + random trace id), not `01`; assert on the `00-<trace>-<span>-` prefix.
- **`BaseChatModel.stream` / `astream` call `_stream` / `_astream` WITHOUT `run_manager`** (only the `_generate`-streaming path passes it), so the run id is taken from `kwargs["run_manager"]` when present and otherwise from the active `_SuppressWhileRunning` flag in the current context (`LaminarCallbackHandler.current_llm_run_id`). `on_chat_model_start` runs inline before `_stream` is iterated, so the flag is visible there. Streamed spans get LangChain's `lc_run--` message id as `gen_ai.response.id`, not OpenAI's.

## Tests

- `tests/test_instrumentations/test_langchain.py` uses fake chat models / `httpx.MockTransport`, no cassettes.
