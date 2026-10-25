# mcp instrumentation

Instruments the MCP Python SDK (`mcp`), 1.x and 2.x, plus FastMCP 2.x, which runs on mcp 1.x's `ClientSession` and low-level `Server`. Ported from OpenLLMetry's `opentelemetry-instrumentation-mcp`, which we no longer depend on (the `lmnr[mcp]` extra is kept as an empty no-op so old install commands still resolve).

## Layout and version selection

- `v1.py` and `v2.py` hold one `WrappedFunctionSpec` list per major. `utils.py` holds what both share: naming, input / output, error recording and `_meta` carriage.
- `McpInstrumentor` installs only the list for the installed major (`wrapped_functions(version)`; 2.x when unparsable). Registering both is not safe: `wrap_function_wrapper` resolves inherited attributes, so 2.x's `ClientSession.send_request` target would also wrap 1.x's inherited `BaseSession.send_request` and double every client span.
- `tests/test_instrumentor_targets.py` pins both lists and picks one by the installed major.

## Span shape (both majors)

- Span names: `{tool}.tool` (span type `TOOL`, `gen_ai.tool.name` set) for `tools/call`, `{method}.mcp` otherwise. `mcp.method.name` is always set, `mcp.request.id` on server spans. Client and server use the same name, matching upstream; in a one-process trace they nest client → server.
- Requests a server sends back to its client (sampling, elicitation, roots) get CLIENT spans that nest under the server span of the request being handled.
- Notifications are not traced.
- Input / output go to `lmnr.span.input` / `lmnr.span.output`, gated on `should_send_prompts()`. Trace propagation is not gated.

## mcp 1.x (`v1.py`)

- **Two targets, both transport-agnostic** (stdio, SSE, streamable HTTP, in-memory):
  - `mcp.shared.session.BaseSession.send_request` emits a CLIENT span per outgoing request, on `ClientSession` and `ServerSession` alike.
  - `mcp.server.lowlevel.server.Server._handle_request` emits a SERVER span per incoming request; the handler (so the tool's own code) runs inside it.
- **`initialize` has no server span**: `ServerSession._received_request` answers it itself, so it never reaches `_handle_request`.
- `Server._handle_request` never raises handler errors (unless `raise_exceptions=True`): it turns them into `ErrorData` and calls `message.respond`. So the server wrapper swaps `message` for `_TracedResponder`, a wrapt proxy whose `respond` records the output / error on the span. `ErrorData` is recorded as an `exception` event of type `McpError`.
- FastMCP 3+ ships native spans of its own; mcp 1.x + FastMCP 3 would put both in Laminar's tracer provider. Not handled.

## mcp 2.x (`v2.py`)

mcp 2.x opens its own spans (tracer `mcp-python-sdk`, client span `MCP send {method}`). They have no input / output and no span type, and they mark failures with a status only, which Laminar does not read as a failure. They are also opened with `start_as_current_span`, which sets only the global OTel context, so Laminar spans the tool's code opens don't nest under them. The wrappers replace them rather than enrich them:

- **`mcp.server._otel.OpenTelemetryMiddleware.__call__`**: `Server.middleware` defaults to `[OpenTelemetryMiddleware()]`, and `runner._on_request` runs the request through that chain. Our wrapper never calls the native method; it opens a Laminar SERVER span and calls `call_next` inside it. That span covers spec validation, `initialize` (unlike 1.x, `initialize` has a server span here), handler lookup and params validation. Servers that removed the middleware get no server spans, just as they would get no native ones.
- **`ClientSession.send_request` / `ServerSession.send_request`**: the same CLIENT wrapper as 1.x (`v1.wrap_send_request`). The typed request and result give the span its input / output. It also covers `Client(server)`, which dispatches in-process through `DirectDispatcher` and has no native spans at all.
- **`ClientSession.send_discover`**: in `mode="auto"` (the default), `Client` connects by sending `server/discover` through `_dispatcher.send_raw_request`, bypassing `send_request`, so it gets its own CLIENT span. A JSON-RPC error there is how a legacy server answers (the client falls back to `initialize`), so it records `rpc.response.status_code` with no `exception` event.
- **`mcp.shared.jsonrpc_dispatcher.otel_span`** (a module-level binding, `object_name=None`): while `ACTIVE_CLIENT_SPAN` is our client span and is still the current span, it yields ours instead of opening a native span. The `traceparent` the dispatcher injects into `_meta` is then ours. The native span is deliberately not dropped in the span processor instead: the server span would point at a parent that is never exported.
- **Handshake modes**: modern (`auto` / discover) requests must carry the `_meta` envelope `send_request` stamps, so a raw `send_raw_request` without it is rejected before the middleware runs (no server span). Server-to-client requests (`ctx.session.list_roots()` and the like) need `Client(..., mode="legacy")`.
- **Error mapping**: `MCPServer` (2.x's FastMCP) masks tool exceptions as `"Error executing tool {name}"` in an `is_error` result. A `pydantic.ValidationError` from params validation is recorded as `ValidationError` with status code `-32602`, and its message (which quotes the client's input) is replaced when content capture is off. Other errors are `MCPError` (status code from its `ErrorData`) or whatever the handler raised.
- **Caller-set `traceparent`**: on the JSON-RPC dispatcher, mcp itself calls `inject(out_meta)` with the current context and overwrites a `traceparent` the caller put in `_meta`, with or without us. With us, the current span is our client span, so the server nests under it. Only `DirectDispatcher` keeps the caller's value. We leave mcp's propagation alone.
- **claude_agent**: `claude_agent/sdk_mcp.py` wraps the middleware instance in `_ToolCallsOnly`, which still calls the instance's `__call__`. With both instrumentors on, SDK MCP servers get our `tools/call` TOOL span (parented through the `_meta.traceparent` claude_agent stamps), not the native one.

## Context propagation

- The client wrapper injects W3C `traceparent` / `tracestate` into `params._meta` of a COPY of the request (`model_dump(by_alias=True)` → `model_validate`), so the caller's request object is never mutated. A `traceparent` the caller set is left alone (but see the 2.x JSON-RPC note above).
- 1.x servers read it back from `message.request_meta`, falling back to `req.params.meta`. `RequestParams.Meta` has `extra="allow"`, so the headers live in `model_extra`. 2.x servers read `ctx.meta`, a plain dict. Either way the server handles each request in a task of its own, so `_meta` is the only way the client's context arrives, even in-process.
- A remote parent from another process just makes the server span the start of a new span path; the trace id is shared.

## Results and errors

- Tool exceptions come back as `CallToolResult` with `isError` (1.x) / `is_error` (2.x) set, not as errors: both sides record an `exception` event of type `tool_error` whose message is the first text block (or a fixed message when content capture is off).
- Client-side failures (`McpError` / `MCPError` for a JSON-RPC error, timeouts, closed streams) are recorded and re-raised. `rpc.response.status_code` is set from the error's `ErrorData.code`.
- On 1.x, `ClientSession.call_tool` may call `list_tools` after the result (output-schema validation); that shows as a separate root `tools/list.mcp` span, not a child.

## Dropped from upstream

- Transport-specific wrappers (`stdio_client`, `sse_client`, streamable HTTP, `ResponseStreamWriter` spans) and the `ServerSession` stream proxies that saved context; `_meta` extraction on the server side replaces them.
- FastMCP-specific wrappers (`ToolManager.call_tool`, `FastMCP.__init__`, the `Client.__aenter__` / `__aexit__` session span): the session / server targets already cover FastMCP 2.x.
- The `Config` singleton, `exception_logger`, `traceloop.*` attributes and `mcp.response.value`.

## Tests

- The dev dependency pins mcp **2.x**. `tests/test_instrumentations/test_mcp/conftest.py` uses `collect_ignore` to skip whichever of `test_mcp_v1.py` / `test_mcp_v2.py` can't import under the installed major. `run-tests.yml` runs the 1.x module (plus the mcp targets test) as a separate `uv run --with mcp==1.30.0 ... -k mcp` step. Run that locally after touching `v1.py` or `utils.py`.
- `test_mcp_v1.py` starts with a file-level `# pyright:` comment because basedpyright checks it against the installed 2.x.
- Neither suite needs cassettes or subprocesses; client and server run in-process through the real SDK.
  - 1.x: FastMCP over `mcp.shared.memory.create_connected_server_and_client_session`.
  - 2.x: every test is parametrized over `Client(server)` (`DirectDispatcher`) and `Client(InMemoryTransport(server))` (JSON-RPC dispatcher, from `mcp.client._memory`), because only the latter has native spans.
- For a JSON-RPC error response, use a low-level `Server` with no prompt handlers (`prompts/list` → `-32601`).
- A cross-process stdio check (server subprocess) gives the same trace id on both sides but is not in the suite.
