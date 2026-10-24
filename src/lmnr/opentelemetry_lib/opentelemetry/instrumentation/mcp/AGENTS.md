# mcp instrumentation

Instruments the MCP Python SDK (`mcp` 1.x), FastMCP 2.x included since it runs on mcp's `ClientSession` and low-level `Server`. Ported from OpenLLMetry's `opentelemetry-instrumentation-mcp`, which we no longer depend on (the `lmnr[mcp]` extra is kept as an empty no-op so old install commands still resolve).

## What is wrapped

- **Two targets, both transport-agnostic** (stdio, SSE, streamable HTTP, in-memory). `tests/test_instrumentor_targets.py` pins them.
  - `mcp.shared.session.BaseSession.send_request` emits a CLIENT span per outgoing request. It also covers requests a server sends back to its client (`ServerSession`: sampling, elicitation, roots), which nest under the server span of the request being handled.
  - `mcp.server.lowlevel.server.Server._handle_request` emits a SERVER span per incoming request; the handler (so the tool's own code) runs inside it.
- **`initialize` has no server span**: `ServerSession._received_request` answers it itself and it never reaches `_handle_request`. Notifications are not traced.
- Span names: `{tool}.tool` (span type `TOOL`, `gen_ai.tool.name` set) for `tools/call`, `{method}.mcp` otherwise; `mcp.method.name` always, `mcp.request.id` on server spans. Client and server use the same name, matching upstream; in a one-process trace they nest client → server.
- **mcp 2.x has no `BaseSession` and ships native OTel spans**, so neither target exists there and `BaseLaminarInstrumentor` silently installs nothing. FastMCP 3+ also has native spans. If those land in Laminar's global tracer provider alongside ours (mcp 1.x + fastmcp 3), expect nesting/duplicates; not handled yet.

## Context propagation

- The client wrapper injects W3C `traceparent` / `tracestate` into `params._meta` of a COPY of the request (`model_dump(by_alias=True)` → `model_validate`), so the caller's request object is never mutated. A `traceparent` the caller set is left alone. Propagation is not gated on content capture.
- The server reads it back from `message.request_meta` (falling back to `req.params.meta`). `RequestParams.Meta` has `extra="allow"`, so the headers live in `model_extra`. The server handles each request in a task spawned from its run loop, so `_meta` is the only way the client's context arrives, even in-process.
- A remote parent from another process just makes the server span the start of a new span path; the trace id is shared.

## Results and errors

- `Server._handle_request` never raises handler errors (unless `raise_exceptions=True`): it turns them into `ErrorData` and calls `message.respond`. So the server wrapper swaps `message` for `_TracedResponder`, a wrapt proxy whose `respond` records the output / error on the span. `ErrorData` is recorded as an `exception` event of type `McpError`.
- FastMCP tool exceptions come back as `CallToolResult(isError=True)`, not as errors: both sides record an `exception` event of type `tool_error` whose message is the first text block (or a fixed message when content capture is off).
- Client-side failures (`McpError` for a JSON-RPC error, timeouts, closed streams) are recorded and re-raised.
- `ClientSession.call_tool` may call `list_tools` after the result (output-schema validation); that shows as a separate root `tools/list.mcp` span, not a child.

## Dropped from upstream

- Transport-specific wrappers (`stdio_client`, `sse_client`, streamable HTTP, `ResponseStreamWriter` spans) and the `ServerSession` stream proxies that saved context; `_meta` extraction in `_handle_request` replaces them.
- FastMCP-specific wrappers (`ToolManager.call_tool`, `FastMCP.__init__`, the `Client.__aenter__` / `__aexit__` session span): the two targets above already cover FastMCP 2.x.
- The `Config` singleton, `exception_logger`, `traceloop.*` attributes and `mcp.response.value`. Input / output go to `lmnr.span.input` / `lmnr.span.output`, gated on `should_send_prompts()`.

## Tests

- `tests/test_instrumentations/test_mcp/` runs a real FastMCP server against a real `ClientSession` over `mcp.shared.memory.create_connected_server_and_client_session`, so there are no cassettes and no subprocesses. Use a low-level `Server` (without prompt handlers) when a test needs a JSON-RPC error response.
- A cross-process stdio check (server subprocess) gives the same trace id on both sides but is not in the suite.
