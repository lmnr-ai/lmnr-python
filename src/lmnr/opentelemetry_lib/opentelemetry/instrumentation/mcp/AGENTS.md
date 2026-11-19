# mcp instrumentation

Instruments the MCP Python SDK (`mcp`), 1.x and 2.x, plus FastMCP 2.x, which runs on mcp 1.x's `ClientSession` and low-level `Server`. Ported from OpenLLMetry's `opentelemetry-instrumentation-mcp`, which we no longer depend on (the `lmnr[mcp]` extra is kept as an empty no-op so old install commands still resolve).

It also owns MCP server observability (LAM-2563): `Laminar.initialize(mcp=True | McpOptions)` switches every MCP server in the process into **MCP observability mode** (root span per call, one trace per user turn, `context` field). The span contract (names, attribute keys, `error.type` strings, `context` description, redaction vectors, turn rules) is shared with the TS SDK (`mcp` option) and app-server's own MCP server; the MCP dashboard queries those exact names, so change them in all three or not at all.

## Layout and version selection

- `v1.py` and `v2.py` hold one `WrappedFunctionSpec` list per major. `utils.py` holds what both share: CLIENT spans, `_meta` carriage, `record_error` / `set_status_code` / `is_tool_error` / `first_text` / `dump` (the server span uses them too).
- SERVER spans have ONE code path for both majors and both modes: `server.py` (`ServerSpan`), `settings.py` (`McpServerSettings`, `current_settings()`, `McpRequest`), `context_field.py`, `turns.py`, `redaction.py`, `fastmcp_compat.py`. The 1.x / 2.x hooks only read an `McpRequest` off their SDK and drive it. Never add a second span path (an earlier `wrap_mcp_server` had its own middleware / handler dict plus markers so the instrumentor stood aside; it duplicated every attribute and error rule).
- The option itself lives in the leaf `tracing/mcp_options.py` (`resolve_mcp_options`, `set_mcp_options` / `reset_mcp_options` / `get_mcp_options`; turn-gap default 20 s, clamped to 25 s with a warning). `laminar.py` can't import this package (cycle via `shared.utils`), hence the leaf. `Laminar.initialize` sets it, `shutdown` resets it. `current_settings()` builds `McpServerSettings` from it, cached by the identity of the resolved options, so a re-initialize gets a fresh `TurnTracker`; `OFF` without it. Read per request, never at instrument time (instrumentors outlive `initialize` / `shutdown`).
- `McpInstrumentor` installs only the list for the installed major (`wrapped_functions(version)`; 2.x when unparsable). Registering both is not safe: `wrap_function_wrapper` resolves inherited attributes, so 2.x's `ClientSession.send_request` target would also wrap 1.x's inherited `BaseSession.send_request` and double every client span.
- `tests/test_instrumentor_targets.py` pins both lists and picks one by the installed major.
- Consequence: observability mode needs the MCP instrumentation, which is on by default. With `Instruments.MCP` disabled the option does nothing.

## Behaviour that must not change

Each was a decision, pinned by `test_mcp_mode_v1.py` / `test_mcp_mode_v2.py` / `test_mcp_mode_units.py` (mode on) and `test_mcp_v1.py` / `test_mcp_v2.py` (mode off).

- Mode on: one ROOT SERVER span per `tools/call` / `prompts/get` / `resources/read`, including failures no handler saw (unknown tool, bad arguments, missing handler). None for `initialize`, `ping`, listing, notifications.
- Mode off (default): every request a server handles gets a SERVER span parented on the client's `_meta.traceparent` (else the current context), and tool schemas are never changed. claude_agent relies on the parenting to nest SDK MCP tool calls under its agent span (`claude_agent/sdk_mcp.py`, which also limits those servers to `tools/call`).
- Both modes: name `{method} {target}`, the same attributes (session, client, protocol, request id), payload keys, redaction of secret-looking keys and `error.type` values, so the MCP dashboard counts mode-off servers too (no turns, no `context`).
- Trace id order (mode on): `turnId` hash → session tracker → `traceparent` (no session) → random; the `traceparent` is always a link. Gap 20 s default, clamped to 25 s.
- Handler spans nest under the request span; FastMCP native spans suppressed in mode on only (subclasses included); results reach the client unchanged; `context` stripping works even when Laminar is not initialized.
- Mode on ignores OTel's suppress-instrumentation flag (an in-process caller's flag is not the server's); mode off honours it.
- `record_payloads=False` keeps no arguments, results, context, URIs or exception text. Degrade to a random trace id when the provider can't force one; never raise.

## Span shape

- CLIENT spans: `{tool}.tool` (span type `TOOL`, `gen_ai.tool.name` set) for `tools/call`, `{method}.mcp` otherwise, input / output in `lmnr.span.input` / `lmnr.span.output`.
- SERVER spans: `{method} {target}` (`tools/call add`), kind SERVER, `mcp.method.name`, `jsonrpc.request.id`, `mcp.session.id` + `lmnr.association.properties.session_id` (children inherit it via `use_span`), `lmnr.mcp.client.name/version`, `mcp.protocol.version`, `network.transport` / `network.protocol.name` (`tcp` / `http`, only when an HTTP request is visible), `gen_ai.operation.name=execute_tool` + `gen_ai.tool.name` / `gen_ai.prompt.name`, `mcp.resource.uri`, `lmnr.mcp.context`. Tool arguments / result in `gen_ai.tool.call.arguments` / `.result` (ingestion lifts them into input / output and renames the span to the tool name); prompts / resources in `lmnr.span.input` / `.output`. All redacted.
- Requests a server sends back to its client (sampling, elicitation, roots) get CLIENT spans that nest under the server span of the request being handled.
- Payloads are gated on `should_send_prompts()` and, in mode on, `record_payloads`. Trace propagation is not gated.

## mcp 1.x (`v1.py`)

- **Two targets, both transport-agnostic** (stdio, SSE, streamable HTTP, in-memory):
  - `mcp.shared.session.BaseSession.send_request` emits a CLIENT span per outgoing request, on `ClientSession` and `ServerSession` alike.
  - `mcp.server.lowlevel.server.Server._handle_request` emits the SERVER span per incoming request; the handler (so the tool's own code) runs inside it. The hook's `instance` is the low-level `Server`, so FastMCP suppression and the schema lookup come from it.
- **`initialize` has no server span**: `ServerSession._received_request` answers it itself.
- `Server._handle_request` never raises handler errors (unless `raise_exceptions=True`): it turns them into `ErrorData` and calls `message.respond`. So the hook swaps `message` for `_Responder`, a wrapt proxy whose `respond` passes the response through a callback: it records the output / error (`ErrorData` → its code as `error.type`; `0` for a plain handler exception) and, in mode on, adds the `context` field to `tools/list`. Everything else the span needs (request id, `_meta`, HTTP headers, client info) comes from `_handle_request`'s own arguments: `server.request_context` is not set yet when the hook starts.

## mcp 2.x (`v2.py`)

mcp 2.x opens its own spans (tracer `mcp-python-sdk`, client span `MCP send {method}`). They have no input / output and no span type, and they mark failures with a status only, which Laminar does not read as a failure. They are also opened with `start_as_current_span`, which sets only the global OTel context, so Laminar spans the tool's code opens don't nest under them. The wrappers replace them rather than enrich them:

- **`mcp.server._otel.OpenTelemetryMiddleware.__call__`** (mode off): `Server.middleware` defaults to `[OpenTelemetryMiddleware()]`, and the runner runs each request through that chain. Our wrapper never calls the native method; it opens the SERVER span and calls `call_next` inside it. That span covers spec validation, `initialize` (unlike 1.x), handler lookup and params validation. Servers that removed the middleware (FastMCP 4) get no server spans, as they would get no native ones; without `Laminar.initialize` mcp's native span is left in place. In mode on it just calls `call_next`.
- **`mcp.server.runner.ServerRunner._compose_server_middleware(inner)`** (mode on): builds the chain per inbound message (requests and notifications) and knows `self.server`, which `ServerRequestContext` does not. We wrap the composed chain, so the span runs before params validation (where `context` must already be gone) and around FastMCP 4's own dispatch middleware (FastMCP 4 removes the SDK middleware and adds `FastMCPServerMiddleware`, which opens its "seam" span). The runner is the only caller of `server.middleware` in mcp 2.3. Mode off returns the chain unchanged.
- **`ClientSession.send_request` / `ServerSession.send_request`**: the same CLIENT wrapper as 1.x (`v1.wrap_send_request`). It also covers `Client(server)`, which dispatches in-process through `DirectDispatcher` and has no native spans at all.
- **`ClientSession.send_discover`**: in `mode="auto"` (the default), `Client` connects by sending `server/discover` through `_dispatcher.send_raw_request`, bypassing `send_request`, so it gets its own CLIENT span. A JSON-RPC error there is how a legacy server answers (the client falls back to `initialize`), so it records `rpc.response.status_code` with no `exception` event.
- **`mcp.shared.jsonrpc_dispatcher.otel_span`** (a module-level binding, `object_name=None`): while `ACTIVE_CLIENT_SPAN` is our client span and is still the current span, it yields ours instead of opening a native span. The `traceparent` the dispatcher injects into `_meta` is then ours. The native span is deliberately not dropped in the span processor instead: the server span would point at a parent that is never exported.
- **Handshake modes**: modern (`auto` / discover) requests must carry the `_meta` envelope `send_request` stamps, so a raw `send_raw_request` without it is rejected before the middleware runs (no server span). Server-to-client requests (`ctx.session.list_roots()` and the like) need `Client(..., mode="legacy")`.
- **Caller-set `traceparent`**: on the JSON-RPC dispatcher, mcp itself calls `inject(out_meta)` with the current context and overwrites a `traceparent` the caller put in `_meta`, with or without us. Only `DirectDispatcher` keeps the caller's value. We leave mcp's propagation alone.
- **claude_agent**: `claude_agent/sdk_mcp.py` wraps the middleware instance in `_ToolCallsOnly`, which still calls the instance's `__call__`. In mode off, SDK MCP servers get our `tools/call` TOOL span parented through the `_meta.traceparent` claude_agent stamps. Mode on is process-wide, so they become mode-on servers too (root spans, `context` field).

## Native OTel in the SDKs (checked against mcp 2.3.0, fastmcp 4.2.2 / 3.4.9)

- **mcp 2.x**: `OpenTelemetryMiddleware` opens a SERVER span per inbound request, `initialize` / `tools/list` / `server/discover` included, with the semconv attributes, parented on `_meta.traceparent`. No session id, client info, payloads, span type or `exception` event; unknown tool / bad arguments come out as `tool_error` (MCPServer folds them into `is_error`).
- **FastMCP 4** (on mcp 2.x): drops the SDK middleware and opens its own seam SERVER span per request (plus `mcp.session.id`, `fastmcp.*`); `error.type` is `ValidationError` / `NotFoundError` / `tool_error`. Ships `fastmcp.telemetry.suppress_fastmcp_telemetry()`.
- **FastMCP 3** (on mcp 1.x): SERVER spans per high-level op (tools/list too), no protocol version / request id, no switch to turn them off.
- **mcp 1.x**: nothing native.
- What mode on adds over all of them: root span + turn grouping (trace-id forcing), links instead of parents, Laminar span type, redacted payloads, `context`, client name/version, session for Laminar, exception events, `-32602` classification of folded errors, no spans for listing / handshake.

## FastMCP suppression (`fastmcp_compat.py`, mode on only)

- Every request of a FastMCP server runs with FastMCP's own spans suppressed: 4 via its own switch; 3 by swapping `fastmcp.server.telemetry.get_tracer`, once, for one that returns a pass-through tracer while our contextvar is set and the original otherwise. `native_span_suppression(server)` is cached per server class; `nullcontext` for other servers.
- "Is it FastMCP?" walks the class's MRO for a `fastmcp` module: a user's `FastMCP` subclass lives in their own package, and checking only the class kept its native spans on (two SERVER spans per request). The low-level servers both FastMCPs build are `fastmcp` classes, which is what the hooks see.
- **`Laminar.use_span` attaches Laminar's isolated context as the global OTel context, dropping values set on the previous global context.** FastMCP 4's suppression key is such a value, so `ServerSpan.activate` re-enters the suppression inside `use_span`; without that, the handler's part of the request got a FastMCP span again.

## Trace ids, roots and the id generator (mode on)

- Every span is started with an empty `Context()`: always a root, also when the server runs in the caller's context (`Client(server)` / `DirectDispatcher`). A client `traceparent` is always a span link; its trace id is used only to start a session's turn or when there is no session. Never fake a remote parent: the dangling parent id hides the span in the trace view.
- Order: `turnId` hash (`sha256("{session or ''}\n{turnId}")[:16]`) → `TurnTracker` (when the request has a session; a NEW turn starts with the call's `traceparent` trace id, random without one, and later calls within the gap join it) → `traceparent` trace id (no session) → random. The tracker is updated on every call with a session, including turn-id calls.
- **Why the tracker beats `traceparent`** (contract change after a live run): an OTel-instrumented client sends a fresh `traceparent` per call, so with `traceparent` first every call of a turn became its own trace. Now it only seeds a new turn; each span still links to its own `traceparent`.
- Forcing: `LaminarIdGenerator` (Laminar's own `TracerProvider`) returns the id set by `forced_trace_id(...)` (a contextvar set only around `start_span`; `tracing/AGENTS.md`). `can_force_trace_id(provider)` is false for a provider Laminar did not build; then the span gets a random trace id and a debug log is written once.
- With the instrumentor on, a client in this process sends its CLIENT span as `traceparent`. Tests that need no `traceparent` suppress OTel instrumentation around a `Client(server)` call (the JSON-RPC dispatcher would still inject mcp's own client span).
- No session (stdio, 2026-07-28 stateless HTTP without `openai/session`) means no tracker.

## Requests and sessions

- Session id: 2.x `ctx.session._connection.session_id`, else the `mcp-session-id` header; 1.x the header of the HTTP request in `message.message_metadata.request_context`; then `_meta["openai/session"]`.
- Client name / version: `session.client_params.client_info` (2.x; 2026-07-28 requests synthesize it from the envelope) / `.clientInfo` (1.x).
- Protocol version: 2.x `ctx.protocol_version`; 1.x the `mcp-protocol-version` header, else what `ServerSession` negotiated from the client's `initialize`.
- `_meta` carriage: the client wrapper injects W3C `traceparent` / `tracestate` into `params._meta` of a COPY of the request (`model_dump(by_alias=True)` → `model_validate`); a `traceparent` the caller set is left alone. 1.x servers read it from `RequestParams.Meta.model_extra`, 2.x from `ctx.meta`. The server handles each request in a task of its own, so `_meta` is the only way the client's context arrives, even in-process.

## The `context` field (mode on)

- Injected into `tools/list` results (copies; dicts on 2.x, models / `ServerResult` on 1.x), never into `required`. Off with `context: False` or when payloads are off (including `LMNR_TRACE_CONTENT=false`). 2.x injects into the result the chain returns, 1.x into the response passed to `message.respond`, so the server's own tool cache keeps the real schemas.
- Stripped from `tools/call` arguments before the SDK validates them: 2.x by `dataclasses.replace(ctx, params=...)` before the chain; 1.x by `model_copy` of `req` before `_handle_request` hands it to the handler (where `validate_input` runs). FastMCP 3/4 reject unexpected arguments outright, so this is load-bearing.
- "Does the tool declare its own `context`?" is answered by the low-level server's registry when there is one (another server copy may have served `tools/list`): 2.x `Server.get_tool_input_schema` (set by `MCPServer` and FastMCP 4), 1.x `Server._get_cached_tool_definition` (refreshed from the list handler, so it covers mcp 1.x `FastMCP` and FastMCP 3). Otherwise what this process saw in `tools/list`; otherwise strip.
- `lmnr.mcp.context` = the stripped value (trimmed, 1000 chars) or `_meta["io.modelcontextprotocol/aiInvocation"].userIntent`. The recorded `gen_ai.tool.call.arguments` are the client's arguments, `context` included.

## Errors

- `isError` results: `-32602` when the first text block reads like an unknown tool / validation failure (`_INVALID_PARAMS_TEXT`, matched against what mcp 1.x/2.x and FastMCP 3/4 actually return), else `tool_error`; no `rpc.response.status_code`.
- Raised `McpError` / `MCPError`, or an `ErrorData` response (1.x): the code, plus `rpc.response.status_code`. 2.x `ValidationError` (params validation inside the chain): `-32602`, and its message (which quotes the client's input) is withheld without payloads. Anything else raised (2.x): the class name. On 1.x `_handle_request` answers any other handler exception with code `0`, so that is the `error.type` (no extra hook just to recover the class name).
- A traced method with no handler is still a span with `-32601`.
- Failures go through `utils.record_error`: the `exception` event (not the status) is what marks a span failed in Laminar. Without payloads the event never carries the raised exception or its text (fixed messages: `Tool returned an error`, `JSON-RPC error {code}`, the class name).
- CLIENT spans: tool `isError` results are `tool_error` with the first text block; client-side failures (`McpError` / `MCPError`, timeouts, closed streams) are recorded and re-raised, with `rpc.response.status_code` from the error's code. On 1.x, `ClientSession.call_tool` may call `list_tools` after the result (output-schema validation); that shows as a separate root `tools/list.mcp` span.

## Dropped from upstream

- Transport-specific wrappers (`stdio_client`, `sse_client`, streamable HTTP, `ResponseStreamWriter` spans) and the `ServerSession` stream proxies that saved context; `_meta` extraction on the server side replaces them.
- FastMCP-specific wrappers (`ToolManager.call_tool`, `FastMCP.__init__`, the `Client.__aenter__` / `__aexit__` session span): the session / server targets already cover FastMCP 2.x.
- The `Config` singleton, `exception_logger`, `traceloop.*` attributes and `mcp.response.value`.

## Tests

- The dev dependency pins mcp **2.x** (and `fastmcp` 4.x, which needs it). `conftest.py` uses `collect_ignore` to skip the modules of the other major, and provides the `mcp_mode` fixture (sets the option as `Laminar.initialize(mcp=True)` would, call it with options to change them, reset on teardown). `test_mcp_mode_v*.py` use it module-wide (`pytestmark`).
- `run-tests.yml` runs the 1.x modules (plus the targets test) as a separate `uv run --with mcp==1.30.0 ... -k mcp` step. Run that locally after touching `v1.py`, `server.py` or `utils.py`; the FastMCP 3 test needs `--with fastmcp==3.4.9` too (skipped otherwise). Under the 1.x overlay the 4.x package is still installed but unimportable; nothing imports it unless the server is a FastMCP one.
- `test_mcp_v1.py` / `test_mcp_mode_v1.py` start with file-level `# pyright:` comments because basedpyright checks them against the installed 2.x.
- Neither suite needs cassettes or subprocesses; client and server run in-process through the real SDK. 1.x: `create_connected_server_and_client_session`. 2.x: every test is parametrized over `Client(server)` (`DirectDispatcher`) and `Client(InMemoryTransport(server))` (JSON-RPC dispatcher), because only the latter has native spans; streamable HTTP via `httpx2.ASGITransport` (the URL's host must include a port to pass the DNS-rebinding check, and `json_response=True` because ASGITransport buffers).
- For a JSON-RPC error response, use a low-level `Server` with no prompt handlers (`prompts/list` → `-32601`).
