"""Unit tests for the deepagents instrumentation.

These tests cover the pure helpers (`_summarize_messages`, `_tool_result_to_json`,
`_extract_messages`, `_set_output_from_result`), the middleware span helpers
(`LaminarMiddleware._tool_span_name`, `LaminarMiddleware._tool_span_input`,
`LaminarMiddleware.wrap_tool_call`, `awrap_tool_call`), the `_SpanHandle`
context manager, and the graph wrappers (`_wrap_graph_invoke`,
`_awrap_graph_invoke`, `_wrap_graph_stream`, `_awrap_graph_stream`,
`_inject_middleware`).

They do NOT spin up a real deep agent — the wrappers are exercised against
mock callables that mimic the shapes produced by `Pregel.invoke` /
`Pregel.stream`. That keeps the tests hermetic (no LLM calls, no VCR
cassettes) and fast, while still covering all the invariants the code
relies on: span types, parent/child relationships, `_root_active` reuse
across nested `invoke`→`stream`, lazy setup in stream wrappers, graceful
handling of `GeneratorExit`, and idempotent middleware injection.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.deepagents import (
    LaminarMiddleware,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.deepagents.instrumentor import (
    _awrap_graph_invoke,
    _awrap_graph_stream,
    _extract_messages,
    _inject_middleware,
    _root_active,
    _set_output_from_result,
    _wrap_graph_invoke,
    _wrap_graph_methods,
    _wrap_graph_stream,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.deepagents.middleware import (
    _SpanHandle,
    _summarize_messages,
    _tool_result_to_json,
)


# --------------------------------------------------------------------- #
# Pure helpers                                                           #
# --------------------------------------------------------------------- #


class _FakeMessage:
    def __init__(self, type_: str | None = None, role: str | None = None, content: Any = None):
        if type_ is not None:
            self.type = type_
        if role is not None:
            self.role = role
        self.content = content


def test_summarize_messages_handles_objects_with_type():
    msgs = [_FakeMessage(type_="human", content="hi")]
    result = _summarize_messages(msgs)
    assert result == [{"role": "human", "content": "hi"}]


def test_summarize_messages_falls_back_to_role():
    msgs = [_FakeMessage(role="assistant", content="ok")]
    result = _summarize_messages(msgs)
    assert result == [{"role": "assistant", "content": "ok"}]


def test_summarize_messages_handles_dicts():
    msgs = [{"role": "user", "content": "q"}, {"type": "ai", "content": "a"}]
    result = _summarize_messages(msgs)
    assert result == [
        {"role": "user", "content": "q"},
        {"role": "ai", "content": "a"},
    ]


def test_summarize_messages_passes_through_unknown_items():
    # An item with neither a role nor a type attribute (and isn't a dict) is
    # passed through untouched.
    sentinel = object()
    result = _summarize_messages([sentinel])
    assert result == [sentinel]


def test_summarize_messages_returns_non_lists_unchanged():
    assert _summarize_messages("not a list") == "not a list"
    assert _summarize_messages(None) is None


def test_tool_result_to_json_with_content():
    class ToolMessage:
        content = "tool output"

    assert _tool_result_to_json(ToolMessage()) == "tool output"


def test_tool_result_to_json_with_update():
    # langgraph `Command` objects expose `.update` as a data attribute
    # (not a callable). The helper must wrap it in {"update": ...}.
    class Command:
        update = {"messages": []}

    assert _tool_result_to_json(Command()) == {"update": {"messages": []}}


def test_tool_result_to_json_passthrough():
    assert _tool_result_to_json(42) == 42
    assert _tool_result_to_json("plain") == "plain"


def test_tool_result_to_json_plain_dict_not_misidentified_as_command():
    # Regression guard: `dict.update` is a callable method on every Python
    # dict, but a langgraph `Command`'s `.update` is a data attribute. The
    # helper must NOT wrap plain dicts in {"update": <bound method>} —
    # that produces meaningless garbage in the tool-span output.
    assert _tool_result_to_json({"raw": 1}) == {"raw": 1}
    assert _tool_result_to_json({}) == {}


def test_extract_messages_returns_list_when_present():
    assert _extract_messages({"messages": [1, 2]}) == [1, 2]


def test_extract_messages_returns_none_for_missing_key():
    assert _extract_messages({}) is None


def test_extract_messages_returns_none_for_non_dict():
    assert _extract_messages("not a dict") is None
    assert _extract_messages(None) is None


def test_set_output_from_result_prefers_last_message_content():
    class _H:
        def __init__(self):
            self.output = None

        def set_output(self, value):
            self.output = value

    handle = _H()
    _set_output_from_result(handle, {"messages": [_FakeMessage(type_="ai", content="final")]})
    assert handle.output == "final"


def test_set_output_from_result_falls_back_to_summary_for_non_content_messages():
    class _H:
        def __init__(self):
            self.output = None

        def set_output(self, value):
            self.output = value

    handle = _H()
    # A last message with neither content nor dict-content still yields the
    # whole summary so the span isn't completely empty.
    _set_output_from_result(handle, {"messages": [{"foo": "bar"}]})
    # The last dict has no content key, so content stays None and we fall
    # back to `_summarize_messages`, which returns the dict list unchanged
    # because role/type are absent.
    assert handle.output == [{"foo": "bar"}]


def test_set_output_from_result_noop_when_no_messages():
    calls = []

    class _H:
        def set_output(self, value):
            calls.append(value)

    _set_output_from_result(_H(), {})
    _set_output_from_result(_H(), "string")
    assert calls == []


# --------------------------------------------------------------------- #
# LaminarMiddleware helpers                                              #
# --------------------------------------------------------------------- #


class _FakeToolCallRequest:
    def __init__(self, name=None, args=None, tool_name=None):
        self.tool_call = {"name": name, "args": args} if name is not None else None
        if tool_name is not None:
            self.tool = types.SimpleNamespace(name=tool_name)


def test_tool_span_name_prefers_tool_call_name():
    req = _FakeToolCallRequest(name="read_file", args={"path": "/tmp/x"})
    assert LaminarMiddleware._tool_span_name(req) == "read_file"


def test_tool_span_name_falls_back_to_request_tool_name():
    req = types.SimpleNamespace(tool_call=None, tool=types.SimpleNamespace(name="fallback"))
    assert LaminarMiddleware._tool_span_name(req) == "fallback"


def test_tool_span_name_defaults_to_generic_tool():
    req = types.SimpleNamespace(tool_call=None, tool=None)
    assert LaminarMiddleware._tool_span_name(req) == "tool"


def test_tool_span_input_pulls_from_tool_call_args():
    req = _FakeToolCallRequest(name="x", args={"q": 1})
    assert LaminarMiddleware._tool_span_input(req) == {"q": 1}


def test_tool_span_input_returns_none_without_tool_call():
    req = types.SimpleNamespace(tool_call=None)
    assert LaminarMiddleware._tool_span_input(req) is None


# --------------------------------------------------------------------- #
# _SpanHandle / wrap_tool_call with TracerWrapper initialized            #
# --------------------------------------------------------------------- #


def _span_by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_span_handle_emits_span_with_type_and_ends_on_exit(
    span_exporter: InMemorySpanExporter,
):
    with _SpanHandle("unit-test", "TOOL") as handle:
        assert handle.span is not None
        handle.set_input({"foo": "bar"})
        handle.set_output("done")

    spans = _span_by_name(span_exporter, "unit-test")
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes["lmnr.span.type"] == "TOOL"
    # set_input / set_output serialize to JSON
    assert s.attributes["lmnr.span.input"] == '{"foo":"bar"}'
    assert s.attributes["lmnr.span.output"] == '"done"'


def test_span_handle_records_exception(span_exporter: InMemorySpanExporter):
    exc = RuntimeError("boom")
    with _SpanHandle("err-span", "TOOL") as handle:
        handle.record_exception(exc)

    spans = _span_by_name(span_exporter, "err-span")
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "ERROR"
    assert s.status.description == "boom"


def test_wrap_tool_call_emits_tool_span_with_name_and_args(
    span_exporter: InMemorySpanExporter,
):
    middleware = LaminarMiddleware()
    request = _FakeToolCallRequest(name="read_file", args={"path": "/etc/hosts"})

    def handler(req):
        return types.SimpleNamespace(content="file contents")

    result = middleware.wrap_tool_call(request, handler)
    assert result.content == "file contents"

    spans = _span_by_name(span_exporter, "read_file")
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes["lmnr.span.type"] == "TOOL"
    assert s.attributes["lmnr.span.input"] == '{"path":"/etc/hosts"}'
    assert s.attributes["lmnr.span.output"] == '"file contents"'


def test_wrap_tool_call_records_exception_and_reraises(
    span_exporter: InMemorySpanExporter,
):
    middleware = LaminarMiddleware()
    request = _FakeToolCallRequest(name="broken", args={})

    def handler(req):
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        middleware.wrap_tool_call(request, handler)

    spans = _span_by_name(span_exporter, "broken")
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"


@pytest.mark.asyncio
async def test_awrap_tool_call_emits_tool_span(span_exporter: InMemorySpanExporter):
    middleware = LaminarMiddleware()
    request = _FakeToolCallRequest(name="async_tool", args={"k": "v"})

    async def handler(req):
        return types.SimpleNamespace(content="async out")

    result = await middleware.awrap_tool_call(request, handler)
    assert result.content == "async out"

    spans = _span_by_name(span_exporter, "async_tool")
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "TOOL"
    assert spans[0].attributes["lmnr.span.output"] == '"async out"'


# --------------------------------------------------------------------- #
# Graph wrappers                                                         #
# --------------------------------------------------------------------- #


def test_wrap_graph_invoke_emits_root_span_with_input_and_output(
    span_exporter: InMemorySpanExporter,
):
    def wrapped(payload):
        return {"messages": [_FakeMessage(type_="ai", content="hello")]}

    result = _wrap_graph_invoke(
        wrapped,
        instance=None,
        args=({"messages": [_FakeMessage(type_="human", content="hi")]},),
        kwargs={},
    )
    assert result["messages"][0].content == "hello"

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes["lmnr.span.type"] == "DEFAULT"
    assert s.attributes["lmnr.span.input"] == '{"messages":[{"role":"human","content":"hi"}]}'
    assert s.attributes["lmnr.span.output"] == '"hello"'


def test_wrap_graph_invoke_skips_when_root_already_active(
    span_exporter: InMemorySpanExporter,
):
    # Simulate Pregel.invoke internally calling self.stream: the outer wrapper
    # sets `_root_active` to True, and the inner wrapper must see it and
    # short-circuit without opening a second root span.
    token = _root_active.set(True)
    try:
        called = []

        def wrapped(payload):
            called.append(payload)
            return {"messages": []}

        _wrap_graph_invoke(wrapped, instance=None, args=({"messages": []},), kwargs={})
    finally:
        _root_active.reset(token)

    # No deep_agent span emitted — the outer-most call owns the root span.
    assert _span_by_name(span_exporter, "deep_agent") == []
    assert len(called) == 1


def test_wrap_graph_invoke_records_exception(span_exporter: InMemorySpanExporter):
    def wrapped(payload):
        raise RuntimeError("graph failed")

    with pytest.raises(RuntimeError, match="graph failed"):
        _wrap_graph_invoke(
            wrapped, instance=None, args=({"messages": []},), kwargs={}
        )

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"


def test_wrap_graph_invoke_resets_root_active_on_success(
    span_exporter: InMemorySpanExporter,
):
    assert _root_active.get() is False

    def wrapped(payload):
        assert _root_active.get() is True
        return {"messages": []}

    _wrap_graph_invoke(wrapped, instance=None, args=({"messages": []},), kwargs={})
    assert _root_active.get() is False


def test_wrap_graph_invoke_resets_root_active_on_exception(
    span_exporter: InMemorySpanExporter,
):
    assert _root_active.get() is False

    def wrapped(payload):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        _wrap_graph_invoke(
            wrapped, instance=None, args=({"messages": []},), kwargs={}
        )
    assert _root_active.get() is False


@pytest.mark.asyncio
async def test_awrap_graph_invoke_emits_root_span(
    span_exporter: InMemorySpanExporter,
):
    async def wrapped(payload):
        return {"messages": [_FakeMessage(type_="ai", content="async ok")]}

    result = await _awrap_graph_invoke(
        wrapped,
        instance=None,
        args=({"messages": []},),
        kwargs={},
    )
    assert result["messages"][0].content == "async ok"

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "DEFAULT"
    assert spans[0].attributes["lmnr.span.output"] == '"async ok"'


# --- stream wrappers: lazy setup + GeneratorExit ---------------------- #


def test_wrap_graph_stream_does_not_open_span_if_never_iterated(
    span_exporter: InMemorySpanExporter,
):
    # If the caller discards the returned generator without iterating it,
    # _SpanHandle must NOT have been entered and `_root_active` must still
    # be False for the next invoke/stream call on the same task.
    def wrapped(payload):
        yield {"messages": []}

    gen = _wrap_graph_stream(
        wrapped, instance=None, args=({"messages": []},), kwargs={}
    )

    # No iteration → no span, no sentinel.
    assert _span_by_name(span_exporter, "deep_agent") == []
    assert _root_active.get() is False

    # Clean up the generator.
    gen.close()

    # Still no span (the generator was never actually run).
    assert _span_by_name(span_exporter, "deep_agent") == []
    assert _root_active.get() is False


def test_wrap_graph_stream_emits_span_when_iterated(
    span_exporter: InMemorySpanExporter,
):
    def wrapped(payload):
        yield {"step": 1}
        yield {"step": 2}

    chunks = list(
        _wrap_graph_stream(
            wrapped, instance=None, args=({"messages": []},), kwargs={}
        )
    )
    assert chunks == [{"step": 1}, {"step": 2}]

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "DEFAULT"
    # `_root_active` must not have leaked out of the generator body — no
    # matter what happened inside, callers must see the default `False`.
    assert _root_active.get() is False


def test_interleaved_sync_streams_both_get_root_spans(
    span_exporter: InMemorySpanExporter,
):
    # Regression guard: if the generator body sets `_root_active=True`
    # before yielding, the mutation leaks to the caller's context on
    # `yield` (documented Python sync-generator behaviour). A second
    # concurrent `graph.stream()` would then see the sentinel on its
    # eager check and skip instrumentation. Interleaving two streams
    # must still emit one root span per stream.
    def wrapped_a(payload):
        yield {"stream": "a", "step": 1}
        yield {"stream": "a", "step": 2}

    def wrapped_b(payload):
        yield {"stream": "b", "step": 1}
        yield {"stream": "b", "step": 2}

    gen_a = _wrap_graph_stream(
        wrapped_a, instance=None, args=({"messages": []},), kwargs={}
    )
    # Pull the first chunk from A (this starts the generator body and,
    # before the fix, would flip `_root_active` on in the caller).
    first_a = next(gen_a)
    assert first_a == {"stream": "a", "step": 1}

    # B must still be instrumented.
    gen_b = _wrap_graph_stream(
        wrapped_b, instance=None, args=({"messages": []},), kwargs={}
    )
    chunks_b = list(gen_b)
    assert chunks_b == [{"stream": "b", "step": 1}, {"stream": "b", "step": 2}]

    # Finish A.
    chunks_a = [first_a] + list(gen_a)
    assert chunks_a == [
        {"stream": "a", "step": 1},
        {"stream": "a", "step": 2},
    ]

    # Two streams → two root spans.
    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 2
    assert all(s.attributes["lmnr.span.type"] == "DEFAULT" for s in spans)


def test_wrap_graph_stream_does_not_mark_error_on_generator_exit(
    span_exporter: InMemorySpanExporter,
):
    # Breaking out of the for-loop sends GeneratorExit into the wrapped
    # generator; the span must be ended cleanly (status UNSET, not ERROR).
    def wrapped(payload):
        yield {"step": 1}
        yield {"step": 2}
        yield {"step": 3}

    for chunk in _wrap_graph_stream(
        wrapped, instance=None, args=({"messages": []},), kwargs={}
    ):
        break  # triggers GeneratorExit on the underlying generator

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].status.status_code.name != "ERROR"
    assert _root_active.get() is False


def test_wrap_graph_stream_records_real_exception(
    span_exporter: InMemorySpanExporter,
):
    def wrapped(payload):
        yield {"step": 1}
        raise RuntimeError("stream boom")

    with pytest.raises(RuntimeError, match="stream boom"):
        list(
            _wrap_graph_stream(
                wrapped, instance=None, args=({"messages": []},), kwargs={}
            )
        )

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
    assert _root_active.get() is False


@pytest.mark.asyncio
async def test_awrap_graph_stream_emits_span_when_iterated(
    span_exporter: InMemorySpanExporter,
):
    async def wrapped(payload):
        yield {"step": 1}
        yield {"step": 2}

    collected = []
    async for chunk in _awrap_graph_stream(
        wrapped, instance=None, args=({"messages": []},), kwargs={}
    ):
        collected.append(chunk)
    assert collected == [{"step": 1}, {"step": 2}]

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "DEFAULT"
    assert _root_active.get() is False


@pytest.mark.asyncio
async def test_awrap_graph_stream_does_not_mark_error_on_generator_exit(
    span_exporter: InMemorySpanExporter,
):
    async def wrapped(payload):
        yield {"step": 1}
        yield {"step": 2}

    agen = _awrap_graph_stream(
        wrapped, instance=None, args=({"messages": []},), kwargs={}
    )
    async for _ in agen:
        break
    # aclose to propagate GeneratorExit-equivalent cleanup
    await agen.aclose()

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].status.status_code.name != "ERROR"
    assert _root_active.get() is False


# --------------------------------------------------------------------- #
# Middleware injection + idempotent graph wrapping                       #
# --------------------------------------------------------------------- #


def test_inject_middleware_adds_laminar_middleware_when_absent():
    received = {}

    def fake_create_deep_agent(*args, **kwargs):
        received.update(kwargs)
        return types.SimpleNamespace(invoke=lambda *a, **k: None, __class__=type("G", (), {}))

    _inject_middleware(fake_create_deep_agent, instance=None, args=(), kwargs={})
    assert any(isinstance(m, LaminarMiddleware) for m in received["middleware"])


def test_inject_middleware_does_not_duplicate_laminar_middleware():
    existing = LaminarMiddleware()
    received = {}

    def fake_create_deep_agent(*args, **kwargs):
        received.update(kwargs)
        return types.SimpleNamespace(invoke=lambda *a, **k: None)

    _inject_middleware(
        fake_create_deep_agent,
        instance=None,
        args=(),
        kwargs={"middleware": (existing,)},
    )
    mws = [m for m in received["middleware"] if isinstance(m, LaminarMiddleware)]
    assert mws == [existing]


def test_wrap_graph_methods_is_idempotent():
    # A graph that was already wrapped by a previous call must not be
    # wrapped a second time — otherwise stream/invoke would emit two root
    # spans per top-level call.
    class FakeGraph:
        def invoke(self, payload):
            return {"messages": []}

    g = FakeGraph()
    _wrap_graph_methods(g)
    invoke_after_first = g.invoke
    _wrap_graph_methods(g)
    assert g.invoke is invoke_after_first


def test_inject_middleware_wraps_returned_graph_invoke(
    span_exporter: InMemorySpanExporter,
):
    # Simulate the full flow: `create_deep_agent` returns a graph whose
    # `invoke` must, after injection, emit a `deep_agent` root span per
    # top-level call.
    class FakeGraph:
        def invoke(self, payload):
            return {"messages": [_FakeMessage(type_="ai", content="answer")]}

    def fake_create_deep_agent(*args, **kwargs):
        return FakeGraph()

    graph = _inject_middleware(
        fake_create_deep_agent, instance=None, args=(), kwargs={}
    )
    result = graph.invoke({"messages": [_FakeMessage(type_="human", content="q")]})
    assert result["messages"][0].content == "answer"

    spans = _span_by_name(span_exporter, "deep_agent")
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "DEFAULT"


def test_nested_stream_inside_invoke_collapses_to_one_root_span(
    span_exporter: InMemorySpanExporter,
):
    # Pregel.invoke delegates to self.stream internally. Both paths are
    # wrapped — the `_root_active` sentinel must collapse them into one
    # root span, matching the behaviour documented in CLAUDE.md.
    def inner_stream(payload):
        yield {"step": 1}

    def outer_invoke(payload):
        # Simulate Pregel's pattern: invoke calls the already-wrapped stream.
        wrapped_stream = _wrap_graph_stream(
            inner_stream, instance=None, args=(payload,), kwargs={}
        )
        return {"messages": list(wrapped_stream)}

    _wrap_graph_invoke(
        outer_invoke, instance=None, args=({"messages": []},), kwargs={}
    )

    assert len(_span_by_name(span_exporter, "deep_agent")) == 1


# --------------------------------------------------------------------- #
# Auto-enable + noise conflicts (deepagents vs langchain/langgraph)      #
# --------------------------------------------------------------------- #


def test_deepagents_auto_enabled_removes_langchain_and_langgraph():
    # When deepagents is installed, the default instrument set keeps
    # DEEPAGENTS and drops LANGCHAIN/LANGGRAPH. This is the contract
    # codified in `_DEEPAGENTS_NOISE_CONFLICTS` and relied on by
    # `init_instrumentations`.
    from lmnr.opentelemetry_lib.tracing.instruments import (
        Instruments,
        _DEEPAGENTS_NOISE_CONFLICTS,
    )

    assert Instruments.LANGCHAIN in _DEEPAGENTS_NOISE_CONFLICTS
    assert Instruments.LANGGRAPH in _DEEPAGENTS_NOISE_CONFLICTS
    # DEEPAGENTS must NOT be in the noise-conflicts set — it's the one
    # instrument we want to KEEP.
    assert Instruments.DEEPAGENTS not in _DEEPAGENTS_NOISE_CONFLICTS


def test_deepagents_initializer_returns_none_when_package_missing(monkeypatch):
    from lmnr.opentelemetry_lib.tracing import _instrument_initializers as inits

    monkeypatch.setattr(inits, "is_package_installed", lambda name: False)
    result = inits.DeepagentsInstrumentorInitializer().init_instrumentor()
    assert result is None


def test_deepagents_initializer_returns_none_when_langchain_missing(monkeypatch):
    from lmnr.opentelemetry_lib.tracing import _instrument_initializers as inits

    def only_deepagents(name):
        return name == "deepagents"

    monkeypatch.setattr(inits, "is_package_installed", only_deepagents)
    result = inits.DeepagentsInstrumentorInitializer().init_instrumentor()
    assert result is None
