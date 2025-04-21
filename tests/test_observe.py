import json
import pytest

from lmnr import observe
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_observe(exporter: InMemorySpanExporter):
    @observe()
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()

    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "x": "arg",
        "y": "arg2",
        "z": "arg3",
        "a": 1,
        "b": 2,
        "c": 3,
    }
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_name(exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("custom_name",)


def test_observe_session_id(exporter: InMemorySpanExporter):
    @observe(session_id="123")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception(exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "test"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception_with_session_id_and_name(exporter: InMemorySpanExporter):
    @observe(session_id="123", name="custom_name")
    def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "custom_name"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("custom_name",)

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "test"


@pytest.mark.asyncio
async def test_observe_async(exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    res = await observed_foo()
    spans = exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_async_exception(exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        await observed_foo()

    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "test"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_nested(exporter: InMemorySpanExporter):
    @observe()
    def observed_bar():
        return "bar"

    @observe(session_id="123")
    def observed_foo():
        return observed_bar()

    result = observed_foo()
    spans = exporter.get_finished_spans()

    assert result == "bar"
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]
    assert bar_span.parent.span_id == foo_span.context.span_id

    assert foo_span.attributes["lmnr.association.properties.session_id"] == "123"
    assert bar_span.attributes["lmnr.association.properties.session_id"] == "123"

    assert foo_span.attributes["lmnr.span.input"] == json.dumps({})
    assert foo_span.attributes["lmnr.span.path"] == ("observed_foo",)
    assert bar_span.attributes["lmnr.span.input"] == json.dumps({})
    assert bar_span.attributes["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert foo_span.attributes["lmnr.span.output"] == json.dumps("bar")
    assert bar_span.attributes["lmnr.span.output"] == json.dumps("bar")

    assert foo_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert bar_span.attributes["lmnr.span.instrumentation_source"] == "python"


def test_observe_skip_input_keys(exporter: InMemorySpanExporter):
    @observe(skip_input_keys=["a"])
    def observed_foo(a, b, c):
        return "foo"

    result = observed_foo(1, 2, 3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert spans[0].attributes["lmnr.span.input"] == json.dumps({"b": 2, "c": 3})


@pytest.mark.asyncio
async def test_observe_skip_input_keys_async(exporter: InMemorySpanExporter):
    @observe(skip_input_keys=["a"])
    async def observed_foo(a, b, c):
        return "foo"

    res = await observed_foo(1, 2, 3)
    spans = exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert spans[0].attributes["lmnr.span.input"] == json.dumps({"b": 2, "c": 3})
