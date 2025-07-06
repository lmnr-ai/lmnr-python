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


def test_observe_user_id(exporter: InMemorySpanExporter):
    @observe(user_id="123")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_metadata(exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.key"] == "value"
    assert json.loads(
        spans[0].attributes["lmnr.association.properties.metadata.nested"]
    ) == {"key2": "value2"}
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

    assert foo_span.attributes["lmnr.span.input"] == json.dumps({})
    assert foo_span.attributes["lmnr.span.path"] == ("observed_foo",)
    assert bar_span.attributes["lmnr.span.input"] == json.dumps({})
    assert bar_span.attributes["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert foo_span.attributes["lmnr.span.output"] == json.dumps("bar")
    assert bar_span.attributes["lmnr.span.output"] == json.dumps("bar")

    assert foo_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert bar_span.attributes["lmnr.span.instrumentation_source"] == "python"


def test_observe_skip_input_keys(exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a"])
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


def test_observe_skip_input_keys_with_kwargs(exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a", "d"])
    def observed_foo(a, b, c, **kwargs):
        return "foo"

    result = observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "b": 2,
        "c": 3,
        "e": 5,
        "f": 6,
    }


@pytest.mark.asyncio
async def test_observe_skip_input_keys_async(exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a"])
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


def test_observe_tags(exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_tags_invalid_type(exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes.get("lmnr.association.properties.tags") is None
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_sequential_spans_async(exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    @observe()
    async def observed_bar():
        return "bar"

    await observed_foo()
    await observed_bar()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert foo_span.get_span_context().trace_id != bar_span.get_span_context().trace_id


@pytest.mark.asyncio
async def test_observe_name_async(exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("custom_name",)


@pytest.mark.asyncio
async def test_observe_session_id_async(exporter: InMemorySpanExporter):
    @observe(session_id="123")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_user_id_async(exporter: InMemorySpanExporter):
    @observe(user_id="123")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_metadata_async(exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.key"] == "value"
    assert json.loads(
        spans[0].attributes["lmnr.association.properties.metadata.nested"]
    ) == {"key2": "value2"}
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_exception_with_session_id_and_name_async(
    exporter: InMemorySpanExporter,
):
    @observe(session_id="123", name="custom_name")
    async def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        await observed_foo()

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
async def test_observe_nested_async(exporter: InMemorySpanExporter):
    @observe()
    async def observed_bar():
        return "bar"

    @observe(session_id="123")
    async def observed_foo():
        return await observed_bar()

    result = await observed_foo()
    spans = exporter.get_finished_spans()

    assert result == "bar"
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]
    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent.span_id == foo_span.context.span_id
    assert foo_span.context.trace_id == bar_span.context.trace_id

    assert foo_span.attributes["lmnr.association.properties.session_id"] == "123"

    assert foo_span.attributes["lmnr.span.input"] == json.dumps({})
    assert foo_span.attributes["lmnr.span.path"] == ("observed_foo",)
    assert bar_span.attributes["lmnr.span.input"] == json.dumps({})
    assert bar_span.attributes["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert foo_span.attributes["lmnr.span.output"] == json.dumps("bar")
    assert bar_span.attributes["lmnr.span.output"] == json.dumps("bar")

    assert foo_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert bar_span.attributes["lmnr.span.instrumentation_source"] == "python"


@pytest.mark.asyncio
async def test_observe_skip_input_keys_with_kwargs_async(
    exporter: InMemorySpanExporter,
):
    @observe(ignore_inputs=["a", "d"])
    async def observed_foo(a, b, c, **kwargs):
        return "foo"

    result = await observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "b": 2,
        "c": 3,
        "e": 5,
        "f": 6,
    }


@pytest.mark.asyncio
async def test_observe_tags_async(exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_tags_invalid_type_async(exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes.get("lmnr.association.properties.tags") is None
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_sequential_spans(exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    @observe()
    def observed_bar():
        return "bar"

    observed_foo()
    observed_bar()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert foo_span.get_span_context().trace_id != bar_span.get_span_context().trace_id
