import json
import pytest

from lmnr import observe
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()

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


def test_observe_name(span_exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("custom_name",)


def test_observe_session_id(span_exporter: InMemorySpanExporter):
    @observe(session_id="123")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_user_id(span_exporter: InMemorySpanExporter):
    @observe(user_id="123")
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_metadata(span_exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.key"] == "value"
    assert json.loads(
        spans[0].attributes["lmnr.association.properties.metadata.nested"]
    ) == {"key2": "value2"}
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "test"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception_with_session_id_and_name(
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="123", name="custom_name")
    def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()

    spans = span_exporter.get_finished_spans()
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
async def test_observe_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    res = await observed_foo()
    spans = span_exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_async_exception(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        await observed_foo()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "test"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_nested(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_bar():
        return "bar"

    @observe(session_id="123")
    def observed_foo():
        return observed_bar()

    result = observed_foo()
    spans = span_exporter.get_finished_spans()

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


def test_observe_skip_input_keys(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a"])
    def observed_foo(a, b, c):
        return "foo"

    result = observed_foo(1, 2, 3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert spans[0].attributes["lmnr.span.input"] == json.dumps({"b": 2, "c": 3})


def test_observe_skip_input_keys_with_kwargs(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a", "d"])
    def observed_foo(a, b, c, **kwargs):
        return "foo"

    result = observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = span_exporter.get_finished_spans()
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
async def test_observe_skip_input_keys_async(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["a"])
    async def observed_foo(a, b, c):
        return "foo"

    res = await observed_foo(1, 2, 3)
    spans = span_exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert spans[0].attributes["lmnr.span.input"] == json.dumps({"b": 2, "c": 3})


def test_observe_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_tags_invalid_type(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes.get("lmnr.association.properties.tags") is None
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_sequential_spans_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    @observe()
    async def observed_bar():
        return "bar"

    await observed_foo()
    await observed_bar()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert foo_span.get_span_context().trace_id != bar_span.get_span_context().trace_id


@pytest.mark.asyncio
async def test_observe_name_async(span_exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("custom_name",)


@pytest.mark.asyncio
async def test_observe_session_id_async(span_exporter: InMemorySpanExporter):
    @observe(session_id="123")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_user_id_async(span_exporter: InMemorySpanExporter):
    @observe(user_id="123")
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_metadata_async(span_exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
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
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="123", name="custom_name")
    async def observed_foo():
        raise ValueError("test")

    with pytest.raises(ValueError):
        await observed_foo()

    spans = span_exporter.get_finished_spans()
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
async def test_observe_nested_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_bar():
        return "bar"

    @observe(session_id="123")
    async def observed_foo():
        return await observed_bar()

    result = await observed_foo()
    spans = span_exporter.get_finished_spans()

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
    span_exporter: InMemorySpanExporter,
):
    @observe(ignore_inputs=["a", "d"])
    async def observed_foo(a, b, c, **kwargs):
        return "foo"

    result = await observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = span_exporter.get_finished_spans()
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


def test_observe_input_formatter(exporter: InMemorySpanExporter):
    def input_formatter(x):
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"x": 2}


def test_observe_input_formatter_exception(exporter: InMemorySpanExporter):
    def input_formatter(x):
        raise ValueError("test")

    @observe(input_formatter=input_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert "lmnr.span.input" not in spans[0].attributes


def test_observe_input_formatter_with_kwargs(exporter: InMemorySpanExporter):
    def input_formatter(x, **kwargs):
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    def observed_foo(x, **kwargs):
        return x

    result = observed_foo(1, a=1, b=2)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "x": 2,
        "custom-A": "1--",
    }


@pytest.mark.asyncio
async def test_observe_tags_async(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_tags_invalid_type_async(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])
    async def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes.get("lmnr.association.properties.tags") is None
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.path"] == ("observed_foo",)


def test_observe_sequential_spans(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    @observe()
    def observed_bar():
        return "bar"

    observed_foo()
    observed_bar()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = [span for span in spans if span.name == "observed_foo"][0]
    bar_span = [span for span in spans if span.name == "observed_bar"][0]

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert foo_span.get_span_context().trace_id != bar_span.get_span_context().trace_id


@pytest.mark.asyncio
async def test_observe_input_formatter_async(exporter: InMemorySpanExporter):
    def input_formatter(x):
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x):
        return x

    result = await observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.input"] == json.dumps({"x": 2})


@pytest.mark.asyncio
async def test_observe_input_formatter_with_kwargs_async(
    exporter: InMemorySpanExporter,
):
    def input_formatter(x, **kwargs):
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x, **kwargs):
        return x

    result = await observed_foo(1, a=1, b=2)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "x": 2,
        "custom-A": "1--",
    }


def test_observe_output_formatter(exporter: InMemorySpanExporter):
    def output_formatter(x):
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == {"x": 2}


def test_observe_output_formatter_exception(exporter: InMemorySpanExporter):
    def output_formatter(x):
        raise ValueError("test")

    @observe(output_formatter=output_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert "lmnr.span.output" not in spans[0].attributes


@pytest.mark.asyncio
async def test_observe_output_formatter_async(exporter: InMemorySpanExporter):
    def output_formatter(x):
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    async def observed_foo(x):
        return x

    result = await observed_foo(1)
    spans = exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == {"x": 2}


def test_observe_non_serializable_input(exporter: InMemorySpanExporter):
    class NonSerializable:
        def __init__(self, x: int):
            self.x = x

    @observe()
    def observed_foo(x: NonSerializable, y: int):
        return x

    observed_foo(NonSerializable(1), 2)
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]
    span_input = json.loads(span.attributes["lmnr.span.input"])
    assert span_input["y"] == 2
    assert "NonSerializable object at 0x" in span_input["x"]
    assert "NonSerializable object at 0x" in json.loads(
        span.attributes["lmnr.span.output"]
    )
