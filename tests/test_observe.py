import json
import os
import uuid
import pytest

from lmnr import Laminar, observe, LaminarSpanContext
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace


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
    assert bar_span.parent.span_id == foo_span.get_span_context().span_id

    assert foo_span.attributes["lmnr.association.properties.session_id"] == "123"

    assert foo_span.attributes["lmnr.span.input"] == json.dumps({})
    assert foo_span.attributes["lmnr.span.path"] == ("observed_foo",)
    assert bar_span.attributes["lmnr.span.input"] == json.dumps({})
    assert bar_span.attributes["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert foo_span.attributes["lmnr.span.output"] == json.dumps("bar")
    assert bar_span.attributes["lmnr.span.output"] == json.dumps("bar")

    assert foo_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert bar_span.attributes["lmnr.span.instrumentation_source"] == "python"


def test_observe_deeply_nested_and_sequential(span_exporter: InMemorySpanExporter):
    @observe()
    def level_4():
        return "level_4"

    @observe()
    def level_3():
        return level_4()

    @observe()
    def level_2():
        return level_3()

    @observe()
    def level_1():
        return level_2()

    @observe()
    def after_all():
        return "after_all"

    result = level_1()
    after_all()
    spans = span_exporter.get_finished_spans()
    assert result == "level_4"
    assert len(spans) == 5

    level_1_span = [span for span in spans if span.name == "level_1"][0]
    level_2_span = [span for span in spans if span.name == "level_2"][0]
    level_3_span = [span for span in spans if span.name == "level_3"][0]
    level_4_span = [span for span in spans if span.name == "level_4"][0]
    after_all_span = [span for span in spans if span.name == "after_all"][0]

    assert level_1_span.parent is None or level_1_span.parent.span_id == 0
    assert level_2_span.parent.span_id == level_1_span.get_span_context().span_id
    assert level_3_span.parent.span_id == level_2_span.get_span_context().span_id
    assert level_4_span.parent.span_id == level_3_span.get_span_context().span_id
    assert after_all_span.parent is None or after_all_span.parent.span_id == 0

    assert level_1_span.attributes["lmnr.span.path"] == ("level_1",)
    assert level_2_span.attributes["lmnr.span.path"] == ("level_1", "level_2")
    assert level_3_span.attributes["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
    )
    assert level_4_span.attributes["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
        "level_4",
    )
    assert after_all_span.attributes["lmnr.span.path"] == ("after_all",)

    assert level_1_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
    )
    assert level_2_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
    )
    assert level_3_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_3_span.get_span_context().span_id)),
    )
    assert level_4_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_3_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_4_span.get_span_context().span_id)),
    )
    assert after_all_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=after_all_span.get_span_context().span_id)),
    )

    assert (
        level_1_span.get_span_context().trace_id
        == level_2_span.get_span_context().trace_id
        == level_3_span.get_span_context().trace_id
        == level_4_span.get_span_context().trace_id
    )
    assert (
        after_all_span.get_span_context().trace_id
        != level_4_span.get_span_context().trace_id
    )


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
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"b": 2, "c": 3}


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
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"b": 2, "c": 3}


def test_observe_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert sorted(span.attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]
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
    assert bar_span.parent.span_id == foo_span.get_span_context().span_id
    assert foo_span.get_span_context().trace_id == bar_span.get_span_context().trace_id

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
async def test_observe_deeply_nested_and_sequential_async(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def level_4():
        return "level_4"

    @observe()
    async def level_3():
        return await level_4()

    @observe()
    async def level_2():
        return await level_3()

    @observe()
    async def level_1():
        return await level_2()

    @observe()
    async def after_all():
        return "after_all"

    result = await level_1()
    await after_all()
    spans = span_exporter.get_finished_spans()
    assert result == "level_4"
    assert len(spans) == 5

    level_1_span = [span for span in spans if span.name == "level_1"][0]
    level_2_span = [span for span in spans if span.name == "level_2"][0]
    level_3_span = [span for span in spans if span.name == "level_3"][0]
    level_4_span = [span for span in spans if span.name == "level_4"][0]
    after_all_span = [span for span in spans if span.name == "after_all"][0]

    assert level_1_span.parent is None or level_1_span.parent.span_id == 0
    assert level_2_span.parent.span_id == level_1_span.get_span_context().span_id
    assert level_3_span.parent.span_id == level_2_span.get_span_context().span_id
    assert level_4_span.parent.span_id == level_3_span.get_span_context().span_id
    assert after_all_span.parent is None or after_all_span.parent.span_id == 0

    assert level_1_span.attributes["lmnr.span.path"] == ("level_1",)
    assert level_2_span.attributes["lmnr.span.path"] == ("level_1", "level_2")
    assert level_3_span.attributes["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
    )
    assert level_4_span.attributes["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
        "level_4",
    )
    assert after_all_span.attributes["lmnr.span.path"] == ("after_all",)

    assert level_1_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
    )
    assert level_2_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
    )
    assert level_3_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_3_span.get_span_context().span_id)),
    )
    assert level_4_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=level_1_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_2_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_3_span.get_span_context().span_id)),
        str(uuid.UUID(int=level_4_span.get_span_context().span_id)),
    )
    assert after_all_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=after_all_span.get_span_context().span_id)),
    )

    assert (
        level_1_span.get_span_context().trace_id
        == level_2_span.get_span_context().trace_id
        == level_3_span.get_span_context().trace_id
        == level_4_span.get_span_context().trace_id
    )
    assert (
        after_all_span.get_span_context().trace_id
        != level_4_span.get_span_context().trace_id
    )


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


def test_observe_input_formatter(span_exporter: InMemorySpanExporter):
    def input_formatter(x):
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"x": 2}


def test_observe_input_formatter_exception(span_exporter: InMemorySpanExporter):
    def input_formatter(x):
        raise ValueError("test")

    @observe(input_formatter=input_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)
    assert "lmnr.span.input" not in spans[0].attributes


def test_observe_input_formatter_with_kwargs(span_exporter: InMemorySpanExporter):
    def input_formatter(x, **kwargs):
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    def observed_foo(x, **kwargs):
        return x

    result = observed_foo(1, a=1, b=2)
    spans = span_exporter.get_finished_spans()
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

    assert sorted(span.attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]
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


@pytest.mark.asyncio
async def test_observe_input_formatter_async(span_exporter: InMemorySpanExporter):
    def input_formatter(x):
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x):
        return x

    result = await observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"x": 2}


@pytest.mark.asyncio
async def test_observe_input_formatter_with_kwargs_async(
    span_exporter: InMemorySpanExporter,
):
    def input_formatter(x, **kwargs):
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x, **kwargs):
        return x

    result = await observed_foo(1, a=1, b=2)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {
        "x": 2,
        "custom-A": "1--",
    }


def test_observe_output_formatter(span_exporter: InMemorySpanExporter):
    def output_formatter(x):
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == {"x": 2}


def test_observe_output_formatter_exception(span_exporter: InMemorySpanExporter):
    def output_formatter(x):
        raise ValueError("test")

    @observe(output_formatter=output_formatter)
    def observed_foo(x):
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert "lmnr.span.output" not in spans[0].attributes


@pytest.mark.asyncio
async def test_observe_output_formatter_async(span_exporter: InMemorySpanExporter):
    def output_formatter(x):
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    async def observed_foo(x):
        return x

    result = await observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == {"x": 2}


def test_observe_complex_nested_input(span_exporter: InMemorySpanExporter):
    import dataclasses
    from typing import List

    @dataclasses.dataclass
    class Address:
        street: str
        city: str
        zipcode: str

    @dataclasses.dataclass
    class Person:
        name: str
        age: int
        address: Address
        hobbies: List[str]

    @observe()
    def observed_foo(person: Person, data: dict):
        return {
            "processed_person": person.name,
            "data_count": len(data),
        }

    address = Address(street="123 Main St", city="Anytown", zipcode="12345")
    person = Person(
        name="Alice", age=30, address=address, hobbies=["reading", "coding"]
    )
    complex_data = {
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "set": {7, 8, 9},
        "nested": {"inner": [10, 11, 12]},
    }

    observed_foo(person, complex_data)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]

    # Check input serialization
    span_input = json.loads(span.attributes["lmnr.span.input"])
    assert span_input["person"]["name"] == "Alice"
    assert span_input["person"]["age"] == 30
    assert span_input["person"]["address"]["street"] == "123 Main St"
    assert span_input["person"]["address"]["city"] == "Anytown"
    assert span_input["person"]["address"]["zipcode"] == "12345"
    assert span_input["person"]["hobbies"] == ["reading", "coding"]

    # Check various data types in the input
    assert span_input["data"]["list"] == [1, 2, 3]
    assert span_input["data"]["tuple"] == [4, 5, 6]  # tuple becomes list in JSON
    assert set(span_input["data"]["set"]) == {7, 8, 9}  # set order may vary
    assert span_input["data"]["nested"]["inner"] == [10, 11, 12]

    # Check output serialization
    span_output = json.loads(span.attributes["lmnr.span.output"])
    assert span_output["processed_person"] == "Alice"
    assert span_output["data_count"] == 4


def test_observe_complex_nested_output(span_exporter: InMemorySpanExporter):
    import dataclasses
    from typing import List

    @dataclasses.dataclass
    class Result:
        success: bool
        message: str
        data: List[int]

    class ProcessedData:
        def __init__(self, items: List[str]):
            self.items = items
            self.count = len(items)

    @observe()
    def observed_foo(input_data: dict):
        # Return complex nested structure
        result = Result(
            success=True, message="Processing complete", data=[1, 2, 3, 4, 5]
        )
        processed = ProcessedData(["item1", "item2", "item3"])

        return {
            "result": result,
            "processed": processed,
            "mixed_data": {
                "tuples": [(1, 2), (3, 4), (5, 6)],
                "sets": [{"a", "b"}, {"c", "d"}],
                "nested_dict": {"level1": {"level2": [result, processed]}},
            },
            "simple_types": [1, "string", True, None, 3.14],
        }

    input_data = {"simple": "input"}
    observed_foo(input_data)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]

    # Check input serialization (simple case)
    span_input = json.loads(span.attributes["lmnr.span.input"])
    assert span_input["input_data"]["simple"] == "input"

    # Check complex output serialization
    span_output = json.loads(span.attributes["lmnr.span.output"])

    # Check dataclass serialization
    assert span_output["result"]["success"] is True
    assert span_output["result"]["message"] == "Processing complete"
    assert span_output["result"]["data"] == [1, 2, 3, 4, 5]

    # Check custom object serialization (falls back to string)
    assert isinstance(span_output["processed"], str)
    assert "ProcessedData" in span_output["processed"]

    # Check mixed data types
    assert span_output["mixed_data"]["tuples"] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]  # tuples become lists

    # Sets become lists (order may vary)
    sets_data = span_output["mixed_data"]["sets"]
    assert len(sets_data) == 2
    assert set(sets_data[0]) in [{"a", "b"}, {"c", "d"}]
    assert set(sets_data[1]) in [{"a", "b"}, {"c", "d"}]

    # Check deeply nested structure
    nested_level2 = span_output["mixed_data"]["nested_dict"]["level1"]["level2"]
    assert len(nested_level2) == 2
    # First item should be the dataclass (serialized)
    assert nested_level2[0]["success"] is True
    assert nested_level2[0]["message"] == "Processing complete"
    # Second item should be the custom object (string representation)
    assert isinstance(nested_level2[1], str)
    assert "ProcessedData" in nested_level2[1]

    # Check simple types
    assert span_output["simple_types"] == [1, "string", True, None, 3.14]


def test_observe_non_serializable_fallback(span_exporter: InMemorySpanExporter):
    class NonSerializable:
        def __init__(self, x: int):
            self.x = x

    @observe()
    def observed_foo(x: NonSerializable, y: int):
        return x

    observed_foo(NonSerializable(1), 2)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]
    span_input = json.loads(span.attributes["lmnr.span.input"])
    assert span_input["y"] == 2
    assert "NonSerializable object at 0x" in span_input["x"]
    assert "NonSerializable object at 0x" in json.loads(
        span.attributes["lmnr.span.output"]
    )


def test_observe_tags_deduplication(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", "foo"])
    def observed_foo(x, y, z, **kwargs):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]


def test_start_as_current_span_inside_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def foo():
        with Laminar.start_as_current_span("test", input="my_input"):
            Laminar.set_span_output("foo")
            pass

    foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(span for span in spans if span.name == "foo")
    inner_span = next(span for span in spans if span.name == "test")
    assert json.loads(inner_span.attributes["lmnr.span.output"]) == "foo"
    assert json.loads(inner_span.attributes["lmnr.span.input"]) == "my_input"
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id
    assert outer_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert outer_span.attributes["lmnr.span.path"] == ("foo",)
    assert inner_span.attributes["lmnr.span.path"] == ("foo", "test")


def test_observe_preserve_global_context(span_exporter: InMemorySpanExporter):
    @observe(preserve_global_context=True)
    def observed_preserve_global():
        return "foo_global"

    @observe()
    def observe_isolated():
        return "foo_isolated"

    # Start a span in the global context
    with trace.get_tracer(__name__).start_as_current_span("outer"):
        result = observed_preserve_global()
        assert result == "foo_global"

        result = observe_isolated()
        assert result == "foo_isolated"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    outer_span = [span for span in spans if span.name == "outer"][0]
    isolated_span = [span for span in spans if span.name == "observe_isolated"][0]
    preserve_span = [span for span in spans if span.name == "observed_preserve_global"][
        0
    ]

    assert (
        outer_span.get_span_context().trace_id
        == preserve_span.get_span_context().trace_id
    )
    assert (
        outer_span.get_span_context().trace_id
        != isolated_span.get_span_context().trace_id
    )

    assert preserve_span.parent.span_id == outer_span.get_span_context().span_id
    assert isolated_span.parent is None


@pytest.mark.asyncio
async def test_observe_preserve_global_context_async(
    span_exporter: InMemorySpanExporter,
):
    @observe(preserve_global_context=True)
    def observed_preserve_global():
        return "foo_global"

    @observe()
    def observe_isolated():
        return "foo_isolated"

    # Start a span in the global context
    with trace.get_tracer(__name__).start_as_current_span("outer"):
        result = observed_preserve_global()
        assert result == "foo_global"

        result = observe_isolated()
        assert result == "foo_isolated"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    outer_span = [span for span in spans if span.name == "outer"][0]
    isolated_span = [span for span in spans if span.name == "observe_isolated"][0]
    preserve_span = [span for span in spans if span.name == "observed_preserve_global"][
        0
    ]

    assert (
        outer_span.get_span_context().trace_id
        == preserve_span.get_span_context().trace_id
    )
    assert (
        outer_span.get_span_context().trace_id
        != isolated_span.get_span_context().trace_id
    )

    assert preserve_span.parent.span_id == outer_span.get_span_context().span_id
    assert isolated_span.parent is None


def test_observe_simple_generator(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        yield "foo"
        yield "bar"

    results = [r for r in observed_foo()]
    assert results == ["foo", "bar"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == ["foo", "bar"]
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_simple_generator_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        yield "foo"
        yield "bar"

    results = [r async for r in observed_foo()]
    assert results == ["foo", "bar"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == ["foo", "bar"]
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("observed_foo",)


def test_start_active_span_with_observe(span_exporter: InMemorySpanExporter):
    """Test start_active_span with observe decorator."""

    @observe()
    def observed_func():
        return "observed_output"

    span = Laminar.start_active_span("outer")
    result = observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = [s for s in spans if s.name == "outer"][0]
    observed_span = [s for s in spans if s.name == "observed_func"][0]

    # Check parent-child relationship
    assert observed_span.parent.span_id == outer_span.get_span_context().span_id
    assert (
        observed_span.get_span_context().trace_id
        == outer_span.get_span_context().trace_id
    )

    # Check span paths
    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert observed_span.attributes["lmnr.span.path"] == ("outer", "observed_func")

    # Check output
    assert json.loads(observed_span.attributes["lmnr.span.output"]) == "observed_output"


def test_start_active_span_with_nested_observe(span_exporter: InMemorySpanExporter):
    """Test start_active_span with nested observe decorators."""

    @observe()
    def inner_func():
        return "inner_output"

    @observe()
    def outer_func():
        return inner_func()

    span = Laminar.start_active_span("root")
    result = outer_func()
    span.end()

    assert result == "inner_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = [s for s in spans if s.name == "root"][0]
    outer_span = [s for s in spans if s.name == "outer_func"][0]
    inner_span = [s for s in spans if s.name == "inner_func"][0]

    # Check parent-child relationships
    assert outer_span.parent.span_id == root_span.get_span_context().span_id
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id

    # Check trace ids
    assert (
        root_span.get_span_context().trace_id
        == outer_span.get_span_context().trace_id
        == inner_span.get_span_context().trace_id
    )

    # Check span paths
    assert root_span.attributes["lmnr.span.path"] == ("root",)
    assert outer_span.attributes["lmnr.span.path"] == ("root", "outer_func")
    assert inner_span.attributes["lmnr.span.path"] == (
        "root",
        "outer_func",
        "inner_func",
    )


def test_start_active_span_multiple_observe_calls(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with multiple sequential observe calls."""

    @observe()
    def func1():
        return "output1"

    @observe()
    def func2():
        return "output2"

    span = Laminar.start_active_span("parent")
    result1 = func1()
    result2 = func2()
    span.end()

    assert result1 == "output1"
    assert result2 == "output2"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    parent_span = [s for s in spans if s.name == "parent"][0]
    func1_span = [s for s in spans if s.name == "func1"][0]
    func2_span = [s for s in spans if s.name == "func2"][0]

    # Both should be children of parent
    assert func1_span.parent.span_id == parent_span.get_span_context().span_id
    assert func2_span.parent.span_id == parent_span.get_span_context().span_id

    # All should share the same trace_id
    assert (
        parent_span.get_span_context().trace_id
        == func1_span.get_span_context().trace_id
        == func2_span.get_span_context().trace_id
    )

    # Check span paths
    assert parent_span.attributes["lmnr.span.path"] == ("parent",)
    assert func1_span.attributes["lmnr.span.path"] == ("parent", "func1")
    assert func2_span.attributes["lmnr.span.path"] == ("parent", "func2")


def test_start_active_span_with_observe_and_context_manager(
    span_exporter: InMemorySpanExporter,
):
    """Test mixing start_active_span, observe, and start_as_current_span."""

    @observe()
    def observed_func():
        with Laminar.start_as_current_span("manual_span"):
            Laminar.set_span_output("manual_output")
        return "observed_output"

    span = Laminar.start_active_span("root")
    result = observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = [s for s in spans if s.name == "root"][0]
    observed_span = [s for s in spans if s.name == "observed_func"][0]
    manual_span = [s for s in spans if s.name == "manual_span"][0]

    # Check parent-child relationships
    assert observed_span.parent.span_id == root_span.get_span_context().span_id
    assert manual_span.parent.span_id == observed_span.get_span_context().span_id

    # Check trace ids
    assert (
        root_span.get_span_context().trace_id
        == observed_span.get_span_context().trace_id
        == manual_span.get_span_context().trace_id
    )

    # Check span paths
    assert root_span.attributes["lmnr.span.path"] == ("root",)
    assert observed_span.attributes["lmnr.span.path"] == ("root", "observed_func")
    assert manual_span.attributes["lmnr.span.path"] == (
        "root",
        "observed_func",
        "manual_span",
    )


@pytest.mark.asyncio
async def test_start_active_span_with_observe_async(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with async observe decorator."""

    @observe()
    async def observed_func():
        return "observed_output"

    span = Laminar.start_active_span("outer")
    result = await observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = [s for s in spans if s.name == "outer"][0]
    observed_span = [s for s in spans if s.name == "observed_func"][0]

    # Check parent-child relationship
    assert observed_span.parent.span_id == outer_span.get_span_context().span_id
    assert (
        observed_span.get_span_context().trace_id
        == outer_span.get_span_context().trace_id
    )

    # Check span paths
    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert observed_span.attributes["lmnr.span.path"] == ("outer", "observed_func")


@pytest.mark.asyncio
async def test_start_active_span_with_nested_observe_async(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with nested async observe decorators."""

    @observe()
    async def inner_func():
        return "inner_output"

    @observe()
    async def middle_func():
        return await inner_func()

    span = Laminar.start_active_span("root")
    result = await middle_func()
    span.end()

    assert result == "inner_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = [s for s in spans if s.name == "root"][0]
    middle_span = [s for s in spans if s.name == "middle_func"][0]
    inner_span = [s for s in spans if s.name == "inner_func"][0]

    # Check parent-child relationships
    assert middle_span.parent.span_id == root_span.get_span_context().span_id
    assert inner_span.parent.span_id == middle_span.get_span_context().span_id

    # Check trace ids
    assert (
        root_span.get_span_context().trace_id
        == middle_span.get_span_context().trace_id
        == inner_span.get_span_context().trace_id
    )

    # Check span paths
    assert root_span.attributes["lmnr.span.path"] == ("root",)
    assert middle_span.attributes["lmnr.span.path"] == ("root", "middle_func")
    assert inner_span.attributes["lmnr.span.path"] == (
        "root",
        "middle_func",
        "inner_func",
    )


@pytest.mark.asyncio
async def test_start_active_span_async_multiple_observe(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with multiple sequential async observe calls."""

    @observe()
    async def func1():
        return "output1"

    @observe()
    async def func2():
        return "output2"

    span = Laminar.start_active_span("parent")
    result1 = await func1()
    result2 = await func2()
    span.end()

    assert result1 == "output1"
    assert result2 == "output2"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    parent_span = [s for s in spans if s.name == "parent"][0]
    func1_span = [s for s in spans if s.name == "func1"][0]
    func2_span = [s for s in spans if s.name == "func2"][0]

    # Both should be children of parent
    assert func1_span.parent.span_id == parent_span.get_span_context().span_id
    assert func2_span.parent.span_id == parent_span.get_span_context().span_id

    # All should share the same trace_id
    assert (
        parent_span.get_span_context().trace_id
        == func1_span.get_span_context().trace_id
        == func2_span.get_span_context().trace_id
    )

    # Check span paths
    assert parent_span.attributes["lmnr.span.path"] == ("parent",)
    assert func1_span.attributes["lmnr.span.path"] == ("parent", "func1")
    assert func2_span.attributes["lmnr.span.path"] == ("parent", "func2")


@pytest.mark.asyncio
async def test_start_active_span_deeply_nested_async(
    span_exporter: InMemorySpanExporter,
):
    """Test deeply nested async structure with start_active_span and observe."""

    @observe()
    async def nested_level3():
        with Laminar.start_as_current_span("level4"):
            pass
        return "level3_output"

    @observe()
    async def nested_level2():
        return await nested_level3()

    async def nested_level1():
        span = Laminar.start_active_span("level1")
        result = await nested_level2()
        span.end()
        return result

    span = Laminar.start_active_span("level0")
    result = await nested_level1()
    span.end()

    assert result == "level3_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5

    level0 = [s for s in spans if s.name == "level0"][0]
    level1 = [s for s in spans if s.name == "level1"][0]
    level2 = [s for s in spans if s.name == "nested_level2"][0]
    level3 = [s for s in spans if s.name == "nested_level3"][0]
    level4 = [s for s in spans if s.name == "level4"][0]

    # Check parent-child relationships
    assert level1.parent.span_id == level0.get_span_context().span_id
    assert level2.parent.span_id == level1.get_span_context().span_id
    assert level3.parent.span_id == level2.get_span_context().span_id
    assert level4.parent.span_id == level3.get_span_context().span_id

    # Check trace ids
    assert (
        level0.get_span_context().trace_id
        == level1.get_span_context().trace_id
        == level2.get_span_context().trace_id
        == level3.get_span_context().trace_id
        == level4.get_span_context().trace_id
    )

    # Check span paths
    assert level0.attributes["lmnr.span.path"] == ("level0",)
    assert level1.attributes["lmnr.span.path"] == ("level0", "level1")
    assert level2.attributes["lmnr.span.path"] == ("level0", "level1", "nested_level2")
    assert level3.attributes["lmnr.span.path"] == (
        "level0",
        "level1",
        "nested_level2",
        "nested_level3",
    )
    assert level4.attributes["lmnr.span.path"] == (
        "level0",
        "level1",
        "nested_level2",
        "nested_level3",
        "level4",
    )


def test_start_active_span_ids_path_with_observe(span_exporter: InMemorySpanExporter):
    """Test that lmnr.span.ids_path is correctly set with start_active_span and observe."""

    @observe()
    def func1():
        @observe()
        def func2():
            return "result"

        return func2()

    span = Laminar.start_active_span("root")
    result = func1()
    span.end()

    assert result == "result"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = [s for s in spans if s.name == "root"][0]
    func1_span = [s for s in spans if s.name == "func1"][0]
    func2_span = [s for s in spans if s.name == "func2"][0]

    # Check ids_path
    assert root_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=root_span.get_span_context().span_id)),
    )
    assert func1_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=root_span.get_span_context().span_id)),
        str(uuid.UUID(int=func1_span.get_span_context().span_id)),
    )
    assert func2_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=root_span.get_span_context().span_id)),
        str(uuid.UUID(int=func1_span.get_span_context().span_id)),
        str(uuid.UUID(int=func2_span.get_span_context().span_id)),
    )


def test_span_context_from_env_variables_observe(span_exporter: InMemorySpanExporter):
    test_trace_id = "01234567-89ab-cdef-0123-456789abcdef"
    test_span_id = "00000000-0000-0000-0123-456789abcdef"
    test_span_id2 = "00000000-0000-0000-fedc-ba9876543210"
    old_val = os.getenv("LMNR_SPAN_CONTEXT")
    test_context = LaminarSpanContext(
        trace_id=test_trace_id,
        span_id=test_span_id2,
        span_path=["grandparent", "parent"],
        span_ids_path=[test_span_id, test_span_id2],
    )

    os.environ["LMNR_SPAN_CONTEXT"] = str(test_context)

    Laminar._initialize_context_from_env()

    @observe()
    def test():
        pass

    test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_id = spans[0].get_span_context().span_id
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == (
        "grandparent",
        "parent",
        "test",
    )
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(test_span_id)),
        str(uuid.UUID(test_span_id2)),
        str(uuid.UUID(int=span_id)),
    )
    assert spans[0].get_span_context().trace_id == uuid.UUID(test_trace_id).int
    assert spans[0].parent.span_id == uuid.UUID(test_span_id2).int
    if old_val:
        os.environ["LMNR_SPAN_CONTEXT"] = old_val
    else:
        os.environ.pop("LMNR_SPAN_CONTEXT", None)


def test_add_span_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo"])
    def test():
        Laminar.add_span_tags(["bar", "baz", "foo"])
        Laminar.add_span_tags(["qux", "bar"])

    test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "baz",
        "foo",
        "qux",
    ]


def test_set_span_tags_add_span_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo"])
    def test():
        Laminar.set_span_tags(["bar", "baz"])
        Laminar.add_span_tags(["qux", "bar"])

    test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "baz",
        "qux",
    ]


def test_observe_disable_tracing_simple(span_exporter: InMemorySpanExporter):
    """Simple test: @observe decorated functions should not create spans when LMNR_DISABLE_TRACING=true."""
    old_val = os.getenv("LMNR_DISABLE_TRACING")

    try:
        # Set env var to disable tracing
        os.environ["LMNR_DISABLE_TRACING"] = "true"

        @observe()
        def disabled_func(x, y):
            return x + y

        result = disabled_func(1, 2)

        # Function should still work normally
        assert result == 3

        # But no spans should be exported
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0
    finally:
        # Restore original value
        if old_val:
            os.environ["LMNR_DISABLE_TRACING"] = old_val
        else:
            os.environ.pop("LMNR_DISABLE_TRACING", None)


def test_observe_disable_tracing_nested_toggle(span_exporter: InMemorySpanExporter):
    """Corner case: nested @observe functions with dynamic tracing toggle.

    Tests that:
    1. Nested observed functions work when tracing is enabled
    2. Nested observed functions don't create spans when disabled
    3. Toggling mid-execution affects only new spans
    4. Function execution is unaffected by tracing state
    """
    old_val = os.getenv("LMNR_DISABLE_TRACING")

    try:

        @observe()
        def inner_func():
            return "inner_result"

        @observe()
        def outer_func():
            return inner_func()

        # Test 1: Enabled tracing - should create nested spans
        os.environ.pop("LMNR_DISABLE_TRACING", None)
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        outer_span = [s for s in spans if s.name == "outer_func"][0]
        inner_span = [s for s in spans if s.name == "inner_func"][0]
        assert inner_span.parent.span_id == outer_span.get_span_context().span_id
        span_exporter.clear()

        # Test 2: Disabled tracing - no spans created but functions still work
        os.environ["LMNR_DISABLE_TRACING"] = "true"
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0
        span_exporter.clear()

        # Test 3: Re-enable tracing - spans created again with correct structure
        os.environ.pop("LMNR_DISABLE_TRACING", None)
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        outer_span = [s for s in spans if s.name == "outer_func"][0]
        inner_span = [s for s in spans if s.name == "inner_func"][0]
        # Verify proper nesting after re-enabling
        assert inner_span.parent.span_id == outer_span.get_span_context().span_id
        assert inner_span.attributes["lmnr.span.path"] == ("outer_func", "inner_func")

    finally:
        # Restore original value
        if old_val:
            os.environ["LMNR_DISABLE_TRACING"] = old_val
        else:
            os.environ.pop("LMNR_DISABLE_TRACING", None)
