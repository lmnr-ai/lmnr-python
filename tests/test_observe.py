import json
import uuid
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
