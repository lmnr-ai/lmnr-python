import json
import os
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from lmnr import Laminar, LaminarSpanContext, observe


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def _parent(span: ReadableSpan) -> SpanContext:
    assert span.parent is not None
    return span.parent


def test_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo(_x: str, _y: str, _z:str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()

    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {
        "_x": "arg",
        "_y": "arg2",
        "_z": "arg3",
        "a": 1,
        "b": 2,
        "c": 3,
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == "foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_name(span_exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    def observed_foo(_x: str, _y: str, _z:str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("custom_name",)


def test_observe_session_id(span_exporter: InMemorySpanExporter):
    @observe(session_id="123")
    def observed_foo(_x: str, _y: str, _z:str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.session_id"] == "123"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_user_id(span_exporter: InMemorySpanExporter):
    @observe(user_id="123")
    def observed_foo(_x: str, _y: str, _z:str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.user_id"] == "123"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_metadata(span_exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    def observed_foo(_x: str, _y: str, _z:str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.metadata.key"] == "value"
    assert json.loads(
        cast(str, (spans[0].attributes or {})["lmnr.association.properties.metadata.nested"]
    )) == {"key2": "value2"}
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo() -> None:
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert (events[0].attributes or {})["exception.type"] == "ValueError"
    assert (events[0].attributes or {})["exception.message"] == "test"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_exception_with_session_id_and_name(
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="123", name="custom_name")
    def observed_foo() -> None:
        raise ValueError("test")

    with pytest.raises(ValueError):
        observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "custom_name"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("custom_name",)

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert (events[0].attributes or {})["exception.type"] == "ValueError"
    assert (events[0].attributes or {})["exception.message"] == "test"


# At the time of writing this test, an erroring observed function would break the context for
# the following sibling spans
def test_observe_exception_preserves_context(span_exporter: InMemorySpanExporter):
    @observe()
    def err() -> None:
        raise ValueError("test")

    @observe()
    def success():
        pass

    @observe()
    def parent():
        try:
            err()
        except Exception:
            print("error")
        success()

    parent()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    err_span = next(span for span in spans if span.name == "err")
    success_span = next(span for span in spans if span.name == "success")
    parent_span = next(span for span in spans if span.name == "parent")
    assert _ctx(err_span).trace_id == _ctx(parent_span).trace_id
    assert _ctx(success_span).trace_id == _ctx(parent_span).trace_id
    assert _parent(err_span).span_id == _ctx(parent_span).span_id
    assert _parent(success_span).span_id == _ctx(parent_span).span_id
    assert (err_span.attributes or {}).get("lmnr.span.path") == ("parent", "err")
    assert (success_span.attributes or {}).get("lmnr.span.path") == ("parent", "success")


# Async counterpart of test_observe_exception_preserves_context above.
@pytest.mark.asyncio
async def test_observe_async_exception_preserves_context(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def err() -> None:
        raise ValueError("test")

    @observe()
    async def success():
        pass

    @observe()
    async def parent():
        try:
            await err()
        except Exception:
            print("error")
        await success()

    await parent()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    err_span = next(span for span in spans if span.name == "err")
    success_span = next(span for span in spans if span.name == "success")
    parent_span = next(span for span in spans if span.name == "parent")
    assert _ctx(err_span).trace_id == _ctx(parent_span).trace_id
    assert _ctx(success_span).trace_id == _ctx(parent_span).trace_id
    assert _parent(err_span).span_id == _ctx(parent_span).span_id
    assert _parent(success_span).span_id == _ctx(parent_span).span_id
    assert (err_span.attributes or {}).get("lmnr.span.path") == ("parent", "err")
    assert (success_span.attributes or {}).get("lmnr.span.path") == ("parent", "success")


@pytest.mark.asyncio
async def test_observe_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo() -> str:
        return "foo"

    res = await observed_foo()
    spans = span_exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == "foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


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
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert (events[0].attributes or {})["exception.type"] == "ValueError"
    assert (events[0].attributes or {})["exception.message"] == "test"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_nested(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_bar() -> str:
        return "bar"

    @observe(session_id="123")
    def observed_foo() -> str:
        return observed_bar()

    result = observed_foo()
    spans = span_exporter.get_finished_spans()

    assert result == "bar"
    assert len(spans) == 2

    foo_span = next(span for span in spans if span.name == "observed_foo")
    bar_span = next(span for span in spans if span.name == "observed_bar")
    assert _parent(bar_span).span_id == _ctx(foo_span).span_id

    assert (foo_span.attributes or {})["lmnr.association.properties.session_id"] == "123"

    assert (foo_span.attributes or {})["lmnr.span.input"] == json.dumps({})
    assert (foo_span.attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert (bar_span.attributes or {})["lmnr.span.input"] == json.dumps({})
    assert (bar_span.attributes or {})["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert (foo_span.attributes or {})["lmnr.span.output"] == json.dumps("bar")
    assert (bar_span.attributes or {})["lmnr.span.output"] == json.dumps("bar")

    assert (foo_span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (bar_span.attributes or {})["lmnr.span.instrumentation_source"] == "python"


def test_observe_deeply_nested_and_sequential(span_exporter: InMemorySpanExporter):
    @observe()
    def level_4() -> str:
        return "level_4"

    @observe()
    def level_3() -> str:
        return level_4()

    @observe()
    def level_2() -> str:
        return level_3()

    @observe()
    def level_1() -> str:
        return level_2()

    @observe()
    def after_all() -> str:
        return "after_all"

    result = level_1()
    _ = after_all()
    spans = span_exporter.get_finished_spans()
    assert result == "level_4"
    assert len(spans) == 5

    level_1_span = next(span for span in spans if span.name == "level_1")
    level_2_span = next(span for span in spans if span.name == "level_2")
    level_3_span = next(span for span in spans if span.name == "level_3")
    level_4_span = next(span for span in spans if span.name == "level_4")
    after_all_span = next(span for span in spans if span.name == "after_all")

    assert level_1_span.parent is None or level_1_span.parent.span_id == 0
    assert _parent(level_2_span).span_id == _ctx(level_1_span).span_id
    assert _parent(level_3_span).span_id == _ctx(level_2_span).span_id
    assert _parent(level_4_span).span_id == _ctx(level_3_span).span_id
    assert after_all_span.parent is None or after_all_span.parent.span_id == 0

    assert (level_1_span.attributes or {})["lmnr.span.path"] == ("level_1",)
    assert (level_2_span.attributes or {})["lmnr.span.path"] == ("level_1", "level_2")
    assert (level_3_span.attributes or {})["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
    )
    assert (level_4_span.attributes or {})["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
        "level_4",
    )
    assert (after_all_span.attributes or {})["lmnr.span.path"] == ("after_all",)

    assert (level_1_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
    )
    assert (level_2_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
    )
    assert (level_3_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
        str(uuid.UUID(int=_ctx(level_3_span).span_id)),
    )
    assert (level_4_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
        str(uuid.UUID(int=_ctx(level_3_span).span_id)),
        str(uuid.UUID(int=_ctx(level_4_span).span_id)),
    )
    assert (after_all_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(after_all_span).span_id)),
    )

    assert (
        _ctx(level_1_span).trace_id
        == _ctx(level_2_span).trace_id
        == _ctx(level_3_span).trace_id
        == _ctx(level_4_span).trace_id
    )
    assert (
        _ctx(after_all_span).trace_id
        != _ctx(level_4_span).trace_id
    )


def test_observe_skip_input_keys(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["_a"])
    def observed_foo(_a: int, _b: int, _c: int) -> str:
        return "foo"

    result = observed_foo(1, 2, 3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {"_b": 2, "_c": 3}


def test_observe_skip_input_keys_with_kwargs(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["_a", "d"])
    def observed_foo(_a: int, _b: int, _c: int, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {
        "_b": 2,
        "_c": 3,
        "e": 5,
        "f": 6,
    }


@pytest.mark.asyncio
async def test_observe_skip_input_keys_async(span_exporter: InMemorySpanExporter):
    @observe(ignore_inputs=["_a"])
    async def observed_foo(_a: int, _b: int, _c: int) -> str:
        return "foo"

    res = await observed_foo(1, 2, 3)
    spans = span_exporter.get_finished_spans()
    assert res == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {"_b": 2, "_c": 3}


def test_observe_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int):
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert sorted(cast(list[str], (span.attributes or {})["lmnr.association.properties.tags"])) == [
        "bar",
        "foo",
    ]
    assert (span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (span.attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_tags_invalid_type(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])  # pyright: ignore[reportArgumentType] intentional
    def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert (span.attributes or {}).get("lmnr.association.properties.tags") is None
    assert (span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (span.attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_observe_sequential_spans(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo() -> str:
        return "foo"

    @observe()
    def observed_bar() -> str:
        return "bar"

    _foo = observed_foo()
    _bar = observed_bar()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = next(span for span in spans if span.name == "observed_foo")
    bar_span = next(span for span in spans if span.name == "observed_bar")

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert _ctx(foo_span).trace_id != _ctx(bar_span).trace_id


@pytest.mark.asyncio
async def test_observe_sequential_spans_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo() -> str:
        return "foo"

    @observe()
    async def observed_bar() -> str:
        return "bar"

    _foo = await observed_foo()
    _bar = await observed_bar()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    foo_span = next(span for span in spans if span.name == "observed_foo")
    bar_span = next(span for span in spans if span.name == "observed_bar")

    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert bar_span.parent is None or bar_span.parent.span_id == 0

    assert _ctx(foo_span).trace_id != _ctx(bar_span).trace_id


@pytest.mark.asyncio
async def test_observe_name_async(span_exporter: InMemorySpanExporter):
    @observe(name="custom_name")
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"
    assert result == "foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("custom_name",)


@pytest.mark.asyncio
async def test_observe_session_id_async(span_exporter: InMemorySpanExporter):
    @observe(session_id="123")
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.session_id"] == "123"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_user_id_async(span_exporter: InMemorySpanExporter):
    @observe(user_id="123")
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.user_id"] == "123"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_metadata_async(span_exporter: InMemorySpanExporter):
    @observe(metadata={"key": "value", "nested": {"key2": "value2"}})
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.metadata.key"] == "value"
    assert json.loads(
        cast(str, (spans[0].attributes or {})["lmnr.association.properties.metadata.nested"]
    )) == {"key2": "value2"}
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


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
    assert (spans[0].attributes or {})["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "custom_name"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("custom_name",)

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert (events[0].attributes or {})["exception.type"] == "ValueError"
    assert (events[0].attributes or {})["exception.message"] == "test"


@pytest.mark.asyncio
async def test_observe_nested_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_bar() -> str:
        return "bar"

    @observe(session_id="123")
    async def observed_foo() -> str:
        return await observed_bar()

    result = await observed_foo()
    spans = span_exporter.get_finished_spans()

    assert result == "bar"
    assert len(spans) == 2

    foo_span = next(span for span in spans if span.name == "observed_foo")
    bar_span = next(span for span in spans if span.name == "observed_bar")
    assert foo_span.parent is None or foo_span.parent.span_id == 0
    assert _parent(bar_span).span_id == _ctx(foo_span).span_id
    assert _ctx(foo_span).trace_id == _ctx(bar_span).trace_id

    assert (foo_span.attributes or {})["lmnr.association.properties.session_id"] == "123"

    assert (foo_span.attributes or {})["lmnr.span.input"] == json.dumps({})
    assert (foo_span.attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert (bar_span.attributes or {})["lmnr.span.input"] == json.dumps({})
    assert (bar_span.attributes or {})["lmnr.span.path"] == ("observed_foo", "observed_bar")

    assert (foo_span.attributes or {})["lmnr.span.output"] == json.dumps("bar")
    assert (bar_span.attributes or {})["lmnr.span.output"] == json.dumps("bar")

    assert (foo_span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (bar_span.attributes or {})["lmnr.span.instrumentation_source"] == "python"


@pytest.mark.asyncio
async def test_observe_deeply_nested_and_sequential_async(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def level_4() -> str:
        return "level_4"

    @observe()
    async def level_3() -> str:
        return await level_4()

    @observe()
    async def level_2() -> str:
        return await level_3()

    @observe()
    async def level_1() -> str:
        return await level_2()

    @observe()
    async def after_all() -> str:
        return "after_all"

    result = await level_1()
    _ = await after_all()
    spans = span_exporter.get_finished_spans()
    assert result == "level_4"
    assert len(spans) == 5

    level_1_span = next(span for span in spans if span.name == "level_1")
    level_2_span = next(span for span in spans if span.name == "level_2")
    level_3_span = next(span for span in spans if span.name == "level_3")
    level_4_span = next(span for span in spans if span.name == "level_4")
    after_all_span = next(span for span in spans if span.name == "after_all")

    assert level_1_span.parent is None or level_1_span.parent.span_id == 0
    assert _parent(level_2_span).span_id == _ctx(level_1_span).span_id
    assert _parent(level_3_span).span_id == _ctx(level_2_span).span_id
    assert _parent(level_4_span).span_id == _ctx(level_3_span).span_id
    assert after_all_span.parent is None or after_all_span.parent.span_id == 0

    assert (level_1_span.attributes or {})["lmnr.span.path"] == ("level_1",)
    assert (level_2_span.attributes or {})["lmnr.span.path"] == ("level_1", "level_2")
    assert (level_3_span.attributes or {})["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
    )
    assert (level_4_span.attributes or {})["lmnr.span.path"] == (
        "level_1",
        "level_2",
        "level_3",
        "level_4",
    )
    assert (after_all_span.attributes or {})["lmnr.span.path"] == ("after_all",)

    assert (level_1_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
    )
    assert (level_2_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
    )
    assert (level_3_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
        str(uuid.UUID(int=_ctx(level_3_span).span_id)),
    )
    assert (level_4_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(level_1_span).span_id)),
        str(uuid.UUID(int=_ctx(level_2_span).span_id)),
        str(uuid.UUID(int=_ctx(level_3_span).span_id)),
        str(uuid.UUID(int=_ctx(level_4_span).span_id)),
    )
    assert (after_all_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(after_all_span).span_id)),
    )

    assert (
        _ctx(level_1_span).trace_id
        == _ctx(level_2_span).trace_id
        == _ctx(level_3_span).trace_id
        == _ctx(level_4_span).trace_id
    )
    assert (
        _ctx(after_all_span).trace_id
        != _ctx(level_4_span).trace_id
    )


@pytest.mark.asyncio
async def test_observe_skip_input_keys_with_kwargs_async(
    span_exporter: InMemorySpanExporter,
):
    @observe(ignore_inputs=["_a", "d"])
    async def observed_foo(_a: int, _b: int, _c: int, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo(1, 2, 3, d=4, e=5, f=6)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {
        "_b": 2,
        "_c": 3,
        "e": 5,
        "f": 6,
    }


def test_observe_input_formatter(span_exporter: InMemorySpanExporter):
    def input_formatter(x: int) -> dict[str, int]:
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    def observed_foo(x: int) -> int:
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {"x": 2}


def test_observe_input_formatter_exception(span_exporter: InMemorySpanExporter):
    def input_formatter(x: int):
        raise ValueError("test")

    @observe(input_formatter=input_formatter)
    def observed_foo(x: int) -> int:
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)
    assert "lmnr.span.input" not in (spans[0].attributes or {})


def test_observe_input_formatter_with_kwargs(span_exporter: InMemorySpanExporter):
    def input_formatter(x: int, **kwargs: dict[str, int]) -> dict[str, str | int]:
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    def observed_foo(x: int, **kwargs: int):
        return x

    result = observed_foo(1, a=1, b=2)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {
        "x": 2,
        "custom-A": "1--",
    }


@pytest.mark.asyncio
async def test_observe_tags_async(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar"])
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert sorted(cast(list[str], (span.attributes or {})["lmnr.association.properties.tags"])) == [
        "bar",
        "foo",
    ]
    assert (span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (span.attributes or {})["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_tags_invalid_type_async(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", 1])  # pyright: ignore[reportArgumentType] intentional
    async def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = await observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    span = spans[0]

    assert (span.attributes or {}).get("lmnr.association.properties.tags") is None
    assert (span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (span.attributes or {})["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_input_formatter_async(span_exporter: InMemorySpanExporter):
    def input_formatter(x: int) -> dict[str, int]:
        return {"x": x + 1}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x: int) -> int:
        return x

    result = await observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {"x": 2}


@pytest.mark.asyncio
async def test_observe_input_formatter_with_kwargs_async(
    span_exporter: InMemorySpanExporter,
):
    def input_formatter(x: int, **kwargs: dict[str, int]) -> dict[str, str | int]:
        return {"x": x + 1, "custom-A": f"{kwargs.get('a')}--"}

    @observe(input_formatter=input_formatter)
    async def observed_foo(x: int, **kwargs: int) -> int:
        return x

    result = await observed_foo(1, a=1, b=2)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.input"])) == {
        "x": 2,
        "custom-A": "1--",
    }


def test_observe_output_formatter(span_exporter: InMemorySpanExporter):
    def output_formatter(x: int) -> dict[str, int]:
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    def observed_foo(x: int) -> int:
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == {"x": 2}


def test_observe_output_formatter_exception(span_exporter: InMemorySpanExporter):
    def output_formatter(x: int):
        raise ValueError("test")

    @observe(output_formatter=output_formatter)
    def observed_foo(x: int) -> int:
        return x

    result = observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert "lmnr.span.output" not in (spans[0].attributes or {})


@pytest.mark.asyncio
async def test_observe_output_formatter_async(span_exporter: InMemorySpanExporter):
    def output_formatter(x: int) -> dict[str, int]:
        return {"x": x + 1}

    @observe(output_formatter=output_formatter)
    async def observed_foo(x: int) -> int:
        return x

    result = await observed_foo(1)
    spans = span_exporter.get_finished_spans()
    assert result == 1
    assert len(spans) == 1
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == {"x": 2}


def test_observe_complex_nested_input(span_exporter: InMemorySpanExporter):
    import dataclasses

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
        hobbies: list[str]

    @observe()
    def observed_foo(person: Person, data: dict[str, Any]) -> dict[str, str | int]:
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

    _result = observed_foo(person, complex_data)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]

    # Check input serialization
    span_input = json.loads(cast(str, (span.attributes or {})["lmnr.span.input"]))
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
    span_output = json.loads(cast(str, (span.attributes or {})["lmnr.span.output"]))
    assert span_output["processed_person"] == "Alice"
    assert span_output["data_count"] == 4


def test_observe_complex_nested_output(span_exporter: InMemorySpanExporter):
    import dataclasses

    @dataclasses.dataclass
    class Result:
        success: bool
        message: str
        data: list[int]

    class ProcessedData:
        def __init__(self, items: list[str]):
            self.items: list[str] = items
            self.count: int = len(items)

    @observe()
    def observed_foo(_input_data: dict[str, Any]) -> dict[str, Any]:
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
    _result = observed_foo(input_data)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]

    # Check input serialization (simple case)
    span_input = json.loads(cast(str, (span.attributes or {})["lmnr.span.input"]))
    assert span_input["_input_data"]["simple"] == "input"

    # Check complex output serialization
    span_output = json.loads(cast(str, (span.attributes or {})["lmnr.span.output"]))

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
            self.x: int = x

    @observe()
    def observed_foo(x: NonSerializable, y: int):
        return x

    _result = observed_foo(NonSerializable(1), 2)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    span = spans[0]
    span_input = json.loads(cast(str, (span.attributes or {})["lmnr.span.input"]))
    assert span_input["y"] == 2
    assert "NonSerializable object at 0x" in span_input["x"]
    assert "NonSerializable object at 0x" in json.loads(
        cast(str, (span.attributes or {})["lmnr.span.output"]
    ))


def test_observe_tags_deduplication(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo", "bar", "foo"])
    def observed_foo(_x: str, _y: str, _z: str, **_kwargs: int) -> str:
        return "foo"

    result = observed_foo("arg", "arg2", "arg3", a=1, b=2, c=3)
    spans = span_exporter.get_finished_spans()
    assert result == "foo"
    assert len(spans) == 1
    assert sorted(cast(list[str], (spans[0].attributes or {})["lmnr.association.properties.tags"])) == [
        "bar",
        "foo",
    ]


def test_start_as_current_span_inside_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def foo():
        with Laminar.start_as_current_span("test", input="my_input"):
            Laminar.set_span_output("foo")

    foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(span for span in spans if span.name == "foo")
    inner_span = next(span for span in spans if span.name == "test")
    assert json.loads(cast(str, (inner_span.attributes or {})["lmnr.span.output"])) == "foo"
    assert json.loads(cast(str, (inner_span.attributes or {})["lmnr.span.input"])) == "my_input"
    assert (
        _ctx(inner_span).trace_id == _ctx(outer_span).trace_id
    )
    assert _parent(inner_span).span_id == _ctx(outer_span).span_id
    assert (outer_span.attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (outer_span.attributes or {})["lmnr.span.path"] == ("foo",)
    assert (inner_span.attributes or {})["lmnr.span.path"] == ("foo", "test")


def test_observe_preserve_global_context(span_exporter: InMemorySpanExporter):
    @observe(preserve_global_context=True)
    def observed_preserve_global() -> str:
        return "foo_global"

    @observe()
    def observe_isolated() -> str:
        return "foo_isolated"

    # Start a span in the global context
    with trace.get_tracer(__name__).start_as_current_span("outer"):
        result = observed_preserve_global()
        assert result == "foo_global"

        result = observe_isolated()
        assert result == "foo_isolated"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    outer_span = next(span for span in spans if span.name == "outer")
    isolated_span = next(span for span in spans if span.name == "observe_isolated")
    preserve_span = next(span for span in spans if span.name == "observed_preserve_global")

    assert (
        _ctx(outer_span).trace_id
        == _ctx(preserve_span).trace_id
    )
    assert (
        _ctx(outer_span).trace_id
        != _ctx(isolated_span).trace_id
    )

    assert _parent(preserve_span).span_id == _ctx(outer_span).span_id
    assert isolated_span.parent is None


@pytest.mark.asyncio
async def test_observe_preserve_global_context_async(
    span_exporter: InMemorySpanExporter,
):
    @observe(preserve_global_context=True)
    def observed_preserve_global() -> str:
        return "foo_global"

    @observe()
    def observe_isolated() -> str:
        return "foo_isolated"

    # Start a span in the global context
    with trace.get_tracer(__name__).start_as_current_span("outer"):
        result = observed_preserve_global()
        assert result == "foo_global"

        result = observe_isolated()
        assert result == "foo_isolated"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    outer_span = next(span for span in spans if span.name == "outer")
    isolated_span = next(span for span in spans if span.name == "observe_isolated")
    preserve_span = next(span for span in spans if span.name == "observed_preserve_global")

    assert (
        _ctx(outer_span).trace_id
        == _ctx(preserve_span).trace_id
    )
    assert (
        _ctx(outer_span).trace_id
        != _ctx(isolated_span).trace_id
    )

    assert _parent(preserve_span).span_id == _ctx(outer_span).span_id
    assert isolated_span.parent is None


def test_observe_simple_generator(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo() -> Generator[str]:
        yield "foo"
        yield "bar"

    results = [r for r in observed_foo()]
    assert results == ["foo", "bar"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == ["foo", "bar"]
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


@pytest.mark.asyncio
async def test_observe_simple_generator_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo() -> AsyncGenerator[str]:
        yield "foo"
        yield "bar"

    results = [r async for r in observed_foo()]
    assert results == ["foo", "bar"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert json.loads(cast(str, (spans[0].attributes or {})["lmnr.span.output"])) == ["foo", "bar"]
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("observed_foo",)


def test_start_active_span_with_observe(span_exporter: InMemorySpanExporter):
    """Test start_active_span with observe decorator."""

    @observe()
    def observed_func() -> str:
        return "observed_output"

    span = Laminar.start_active_span("outer")
    result = observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = next(s for s in spans if s.name == "outer")
    observed_span = next(s for s in spans if s.name == "observed_func")

    # Check parent-child relationship
    assert _parent(observed_span).span_id == _ctx(outer_span).span_id
    assert (
        _ctx(observed_span).trace_id
        == _ctx(outer_span).trace_id
    )

    # Check span paths
    assert (outer_span.attributes or {})["lmnr.span.path"] == ("outer",)
    assert (observed_span.attributes or {})["lmnr.span.path"] == ("outer", "observed_func")

    # Check output
    assert json.loads(cast(str, (observed_span.attributes or {})["lmnr.span.output"])) == "observed_output"


def test_start_active_span_with_nested_observe(span_exporter: InMemorySpanExporter):
    """Test start_active_span with nested observe decorators."""

    @observe()
    def inner_func() -> str:
        return "inner_output"

    @observe()
    def outer_func() -> str:
        return inner_func()

    span = Laminar.start_active_span("root")
    result = outer_func()
    span.end()

    assert result == "inner_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = next(s for s in spans if s.name == "root")
    outer_span = next(s for s in spans if s.name == "outer_func")
    inner_span = next(s for s in spans if s.name == "inner_func")

    # Check parent-child relationships
    assert _parent(outer_span).span_id == _ctx(root_span).span_id
    assert _parent(inner_span).span_id == _ctx(outer_span).span_id

    # Check trace ids
    assert (
        _ctx(root_span).trace_id
        == _ctx(outer_span).trace_id
        == _ctx(inner_span).trace_id
    )

    # Check span paths
    assert (root_span.attributes or {})["lmnr.span.path"] == ("root",)
    assert (outer_span.attributes or {})["lmnr.span.path"] == ("root", "outer_func")
    assert (inner_span.attributes or {})["lmnr.span.path"] == (
        "root",
        "outer_func",
        "inner_func",
    )


def test_start_active_span_multiple_observe_calls(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with multiple sequential observe calls."""

    @observe()
    def func1() -> str:
        return "output1"

    @observe()
    def func2() -> str:
        return "output2"

    span = Laminar.start_active_span("parent")
    result1 = func1()
    result2 = func2()
    span.end()

    assert result1 == "output1"
    assert result2 == "output2"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    parent_span = next(s for s in spans if s.name == "parent")
    func1_span = next(s for s in spans if s.name == "func1")
    func2_span = next(s for s in spans if s.name == "func2")

    # Both should be children of parent
    assert _parent(func1_span).span_id == _ctx(parent_span).span_id
    assert _parent(func2_span).span_id == _ctx(parent_span).span_id

    # All should share the same trace_id
    assert (
        _ctx(parent_span).trace_id
        == _ctx(func1_span).trace_id
        == _ctx(func2_span).trace_id
    )

    # Check span paths
    assert (parent_span.attributes or {})["lmnr.span.path"] == ("parent",)
    assert (func1_span.attributes or {})["lmnr.span.path"] == ("parent", "func1")
    assert (func2_span.attributes or {})["lmnr.span.path"] == ("parent", "func2")


def test_start_active_span_with_observe_and_context_manager(
    span_exporter: InMemorySpanExporter,
):
    """Test mixing start_active_span, observe, and start_as_current_span."""

    @observe()
    def observed_func() -> str:
        with Laminar.start_as_current_span("manual_span"):
            Laminar.set_span_output("manual_output")
        return "observed_output"

    span = Laminar.start_active_span("root")
    result = observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = next(s for s in spans if s.name == "root")
    observed_span = next(s for s in spans if s.name == "observed_func")
    manual_span = next(s for s in spans if s.name == "manual_span")

    # Check parent-child relationships
    assert _parent(observed_span).span_id == _ctx(root_span).span_id
    assert _parent(manual_span).span_id == _ctx(observed_span).span_id

    # Check trace ids
    assert (
        _ctx(root_span).trace_id
        == _ctx(observed_span).trace_id
        == _ctx(manual_span).trace_id
    )

    # Check span paths
    assert (root_span.attributes or {})["lmnr.span.path"] == ("root",)
    assert (observed_span.attributes or {})["lmnr.span.path"] == ("root", "observed_func")
    assert (manual_span.attributes or {})["lmnr.span.path"] == (
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
    async def observed_func() -> str:
        return "observed_output"

    span = Laminar.start_active_span("outer")
    result = await observed_func()
    span.end()

    assert result == "observed_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = next(s for s in spans if s.name == "outer")
    observed_span = next(s for s in spans if s.name == "observed_func")

    # Check parent-child relationship
    assert _parent(observed_span).span_id == _ctx(outer_span).span_id
    assert (
        _ctx(observed_span).trace_id
        == _ctx(outer_span).trace_id
    )

    # Check span paths
    assert (outer_span.attributes or {})["lmnr.span.path"] == ("outer",)
    assert (observed_span.attributes or {})["lmnr.span.path"] == ("outer", "observed_func")


@pytest.mark.asyncio
async def test_start_active_span_with_nested_observe_async(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with nested async observe decorators."""

    @observe()
    async def inner_func() -> str:
        return "inner_output"

    @observe()
    async def middle_func() -> str:
        return await inner_func()

    span = Laminar.start_active_span("root")
    result = await middle_func()
    span.end()

    assert result == "inner_output"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = next(s for s in spans if s.name == "root")
    middle_span = next(s for s in spans if s.name == "middle_func")
    inner_span = next(s for s in spans if s.name == "inner_func")

    # Check parent-child relationships
    assert _parent(middle_span).span_id == _ctx(root_span).span_id
    assert _parent(inner_span).span_id == _ctx(middle_span).span_id

    # Check trace ids
    assert (
        _ctx(root_span).trace_id
        == _ctx(middle_span).trace_id
        == _ctx(inner_span).trace_id
    )

    # Check span paths
    assert (root_span.attributes or {})["lmnr.span.path"] == ("root",)
    assert (middle_span.attributes or {})["lmnr.span.path"] == ("root", "middle_func")
    assert (inner_span.attributes or {})["lmnr.span.path"] == (
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
    async def func1() -> str:
        return "output1"

    @observe()
    async def func2() -> str:
        return "output2"

    span = Laminar.start_active_span("parent")
    result1 = await func1()
    result2 = await func2()
    span.end()

    assert result1 == "output1"
    assert result2 == "output2"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    parent_span = next(s for s in spans if s.name == "parent")
    func1_span = next(s for s in spans if s.name == "func1")
    func2_span = next(s for s in spans if s.name == "func2")

    # Both should be children of parent
    assert _parent(func1_span).span_id == _ctx(parent_span).span_id
    assert _parent(func2_span).span_id == _ctx(parent_span).span_id

    # All should share the same trace_id
    assert (
        _ctx(parent_span).trace_id
        == _ctx(func1_span).trace_id
        == _ctx(func2_span).trace_id
    )

    # Check span paths
    assert (parent_span.attributes or {})["lmnr.span.path"] == ("parent",)
    assert (func1_span.attributes or {})["lmnr.span.path"] == ("parent", "func1")
    assert (func2_span.attributes or {})["lmnr.span.path"] == ("parent", "func2")


@pytest.mark.asyncio
async def test_start_active_span_deeply_nested_async(
    span_exporter: InMemorySpanExporter,
):
    """Test deeply nested async structure with start_active_span and observe."""

    @observe()
    async def nested_level3() -> str:
        with Laminar.start_as_current_span("level4"):
            pass
        return "level3_output"

    @observe()
    async def nested_level2() -> str:
        return await nested_level3()

    async def nested_level1() -> str:
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

    level0 = next(s for s in spans if s.name == "level0")
    level1 = next(s for s in spans if s.name == "level1")
    level2 = next(s for s in spans if s.name == "nested_level2")
    level3 = next(s for s in spans if s.name == "nested_level3")
    level4 = next(s for s in spans if s.name == "level4")

    # Check parent-child relationships
    assert _parent(level1).span_id == _ctx(level0).span_id
    assert _parent(level2).span_id == _ctx(level1).span_id
    assert _parent(level3).span_id == _ctx(level2).span_id
    assert _parent(level4).span_id == _ctx(level3).span_id

    # Check trace ids
    assert (
        _ctx(level0).trace_id
        == _ctx(level1).trace_id
        == _ctx(level2).trace_id
        == _ctx(level3).trace_id
        == _ctx(level4).trace_id
    )

    # Check span paths
    assert (level0.attributes or {})["lmnr.span.path"] == ("level0",)
    assert (level1.attributes or {})["lmnr.span.path"] == ("level0", "level1")
    assert (level2.attributes or {})["lmnr.span.path"] == ("level0", "level1", "nested_level2")
    assert (level3.attributes or {})["lmnr.span.path"] == (
        "level0",
        "level1",
        "nested_level2",
        "nested_level3",
    )
    assert (level4.attributes or {})["lmnr.span.path"] == (
        "level0",
        "level1",
        "nested_level2",
        "nested_level3",
        "level4",
    )


def test_start_active_span_ids_path_with_observe(span_exporter: InMemorySpanExporter):
    """Test that lmnr.span.ids_path is correctly set with start_active_span and observe."""

    @observe()
    def func1() -> str:
        @observe()
        def func2() -> str:
            return "result"

        return func2()

    span = Laminar.start_active_span("root")
    result = func1()
    span.end()

    assert result == "result"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    root_span = next(s for s in spans if s.name == "root")
    func1_span = next(s for s in spans if s.name == "func1")
    func2_span = next(s for s in spans if s.name == "func2")

    # Check ids_path
    assert (root_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(root_span).span_id)),
    )
    assert (func1_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(root_span).span_id)),
        str(uuid.UUID(int=_ctx(func1_span).span_id)),
    )
    assert (func2_span.attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=_ctx(root_span).span_id)),
        str(uuid.UUID(int=_ctx(func1_span).span_id)),
        str(uuid.UUID(int=_ctx(func2_span).span_id)),
    )


def test_span_context_from_env_variables_observe(span_exporter: InMemorySpanExporter):
    test_trace_id = "01234567-89ab-cdef-0123-456789abcdef"
    test_span_id = "00000000-0000-0000-0123-456789abcdef"
    test_span_id2 = "00000000-0000-0000-fedc-ba9876543210"
    old_val = os.getenv("LMNR_SPAN_CONTEXT")
    test_context = LaminarSpanContext(
        trace_id=uuid.UUID(test_trace_id),
        span_id=uuid.UUID(test_span_id2),
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
    span_id = _ctx(spans[0]).span_id
    assert spans[0].name == "test"
    assert (spans[0].attributes or {})["lmnr.span.instrumentation_source"] == "python"
    assert (spans[0].attributes or {})["lmnr.span.path"] == (
        "grandparent",
        "parent",
        "test",
    )
    assert (spans[0].attributes or {})["lmnr.span.ids_path"] == (
        str(uuid.UUID(test_span_id)),
        str(uuid.UUID(test_span_id2)),
        str(uuid.UUID(int=span_id)),
    )
    assert _ctx(spans[0]).trace_id == uuid.UUID(test_trace_id).int
    assert _parent(spans[0]).span_id == uuid.UUID(test_span_id2).int
    if old_val:
        os.environ["LMNR_SPAN_CONTEXT"] = old_val
    else:
        _popped_val = os.environ.pop("LMNR_SPAN_CONTEXT", None)


def test_add_span_tags(span_exporter: InMemorySpanExporter):
    @observe(tags=["foo"])
    def test():
        Laminar.add_span_tags(["bar", "baz", "foo"])
        Laminar.add_span_tags(["qux", "bar"])

    test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert sorted(cast(list[str], (spans[0].attributes or {})["lmnr.association.properties.tags"])) == [
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
    assert sorted(cast(list[str], (spans[0].attributes or {})["lmnr.association.properties.tags"])) == [
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
        def disabled_func(x: int, y: int) -> int:
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
            _popped_val = os.environ.pop("LMNR_DISABLE_TRACING", None)


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
        def inner_func() -> str:
            return "inner_result"

        @observe()
        def outer_func() -> str:
            return inner_func()

        # Test 1: Enabled tracing - should create nested spans
        _popped_val = os.environ.pop("LMNR_DISABLE_TRACING", None)
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        outer_span = next(s for s in spans if s.name == "outer_func")
        inner_span = next(s for s in spans if s.name == "inner_func")
        assert _parent(inner_span).span_id == _ctx(outer_span).span_id
        span_exporter.clear()

        # Test 2: Disabled tracing - no spans created but functions still work
        os.environ["LMNR_DISABLE_TRACING"] = "true"
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0
        span_exporter.clear()

        # Test 3: Re-enable tracing - spans created again with correct structure
        _popped_val = os.environ.pop("LMNR_DISABLE_TRACING", None)
        result = outer_func()
        assert result == "inner_result"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        outer_span = next(s for s in spans if s.name == "outer_func")
        inner_span = next(s for s in spans if s.name == "inner_func")
        # Verify proper nesting after re-enabling
        assert _parent(inner_span).span_id == _ctx(outer_span).span_id
        assert (inner_span.attributes or {})["lmnr.span.path"] == ("outer_func", "inner_func")

    finally:
        # Restore original value
        if old_val:
            os.environ["LMNR_DISABLE_TRACING"] = old_val
        else:
            _popped_val = os.environ.pop("LMNR_DISABLE_TRACING", None)
