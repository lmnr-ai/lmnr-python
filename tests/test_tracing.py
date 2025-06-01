import json
import pytest
import uuid

from lmnr import Attributes, Laminar, observe, TracingLevel, use_span
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.sdk.types import LaminarSpanContext


def test_start_as_current_span(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", input="my_input"):
        Laminar.set_span_output("foo")
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_start_as_current_span_exception(exporter: InMemorySpanExporter):
    with pytest.raises(ValueError):
        with Laminar.start_as_current_span("test", input="my_input"):
            Laminar.set_span_output("foo")
            raise ValueError("error")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "error"


def test_start_as_current_span_span_type(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", span_type="LLM"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "LLM"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_start_as_current_span_labels(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", labels=["foo", "bar"]):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.labels"] == (
        "foo",
        "bar",
    )
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_set_span_attributes(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", input="my_input"):
        Laminar.set_span_attributes(
            {
                Attributes.PROVIDER: "openai",
                Attributes.REQUEST_MODEL: "gpt-4o-date-version",
                Attributes.RESPONSE_MODEL: "gpt-4o",
                Attributes.INPUT_TOKEN_COUNT: 100,
                Attributes.OUTPUT_TOKEN_COUNT: 200,
            },
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"

    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4o-date-version"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4o"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 200
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_set_attributes(exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with use_span(span, end_on_exit=True):
        Laminar.set_span_attributes(
            {
                Attributes.PROVIDER: "openai",
                Attributes.REQUEST_MODEL: "gpt-4o-date-version",
                Attributes.RESPONSE_MODEL: "gpt-4o",
                Attributes.INPUT_TOKEN_COUNT: 100,
                Attributes.OUTPUT_TOKEN_COUNT: 200,
            },
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"

    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4o-date-version"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4o"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 200
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_end_on_exit(exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with use_span(span, end_on_exit=True):
        Laminar.set_span_output("foo")
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_manual_end(exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with use_span(span) as inner_span:
        Laminar.set_span_output("foo")
        inner_span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_exception(exporter: InMemorySpanExporter):
    def foo(span):
        with use_span(span, end_on_exit=True):
            raise ValueError("error")

    span = Laminar.start_span("test", input="my_input")
    with pytest.raises(ValueError):
        foo(span)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)

    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "error"


def test_use_span_nested_path(exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")
    with use_span(span, end_on_exit=True):
        with Laminar.start_as_current_span("foo"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = [span for span in spans if span.name == "test"][0]
    inner_span = [span for span in spans if span.name == "foo"][0]

    assert outer_span.attributes["lmnr.span.path"] == ("test",)
    assert inner_span.attributes["lmnr.span.path"] == ("test", "foo")


def test_use_span_suppress_exception(exporter: InMemorySpanExporter):
    def foo(span):
        with use_span(span, end_on_exit=True, record_exception=False):
            raise ValueError("error")

    span = Laminar.start_span("test", input="my_input")
    with pytest.raises(ValueError):
        foo(span)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)

    events = spans[0].events
    assert len(events) == 0


def test_session_id(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_session_id("123")
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_session_id_doesnt_leak(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("no_session"):
        pass

    with Laminar.start_as_current_span("in_session"):
        Laminar.set_trace_session_id("123")
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    in_session_span = [span for span in spans if span.name == "in_session"][0]
    no_session_span = [span for span in spans if span.name == "no_session"][0]

    assert in_session_span.attributes["lmnr.association.properties.session_id"] == "123"
    assert in_session_span.attributes["lmnr.span.instrumentation_source"] == "python"

    assert (
        no_session_span.attributes.get("lmnr.association.properties.session_id") is None
    )
    assert no_session_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert no_session_span.attributes["lmnr.span.path"] == ("no_session",)


def test_user_id(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_user_id("123")
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_metadata(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_metadata({"foo": "bar"})
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_metadata_does_not_leak(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("with_metadata"):
        Laminar.set_trace_metadata({"foo": "bar"})
        pass

    with Laminar.start_as_current_span("no_metadata"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    with_metadata_span = [span for span in spans if span.name == "with_metadata"][0]
    no_metadata_span = [span for span in spans if span.name == "no_metadata"][0]

    assert (
        with_metadata_span.attributes["lmnr.association.properties.metadata.foo"]
        == "bar"
    )

    assert (
        no_metadata_span.attributes.get("lmnr.association.properties.metadata.foo")
        is None
    )
    assert with_metadata_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert no_metadata_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert no_metadata_span.attributes["lmnr.span.path"] == ("no_metadata",)


def test_tags(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_span_tags(["foo", "bar"])
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_tracing_level_attribute(exporter: InMemorySpanExporter):
    with Laminar.set_tracing_level(TracingLevel.META_ONLY):
        with Laminar.start_as_current_span("test"):
            pass

    with Laminar.set_tracing_level(TracingLevel.OFF):
        with Laminar.start_as_current_span("test2"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    first_span = [span for span in spans if span.name == "test"][0]
    second_span = [span for span in spans if span.name == "test2"][0]
    assert first_span.attributes["lmnr.internal.tracing_level"] == "meta_only"
    assert second_span.attributes["lmnr.internal.tracing_level"] == "off"
    assert first_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert second_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert first_span.attributes["lmnr.span.path"] == ("test",)
    assert second_span.attributes["lmnr.span.path"] == ("test2",)


def test_1k_attributes(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test") as span:
        for i in range(1000):
            span.set_attribute(f"foo_{i}", f"bar{i}")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)
    for i in range(1000):
        assert spans[0].attributes[f"foo_{i}"] == f"bar{i}"


def test_span_context(exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context(span))
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )


def test_span_context_dict(exporter: InMemorySpanExporter):
    def foo(context: dict):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )


def test_span_context_str(exporter: InMemorySpanExporter):
    def foo(context: str):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )


def test_span_context_ended_span(exporter: InMemorySpanExporter):
    # TODO: check with opentelemetry standards if we should allow this
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    span.end()
    foo(Laminar.get_laminar_span_context(span))

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )


def test_span_context_otel_fallback(exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(span.get_span_context())
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )


def test_span_context_dict_fallback(exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )
