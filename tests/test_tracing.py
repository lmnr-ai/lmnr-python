import json
import pytest
import uuid

from lmnr import Attributes, Laminar, observe, TracingLevel, use_span
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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


def test_start_as_current_span_labels(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", labels={"foo": "bar"}):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        json.loads(spans[0].attributes["lmnr.association.properties.label.foo"])
        == "bar"
    )
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"


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

    assert outer_span.attributes["lmnr.span.path"] == "test"
    assert inner_span.attributes["lmnr.span.path"] == "test.foo"


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

    events = spans[0].events
    assert len(events) == 0


def test_session_id(exporter: InMemorySpanExporter):
    Laminar.set_session("123")
    with Laminar.start_as_current_span("test"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"


def test_session_id_clear(exporter: InMemorySpanExporter):
    Laminar.set_session("123")
    with Laminar.start_as_current_span("in_session"):
        pass
    Laminar.clear_session()

    with Laminar.start_as_current_span("no_session"):
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


def test_with_labels(exporter: InMemorySpanExporter):
    with Laminar.with_labels(labels={"foo": "bar"}):
        with Laminar.start_as_current_span("test1"):
            pass
        with Laminar.start_as_current_span("test2"):
            pass

    with Laminar.start_as_current_span("test3"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    first_span = [span for span in spans if span.name == "test1"][0]
    second_span = [span for span in spans if span.name == "test2"][0]
    third_span = [span for span in spans if span.name == "test3"][0]
    assert (
        json.loads(first_span.attributes["lmnr.association.properties.label.foo"])
        == "bar"
    )
    assert (
        json.loads(second_span.attributes["lmnr.association.properties.label.foo"])
        == "bar"
    )
    assert third_span.attributes.get("lmnr.association.properties.label.foo") is None

    assert first_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert second_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert third_span.attributes["lmnr.span.instrumentation_source"] == "python"


def test_with_labels_observe(exporter: InMemorySpanExporter):
    @observe()
    def foo():
        pass

    with Laminar.with_labels(labels={"foo": "bar"}):
        foo()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "foo"
    assert (
        json.loads(spans[0].attributes["lmnr.association.properties.label.foo"])
        == "bar"
    )
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"


def test_metadata(exporter: InMemorySpanExporter):
    Laminar.set_metadata({"foo": "bar"})
    with Laminar.start_as_current_span("test"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        json.loads(spans[0].attributes["lmnr.association.properties.metadata.foo"])
        == "bar"
    )
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"


def test_metadata_clear(exporter: InMemorySpanExporter):
    Laminar.set_metadata({"foo": "bar"})
    with Laminar.start_as_current_span("with_metadata"):
        pass
    Laminar.clear_metadata()

    with Laminar.start_as_current_span("no_metadata"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    with_metadata_span = [span for span in spans if span.name == "with_metadata"][0]
    no_metadata_span = [span for span in spans if span.name == "no_metadata"][0]

    assert (
        json.loads(
            with_metadata_span.attributes["lmnr.association.properties.metadata.foo"]
        )
        == "bar"
    )

    assert (
        no_metadata_span.attributes.get("lmnr.association.properties.metadata.foo")
        is None
    )
    assert with_metadata_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert no_metadata_span.attributes["lmnr.span.instrumentation_source"] == "python"


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


def test_force_trace_id(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span(
        "test", trace_id=uuid.UUID("01234567-890a-bcde-f123-456789abcdef")
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].context.trace_id
        == uuid.UUID("01234567-890a-bcde-f123-456789abcdef").int
    )

    assert spans[0].attributes.get("lmnr.internal.override_parent_span") is True
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"


def test_force_trace_id_does_not_override_if_not_uuid(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", trace_id="not_a_uuid"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("lmnr.internal.override_parent_span") is None
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
