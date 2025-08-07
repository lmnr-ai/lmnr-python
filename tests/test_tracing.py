import json
import pytest
import uuid

from lmnr import Attributes, Laminar
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.sdk.types import LaminarSpanContext


def test_start_as_current_span(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", input="my_input"):
        Laminar.set_span_output("foo")
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_start_as_current_span_exception(span_exporter: InMemorySpanExporter):
    with pytest.raises(ValueError):
        with Laminar.start_as_current_span("test", input="my_input"):
            Laminar.set_span_output("foo")
            raise ValueError("error")

    spans = span_exporter.get_finished_spans()
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


def test_start_as_current_span_span_type(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", span_type="LLM"):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.type"] == "LLM"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_start_as_current_span_tags(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", tags=["foo", "bar"]):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.tags"] == (
        "foo",
        "bar",
    )
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_set_span_attributes(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test", input="my_input"):
        Laminar.set_span_attributes(
            {
                Attributes.PROVIDER: "openai",
                Attributes.REQUEST_MODEL: "gpt-4o-date-version",
                Attributes.RESPONSE_MODEL: "gpt-4o",
                Attributes.INPUT_TOKEN_COUNT: 100,
                Attributes.OUTPUT_TOKEN_COUNT: 200,
                "freeform": "freeform",
            },
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"

    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4o-date-version"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4o"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 200
    assert spans[0].attributes["freeform"] == "freeform"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_set_attributes(span_exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with Laminar.use_span(span, end_on_exit=True):
        Laminar.set_span_attributes(
            {
                Attributes.PROVIDER: "openai",
                Attributes.REQUEST_MODEL: "gpt-4o-date-version",
                Attributes.RESPONSE_MODEL: "gpt-4o",
                Attributes.INPUT_TOKEN_COUNT: 100,
                Attributes.OUTPUT_TOKEN_COUNT: 200,
            },
        )

    spans = span_exporter.get_finished_spans()
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


def test_use_span_end_on_exit(span_exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with Laminar.use_span(span, end_on_exit=True):
        Laminar.set_span_output("foo")
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


@pytest.mark.vcr
def test_use_span_with_auto_instrumentation(span_exporter: InMemorySpanExporter):
    from openai import OpenAI

    # real key was recorded in vcr
    openai_client = OpenAI(api_key="fake")

    span = Laminar.start_span("test", input="my_input")

    with Laminar.use_span(span, end_on_exit=True):
        openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what is the capital of France?"}],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    test_span = [span for span in spans if span.name == "test"][0]
    openai_span = [span for span in spans if span.name == "openai.chat"][0]

    assert json.loads(test_span.attributes["lmnr.span.input"]) == "my_input"
    assert test_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert test_span.attributes["lmnr.span.path"] == ("test",)

    assert openai_span.parent.span_id == test_span.get_span_context().span_id
    assert openai_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert openai_span.attributes["lmnr.span.path"] == ("test", "openai.chat")


def test_use_span_manual_end(span_exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")

    with Laminar.use_span(span) as inner_span:
        Laminar.set_span_output("foo")
        inner_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "foo"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_use_span_exception(span_exporter: InMemorySpanExporter):
    def foo(span):
        with Laminar.use_span(span, end_on_exit=True):
            raise ValueError("error")

    span = Laminar.start_span("test", input="my_input")
    with pytest.raises(ValueError):
        foo(span)

    spans = span_exporter.get_finished_spans()
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


def test_use_span_nested_path(span_exporter: InMemorySpanExporter):
    span = Laminar.start_span("test", input="my_input")
    with Laminar.use_span(span, end_on_exit=True):
        with Laminar.start_as_current_span("foo"):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = [span for span in spans if span.name == "test"][0]
    inner_span = [span for span in spans if span.name == "foo"][0]

    assert outer_span.attributes["lmnr.span.path"] == ("test",)
    assert inner_span.attributes["lmnr.span.path"] == ("test", "foo")


def test_use_span_suppress_exception(span_exporter: InMemorySpanExporter):
    def foo(span):
        with Laminar.use_span(span, end_on_exit=True, record_exception=False):
            raise ValueError("error")

    span = Laminar.start_span("test", input="my_input")
    with pytest.raises(ValueError):
        foo(span)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "my_input"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)

    events = spans[0].events
    assert len(events) == 0


@pytest.mark.vcr
def test_use_span_with_auto_instrumentation_langchain(
    span_exporter: InMemorySpanExporter,
):
    from langchain_openai import ChatOpenAI

    # the real API key was used in the vcr cassette
    openai_client = ChatOpenAI(api_key="test-api-key")

    span = Laminar.start_span("test", input="my_input")

    with Laminar.use_span(span, end_on_exit=True):
        response = openai_client.invoke(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "what is the capital of France?"}],
        )
        Laminar.set_span_output(response.content[0])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    test_span = [span for span in spans if span.name == "test"][0]
    openai_span = [span for span in spans if span.name == "ChatOpenAI.chat"][0]

    assert (
        test_span.get_span_context().trace_id == openai_span.get_span_context().trace_id
    )
    assert openai_span.parent.span_id == test_span.get_span_context().span_id
    assert openai_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert openai_span.attributes["lmnr.span.path"] == ("test", "ChatOpenAI.chat")


def test_session_id(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_session_id("123")
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.session_id"] == "123"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_session_id_doesnt_leak(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("no_session"):
        pass

    with Laminar.start_as_current_span("in_session"):
        Laminar.set_trace_session_id("123")
        pass

    spans = span_exporter.get_finished_spans()
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


def test_user_id(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_user_id("123")
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "123"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_metadata(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_trace_metadata({"foo": "bar"})
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_metadata_does_not_leak(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("with_metadata"):
        Laminar.set_trace_metadata({"foo": "bar"})
        pass

    with Laminar.start_as_current_span("no_metadata"):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    with_metadata_span = [span for span in spans if span.name == "with_metadata"][0]
    no_metadata_span = [span for span in spans if span.name == "no_metadata"][0]

    assert with_metadata_span.parent is None or with_metadata_span.parent.span_id == 0
    assert no_metadata_span.parent is None or no_metadata_span.parent.span_id == 0

    assert (
        with_metadata_span.get_span_context().trace_id
        != no_metadata_span.get_span_context().trace_id
    )

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
    assert with_metadata_span.parent is None or with_metadata_span.parent.span_id == 0
    assert no_metadata_span.parent is None or no_metadata_span.parent.span_id == 0
    assert (
        with_metadata_span.get_span_context().trace_id
        != no_metadata_span.get_span_context().trace_id
    )


def test_tags(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_span_tags(["foo", "bar"])
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("foo", "bar")
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)


def test_1k_attributes(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test") as span:
        for i in range(1000):
            span.set_attribute(f"foo_{i}", f"bar{i}")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("test",)
    for i in range(1000):
        assert spans[0].attributes[f"foo_{i}"] == f"bar{i}"


def test_span_context(span_exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context(span))
    span.end()

    spans = span_exporter.get_finished_spans()
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


def test_span_context_dict(span_exporter: InMemorySpanExporter):
    def foo(context: dict):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = span_exporter.get_finished_spans()
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


def test_span_context_str(span_exporter: InMemorySpanExporter):
    def foo(context: str):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = span_exporter.get_finished_spans()
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


def test_span_context_ended_span(span_exporter: InMemorySpanExporter):
    # TODO: check with opentelemetry standards if we should allow this
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    span.end()
    foo(Laminar.get_laminar_span_context(span))

    spans = span_exporter.get_finished_spans()
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


def test_span_context_otel_fallback(span_exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(span.get_span_context())
    span.end()

    spans = span_exporter.get_finished_spans()
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


def test_span_context_dict_fallback(span_exporter: InMemorySpanExporter):
    def foo(context: LaminarSpanContext):
        with Laminar.start_as_current_span("inner", parent_span_context=context):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.get_laminar_span_context_dict(span))
    span.end()

    spans = span_exporter.get_finished_spans()
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


def test_tags_deduplication(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("test"):
        Laminar.set_span_tags(["foo", "bar", "foo"])
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]
