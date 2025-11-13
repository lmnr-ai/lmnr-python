import json
import pytest
import uuid

from lmnr import Attributes, Laminar
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.tracing.tracer import TracerWrapper
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
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]
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
            model="gpt-5-nano",
            input=[{"role": "user", "content": "what is the capital of France?"}],
        )
        Laminar.set_span_output(response.content[0])

    spans = span_exporter.get_finished_spans()
    # assert len(spans) == 2
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
    assert sorted(spans[0].attributes["lmnr.association.properties.tags"]) == [
        "bar",
        "foo",
    ]
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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


def test_span_context_ser_de(span_exporter: InMemorySpanExporter):
    def foo(context: str):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    span = Laminar.start_span("test")
    foo(Laminar.serialize_span_context(span))
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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


def test_span_context_path_ids_path(span_exporter: InMemorySpanExporter):
    def foo(context: str):
        parent_span_context = Laminar.deserialize_span_context(context)
        with Laminar.start_as_current_span(
            "inner", parent_span_context=parent_span_context
        ):
            pass

    with Laminar.start_as_current_span("outer"):
        span = Laminar.start_span("test")
        # Clear the span processor to ensure the path is not cached
        # This simulates span context being passed across services
        TracerWrapper.instance._span_processor.clear()
        foo(Laminar.serialize_span_context(span))
        span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "outer"][0]
    context_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("outer", "test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=context_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )
    assert inner_span.parent.span_id == context_span.get_span_context().span_id
    assert context_span.parent.span_id == outer_span.get_span_context().span_id


def test_span_context_path_ids_path_start_span(span_exporter: InMemorySpanExporter):
    def foo(context: str):
        parent_span_context = Laminar.deserialize_span_context(context)
        span = Laminar.start_span("inner", parent_span_context=parent_span_context)
        span.end()

    with Laminar.start_as_current_span("outer"):
        span = Laminar.start_span("test")
        # Clear the span processor to ensure the path is not cached
        # This simulates span context being passed across services
        TracerWrapper.instance._span_processor.clear()
        foo(Laminar.serialize_span_context(span))
        span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "outer"][0]
    context_span = [span for span in spans if span.name == "test"][0]

    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("outer", "test", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
        str(uuid.UUID(int=context_span.get_span_context().span_id)),
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )
    assert inner_span.parent.span_id == context_span.get_span_context().span_id
    assert context_span.parent.span_id == outer_span.get_span_context().span_id


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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


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
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id


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


def test_start_active_span_simple(span_exporter: InMemorySpanExporter):
    span = Laminar.start_active_span("outer", input="test_input")
    Laminar.set_span_output("test_output")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "outer"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == "test_input"
    assert json.loads(spans[0].attributes["lmnr.span.output"]) == "test_output"
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.path"] == ("outer",)


def test_start_active_span_with_nested_context_manager(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with nested start_as_current_span."""
    span = Laminar.start_active_span("outer")

    with Laminar.start_as_current_span("inner"):
        Laminar.set_span_output("inner_output")

    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer_span = [s for s in spans if s.name == "outer"][0]
    inner_span = [s for s in spans if s.name == "inner"][0]

    # Check parent-child relationship
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )

    # Check span paths
    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert inner_span.attributes["lmnr.span.path"] == ("outer", "inner")

    # Check output
    assert json.loads(inner_span.attributes["lmnr.span.output"]) == "inner_output"


def test_start_active_span_deeply_nested(span_exporter: InMemorySpanExporter):
    """Test multiple levels of nesting with start_active_span."""
    outer_span = Laminar.start_active_span("outer")

    with Laminar.start_as_current_span("middle"):
        inner_span = Laminar.start_active_span("inner")
        Laminar.set_span_output("inner_output")
        inner_span.end()

    outer_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    outer = [s for s in spans if s.name == "outer"][0]
    middle = [s for s in spans if s.name == "middle"][0]
    inner = [s for s in spans if s.name == "inner"][0]

    # Check parent-child relationships
    assert middle.parent.span_id == outer.get_span_context().span_id
    assert inner.parent.span_id == middle.get_span_context().span_id

    # Check all spans share the same trace_id
    assert (
        outer.get_span_context().trace_id
        == middle.get_span_context().trace_id
        == inner.get_span_context().trace_id
    )

    # Check span paths
    assert outer.attributes["lmnr.span.path"] == ("outer",)
    assert middle.attributes["lmnr.span.path"] == ("outer", "middle")
    assert inner.attributes["lmnr.span.path"] == ("outer", "middle", "inner")

    # Check ids_path
    assert outer.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer.get_span_context().span_id)),
    )
    assert middle.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer.get_span_context().span_id)),
        str(uuid.UUID(int=middle.get_span_context().span_id)),
    )
    assert inner.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer.get_span_context().span_id)),
        str(uuid.UUID(int=middle.get_span_context().span_id)),
        str(uuid.UUID(int=inner.get_span_context().span_id)),
    )


def test_start_active_span_sequential_siblings(span_exporter: InMemorySpanExporter):
    """Test sequential siblings under an active span."""
    parent_span = Laminar.start_active_span("parent")

    with Laminar.start_as_current_span("child1"):
        Laminar.set_span_output("output1")

    with Laminar.start_as_current_span("child2"):
        Laminar.set_span_output("output2")

    parent_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    parent = [s for s in spans if s.name == "parent"][0]
    child1 = [s for s in spans if s.name == "child1"][0]
    child2 = [s for s in spans if s.name == "child2"][0]

    # Both children should have the same parent
    assert child1.parent.span_id == parent.get_span_context().span_id
    assert child2.parent.span_id == parent.get_span_context().span_id

    # All should share the same trace_id
    assert (
        parent.get_span_context().trace_id
        == child1.get_span_context().trace_id
        == child2.get_span_context().trace_id
    )

    # Check span paths
    assert parent.attributes["lmnr.span.path"] == ("parent",)
    assert child1.attributes["lmnr.span.path"] == ("parent", "child1")
    assert child2.attributes["lmnr.span.path"] == ("parent", "child2")


def test_start_active_span_with_tags_and_span_type(
    span_exporter: InMemorySpanExporter,
):
    """Test start_active_span with tags and span_type."""
    span = Laminar.start_active_span(
        "test_span", input={"key": "value"}, span_type="LLM", tags=["tag1", "tag2"]
    )
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"
    assert spans[0].attributes["lmnr.span.type"] == "LLM"
    assert json.loads(spans[0].attributes["lmnr.span.input"]) == {"key": "value"}
    assert spans[0].attributes["lmnr.span.path"] == ("test_span",)

    # Check tags if present (tags might be set via the span processor)
    tags_attr = spans[0].attributes.get("lmnr.association.properties.tags")
    if tags_attr:
        assert sorted(tags_attr) == ["tag1", "tag2"]


def test_start_active_span_multiple_active_spans(span_exporter: InMemorySpanExporter):
    """Test multiple active spans in sequence (not nested)."""
    span1 = Laminar.start_active_span("span1")
    Laminar.set_span_output("output1")
    span1.end()

    span2 = Laminar.start_active_span("span2")
    Laminar.set_span_output("output2")
    span2.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    span1_obj = [s for s in spans if s.name == "span1"][0]
    span2_obj = [s for s in spans if s.name == "span2"][0]

    # They should be in different traces
    assert (
        span1_obj.get_span_context().trace_id != span2_obj.get_span_context().trace_id
    )

    # Both should be root spans
    assert span1_obj.parent is None or span1_obj.parent.span_id == 0
    assert span2_obj.parent is None or span2_obj.parent.span_id == 0

    # Check paths
    assert span1_obj.attributes["lmnr.span.path"] == ("span1",)
    assert span2_obj.attributes["lmnr.span.path"] == ("span2",)


@pytest.mark.asyncio
async def test_start_active_span_async(span_exporter: InMemorySpanExporter):
    """Test start_active_span with async functions."""

    async def async_work():
        with Laminar.start_as_current_span("async_inner"):
            Laminar.set_span_output("async_output")

    span = Laminar.start_active_span("async_outer")
    await async_work()
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    outer = [s for s in spans if s.name == "async_outer"][0]
    inner = [s for s in spans if s.name == "async_inner"][0]

    # Check parent-child relationship
    assert inner.parent.span_id == outer.get_span_context().span_id
    assert inner.get_span_context().trace_id == outer.get_span_context().trace_id

    # Check span paths
    assert outer.attributes["lmnr.span.path"] == ("async_outer",)
    assert inner.attributes["lmnr.span.path"] == ("async_outer", "async_inner")


@pytest.mark.asyncio
async def test_start_active_span_async_nested(span_exporter: InMemorySpanExporter):
    """Test nested active spans with async functions."""

    async def nested_level2():
        with Laminar.start_as_current_span("level2"):
            pass

    async def nested_level1():
        span = Laminar.start_active_span("level1")
        await nested_level2()
        span.end()

    span = Laminar.start_active_span("level0")
    await nested_level1()
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    level0 = [s for s in spans if s.name == "level0"][0]
    level1 = [s for s in spans if s.name == "level1"][0]
    level2 = [s for s in spans if s.name == "level2"][0]

    # Check parent-child relationships
    assert level1.parent.span_id == level0.get_span_context().span_id
    assert level2.parent.span_id == level1.get_span_context().span_id

    # Check trace ids
    assert (
        level0.get_span_context().trace_id
        == level1.get_span_context().trace_id
        == level2.get_span_context().trace_id
    )

    # Check span paths
    assert level0.attributes["lmnr.span.path"] == ("level0",)
    assert level1.attributes["lmnr.span.path"] == ("level0", "level1")
    assert level2.attributes["lmnr.span.path"] == ("level0", "level1", "level2")
