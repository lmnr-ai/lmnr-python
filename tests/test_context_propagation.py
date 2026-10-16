from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN_ID

from lmnr import Laminar, observe


def _assert_association_properties(
    span: ReadableSpan,
    user_id: str,
    session_id: str,
    metadata: dict[str, str],
    trace_type: str,
):
    attributes = span.attributes or {}
    assert attributes["lmnr.association.properties.user_id"] == user_id
    assert attributes["lmnr.association.properties.session_id"] == session_id
    assert attributes["lmnr.association.properties.trace_type"] == trace_type
    for key, value in metadata.items():
        assert attributes[f"lmnr.association.properties.metadata.{key}"] == value


def _assert_same_trace_and_inheritance(
    spans: list[ReadableSpan], expected_parent_span_id: int | None = None
):
    contexts = [span.get_span_context() for span in spans]
    assert all(context is not None for context in contexts)
    trace_ids = [context.trace_id for context in contexts if context is not None]
    assert len(set(trace_ids)) == 1
    first_parent = spans[0].parent
    if expected_parent_span_id is not None:
        assert first_parent is not None
        assert first_parent.span_id == expected_parent_span_id
    else:
        assert (
            first_parent is None
            or first_parent.span_id is None
            or first_parent.span_id == INVALID_SPAN_ID
        )

    for i, span in enumerate(spans[1:]):
        parent = span.parent
        prev_context = contexts[i]
        assert parent is not None
        assert prev_context is not None
        assert parent.span_id == prev_context.span_id


def test_ctx_prop_parent_sc_child_s(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    ):
        p_span = Laminar.start_span("parent")
        with Laminar.use_span(p_span, end_on_exit=True):
            c_span = Laminar.start_span("child")
            c_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_sc_child_sc(span_exporter: InMemorySpanExporter):
    with (
        Laminar.start_as_current_span(
            "grandparent",
            user_id="user_id",
            session_id="session_id",
            span_type="EVALUATION",
            metadata={"foo": "bar"},
        ),
        Laminar.start_as_current_span("parent"),
        Laminar.start_as_current_span("child"),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_sc_child_obs(span_exporter: InMemorySpanExporter):
    @observe()
    def child():
        pass

    @observe()
    def parent():
        child()

    with Laminar.start_as_current_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    ):
        parent()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_sa_child_s(span_exporter: InMemorySpanExporter):
    g_span = Laminar.start_active_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )
    p_span = Laminar.start_span("parent")
    with Laminar.use_span(p_span, end_on_exit=True):
        c_span = Laminar.start_span("child")
        c_span.end()
    g_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_sa_child_sc(span_exporter: InMemorySpanExporter):
    grandparent_span = Laminar.start_active_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )

    with (
        Laminar.start_as_current_span("parent"),
        Laminar.start_as_current_span("child"),
    ):
        pass
    grandparent_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_sa_child_obs(span_exporter: InMemorySpanExporter):
    @observe()
    def child():
        pass

    @observe()
    def parent():
        child()

    g_span = Laminar.start_active_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )
    parent()
    g_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_obs_child_s(span_exporter: InMemorySpanExporter):
    @observe(
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )
    def grandparent():
        p_span = Laminar.start_span("parent")
        with Laminar.use_span(p_span, end_on_exit=True):
            c_span = Laminar.start_span("child")
            c_span.end()

    grandparent()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_parent_use_span(span_exporter: InMemorySpanExporter):
    g_span = Laminar.start_span(
        "grandparent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )
    with Laminar.use_span(g_span, end_on_exit=True):
        p_span = Laminar.start_span("parent")
        with Laminar.use_span(p_span, end_on_exit=True):
            c_span = Laminar.start_span("child")
            c_span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    grandparent_span = next(s for s in spans if s.name == "grandparent")
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")
    _assert_association_properties(
        grandparent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_same_trace_and_inheritance([grandparent_span, parent_span, child_span])


def test_ctx_prop_laminar_span_context(span_exporter: InMemorySpanExporter):
    span = Laminar.start_span(
        "parent",
        user_id="user_id",
        session_id="session_id",
        span_type="EVALUATION",
        metadata={"foo": "bar"},
    )
    span_context = Laminar.get_laminar_span_context(span)
    passed_span_context = Laminar.serialize_span_context(span)
    assert passed_span_context is not None
    span2 = Laminar.start_span(
        "child",
        parent_span_context=Laminar.deserialize_span_context(passed_span_context),
    )
    span2.end()
    span.end()

    assert span_context is not None
    assert span_context.user_id == "user_id"
    assert span_context.session_id == "session_id"
    assert span_context.trace_type is not None
    assert span_context.trace_type.value == "EVALUATION"
    assert span_context.metadata == {"foo": "bar"}

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")

    parent_context = parent_span.get_span_context()
    child_context = child_span.get_span_context()
    assert parent_context is not None
    assert child_context is not None
    assert parent_context.trace_id == child_context.trace_id
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_context.span_id

    _assert_association_properties(
        parent_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
    _assert_association_properties(
        child_span, "user_id", "session_id", {"foo": "bar"}, "EVALUATION"
    )
