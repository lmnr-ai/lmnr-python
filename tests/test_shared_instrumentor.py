"""Tests for the shared instrumentor contract in
`opentelemetry_lib/opentelemetry/instrumentation/shared/`.

These cover the generic machinery every migrated instrumentor relies on, so a
regression here would silently affect all of them at once.
"""

import time

import pytest
from opentelemetry.trace import SpanKind

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    add_spec_wrapper,
    stamp_instrumentation_scope,
)


def _spec(**overrides) -> WrappedFunctionSpec:
    base = WrappedFunctionSpec(
        package_name="pkg",
        method_name="meth",
        is_async=False,
        wrapper_function=lambda *a, **k: None,
    )
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class _FakeSpan:
    """Minimal Span stand-in: `set_span_attribute` only needs `set_attribute`."""

    def __init__(self):
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


# ---------------------------------------------------------------------------
# add_spec_wrapper
# ---------------------------------------------------------------------------


def test_add_spec_wrapper_binds_the_spec_as_the_first_argument():
    seen = {}

    def handler(to_wrap, wrapped, instance, args, kwargs):
        seen["spec"] = to_wrap
        return wrapped(*args, **kwargs)

    spec = _spec(span_name="my.span")
    wrapper = add_spec_wrapper(handler, spec)

    assert wrapper(lambda x: x * 2, None, (21,), {}) == 42
    assert seen["spec"] is spec


def test_add_spec_wrapper_normalizes_none_args_and_kwargs():
    """wrapt always passes args/kwargs, but the parameters are declared optional
    for direct callers. The handler must never receive None for either, or every
    wrapper would need to defensively re-normalize (which is exactly the
    `if kwargs is None` boilerplate the litellm wrappers still carry)."""
    seen = {}

    def handler(to_wrap, wrapped, instance, args, kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

    add_spec_wrapper(handler, _spec())(lambda: None, None)

    assert seen["args"] == ()
    assert seen["kwargs"] == {}


def test_add_spec_wrapper_forwards_handler_kwargs():
    """This is the mechanism `wrapper_kwargs()` rides on, so that an
    instrumentor-level collaborator (e.g. the browser group's client) reaches a
    wrapper without being stuffed into per-method spec config."""
    seen = {}

    def handler(to_wrap, wrapped, instance, args, kwargs, *, client):
        seen["client"] = client

    sentinel = object()
    add_spec_wrapper(handler, _spec(), client=sentinel)(lambda: None, None, (), {})

    assert seen["client"] is sentinel


# ---------------------------------------------------------------------------
# stamp_instrumentation_scope
# ---------------------------------------------------------------------------


def test_stamp_instrumentation_scope_records_name_and_version():
    span = _FakeSpan()
    stamp_instrumentation_scope(
        span,
        _spec(
            instrumentation_scope=LaminarInstrumentationScopeAttributes(
                name="litellm", version="1.2.3"
            )
        ),
    )

    assert span.attributes == {
        "lmnr.span.instrumentation_scope.name": "litellm",
        "lmnr.span.instrumentation_scope.version": "1.2.3",
    }


def test_stamp_instrumentation_scope_is_a_noop_without_a_scope():
    """`instrumentation_scope` is an optional spec key, so a spec that omits it
    must not raise — that would abort the wrapper before the real call."""
    span = _FakeSpan()
    stamp_instrumentation_scope(span, _spec())

    assert span.attributes == {}


def test_stamp_instrumentation_scope_tolerates_a_none_span():
    """`safe_start_span` returns None when Laminar is not initialized, and
    wrappers stamp the scope before checking — so None must be a no-op."""
    stamp_instrumentation_scope(
        None,
        _spec(
            instrumentation_scope=LaminarInstrumentationScopeAttributes(
                name="litellm", version="1.2.3"
            )
        ),
    )


# ---------------------------------------------------------------------------
# safe_start_span
# ---------------------------------------------------------------------------


def test_safe_start_span_honors_a_retroactive_start_time(span_exporter):
    """The OpenAI responses/assistants wrappers only learn a call happened once
    it finished, so they backdate the span. `Laminar.start_span` deliberately
    does not expose this, hence the tracer path."""
    start = time.time_ns() - 5_000_000_000  # 5s ago

    span = safe_start_span("backdated", start_time=start)
    assert span is not None
    span.end()

    (exported,) = span_exporter.get_finished_spans()
    assert exported.start_time == start
    assert exported.attributes["lmnr.span.type"] == "DEFAULT"


def test_safe_start_span_honors_span_kind(span_exporter):
    span = safe_start_span("client-call", kind=SpanKind.CLIENT, span_type="LLM")
    assert span is not None
    span.end()

    (exported,) = span_exporter.get_finished_spans()
    assert exported.kind == SpanKind.CLIENT
    assert exported.attributes["lmnr.span.type"] == "LLM"


def test_safe_start_span_default_path_still_goes_through_laminar(span_exporter):
    """Without start_time/kind the helper must keep using the public
    `Laminar.start_span`, which is what applies Laminar's own span bookkeeping."""
    span = safe_start_span("plain", span_type="TOOL")
    assert span is not None
    span.end()

    (exported,) = span_exporter.get_finished_spans()
    assert exported.kind == SpanKind.INTERNAL
    assert exported.attributes["lmnr.span.type"] == "TOOL"


# ---------------------------------------------------------------------------
# BaseLaminarInstrumentor
# ---------------------------------------------------------------------------


def test_base_instrumentor_is_abstract():
    with pytest.raises(TypeError):
        BaseLaminarInstrumentor()  # type: ignore[abstract]


def test_wrapper_kwargs_defaults_to_empty_and_reaches_the_wrapper():
    seen = {}

    def handler(to_wrap, wrapped, instance, args, kwargs, **extra):
        seen["extra"] = extra

    class _Instrumentor(BaseLaminarInstrumentor):
        def instrumentation_dependencies(self):
            return ()

        def instrumentation_scope(self):
            return LaminarInstrumentationScopeAttributes(name="t", version="0")

        def wrapper_kwargs(self):
            return {"client": "sentinel"}

    inst = _Instrumentor()
    inst.instrumentor_config = LaminarInstrumentorConfig(
        wrapped_functions=[_spec(wrapper_function=handler)]
    )

    add_spec_wrapper(handler, _spec(), **inst.wrapper_kwargs())(
        lambda: None, None, (), {}
    )
    assert seen["extra"] == {"client": "sentinel"}
