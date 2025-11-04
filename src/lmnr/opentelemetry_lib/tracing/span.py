from inspect import Traceback
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, Span as SDKSpan
from opentelemetry.sdk.util.instrumentation import (
    InstrumentationInfo,
    InstrumentationScope,
)
from opentelemetry.trace import Link, Span, SpanContext, SpanKind, Status
from opentelemetry.util.types import AttributeValue
from opentelemetry.context import detach

from lmnr.opentelemetry_lib.tracing.context import (
    pop_span_context,
)
from lmnr.sdk.log import get_default_logger


class LaminarSpan(Span, ReadableSpan):
    # We wrap the SDK span in a LaminarSpan instead of inheriting from it,
    # because OpenTelemetry discourages direct initialization of SdkSpan objects.
    # Instead, we rely on the tracer to create the span for us, and then we
    # wrap it in a LaminarSpan.
    span: SDKSpan
    # Whether the span has been popped from the context stack to prevent
    # double popping if span.end() is called multiple times.
    _popped: bool = False

    def __init__(self, span: SDKSpan):
        self.span = span
        self.logger = get_default_logger(__name__)

    def end(self, end_time: int | None = None) -> None:
        self.span.end(end_time)
        if hasattr(self, "_lmnr_ctx_token") and not self._popped:
            try:
                pop_span_context()
                # Internally handles and logs the error
                detach(self._lmnr_ctx_token)
                self._popped = True
            except Exception:
                pass

    ### ========================================================================
    # The below methods are just passthrough of abstract Span methods
    # to the SDK span. If you need to override them, or add additional logic,
    # move them above this section.
    # ==========================================================================

    def get_span_context(self) -> SpanContext:
        return self.span.get_span_context()

    def set_attributes(self, attributes: dict[str, AttributeValue]) -> None:
        self.span.set_attributes(attributes)

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.span.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        attributes: dict[str, AttributeValue] = None,
        timestamp: int | None = None,
    ) -> None:
        self.span.add_event(name, attributes, timestamp)

    def add_link(
        self, context: SpanContext, attributes: dict[str, AttributeValue] = None
    ) -> None:
        self.span.add_link(context, attributes)

    def update_name(self, name: str) -> None:
        self.span.update_name(name)

    def is_recording(self) -> bool:
        return self.span.is_recording()

    def set_status(self, status: Status, description: str | None = None) -> None:
        self.span.set_status(status, description)

    def record_exception(
        self,
        exception: BaseException,
        attributes: dict[str, AttributeValue] = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        self.span.record_exception(exception, attributes, timestamp, escaped)

    def __enter__(self) -> "LaminarSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: Traceback | None,
    ) -> None:
        self.end()

    def _readable_span(self) -> ReadableSpan:
        return self.span._readable_span()

    @property
    def name(self) -> str:
        return self.span.name

    @property
    def context(self) -> SpanContext:
        return self.span.context

    @property
    def start_time(self) -> int:
        return self.span.start_time

    @property
    def end_time(self) -> int:
        return self.span.end_time

    @property
    def dropped_attributes(self) -> int:
        return self.span.dropped_attributes

    @property
    def dropped_events(self) -> int:
        return self.span.dropped_events

    @property
    def dropped_links(self) -> int:
        return self.span.dropped_links

    @property
    def attributes(self) -> dict[str, AttributeValue]:
        return self.span.attributes

    @property
    def events(self) -> list[Event]:
        return self.span.events

    @property
    def links(self) -> list[Link]:
        return self.span.links

    @property
    def status(self) -> Status:
        return self.span.status

    @property
    def parent_span_context(self) -> SpanContext:
        return self.span.parent_span_context

    @property
    def span_context(self) -> SpanContext:
        return self.span.span_context

    @property
    def span_id(self) -> str:
        return self.span.span_id

    @property
    def kind(self) -> SpanKind:
        return self.span.kind

    @property
    def resource(self) -> Resource:
        return self.span.resource

    @property
    def instrumentation_scope(self) -> InstrumentationScope:
        return self.span.instrumentation_scope

    @property
    def instrumentation_info(self) -> InstrumentationInfo:
        return self.span.instrumentation_info

    def to_json(self) -> str:
        return self.span.to_json()
