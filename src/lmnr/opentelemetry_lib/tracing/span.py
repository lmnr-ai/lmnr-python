import asyncio
from logging import Logger
from inspect import Traceback
from typing import Any, Literal
import orjson
import uuid

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, Span as SDKSpan
from opentelemetry.sdk.util.instrumentation import (
    InstrumentationInfo,
    InstrumentationScope,
)
from opentelemetry.trace import Link, Span, SpanContext, SpanKind, Status
from opentelemetry.util.types import AttributeValue
from opentelemetry.context import detach

from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    METADATA,
    SESSION_ID,
    SPAN_IDS_PATH,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_PATH,
    TRACE_TYPE,
    USER_ID,
)
from lmnr.opentelemetry_lib.tracing.context import (
    detach_context,
    pop_span_context,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanContext
from lmnr.sdk.utils import is_otel_attribute_value_type, json_dumps

MAX_MANUAL_SPAN_PAYLOAD_SIZE = 1024 * 1024 * 10  # 10MB


class LaminarSpanInterfaceMixin:
    """Mixin providing Laminar-specific span methods and properties."""

    span: SDKSpan
    logger: Logger

    def set_trace_session_id(self, session_id: str | None = None) -> None:
        """Set the session id for the current trace. Must be called at most once per trace.

        Args:
            session_id (str | None): Session id to set for the span.
        """
        if session_id is not None:
            self.set_attribute(f"{ASSOCIATION_PROPERTIES}.session_id", session_id)

    def set_trace_user_id(self, user_id: str | None = None) -> None:
        """Set the user id for the current trace. Must be called at most once per trace.

        Args:
            user_id (str | None): User id to set for the span.
        """
        if user_id is not None:
            self.span.set_attribute(f"{ASSOCIATION_PROPERTIES}.user_id", user_id)

    def set_trace_metadata(self, metadata: dict[str, AttributeValue]) -> None:
        """Set the metadata for the current trace, merging with any global metadata.
        Must be called at most once per trace.

        Args:
            metadata (dict[str, AttributeValue]): Metadata to set for the trace.
        """
        formatted_metadata = {}
        for key, value in metadata.items():
            if is_otel_attribute_value_type(value):
                formatted_metadata[f"{ASSOCIATION_PROPERTIES}.metadata.{key}"] = value
            else:
                formatted_metadata[f"{ASSOCIATION_PROPERTIES}.metadata.{key}"] = (
                    json_dumps(value)
                )
        self.span.set_attributes(formatted_metadata)

    def get_laminar_span_context(self) -> LaminarSpanContext:
        span_path = []
        span_ids_path = []
        user_id = None
        session_id = None
        trace_type = None
        metadata = {}
        if hasattr(self.span, "attributes"):
            span_path = list(self.span.attributes.get(SPAN_PATH, tuple()))
            span_ids_path = list(self.span.attributes.get(SPAN_IDS_PATH, tuple()))
            user_id = self.span.attributes.get(
                f"{ASSOCIATION_PROPERTIES}.{USER_ID}", None
            )
            session_id = self.span.attributes.get(
                f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}", None
            )
            trace_type = self.span.attributes.get(
                f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}", None
            )
            metadata = {
                k.replace(f"{ASSOCIATION_PROPERTIES}.{METADATA}.", ""): v
                for k, v in self.span.attributes.items()
                if k.startswith(f"{ASSOCIATION_PROPERTIES}.{METADATA}.")
            }
            for k, v in metadata.items():
                try:
                    metadata[k] = orjson.loads(v)
                except Exception:
                    metadata[k] = v
        else:
            self.logger.warning(
                "Attributes object is not available. Most likely the span is not a LaminarSpan "
                "and not an OpenTelemetry default SDK span. Span path and ids path will be empty.",
            )
        return LaminarSpanContext(
            trace_id=uuid.UUID(int=self.span.get_span_context().trace_id),
            span_id=uuid.UUID(int=self.span.get_span_context().span_id),
            is_remote=self.span.get_span_context().is_remote,
            span_path=span_path,
            span_ids_path=span_ids_path,
            user_id=user_id,
            session_id=session_id,
            trace_type=trace_type,
            metadata=metadata,
        )

    def span_id(self, format: Literal["int", "uuid"] = "int") -> int | uuid.UUID:
        if format == "int":
            return self.span.get_span_context().span_id
        elif format == "uuid":
            return uuid.UUID(int=self.span.get_span_context().span_id)
        self.logger.warning(f"Invalid format: {format}. Returning int.")
        return self.span.get_span_context().span_id

    def trace_id(self, format: Literal["int", "uuid"] = "int") -> int | uuid.UUID:
        if format == "int":
            return self.span.get_span_context().trace_id
        elif format == "uuid":
            return uuid.UUID(int=self.span.get_span_context().trace_id)
        self.logger.warning(f"Invalid format: {format}. Returning int.")
        return self.span.get_span_context().trace_id

    def parent_span_id(
        self, format: Literal["int", "uuid"] = "int"
    ) -> int | uuid.UUID | None:
        parent_span_id = self.span.parent.span_id if self.span.parent else None
        if parent_span_id is None:
            return None
        if format == "int":
            return parent_span_id
        elif format == "uuid":
            return uuid.UUID(int=parent_span_id)
        self.logger.warning(f"Invalid format: {format}. Returning int.")
        return parent_span_id

    def set_output(self, output: Any = None) -> None:
        if output is not None:
            serialized_output = json_dumps(output)
            if len(serialized_output) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                self.span.set_attribute(
                    SPAN_OUTPUT,
                    "Laminar: output too large to record",
                )
            else:
                self.span.set_attribute(SPAN_OUTPUT, serialized_output)

    def set_input(self, input: Any = None) -> None:
        if input is not None:
            serialized_input = json_dumps(input)
            if len(serialized_input) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                self.span.set_attribute(
                    SPAN_INPUT,
                    "Laminar: input too large to record",
                )
            else:
                self.span.set_attribute(SPAN_INPUT, serialized_input)

    def add_tags(self, tags: list[str]) -> None:
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            self.logger.warning("Tags must be a list of strings. Tags will be ignored.")
            return
        current_tags = self.tags
        if current_tags is None:
            current_tags = []
        current_tags.extend(tags)
        self.span.set_attribute(
            f"{ASSOCIATION_PROPERTIES}.tags", list(set(current_tags))
        )

    def set_tags(self, tags: list[str]) -> None:
        """Set the tags for the current span.

        Args:
            tags (list[str]): Tags to set for the span.
        """
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            self.logger.warning("Tags must be a list of strings. Tags will be ignored.")
            return
        self.span.set_attribute(f"{ASSOCIATION_PROPERTIES}.tags", list(set(tags)))

    @property
    def tags(self) -> list[str]:
        if not hasattr(self.span, "attributes"):
            self.logger.debug(
                "[LaminarSpan.tags] WARNING. Current span does not have attributes object. "
                "Perhaps, the span was created with a custom OTel SDK. Returning an empty list. "
                "Help: OpenTelemetry API does not guarantee reading attributes from a span, but OTel SDK "
                "allows it by default. Laminar SDK allows to read attributes too.",
            )
            return []
        try:
            return list(self.span.attributes.get(f"{ASSOCIATION_PROPERTIES}.tags", []))
        except Exception:
            return []

    @property
    def laminar_association_properties(self) -> dict[str, Any]:
        if not hasattr(self.span, "attributes"):
            self.logger.debug(
                "[LaminarSpan.laminar_association_properties] WARNING. Current span "
                "does not have attributes object. Perhaps, the span was created with a "
                "custom OTel SDK. Returning an empty dictionary."
                "Help: OpenTelemetry API does not guarantee reading attributes from a span, but OTel SDK "
                "allows it by default. Laminar SDK allows to read attributes too.",
            )
            return {}
        try:
            values = {}
            for key, value in self.span.attributes.items():
                if key.startswith(f"{ASSOCIATION_PROPERTIES}."):
                    if key.startswith(f"{ASSOCIATION_PROPERTIES}.metadata."):
                        meta_key = key.replace(
                            f"{ASSOCIATION_PROPERTIES}.metadata.", ""
                        )
                        try:
                            values[meta_key] = orjson.loads(value)
                        except Exception:
                            values[meta_key] = value
                    else:
                        values[key] = value
            return values
        except Exception:
            return {}


class SpanDelegationMixin:
    """Mixin providing delegation to the wrapped SDK span for standard OpenTelemetry methods."""

    span: SDKSpan

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

    def _readable_span(self) -> ReadableSpan:
        return self.span._readable_span()

    @property
    def name(self) -> str:
        return self.span.name

    @property
    def context(self) -> SpanContext:
        return self.span.context

    @property
    def start_time(self) -> int | None:
        return self.span.start_time

    @property
    def end_time(self) -> int | None:
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


class LaminarSpan(LaminarSpanInterfaceMixin, SpanDelegationMixin, Span, ReadableSpan):
    """
    Laminar's span wrapper that complies with OpenTelemetry's Span and ReadableSpan interfaces.

    We wrap the SDK span instead of inheriting from it, because OpenTelemetry discourages
    direct initialization of SdkSpan objects. Instead, we rely on the tracer to create
    the span for us, and then we wrap it in a LaminarSpan.
    """

    span: SDKSpan
    _popped: bool = False

    def __init__(self, span: SDKSpan):
        if isinstance(span, LaminarSpan):
            span = span.span
        self.logger = get_default_logger(__name__)
        self.span = span

    def end(self, end_time: int | None = None) -> None:
        self.span.end(end_time)
        if hasattr(self, "_lmnr_ctx_token") and not self._popped:
            try:
                pop_span_context()
                try:
                    current_task = asyncio.current_task()
                except Exception:
                    current_task = None
                if (
                    hasattr(self, "_lmnr_task_id")
                    and id(current_task) == self._lmnr_task_id
                ):
                    detach(self._lmnr_ctx_token)
                else:
                    self.logger.debug(
                        "Not detaching global context, not in the same context"
                    )
                self._popped = True
            except Exception:
                self.logger.debug("Failed to pop span context", exc_info=True)
        if hasattr(self, "_lmnr_isolated_ctx_token"):
            try:
                detach_context(self._lmnr_isolated_ctx_token)
            except Exception:
                self.logger.debug("Failed to detach isolated context", exc_info=True)
        if hasattr(self, "_lmnr_assoc_props_token") and self._lmnr_assoc_props_token:
            try:
                detach_context(self._lmnr_assoc_props_token)
            except Exception:
                self.logger.debug(
                    "Failed to detach association properties context", exc_info=True
                )

    def __enter__(self) -> "LaminarSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: Traceback | None,
    ) -> None:
        self.end()
