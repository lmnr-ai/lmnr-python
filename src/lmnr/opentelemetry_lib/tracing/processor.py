import uuid

from opentelemetry.sdk.trace.export import (
    SpanProcessor,
    SpanExporter,
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span
from opentelemetry.context import Context, get_value, get_current, set_value

from lmnr.opentelemetry_lib.tracing.attributes import (
    SPAN_IDS_PATH,
    SPAN_INSTRUMENTATION_SOURCE,
    SPAN_LANGUAGE_VERSION,
    SPAN_PATH,
    SPAN_SDK_VERSION,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.opentelemetry_lib.tracing.context_properties import (
    _set_association_properties_attributes,
)
from lmnr.version import PYTHON_VERSION, __version__


class LaminarSpanProcessor(SpanProcessor):
    instance: BatchSpanProcessor | SimpleSpanProcessor
    __span_id_to_path: dict[int, list[str]] = {}
    __span_id_lists: dict[int, list[str]] = {}

    def __init__(
        self,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        force_http: bool = False,
        max_export_batch_size: int = 512,
        disable_batch: bool = False,
        exporter: SpanExporter | None = None,
    ):
        self.exporter = exporter or LaminarSpanExporter(
            base_url=base_url,
            port=port,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            force_http=force_http,
        )
        self.instance = (
            SimpleSpanProcessor(self.exporter)
            if disable_batch
            else BatchSpanProcessor(
                self.exporter, max_export_batch_size=max_export_batch_size
            )
        )

    def on_start(self, span: Span, parent_context: Context | None = None):
        span_path_in_context = get_value("span_path", parent_context or get_current())
        parent_span_path = span_path_in_context or (
            self.__span_id_to_path.get(span.parent.span_id) if span.parent else None
        )
        parent_span_ids_path = (
            self.__span_id_lists.get(span.parent.span_id, []) if span.parent else []
        )
        span_path = parent_span_path + [span.name] if parent_span_path else [span.name]
        span_ids_path = parent_span_ids_path + [
            str(uuid.UUID(int=span.get_span_context().span_id))
        ]
        span.set_attribute(SPAN_PATH, span_path)
        span.set_attribute(SPAN_IDS_PATH, span_ids_path)
        set_value("span_path", span_path, get_current())
        self.__span_id_to_path[span.get_span_context().span_id] = span_path
        self.__span_id_lists[span.get_span_context().span_id] = span_ids_path

        span.set_attribute(SPAN_INSTRUMENTATION_SOURCE, "python")
        span.set_attribute(SPAN_SDK_VERSION, __version__)
        span.set_attribute(SPAN_LANGUAGE_VERSION, f"python@{PYTHON_VERSION}")

        association_properties = get_value("association_properties")
        if association_properties is not None:
            _set_association_properties_attributes(span, association_properties)
        self.instance.on_start(span, parent_context)

    def on_end(self, span: Span):
        self.instance.on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.instance.force_flush(timeout_millis)

    def shutdown(self):
        self.instance.shutdown()

    def clear(self):
        self.__span_id_to_path = {}
        self.__span_id_lists = {}
