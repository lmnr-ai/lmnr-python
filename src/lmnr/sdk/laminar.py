from contextlib import contextmanager
from contextvars import Context
import warnings
from lmnr.opentelemetry_lib import TracerManager
from lmnr.opentelemetry_lib.tracing import TracerWrapper, get_current_context
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_USER_ID_KEY,
    attach_context,
    get_event_attributes_from_context,
)
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    SPAN_IDS_PATH,
    SPAN_PATH,
    USER_ID,
    Attributes,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib import MAX_MANUAL_SPAN_PAYLOAD_SIZE
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import get_otel_env_var

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.trace import INVALID_TRACE_ID, Span, Status, StatusCode, use_span
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.util.types import AttributeValue

from typing import Any, Iterator, Literal

import datetime
import logging
import os
import re
import uuid

from lmnr.opentelemetry_lib.tracing.attributes import (
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    TRACE_TYPE,
)
from lmnr.sdk.utils import from_env, is_otel_attribute_value_type

from .log import VerboseColorfulFormatter

from .types import (
    LaminarSpanContext,
    SessionRecordingOptions,
    TraceType,
)


class Laminar:
    __project_api_key: str | None = None
    __initialized: bool = False
    __base_http_url: str | None = None

    @classmethod
    def initialize(
        cls,
        project_api_key: str | None = None,
        base_url: str | None = None,
        base_http_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
        instruments: (
            list[Instruments] | set[Instruments] | tuple[Instruments] | None
        ) = None,
        disabled_instruments: (
            list[Instruments] | set[Instruments] | tuple[Instruments] | None
        ) = None,
        disable_batch: bool = False,
        max_export_batch_size: int | None = None,
        export_timeout_seconds: int | None = None,
        set_global_tracer_provider: bool = True,
        otel_logger_level: int = logging.ERROR,
        session_recording_options: SessionRecordingOptions | None = None,
        force_http: bool = False,
    ):
        """Initialize Laminar context across the application.
        This method must be called before using any other Laminar methods or
        decorators.

        Args:
            project_api_key (str | None, optional): Laminar project api key.\
                You can generate one by going to the projects settings page on\
                the Laminar dashboard. If not specified, we will try to read\
                from the LMNR_PROJECT_API_KEY environment variable in os.environ\
                or in .env file. Defaults to None.
            base_url (str | None, optional): Laminar API url. Do NOT include\
                the port number, use `http_port` and `grpc_port`. If not\
                specified, defaults to https://api.lmnr.ai.
            base_http_url (str | None, optional): Laminar API http url. Only\
                set this if your Laminar backend HTTP is proxied through a\
                different host. If not specified, defaults to\
                https://api.lmnr.ai.
            http_port (int | None, optional): Laminar API http port. If not\
                specified, defaults to 443.
            grpc_port (int | None, optional): Laminar API grpc port. If not\
                specified, defaults to 8443.
            instruments (set[Instruments] | list[Instruments] | tuple[Instruments] | None, optional):
                Instruments to enable. Defaults to all instruments. You can pass\
                an empty set to disable all instruments. Read more:\
                https://docs.lmnr.ai/tracing/automatic-instrumentation
            disabled_instruments (set[Instruments] | list[Instruments] | tuple[Instruments] | None, optional):
                Instruments to disable. Defaults to None.
            disable_batch (bool, optional): If set to True, spans will be sent\
                immediately to the backend. Useful for debugging, but may cause\
                performance overhead in production. Defaults to False.
            max_export_batch_size (int | None, optional): Maximum number of spans\
                to export in a single batch. If not specified, defaults to 64\
                (lower than the OpenTelemetry default of 512). If you see\
                `DEADLINE_EXCEEDED` errors, try reducing this value.
            export_timeout_seconds (int | None, optional): Timeout for the OTLP\
                exporter. Defaults to 30 seconds (unlike the OpenTelemetry\
                default of 10 seconds). Defaults to None.
            set_global_tracer_provider (bool, optional): If set to True, the\
                Laminar tracer provider will be set as the global tracer provider.\
                OpenTelemetry allows only one tracer provider per app, so set this\
                to False, if you are using another tracing library. Setting this to\
                False may break some external instrumentations, e.g. LiteLLM.\
                Defaults to True.
            otel_logger_level (int, optional): OpenTelemetry logger level. Defaults\
                to logging.ERROR.
            session_recording_options (SessionRecordingOptions | None, optional): Options\
                for browser session recording. Currently supports 'mask_input'\
                (bool) to control whether input fields are masked during recording.\
                Defaults to None (uses default masking behavior).
            force_http (bool, optional): If set to True, the HTTP OTEL exporter will be\
                used instead of the gRPC OTEL exporter. Defaults to False.
        Raises:
            ValueError: If project API key is not set
        """
        if cls.is_initialized():
            cls.__logger.info(
                "Laminar is already initialized. Skipping initialization."
            )
            return

        cls.__project_api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")

        if (
            not cls.__project_api_key
            and not get_otel_env_var("ENDPOINT")
            and not get_otel_env_var("HEADERS")
        ):
            raise ValueError(
                "Please initialize the Laminar object with"
                " your project API key or set the LMNR_PROJECT_API_KEY"
                " environment variable in your environment or .env file"
            )

        cls._initialize_logger()

        url = base_url or from_env("LMNR_BASE_URL")
        if url:
            url = url.rstrip("/")
            if not url.startswith("http:") and not url.startswith("https:"):
                url = f"https://{url}"
            if match := re.search(r":(\d{1,5})$", url):
                url = url[: -len(match.group(0))]
                cls.__logger.info(f"Ignoring port in base URL: {match.group(1)}")
        http_url = base_http_url or url or "https://api.lmnr.ai"
        if not http_url.startswith("http:") and not http_url.startswith("https:"):
            http_url = f"https://{http_url}"
        if match := re.search(r":(\d{1,5})$", http_url):
            http_url = http_url[: -len(match.group(0))]
            if http_port is None:
                cls.__logger.info(f"Using HTTP port from base URL: {match.group(1)}")
                http_port = int(match.group(1))
            else:
                cls.__logger.info(f"Using HTTP port passed as an argument: {http_port}")

        cls.__initialized = True
        cls.__base_http_url = f"{http_url}:{http_port or 443}"

        if not os.getenv("OTEL_ATTRIBUTE_COUNT_LIMIT"):
            # each message is at least 2 attributes: role and content,
            # but the default attribute limit is 128, so raise it
            os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "10000"

        TracerManager.init(
            base_url=url,
            http_port=http_port or 443,
            port=grpc_port or 8443,
            project_api_key=cls.__project_api_key,
            instruments=set(instruments) if instruments is not None else None,
            block_instruments=(
                set(disabled_instruments) if disabled_instruments is not None else None
            ),
            disable_batch=disable_batch,
            max_export_batch_size=max_export_batch_size,
            timeout_seconds=export_timeout_seconds,
            set_global_tracer_provider=set_global_tracer_provider,
            otel_logger_level=otel_logger_level,
            session_recording_options=session_recording_options,
            force_http=force_http,
        )

    @classmethod
    def is_initialized(cls):
        """Check if Laminar is initialized. A utility to make sure other
        methods are called after initialization.

        Returns:
            bool: True if Laminar is initialized, False otherwise
        """
        return cls.__initialized

    @classmethod
    def _initialize_logger(cls):
        cls.__logger = logging.getLogger(__name__)
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(VerboseColorfulFormatter())
        cls.__logger.addHandler(console_log_handler)

    @classmethod
    def event(
        cls,
        name: str,
        attributes: dict[str, AttributeValue] | None = None,
        timestamp: datetime.datetime | int | None = None,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Associate an event with the current span. This is a wrapper around
        `span.add_event()` that adds the event to the current span.

        Args:
            name (str): event name
            attributes (dict[str, AttributeValue] | None, optional): event attributes.
                Defaults to None.
            timestamp (datetime.datetime | int | None, optional): If int, must\
                be epoch nanoseconds. If not specified, relies on the underlying\
                OpenTelemetry implementation. Defaults to None.
        """
        if not cls.is_initialized():
            return

        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)

        extra_attributes = get_event_attributes_from_context()

        # override the user_id and session_id from the context with the ones
        # passed as arguments
        if user_id is not None:
            extra_attributes["lmnr.event.user_id"] = user_id
        if session_id is not None:
            extra_attributes["lmnr.event.session_id"] = session_id

        current_span = trace.get_current_span(context=get_current_context())
        if current_span == trace.INVALID_SPAN:
            with cls.start_as_current_span(name) as span:
                span.add_event(
                    name, {**(attributes or {}), **extra_attributes}, timestamp
                )
            return

        current_span.add_event(
            name, {**(attributes or {}), **extra_attributes}, timestamp
        )

    @classmethod
    @contextmanager
    def start_as_current_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
        context: Context | None = None,
        labels: list[str] | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Span]:
        """Start a new span as the current span. Useful for manual
        instrumentation. If `span_type` is set to `"LLM"`, you should report
        usage and response attributes manually. See `Laminar.set_span_attributes`
        for more information.

        Usage example:
        ```python
        with Laminar.start_as_current_span("my_span", input="my_input") as span:
            await my_async_function()
            Laminar.set_span_output("my_output")`
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            labels (list[str] | None, optional): [DEPRECATED] Use tags\
                instead. Labels to set for the span. Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
        """

        if not cls.is_initialized():
            yield trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=RandomIdGenerator().generate_trace_id(),
                    span_id=RandomIdGenerator().generate_span_id(),
                    is_remote=False,
                )
            )
            return

        wrapper = TracerWrapper()

        with get_tracer_with_context() as (tracer, isolated_context):
            ctx = context or isolated_context
            path = []
            span_ids_path = []
            if parent_span_context is not None:
                if isinstance(parent_span_context, (dict, str)):
                    try:
                        laminar_span_context = LaminarSpanContext.deserialize(
                            parent_span_context
                        )
                        path = laminar_span_context.span_path
                        span_ids_path = laminar_span_context.span_ids_path
                    except Exception:
                        cls.__logger.warning(
                            f"`start_as_current_span` Could not deserialize parent_span_context: {parent_span_context}. "
                            "Will use it as is."
                        )
                        laminar_span_context = parent_span_context
                else:
                    laminar_span_context = parent_span_context
                    if isinstance(laminar_span_context, LaminarSpanContext):
                        path = laminar_span_context.span_path
                        span_ids_path = laminar_span_context.span_ids_path
                span_context = LaminarSpanContext.try_to_otel_span_context(
                    laminar_span_context, cls.__logger
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            ctx_token = context_api.attach(ctx)
            label_props = {}
            try:
                if labels:
                    warnings.warn(
                        "`Laminar.start_as_current_span` `labels` is deprecated. Use `tags` instead.",
                        DeprecationWarning,
                    )
                    label_props = {f"{ASSOCIATION_PROPERTIES}.labels": labels}
            except Exception:
                cls.__logger.warning(
                    f"`start_as_current_span` Could not set labels: {labels}. "
                    "They will be propagated to the next span."
                )
            tag_props = {}
            if tags:
                if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
                    tag_props = {f"{ASSOCIATION_PROPERTIES}.tags": tags}
                else:
                    cls.__logger.warning(
                        f"`start_as_current_span` Could not set tags: {tags}. Tags must be a list of strings. "
                        "Tags will be ignored."
                    )

            with tracer.start_as_current_span(
                name,
                context=ctx,
                attributes={
                    SPAN_TYPE: span_type,
                    PARENT_SPAN_PATH: path,
                    PARENT_SPAN_IDS_PATH: span_ids_path,
                    **(label_props),
                    **(tag_props),
                },
            ) as span:
                wrapper.push_span_context(span)
                if input is not None:
                    serialized_input = json_dumps(input)
                    if len(serialized_input) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                        span.set_attribute(
                            SPAN_INPUT,
                            "Laminar: input too large to record",
                        )
                    else:
                        span.set_attribute(
                            SPAN_INPUT,
                            serialized_input,
                        )
                yield span

            wrapper.pop_span_context()
            try:
                context_api.detach(ctx_token)
            except Exception:
                pass

    @classmethod
    def start_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
        context: Context | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        labels: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ):
        """Start a new span. Useful for manual instrumentation.
        If `span_type` is set to `"LLM"`, you should report usage and response
        attributes manually. See `Laminar.set_span_attributes` for more
        information.

        Note that spans started with this method must be ended manually.
        In addition, they must be ended in LIFO order, e.g.
        span1 = Laminar.start_span("span1")
        span2 = Laminar.start_span("span2")
        span2.end()
        span1.end()
        Otherwise, the behavior is undefined.

        Usage example:
        ```python
        from src.lmnr import Laminar
        def foo(span):
            with Laminar.use_span(span):
                with Laminar.start_as_current_span("foo_inner"):
                    some_function()

        def bar():
            with Laminar.use_span(span):
                openai_client.chat.completions.create()

        span = Laminar.start_span("outer")
        foo(span)
        bar(span)
        # IMPORTANT: End the span manually
        span.end()

        # Results in:
        # | outer
        # |   | foo
        # |   |   | foo_inner
        # |   | bar
        # |   |   | openai.chat
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
            labels (dict[str, str] | None, optional): [DEPRECATED] Use tags\
                instead. Labels to set for the span. Defaults to None.
        """
        if not cls.is_initialized():
            return trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=RandomIdGenerator().generate_trace_id(),
                    span_id=RandomIdGenerator().generate_span_id(),
                    is_remote=False,
                )
            )

        with get_tracer_with_context() as (tracer, isolated_context):
            ctx = context or isolated_context
            path = []
            span_ids_path = []
            if parent_span_context is not None:
                if isinstance(parent_span_context, (dict, str)):
                    try:
                        laminar_span_context = LaminarSpanContext.deserialize(
                            parent_span_context
                        )
                        path = laminar_span_context.span_path
                        span_ids_path = laminar_span_context.span_ids_path
                    except Exception:
                        cls.__logger.warning(
                            f"`start_span` Could not deserialize parent_span_context: {parent_span_context}. "
                            "Will use it as is."
                        )
                        laminar_span_context = parent_span_context
                else:
                    laminar_span_context = parent_span_context
                    if isinstance(laminar_span_context, LaminarSpanContext):
                        path = laminar_span_context.span_path
                        span_ids_path = laminar_span_context.span_ids_path
                span_context = LaminarSpanContext.try_to_otel_span_context(
                    laminar_span_context, cls.__logger
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            label_props = {}
            try:
                if labels:
                    warnings.warn(
                        "`Laminar.start_span` `labels` is deprecated. Use `tags` instead.",
                        DeprecationWarning,
                    )
                    label_props = {
                        f"{ASSOCIATION_PROPERTIES}.labels": json_dumps(labels)
                    }
            except Exception:
                cls.__logger.warning(
                    f"`start_span` Could not set labels: {labels}. They will be "
                    "propagated to the next span."
                )
            tag_props = {}
            if tags:
                if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
                    tag_props = {f"{ASSOCIATION_PROPERTIES}.tags": tags}
                else:
                    cls.__logger.warning(
                        f"`start_span` Could not set tags: {tags}. Tags must be a list of strings. "
                        + "Tags will be ignored."
                    )

            span = tracer.start_span(
                name,
                context=ctx,
                attributes={
                    SPAN_TYPE: span_type,
                    PARENT_SPAN_PATH: path,
                    PARENT_SPAN_IDS_PATH: span_ids_path,
                    **(label_props),
                    **(tag_props),
                },
            )

            if input is not None:
                serialized_input = json_dumps(input)
                if len(serialized_input) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                    span.set_attribute(
                        SPAN_INPUT,
                        "Laminar: input too large to record",
                    )
                else:
                    span.set_attribute(
                        SPAN_INPUT,
                        serialized_input,
                    )
            return span

    @classmethod
    @contextmanager
    def use_span(
        cls,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Iterator[Span]:
        """Use a span as the current span. Useful for manual instrumentation.

        Fully copies the implementation of `use_span` from opentelemetry.trace
        and replaces the context API with Laminar's isolated context.

        Args:
            span: The span that should be activated in the current context.
            end_on_exit: Whether to end the span automatically when leaving the
                context manager scope.
            record_exception: Whether to record any exceptions raised within the
                context as error event on the span.
            set_status_on_exception: Only relevant if the returned span is used
                in a with/context manager. Defines whether the span status will
                be automatically set to ERROR when an uncaught exception is
                raised in the span with block. The span status won't be set by
                this mechanism if it was previously set manually.
        """
        if not cls.is_initialized():
            with use_span(
                span, end_on_exit, record_exception, set_status_on_exception
            ) as s:
                yield s
            return

        wrapper = TracerWrapper()

        try:
            context = wrapper.push_span_context(span)
            # Some auto-instrumentations are not under our control, so they
            # don't have access to our isolated context. We attach the context
            # to the OTEL global context, so that spans know their parent
            # span and trace_id.
            context_token = context_api.attach(context)
            try:
                yield span
            finally:
                context_api.detach(context_token)
                wrapper.pop_span_context()

        # Record only exceptions that inherit Exception class but not BaseException, because
        # classes that directly inherit BaseException are not technically errors, e.g. GeneratorExit.
        # See https://github.com/open-telemetry/opentelemetry-python/issues/4484
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if isinstance(span, Span) and span.is_recording():
                # Record the exception as an event
                if record_exception:
                    span.record_exception(
                        exc, attributes=get_event_attributes_from_context()
                    )

                # Set status in case exception was raised
                if set_status_on_exception:
                    span.set_status(
                        Status(
                            status_code=StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )

            # This causes parent spans to set their status to ERROR and to record
            # an exception as an event if a child span raises an exception even if
            # such child span was started with both record_exception and
            # set_status_on_exception attributes set to False.
            raise

        finally:
            if end_on_exit:
                span.end()

    @classmethod
    def start_active_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
        context: Context | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        tags: list[str] | None = None,
    ) -> Span:
        """Start a span and mark it as active within the current context.
        All spans started after this one will be children of this span.
        Useful for manual instrumentation. Must be ended manually.
        If `span_type` is set to `"LLM"`, you should report usage and response
        attributes manually. See `Laminar.set_span_attributes` for more
        information. Returns the span object.

        Note that ending the started span in a different async context yields
        unexpected results. When propagating spans across different async or
        threading contexts, it is recommended to either:
        - Make sure to start and end the span in the same async context or thread, or
        - Use `Laminar.start_span` + `Laminar.use_span` where possible.

        Note that spans started with this method must be ended manually.
        In addition, they must be ended in LIFO order, e.g.
        span1 = Laminar.start_active_span("span1")
        span2 = Laminar.start_active_span("span2")
        span2.end()
        span1.end()
        Otherwise, the behavior is undefined.

        Usage example:
        ```python
        from src.lmnr import Laminar, observe

        @observe()
        def foo():
            with Laminar.start_as_current_span("foo_inner"):
                some_function()
        
        @observe()
        def bar():
            openai_client.chat.completions.create()
        
        span = Laminar.start_active_span("outer")
        foo()
        bar()
        # IMPORTANT: End the span manually
        span.end()
        
        # Results in:
        # | outer
        # |   | foo
        # |   |   | foo_inner
        # |   | bar
        # |   |   | openai.chat
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
        """
        span = cls.start_span(
            name=name,
            input=input,
            span_type=span_type,
            context=context,
            parent_span_context=parent_span_context,
            tags=tags,
        )
        if not cls.is_initialized():
            return span
        wrapper = TracerWrapper()
        context = wrapper.push_span_context(span)
        context_token = context_api.attach(context)
        span._lmnr_ctx_token = context_token
        return span

    @classmethod
    def set_span_output(cls, output: Any = None):
        """Set the output of the current span. Useful for manual
        instrumentation.

        Args:
            output (Any, optional): output of the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
        """
        span = trace.get_current_span(context=get_current_context())
        if output is not None and span != trace.INVALID_SPAN:
            serialized_output = json_dumps(output)
            if len(serialized_output) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                span.set_attribute(
                    SPAN_OUTPUT,
                    "Laminar: output too large to record",
                )
            else:
                span.set_attribute(SPAN_OUTPUT, serialized_output)

    @classmethod
    def set_span_attributes(
        cls,
        attributes: dict[Attributes | str, Any],
    ):
        """Set attributes for the current span. Useful for manual
        instrumentation.
        Example:
        ```python
        with Laminar.start_as_current_span(
            name="my_span_name", input=input["messages"], span_type="LLM"
        ):
            response = await my_custom_call_to_openai(input)
            Laminar.set_span_output(response["choices"][0]["message"]["content"])
            Laminar.set_span_attributes({
                Attributes.PROVIDER: 'openai',
                Attributes.REQUEST_MODEL: input["model"],
                Attributes.RESPONSE_MODEL: response["model"],
                Attributes.INPUT_TOKEN_COUNT: response["usage"]["prompt_tokens"],
                Attributes.OUTPUT_TOKEN_COUNT: response["usage"]["completion_tokens"],
            })
            # ...
        ```

        Args:
            attributes (dict[Attributes | str, Any]): attributes to set for the span
        """
        span = trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            return

        for key, value in attributes.items():
            if isinstance(key, Attributes):
                key = key.value
            if not is_otel_attribute_value_type(value):
                span.set_attribute(key, json_dumps(value))
            else:
                span.set_attribute(key, value)

    @classmethod
    def get_laminar_span_context(
        cls, span: trace.Span | None = None
    ) -> LaminarSpanContext | None:
        """Get the laminar span context for a given span.
        If no span is provided, the current active span will be used.
        """
        if not cls.is_initialized():
            return None

        span = span or trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            return None
        return LaminarSpanContext(
            trace_id=uuid.UUID(int=span.get_span_context().trace_id),
            span_id=uuid.UUID(int=span.get_span_context().span_id),
            is_remote=span.get_span_context().is_remote,
            span_path=span.attributes.get(SPAN_PATH, []),
            span_ids_path=span.attributes.get(SPAN_IDS_PATH, []),
        )

    @classmethod
    def get_laminar_span_context_dict(
        cls, span: trace.Span | None = None
    ) -> dict | None:
        span_context = cls.get_laminar_span_context(span)
        if span_context is None:
            return None
        return span_context.model_dump()

    @classmethod
    def serialize_span_context(cls, span: trace.Span | None = None) -> str | None:
        """Get the laminar span context for a given span as a string.
        If no span is provided, the current active span will be used.

        This is useful for continuing a trace across services.

        Example:
        ```python
        # service A:
        with Laminar.start_as_current_span("service_a"):
            span_context = Laminar.serialize_span_context()
            # send span_context to service B
            call_service_b(request, headers={"laminar-span-context": span_context})

        # service B:
        def call_service_b(request, headers):
            span_context = Laminar.deserialize_span_context(headers["laminar-span-context"])
            with Laminar.start_as_current_span("service_b", parent_span_context=span_context):
                # rest of the function
                pass
        ```

        This will result in a trace like:
        ```
        service_a
          service_b
        ```
        """
        span_context = cls.get_laminar_span_context(span)
        if span_context is None:
            return None
        return str(span_context)

    @classmethod
    def deserialize_span_context(cls, span_context: dict | str) -> LaminarSpanContext:
        return LaminarSpanContext.deserialize(span_context)

    @classmethod
    def flush(cls) -> bool:
        """Flush the internal tracer.

        Returns:
            bool: True if the tracer was flushed, False otherwise
            (e.g. no tracer or timeout).
        """
        if not cls.is_initialized():
            return False
        return TracerManager.flush()

    @classmethod
    def force_flush(cls):
        """Force flush the internal tracer. WARNING: Any active spans are
        removed from context; that is, spans started afterwards will start
        a new trace.

        Actually shuts down the span processor and re-initializes it as long
        as it is a LaminarSpanProcessor. This is not recommended in production
        workflows, but is useful at the end of Lambda functions, where a regular
        flush might be killed by the Lambda runtime, because the actual export
        inside it runs in a background thread.
        """
        if not cls.is_initialized():
            return
        TracerManager.force_reinit_processor()

    @classmethod
    def shutdown(cls):
        if cls.is_initialized():
            TracerManager.shutdown()
            cls.__initialized = False

    @classmethod
    def set_span_tags(cls, tags: list[str]):
        """Set the tags for the current span.

        Args:
            tags (list[str]): Tags to set for the span.
        """
        if not cls.is_initialized():
            return

        span = trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set tags on")
            return
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            cls.__logger.warning(
                "Tags must be a list of strings. Tags will be ignored."
            )
            return
        # list(set(tags)) to deduplicate tags
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.tags", list(set(tags)))

    @classmethod
    def set_trace_session_id(cls, session_id: str | None = None):
        """Set the session id for the current trace.
        Overrides any existing session id.

        Args:
            session_id (str | None, optional): Custom session id. Defaults to None.
        """
        if not cls.is_initialized():
            return

        context = get_current_context()
        context = context_api.set_value(CONTEXT_SESSION_ID_KEY, session_id, context)
        attach_context(context)

        span = trace.get_current_span(context=context)
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set session id on")
            return
        if session_id is not None:
            span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}", session_id)

    @classmethod
    def set_trace_user_id(cls, user_id: str | None = None):
        """Set the user id for the current trace.
        Overrides any existing user id.

        Args:
            user_id (str | None, optional): Custom user id. Defaults to None.
        """
        if not cls.is_initialized():
            return

        context = get_current_context()
        context = context_api.set_value(CONTEXT_USER_ID_KEY, user_id, context)
        attach_context(context)

        span = trace.get_current_span(context=context)
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set user id on")
            return
        if user_id is not None:
            span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{USER_ID}", user_id)

    @classmethod
    def set_trace_metadata(cls, metadata: dict[str, AttributeValue]):
        """Set the metadata for the current trace.

        Args:
            metadata (dict[str, AttributeValue]): Metadata to set for the trace.
        """
        if not cls.is_initialized():
            return

        span = trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set metadata on")
            return
        for key, value in metadata.items():
            if is_otel_attribute_value_type(value):
                span.set_attribute(f"{ASSOCIATION_PROPERTIES}.metadata.{key}", value)
            else:
                span.set_attribute(
                    f"{ASSOCIATION_PROPERTIES}.metadata.{key}", json_dumps(value)
                )

    @classmethod
    def get_base_http_url(cls):
        return cls.__base_http_url

    @classmethod
    def get_project_api_key(cls):
        return cls.__project_api_key

    @classmethod
    def get_trace_id(cls) -> uuid.UUID | None:
        """Get the trace id for the current active span represented as a UUID.
        Returns None if there is no active span.

        Returns:
            uuid.UUID | None: The trace id for the current span, or None if\
            there is no active span.
        """
        trace_id = (
            trace.get_current_span(context=get_current_context())
            .get_span_context()
            .trace_id
        )
        if trace_id == INVALID_TRACE_ID:
            return None
        return uuid.UUID(int=trace_id)

    @classmethod
    def _headers(cls):
        assert cls.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + cls.__project_api_key,
            "Content-Type": "application/json",
        }

    @classmethod
    def _set_trace_type(
        cls,
        trace_type: TraceType,
    ):
        """Set the trace_type for the current span and the context
        Args:
            trace_type (TraceType): Type of the trace
        """
        if not cls.is_initialized():
            return

        span = trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set trace type on")
            return
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}", trace_type.value)
