from contextlib import contextmanager
from contextvars import Context
from lmnr.openllmetry_sdk import Traceloop
from lmnr.openllmetry_sdk.instruments import Instruments
from lmnr.openllmetry_sdk.tracing import get_tracer
from lmnr.openllmetry_sdk.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    Attributes,
    SPAN_TYPE,
    OVERRIDE_PARENT_SPAN,
)
from lmnr.openllmetry_sdk.config import MAX_MANUAL_SPAN_PAYLOAD_SIZE
from lmnr.openllmetry_sdk.decorators.base import json_dumps
from opentelemetry import context as context_api, trace
from opentelemetry.context import attach, detach
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
    Compression,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.util.types import AttributeValue

from typing import Any, Awaitable, Literal, Optional, Set, Union

import atexit
import copy
import datetime
import dotenv
import json
import logging
import os
import random
import re
import uuid
import warnings

from lmnr.openllmetry_sdk.tracing.attributes import (
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    TRACE_TYPE,
)
from lmnr.openllmetry_sdk.tracing.tracing import (
    get_association_properties,
    remove_association_properties,
    set_association_properties,
    update_association_properties,
)
from lmnr.sdk.client import LaminarClient

from .log import VerboseColorfulFormatter

from .types import (
    LaminarSpanContext,
    PipelineRunResponse,
    NodeInput,
    SemanticSearchResponse,
    TraceType,
    TracingLevel,
)


class Laminar:
    __base_http_url: str
    __base_grpc_url: str
    __project_api_key: Optional[str] = None
    __env: dict[str, str] = {}
    __initialized: bool = False

    @classmethod
    def initialize(
        cls,
        project_api_key: Optional[str] = None,
        env: dict[str, str] = {},
        base_url: Optional[str] = None,
        http_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        instruments: Optional[Set[Instruments]] = None,
        disable_batch: bool = False,
        max_export_batch_size: Optional[int] = None,
        export_timeout_seconds: Optional[int] = None,
    ):
        """Initialize Laminar context across the application.
        This method must be called before using any other Laminar methods or
        decorators.

        Args:
            project_api_key (Optional[str], optional): Laminar project api key.\
                            You can generate one by going to the projects\
                            settings page on the Laminar dashboard.\
                            If not specified, it will try to read from the\
                            LMNR_PROJECT_API_KEY environment variable\
                            in os.environ or in .env file.
                            Defaults to None.
            env (dict[str, str], optional): Default environment passed to\
                            `run` requests, unless overriden at request time.\
                            Usually, model provider keys are stored here.
                            Defaults to {}.
            base_url (Optional[str], optional): Laminar API url. Do NOT include\
                            the port number, use `http_port` and `grpc_port`.\
                            If not specified, defaults to https://api.lmnr.ai.
            http_port (Optional[int], optional): Laminar API http port.\
                            If not specified, defaults to 443.
            grpc_port (Optional[int], optional): Laminar API grpc port.\
                            If not specified, defaults to 8443.
            instruments (Optional[Set[Instruments]], optional): Instruments to\
                        enable. Defaults to all instruments. You can pass\
                        an empty set to disable all instruments. Read more:\
                        https://docs.lmnr.ai/tracing/automatic-instrumentation
            disable_batch (bool, optional): If set to True, spans will be sent\
                        immediately to the backend. Useful for debugging, but\
                        may cause performance overhead in production.
                        Defaults to False.
            export_timeout_seconds (Optional[int], optional): Timeout for the OTLP\
                        exporter. Defaults to 30 seconds (unlike the\
                        OpenTelemetry default of 10 seconds).
                        Defaults to None.

        Raises:
            ValueError: If project API key is not set
        """
        cls.__project_api_key = project_api_key or os.environ.get(
            "LMNR_PROJECT_API_KEY"
        )
        if not cls.__project_api_key:
            dotenv_path = dotenv.find_dotenv(usecwd=True)
            cls.__project_api_key = dotenv.get_key(
                dotenv_path=dotenv_path, key_to_get="LMNR_PROJECT_API_KEY"
            )
        if not cls.__project_api_key:
            raise ValueError(
                "Please initialize the Laminar object with"
                " your project API key or set the LMNR_PROJECT_API_KEY"
                " environment variable in your environment or .env file"
            )
        url = re.sub(r"/$", "", base_url or "https://api.lmnr.ai")
        if re.search(r":\d{1,5}$", url):
            raise ValueError(
                "Please provide the `base_url` without the port number. "
                "Use the `http_port` and `grpc_port` arguments instead."
            )
        cls.__base_http_url = f"{url}:{http_port or 443}"
        cls.__base_grpc_url = f"{url}:{grpc_port or 8443}"

        cls.__env = env
        cls.__initialized = True
        cls._initialize_logger()
        LaminarClient.initialize(
            base_url=cls.__base_http_url,
            project_api_key=cls.__project_api_key,
        )
        atexit.register(LaminarClient.shutdown)
        if not os.environ.get("OTEL_ATTRIBUTE_COUNT_LIMIT"):
            # each message is at least 2 attributes: role and content,
            # but the default attribute limit is 128, so raise it
            os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "10000"

        # if not is_latest_version():
        #     cls.__logger.warning(
        #         "You are using an older version of the Laminar SDK. "
        #         f"Latest version: {get_latest_pypi_version()}, current version: {SDK_VERSION}.\n"
        #         "Please update to the latest version by running "
        #         "`pip install --upgrade lmnr`."
        #     )

        Traceloop.init(
            base_http_url=cls.__base_http_url,
            project_api_key=cls.__project_api_key,
            exporter=OTLPSpanExporter(
                endpoint=cls.__base_grpc_url,
                headers={"authorization": f"Bearer {cls.__project_api_key}"},
                compression=Compression.Gzip,
                # default timeout is 10 seconds, increase it to 30 seconds
                timeout=export_timeout_seconds or 30,
            ),
            instruments=instruments,
            disable_batch=disable_batch,
            max_export_batch_size=max_export_batch_size,
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
    def run(
        cls,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        parent_span_id: Optional[uuid.UUID] = None,
        trace_id: Optional[uuid.UUID] = None,
    ) -> Union[PipelineRunResponse, Awaitable[PipelineRunResponse]]:
        """Runs the pipeline with the given inputs. If called from an async
        function, must be awaited.

        Args:
            pipeline (str): name of the Laminar pipeline.\
                The pipeline must have a target version set.
            inputs (dict[str, NodeInput]):
                inputs to the endpoint's target pipeline.\
                Keys in the dictionary must match input node names
            env (dict[str, str], optional):
                Environment variables for the pipeline execution.
                Defaults to {}.
            metadata (dict[str, str], optional):
                any custom metadata to be stored with execution trace.
                Defaults to {}.
            parent_span_id (Optional[uuid.UUID], optional): parent span id for\
                the resulting span.
                Defaults to None.
            trace_id (Optional[uuid.UUID], optional): trace id for the\
                resulting trace.
                Defaults to None.

        Returns:
            PipelineRunResponse: response object containing the outputs

        Raises:
            ValueError: if project API key is not set
            PipelineRunError: if the endpoint run fails
        """
        return LaminarClient.run_pipeline(
            pipeline=pipeline,
            inputs=inputs,
            env=env or cls.__env,
            metadata=metadata,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
        )

    @classmethod
    def semantic_search(
        cls,
        query: str,
        dataset_id: uuid.UUID,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResponse:
        """Perform a semantic search on a dataset. If called from an async
        function, must be awaited.

        Args:
            query (str): query string to search by
            dataset_id (uuid.UUID): id of the dataset to search in
            limit (Optional[int], optional): maximum number of results to\
                return. Defaults to None.
            threshold (Optional[float], optional): minimum score for a result\
                to be returned. Defaults to None.

        Returns:
            SemanticSearchResponse: response object containing the search results sorted by score in descending order
        """
        return LaminarClient.semantic_search(
            query=query,
            dataset_id=dataset_id,
            limit=limit,
            threshold=threshold,
        )

    @classmethod
    def event(
        cls,
        name: str,
        value: Optional[AttributeValue] = None,
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Associate an event with the current span. If using manual\
        instrumentation, use raw OpenTelemetry `span.add_event()` instead.\
       `value` will be saved as a `lmnr.event.value` attribute.

        Args:
            name (str): event name
            value (Optional[AttributeValue]): event value. Must be a primitive\
                            type. Boolean true is assumed in the backend if\
                            `value` is None.
                            Defaults to None.
            timestamp (Optional[Union[datetime.datetime, int]], optional):\
                            If int, must be epoch nanoseconds. If not\
                            specified, relies on the underlying OpenTelemetry\
                            implementation. Defaults to None.
        """
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)

        event = {
            "lmnr.event.type": "default",
        }
        if value is not None:
            event["lmnr.event.value"] = value

        current_span = trace.get_current_span()
        if current_span == trace.INVALID_SPAN:
            cls.__logger.warning(
                "`Laminar().event()` called outside of span context. "
                f"Event '{name}' will not be recorded in the trace. "
                "Make sure to annotate the function with a decorator"
            )
            return

        current_span.add_event(name, event, timestamp)

    @classmethod
    @contextmanager
    def start_as_current_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Union[
            Literal["DEFAULT"], Literal["LLM"], Literal["TOOL"]
        ] = "DEFAULT",
        context: Optional[Context] = None,
        labels: Optional[list[str]] = None,
        parent_span_context: Optional[LaminarSpanContext] = None,
        # deprecated, use parent_span_context instead
        trace_id: Optional[uuid.UUID] = None,
    ):
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
            span_type (Union[Literal["DEFAULT"], Literal["LLM"]], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Optional[Context], optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (Optional[LaminarSpanContext], optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            labels (Optional[list[str]], optional): labels to set for the\
                span. Defaults to None.
            trace_id (Optional[uuid.UUID], optional): [Deprecated] override\
                the trace id for the span. If not provided, use the current\
                trace id. Defaults to None.
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

        with get_tracer() as tracer:
            ctx = context or context_api.get_current()
            if trace_id is not None:
                warnings.warn(
                    "trace_id provided to `Laminar.start_as_current_span`"
                    " is deprecated, use parent_span_context instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if parent_span_context is not None:
                span_context = LaminarSpanContext.try_to_otel_span_context(
                    parent_span_context, cls.__logger
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            elif trace_id is not None and isinstance(trace_id, uuid.UUID):
                span_context = trace.SpanContext(
                    trace_id=int(trace_id),
                    span_id=random.getrandbits(64),
                    is_remote=False,
                    trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            ctx_token = attach(ctx)
            label_props = {}
            try:
                if labels:
                    label_props = {f"{ASSOCIATION_PROPERTIES}.labels": labels}
            except Exception:
                cls.__logger.warning(
                    f"`start_as_current_span` Could not set labels: {labels}. "
                    "They will be propagated to the next span."
                )
            with tracer.start_as_current_span(
                name,
                context=ctx,
                attributes={
                    SPAN_TYPE: span_type,
                    **(label_props),
                },
            ) as span:
                if trace_id is not None and isinstance(trace_id, uuid.UUID):
                    span.set_attribute(OVERRIDE_PARENT_SPAN, True)
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

            # TODO: Figure out if this is necessary
            try:
                detach(ctx_token)
            except Exception:
                pass

    @classmethod
    @contextmanager
    def with_labels(cls, labels: list[str], context: Optional[Context] = None):
        """Set labels for spans within this `with` context. This is useful for
        adding labels to the spans created in the auto-instrumentations.

        Requirements:
        - Labels must be created in your project in advance.
        - Keys must be strings from your label names.
        - Values must be strings matching the label's allowed values.

        Usage example:
        ```python
        with Laminar.with_labels({"sentiment": "positive"}):
            openai_client.chat.completions.create()
        ```
        """
        if not cls.is_initialized():
            yield
            return

        with get_tracer():
            label_props = labels.copy()
            prev_labels = get_association_properties(context).get("labels", [])
            update_association_properties(
                {"labels": prev_labels + label_props},
                set_on_current_span=False,
                context=context,
            )
            yield
            try:
                set_association_properties({"labels": prev_labels})
            except Exception:
                cls.__logger.warning(
                    f"`with_labels` Could not remove labels: {labels}. They will be "
                    "propagated to the next span."
                )
                pass

    @classmethod
    def start_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Union[
            Literal["DEFAULT"], Literal["LLM"], Literal["TOOL"]
        ] = "DEFAULT",
        context: Optional[Context] = None,
        parent_span_context: Optional[LaminarSpanContext] = None,
        labels: Optional[dict[str, str]] = None,
        # deprecated, use parent_span_context instead
        trace_id: Optional[uuid.UUID] = None,
    ):
        """Start a new span. Useful for manual instrumentation.
        If `span_type` is set to `"LLM"`, you should report usage and response
        attributes manually. See `Laminar.set_span_attributes` for more
        information.

        Usage example:
        ```python
        from src.lmnr import Laminar, use_span
        def foo(span):
            with use_span(span):
                with Laminar.start_as_current_span("foo_inner"):
                    some_function()
        
        def bar():
            with use_span(span):
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
            span_type (Union[Literal["DEFAULT"], Literal["LLM"]], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Optional[Context], optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (Optional[LaminarSpanContext], optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            labels (Optional[dict[str, str]], optional): labels to set for the\
                span. Defaults to None.
            trace_id (Optional[uuid.UUID], optional): Deprecated, use\
                `parent_span_context` instead. If provided, it will be used to\
                set the trace id for the span.
        """
        if not cls.is_initialized():
            return trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=RandomIdGenerator().generate_trace_id(),
                    span_id=RandomIdGenerator().generate_span_id(),
                    is_remote=False,
                )
            )

        with get_tracer() as tracer:
            ctx = context or context_api.get_current()
            if trace_id is not None:
                warnings.warn(
                    "trace_id provided to `Laminar.start_span`"
                    " is deprecated, use parent_span_context instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if parent_span_context is not None:
                span_context = LaminarSpanContext.try_to_otel_span_context(
                    parent_span_context, cls.__logger
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            elif trace_id is not None and isinstance(trace_id, uuid.UUID):
                span_context = trace.SpanContext(
                    trace_id=int(trace_id),
                    span_id=random.getrandbits(64),
                    is_remote=False,
                    trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                )
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(span_context), ctx
                )
            label_props = {}
            try:
                if labels:
                    label_props = {
                        f"{ASSOCIATION_PROPERTIES}.labels": json_dumps(labels)
                    }
            except Exception:
                cls.__logger.warning(
                    f"`start_span` Could not set labels: {labels}. They will be "
                    "propagated to the next span."
                )
            span = tracer.start_span(
                name,
                context=ctx,
                attributes={
                    SPAN_TYPE: span_type,
                    **(label_props),
                },
            )
            if trace_id is not None and isinstance(trace_id, uuid.UUID):
                span.set_attribute(OVERRIDE_PARENT_SPAN, True)
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
    def set_span_output(cls, output: Any = None):
        """Set the output of the current span. Useful for manual
        instrumentation.

        Args:
            output (Any, optional): output of the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
        """
        span = trace.get_current_span()
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
    @contextmanager
    def set_tracing_level(self, level: TracingLevel):
        """Set the tracing level for the current span and the context
        (i.e. any children spans created from the current span in the current
        thread).

        Tracing level can be one of:
        - `TracingLevel.ALL`: Enable tracing for the current span and all
            children spans.
        - `TracingLevel.META_ONLY`: Enable tracing for the current span and all
            children spans, but only record metadata, e.g. tokens, costs.
        - `TracingLevel.OFF`: Disable recording any spans.

        Example:
        ```python
        from lmnr import Laminar, TracingLevel

        with Laminar.set_tracing_level(TracingLevel.META_ONLY):
            openai_client.chat.completions.create()
        ```
        """
        if level == TracingLevel.ALL:
            yield
        else:
            level = "meta_only" if level == TracingLevel.META_ONLY else "off"
            update_association_properties({"tracing_level": level})
            yield
            try:
                remove_association_properties({"tracing_level": level})
            except Exception:
                pass

    @classmethod
    def set_span_attributes(
        cls,
        attributes: dict[Attributes, Any],
    ):
        """Set attributes for the current span. Useful for manual
        instrumentation.
        Example:
        ```python
        with L.start_as_current_span(
            name="my_span_name", input=input["messages"], span_type="LLM"
        ):
            response = await my_custom_call_to_openai(input)
            L.set_span_output(response["choices"][0]["message"]["content"])
            L.set_span_attributes({
                Attributes.PROVIDER: 'openai',
                Attributes.REQUEST_MODEL: input["model"],
                Attributes.RESPONSE_MODEL: response["model"],
                Attributes.INPUT_TOKEN_COUNT: response["usage"]["prompt_tokens"],
                Attributes.OUTPUT_TOKEN_COUNT: response["usage"]["completion_tokens"],
            })
            # ...
        ```

        Args:
            attributes (dict[ATTRIBUTES, Any]): attributes to set for the span
        """
        span = trace.get_current_span()
        if span == trace.INVALID_SPAN:
            return

        for key, value in attributes.items():
            # Python 3.12+ should do: if key not in Attributes:
            try:
                Attributes(key.value)
            except (TypeError, AttributeError):
                cls.__logger.warning(
                    f"Attribute {key} is not a valid Laminar attribute."
                )
                continue
            if not isinstance(value, (str, int, float, bool)):
                span.set_attribute(key.value, json_dumps(value))
            else:
                span.set_attribute(key.value, value)

    @classmethod
    def get_laminar_span_context(
        cls, span: Optional[trace.Span] = None
    ) -> Optional[LaminarSpanContext]:
        """Get the laminar span context for a given span.
        If no span is provided, the current active span will be used.
        """
        span = span or trace.get_current_span()
        if span == trace.INVALID_SPAN:
            return None
        return LaminarSpanContext(
            trace_id=uuid.UUID(int=span.get_span_context().trace_id),
            span_id=uuid.UUID(int=span.get_span_context().span_id),
            is_remote=span.get_span_context().is_remote,
        )

    @classmethod
    def get_laminar_span_context_dict(
        cls, span: Optional[trace.Span] = None
    ) -> Optional[dict]:
        span_context = cls.get_laminar_span_context(span)
        if span_context is None:
            return None
        return span_context.to_dict()

    @classmethod
    def serialize_span_context(cls, span: Optional[trace.Span] = None) -> Optional[str]:
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
        return json.dumps(span_context.to_dict())

    @classmethod
    def deserialize_span_context(
        cls, span_context: Union[dict, str]
    ) -> LaminarSpanContext:
        return LaminarSpanContext.deserialize(span_context)

    @classmethod
    def shutdown(cls):
        Traceloop.flush()
        LaminarClient.shutdown()

    @classmethod
    async def shutdown_async(cls):
        Traceloop.flush()
        await LaminarClient.shutdown_async()

    @classmethod
    def set_session(
        cls,
        session_id: Optional[str] = None,
    ):
        """Set the session and user id for the current span and the context
        (i.e. any children spans created from the current span in the current
        thread).

        Args:
            session_id (Optional[str], optional): Custom session id.\
                            Useful to debug and group long-running\
                            sessions/conversations.
                            Defaults to None.
        """
        association_properties = {}
        if session_id is not None:
            association_properties[SESSION_ID] = session_id
        update_association_properties(association_properties)

    @classmethod
    def set_metadata(cls, metadata: dict[str, str]):
        """Set the metadata for the current trace.

        Args:
            metadata (dict[str, str]): Metadata to set for the trace. Willl be\
                sent as attributes, so must be json serializable.
        """
        props = {f"metadata.{k}": json_dumps(v) for k, v in metadata.items()}
        update_association_properties(props)

    @classmethod
    def clear_metadata(cls):
        """Clear the metadata from the context"""
        props: dict = copy.copy(context_api.get_value("association_properties"))
        metadata_keys = [k for k in props.keys() if k.startswith("metadata.")]
        for k in metadata_keys:
            props.pop(k)
        set_association_properties(props)

    @classmethod
    def clear_session(cls):
        """Clear the session and user id from  the context"""
        props: dict = copy.copy(context_api.get_value("association_properties"))
        props.pop("session_id", None)
        props.pop("user_id", None)
        set_association_properties(props)

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
        association_properties = {
            TRACE_TYPE: trace_type.value,
        }
        update_association_properties(association_properties)
