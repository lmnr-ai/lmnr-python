from contextlib import contextmanager
from contextvars import Context
from opentelemetry import context, trace
from opentelemetry.util.types import AttributeValue
from opentelemetry.context import set_value, attach, detach
from lmnr.traceloop_sdk import Traceloop
from lmnr.traceloop_sdk.instruments import Instruments
from lmnr.traceloop_sdk.tracing import get_tracer
from lmnr.traceloop_sdk.tracing.attributes import (
    Attributes,
    SPAN_TYPE,
    OVERRIDE_PARENT_SPAN,
)
from lmnr.traceloop_sdk.decorators.base import json_dumps
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from pydantic.alias_generators import to_snake
from typing import Any, Literal, Optional, Set, Union

import aiohttp
import copy
import datetime
import dotenv
import json
import logging
import os
import random
import re
import requests
import urllib.parse
import uuid

from lmnr.traceloop_sdk.tracing.attributes import (
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_PATH,
    TRACE_TYPE,
    USER_ID,
)
from lmnr.traceloop_sdk.tracing.tracing import (
    get_span_path,
    set_association_properties,
    update_association_properties,
)

from .log import VerboseColorfulFormatter

from .types import (
    CreateEvaluationResponse,
    EvaluationResultDatapoint,
    GetDatapointsResponse,
    PipelineRunError,
    PipelineRunResponse,
    NodeInput,
    PipelineRunRequest,
    TraceType,
)


class Laminar:
    __base_http_url: str
    __base_grpc_url: str
    __project_api_key: Optional[str] = None
    __env: dict[str, str] = {}
    __initialized: bool = False
    __http_session: Optional[requests.Session] = None

    @classmethod
    def initialize(
        cls,
        project_api_key: Optional[str] = None,
        env: dict[str, str] = {},
        base_url: Optional[str] = None,
        http_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        instruments: Optional[Set[Instruments]] = None,
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
        url = base_url or "https://api.lmnr.ai"
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
        cls.__http_session = requests.Session()
        Traceloop.init(
            exporter=OTLPSpanExporter(
                endpoint=cls.__base_grpc_url,
                headers={"authorization": f"Bearer {cls.__project_api_key}"},
            ),
            instruments=instruments,
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
    ) -> PipelineRunResponse:
        """Runs the pipeline with the given inputs

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
        if cls.__project_api_key is None:
            raise ValueError(
                "Please initialize the Laminar object with your project "
                "API key or set the LMNR_PROJECT_API_KEY environment variable"
            )
        try:
            current_span = trace.get_current_span()
            if current_span != trace.INVALID_SPAN:
                parent_span_id = parent_span_id or uuid.UUID(
                    int=current_span.get_span_context().span_id
                )
                trace_id = trace_id or uuid.UUID(
                    int=current_span.get_span_context().trace_id
                )
            request = PipelineRunRequest(
                inputs=inputs,
                pipeline=pipeline,
                env=env or cls.__env,
                metadata=metadata,
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            )
        except Exception as e:
            raise ValueError(f"Invalid request: {e}")

        response = (
            cls.__http_session.post(
                cls.__base_http_url + "/v1/pipeline/run",
                data=json.dumps(request.to_dict()),
                headers=cls._headers(),
            )
            if cls.__http_session
            else requests.post(
                cls.__base_http_url + "/v1/pipeline/run",
                data=json.dumps(request.to_dict()),
                headers=cls._headers(),
            )
        )
        if response.status_code != 200:
            raise PipelineRunError(response)
        try:
            resp_json = response.json()
            keys = list(resp_json.keys())
            for key in keys:
                value = resp_json[key]
                del resp_json[key]
                resp_json[to_snake(key)] = value
            return PipelineRunResponse(**resp_json)
        except Exception:
            raise PipelineRunError(response)

    @classmethod
    def event(
        cls,
        name: str,
        value: Optional[AttributeValue] = None,
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Associate an event with the current span. If event with such
        name never existed, Laminar will create a new event and infer its type
        from the value. If the event already exists, Laminar will append the
        value to the event if and only if the value is of a matching type.
        Otherwise, the event won't be recorded.
        Supported types are string, numeric, and boolean. If the value
        is `None`, event is considered a boolean tag with the value of `True`.

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
        span_type: Union[Literal["DEFAULT"], Literal["LLM"]] = "DEFAULT",
        context: Optional[Context] = None,
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
            trace_id (Optional[uuid.UUID], optional): [EXPERIMENTAL] override\
                the trace id for the span. If not provided, use the current\
                trace id. Defaults to None.
        """

        with get_tracer() as tracer:
            span_path = get_span_path(name)
            ctx = set_value("span_path", span_path, context)
            if trace_id is not None:
                if isinstance(trace_id, uuid.UUID):
                    span_context = trace.SpanContext(
                        trace_id=int(trace_id),
                        span_id=random.getrandbits(64),
                        is_remote=False,
                        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                    )
                    ctx = trace.set_span_in_context(
                        trace.NonRecordingSpan(span_context), ctx
                    )
                else:
                    cls.__logger.warning(
                        "trace_id provided to `Laminar.start_as_current_span`"
                        " is not a valid UUID"
                    )
            ctx_token = attach(ctx)
            with tracer.start_as_current_span(
                name,
                context=ctx,
                attributes={SPAN_PATH: span_path, SPAN_TYPE: span_type},
            ) as span:
                if trace_id is not None and isinstance(trace_id, uuid.UUID):
                    span.set_attribute(OVERRIDE_PARENT_SPAN, True)
                if input is not None:
                    span.set_attribute(
                        SPAN_INPUT,
                        json_dumps(input),
                    )
                yield span

            # # TODO: Figure out if this is necessary
            try:
                detach(ctx_token)
            except Exception:
                pass

    @classmethod
    def start_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Union[Literal["DEFAULT"], Literal["LLM"]] = "DEFAULT",
        context: Optional[Context] = None,
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
            trace_id (Optional[uuid.UUID], optional): [EXPERIMENTAL] override\
                the trace id for the span. If not provided, use the current\
                trace id. Defaults to None.
        """
        with get_tracer() as tracer:
            span_path = get_span_path(name)
            ctx = set_value("span_path", span_path, context)
            if trace_id is not None:
                if isinstance(trace_id, uuid.UUID):
                    span_context = trace.SpanContext(
                        trace_id=int(trace_id),
                        span_id=random.getrandbits(64),
                        is_remote=False,
                        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                    )
                    ctx = trace.set_span_in_context(
                        trace.NonRecordingSpan(span_context), ctx
                    )
                else:
                    cls.__logger.warning(
                        "trace_id provided to `Laminar.start_span`"
                        " is not a valid UUID"
                    )
            span = tracer.start_span(
                name,
                context=ctx,
                attributes={SPAN_PATH: span_path, SPAN_TYPE: span_type},
            )
            if trace_id is not None and isinstance(trace_id, uuid.UUID):
                span.set_attribute(OVERRIDE_PARENT_SPAN, True)
            if input is not None:
                span.set_attribute(
                    SPAN_INPUT,
                    json_dumps(input),
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
            span.set_attribute(SPAN_OUTPUT, json_dumps(output))

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
    def set_session(
        cls,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Set the session and user id for the current span and the context
        (i.e. any children spans created from the current span in the current
        thread).

        Args:
            session_id (Optional[str], optional): Custom session id.\
                            Useful to debug and group long-running\
                            sessions/conversations.
                            Defaults to None.
            user_id (Optional[str], optional): Custom user id.\
                            Useful for grouping spans or traces by user.\
                            Defaults to None.
        """
        association_properties = {}
        if session_id is not None:
            association_properties[SESSION_ID] = session_id
        if user_id is not None:
            association_properties[USER_ID] = user_id
        update_association_properties(association_properties)

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

    @classmethod
    def clear_session(cls):
        """Clear the session and user id from  the context"""
        props: dict = copy.copy(context.get_value("association_properties"))
        props.pop("session_id", None)
        props.pop("user_id", None)
        set_association_properties(props)

    @classmethod
    def create_evaluation(
        cls,
        data: list[EvaluationResultDatapoint],
        group_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> CreateEvaluationResponse:
        response = requests.post(
            cls.__base_http_url + "/v1/evaluations",
            data=json.dumps(
                {
                    "groupId": group_id,
                    "name": name,
                    "points": [datapoint.to_dict() for datapoint in data],
                }
            ),
            headers=cls._headers(),
        )
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(f"Error creating evaluation {json.dumps(resp_json)}")
            except requests.exceptions.RequestException:
                raise ValueError(f"Error creating evaluation {response.text}")
        return CreateEvaluationResponse.model_validate(response.json())

    @classmethod
    def get_datapoints(
        cls,
        dataset_name: str,
        offset: int,
        limit: int,
    ) -> GetDatapointsResponse:
        params = {"name": dataset_name, "offset": offset, "limit": limit}
        url = (
            cls.__base_http_url
            + "/v1/datasets/datapoints?"
            + urllib.parse.urlencode(params)
        )
        response = requests.get(url, headers=cls._headers())
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {json.dumps(resp_json)}"
                )
            except requests.exceptions.RequestException:
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {response.text}"
                )
        return GetDatapointsResponse.model_validate(response.json())

    @classmethod
    def _headers(cls):
        assert cls.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + cls.__project_api_key,
            "Content-Type": "application/json",
        }
