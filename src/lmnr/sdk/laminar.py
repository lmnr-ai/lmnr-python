from opentelemetry import context
from opentelemetry.trace import (
    INVALID_SPAN,
    get_current_span,
    set_span_in_context,
    Span,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.util.types import AttributeValue
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import get_tracer

from pydantic.alias_generators import to_snake
from typing import Any, Optional, Tuple, Union

import copy
import datetime
import dotenv
import json
import logging
import os
import requests
import uuid

from .log import VerboseColorfulFormatter

from .types import (
    CreateEvaluationResponse,
    EvaluationResultDatapoint,
    PipelineRunError,
    PipelineRunResponse,
    NodeInput,
    PipelineRunRequest,
)


class Laminar:
    __base_url: str = "https://api.lmnr.ai"
    __project_api_key: Optional[str] = None
    __env: dict[str, str] = {}
    __initialized: bool = False

    @classmethod
    def initialize(
        cls,
        project_api_key: Optional[str] = None,
        env: dict[str, str] = {},
        base_url: Optional[str] = None,
    ):
        """Initialize Laminar context across the application.
        This method must be called before using any other Laminar methods or
        decorators.

        Args:
            project_api_key (Optional[str], optional): Laminar project api key.
                            You can generate one by going to the projects
                            settings page on the Laminar dashboard.
                            If not specified, it will try to read from the
                            LMNR_PROJECT_API_KEY environment variable
                            in os.environ or in .env file.
                            Defaults to None.
            env (dict[str, str], optional): Default environment passed to
                            `run` and `evaluate_event` requests, unless
                            overriden at request time. Usually, model
                            provider keys are stored here.
                            Defaults to {}.
            base_url (Optional[str], optional): Url of Laminar endpoint,
                            or the  customopen telemetry ingester.
                            If not specified, defaults to
                            https://api.lmnr.ai.
                            For locally hosted Laminar, default setting
                            must be http://localhost:8000
                            Defaults to None.

        Raises:
            ValueError: If project API key is not set
        """
        cls.__project_api_key = project_api_key or os.environ.get(
            "LMNR_PROJECT_API_KEY"
        )
        if not project_api_key:
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
        if base_url is not None:
            cls.__base_url = base_url
        cls.__env = env
        cls.__initialized = True
        cls._initialize_logger()
        Traceloop.init(
            api_endpoint=cls.__base_url,
            api_key=cls.__project_api_key,
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
            pipeline (str): name of the Laminar pipeline.
                The pipeline must have a target version set.
            inputs (dict[str, NodeInput]):
                inputs to the endpoint's target pipeline.
                Keys in the dictionary must match input node names
            env (dict[str, str], optional):
                Environment variables for the pipeline execution.
                Defaults to {}.
            metadata (dict[str, str], optional):
                any custom metadata to be stored
                with execution trace. Defaults to {}.
            parent_span_id (Optional[uuid.UUID], optional):
                parent span id for the resulting span.
                Defaults to None.
            trace_id (Optional[uuid.UUID], optional):
                trace id for the resulting trace.
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
            current_span = get_current_span()
            if current_span != INVALID_SPAN:
                parent_span_id = parent_span_id or uuid.UUID(
                    int=current_span.get_span_context().span_id
                )
                trace_id = trace_id or uuid.UUID(
                    int=current_span.get_span_context().trace_id
                )
            request = PipelineRunRequest(
                inputs=inputs,
                pipeline=pipeline,
                env=env,
                metadata=metadata,
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            )
        except Exception as e:
            raise ValueError(f"Invalid request: {e}")

        response = requests.post(
            cls.__base_url + "/v1/pipeline/run",
            data=json.dumps(request.to_dict()),
            headers=cls._headers(),
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
        value: AttributeValue,
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Associate an event with the current span

        Args:
            name (str): event name
            value (AttributeValue): event value. Must be a primitive type
                            or a sequence of values of the same primitive type
            timestamp (Optional[Union[datetime.datetime, int]], optional):
                            If int, must be epoch nanoseconds. If not
                            specified, relies on the underlying OpenTelemetry
                            implementation. Defaults to None.
        """
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)

        event = {
            "lmnr.event.type": "default",
            "lmnr.event.value": value,
        }

        current_span = get_current_span()
        if current_span == INVALID_SPAN:
            cls.__logger.warning(
                "`Laminar().event()` called outside of span context. "
                f"Event '{name}' will not be recorded in the trace. "
                "Make sure to annotate the function with a decorator"
            )
            return

        current_span.add_event(name, event, timestamp)

    @classmethod
    def evaluate_event(
        cls,
        name: str,
        evaluator: str,
        data: dict[str, AttributeValue],
        env: Optional[dict[str, str]] = {},
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Send an event for evaluation to the Laminar backend

        Args:
            name (str): name of the event
            evaluator (str): name of the pipeline that evaluates the event.
                        The pipeline must have a target version set.
            data (dict[str, AttributeValue]): map from input node name to
                        its value in the evaluator pipeline
            env (dict[str, str], optional): environment variables required
                        to run the pipeline. Defaults to {}.
            timestamp (Optional[Union[datetime.datetime, int]], optional):
                        If int, must be epoch nanoseconds.
                        If not specified, relies on the underlying
                        OpenTelemetry implementation. Defaults to None.
        """
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)
        event = {
            "lmnr.event.type": "evaluate",
            "lmnr.event.evaluator": evaluator,
            "lmnr.event.data": json.dumps(data),
            "lmnr.event.env": json.dumps(env if env is not None else cls.__env),
        }
        current_span = get_current_span()
        if current_span == INVALID_SPAN:
            cls.__logger.warning(
                "`Laminar().evaluate_event()` called outside of span context."
                f"Event '{name}' will not be recorded in the trace. "
                "Make sure to annotate the function with a decorator"
            )
            return

        current_span.add_event(name, event)

    @classmethod
    def start_span(
        cls,
        name: str,
        input: Any = None,
    ) -> Tuple[Span, object]:
        """Start a new span with the given name. Useful for manual
        instrumentation.

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an
                attribute, so must be json serializable. Defaults to None.

        Returns:
            Tuple[Span, object]: Span - the started span, object -
                    context token
                    that must be passed to `end_span` to end the span.

        """
        with get_tracer() as tracer:
            span = tracer.start_span(name)
            ctx = set_span_in_context(span)
            token = context.attach(ctx)
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
            if input is not None:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps({"input": input})
                )
            return (span, token)

    @classmethod
    def end_span(cls, span: Span, token: object, output: Any = None):
        """End the span started with `start_span`

        Args:
            span (Span): span returned by `start_span`
            token (object): context token returned by `start_span`
            output (Any, optional): output of the span. Will be sent as an
                attribute, so must be json serializable. Defaults to None.
        """
        if output is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps({"output": output})
            )
        span.end()
        context.detach(token)

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
            session_id (Optional[str], optional): Custom session id.
                            Useful to debug and group long-running
                            sessions/conversations.
                            Defaults to None.
            user_id (Optional[str], optional): Custom user id.
                            Useful for grouping spans or traces by user.
                            Defaults to None.
        """
        current_span = get_current_span()
        if current_span != INVALID_SPAN:
            cls.__logger.debug(
                "Laminar().set_session() called inside a span context. Setting"
                " it manually in the current span."
            )
            if session_id is not None:
                current_span.set_attribute(
                    "traceloop.association.properties.session_id", session_id
                )
            if user_id is not None:
                current_span.set_attribute(
                    "traceloop.association.properties.user_id", user_id
                )
        association_properties = {}
        if session_id is not None:
            association_properties["session_id"] = session_id
        if user_id is not None:
            association_properties["user_id"] = user_id
        Traceloop.set_association_properties(association_properties)

    @classmethod
    def clear_session(cls):
        """Clear the session and user id from  the context"""
        props: dict = copy.copy(context.get_value("association_properties"))
        props.pop("session_id", None)
        props.pop("user_id", None)
        Traceloop.set_association_properties(props)

    @classmethod
    def create_evaluation(cls, name: str) -> CreateEvaluationResponse:
        response = requests.post(
            cls.__base_url + "/v1/evaluations",
            data=json.dumps({"name": name}),
            headers=cls._headers(),
        )
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(f"Error creating evaluation {json.dumps(resp_json)}")
            except Exception:
                raise ValueError(f"Error creating evaluation {response.text}")
        return CreateEvaluationResponse.model_validate(response.json())

    @classmethod
    def post_evaluation_results(
        cls, evaluation_name: str, data: list[EvaluationResultDatapoint]
    ) -> requests.Response:
        body = {
            "name": evaluation_name,
            "points": data,
        }
        response = requests.post(
            cls.__base_url + "/v1/evaluation-datapoints",
            data=json.dumps(body),
            headers=cls._headers(),
        )
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(
                    f"Failed to send evaluation results. Response: {json.dumps(resp_json)}"
                )
            except Exception:
                raise ValueError(
                    f"Failed to send evaluation results. Error: {response.text}"
                )
        return response

    @classmethod
    def update_evaluation_status(
        cls, evaluation_name: str, status: str
    ) -> requests.Response:
        body = {
            "name": evaluation_name,
            "status": status,
        }
        response = requests.put(
            cls.__base_url + "/v1/evaluations/",
            data=json.dumps(body),
            headers=cls._headers(),
        )
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(
                    f"Failed to send evaluation status. Response: {json.dumps(resp_json)}"
                )
            except Exception:
                raise ValueError(
                    f"Failed to send evaluation status. Error: {response.text}"
                )
        return response

    @classmethod
    def _headers(cls):
        return {
            "Authorization": "Bearer " + cls.__project_api_key,
            "Content-Type": "application/json",
        }
