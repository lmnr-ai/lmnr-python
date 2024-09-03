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
    PipelineRunError,
    PipelineRunResponse,
    NodeInput,
    PipelineRunRequest,
)


class APIError(Exception):
    def __init__(self, status: Union[int, str], message: str, details: Any = None):
        self.message = message
        self.status = status
        self.details = details

    def __str__(self):
        msg = "{0} ({1}): {2}"
        return msg.format(self.message, self.status, self.details)


class Laminar:
    # _base_url = "https://api.lmnr.ai"
    _base_url = "http://localhost:8000"
    logger = logging.getLogger(__name__)
    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(VerboseColorfulFormatter())
    logger.addHandler(console_log_handler)

    def __init__(self, project_api_key: Optional[str] = None):
        self.project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
        if not self.project_api_key:
            dotenv_path = dotenv.find_dotenv(usecwd=True)
            self.project_api_key = dotenv.get_key(
                dotenv_path=dotenv_path, key_to_get="LMNR_PROJECT_API_KEY"
            )
        if not self.project_api_key:
            raise ValueError(
                "Please initialize the Laminar object with your project API key or set "
                "the LMNR_PROJECT_API_KEY environment variable in your environment or .env file"
            )
        Traceloop.init(
            api_endpoint=self._base_url,
            api_key=self.project_api_key,
        )

    def run(
        self,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        parent_span_id: Optional[uuid.UUID] = None,
        trace_id: Optional[uuid.UUID] = None,
    ) -> PipelineRunResponse:
        """Runs the pipeline with the given inputs

        Args:
            pipeline (str): name of the Laminar pipeline
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
        if self.project_api_key is None:
            raise ValueError(
                "Please initialize the Laminar object with your project API key or set "
                "the LMNR_PROJECT_API_KEY environment variable"
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
            self._base_url + "/v1/pipeline/run",
            data=json.dumps(request.to_dict()),
            headers=self._headers(),
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

    def event(
        self,
        name: str,
        value: AttributeValue,
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Associate an event with the current span

        Args:
            name (str): event name
            value (AttributeValue): event value. Must be a primitive type
            timestamp (Optional[Union[datetime.datetime, int]], optional): If int, must be epoch nanoseconds.
                If not specified, relies on the underlying otel implementation. Defaults to None.
        """
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)

        event = {
            "lmnr.event.type": "default",
            "lmnr.event.value": value,
        }

        current_span = get_current_span()
        if current_span == INVALID_SPAN:
            self.logger.warning(
                f"`Laminar().event()` called outside of span context. Event '{name}' will not be recorded in the trace. "
                "Make sure to annotate the function with a decorator"
            )
            return

        current_span.add_event(name, event, timestamp)

    def evaluate_event(
        self,
        name: str,
        evaluator: str,
        data: dict[str, AttributeValue],
        env: dict[str, str] = {},
        timestamp: Optional[Union[datetime.datetime, int]] = None,
    ):
        """Send an event for evaluation to the Laminar backend

        Args:
            name (str): name of the event
            evaluator (str): name of the pipeline that evaluates the event
            data (dict[str, AttributeValue]): map from input node name to node value in the evaluator pipeline
            env (dict[str, str], optional): environment variables required to run the pipeline. Defaults to {}.
            timestamp (Optional[Union[datetime.datetime, int]], optional): If int, must be epoch nanoseconds.
                If not specified, relies on the underlying otel implementation. Defaults to None.
        """
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)
        event = {
            "lmnr.event.type": "evaluate",
            "lmnr.event.evaluator": evaluator,
            "lmnr.event.data": json.dumps(data),
            "lmnr.event.env": json.dumps(env),
        }
        current_span = get_current_span()
        if current_span == INVALID_SPAN:
            self.logger.warning(
                f"`Laminar().evaluate_event()` called outside of span context. Event '{name}' will not be recorded in the trace. "
                "Make sure to annotate the function with a decorator"
            )
            return

        current_span.add_event(name, event)

    def start_span(
        self,
        name: str,
        input: Any = None,
    ) -> Tuple[Span, object]:
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

    def end_span(self, span: Span, token: object, output: Any = None):
        if output is not None:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps({"output": output})
            )
        span.end()
        context.detach(token)

    def set_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        current_span = get_current_span()
        if current_span != INVALID_SPAN:
            self.logger.debug(
                "Laminar().set_session() called outside of span context. Setting it manually in the current span."
            )
            if session_id is not None:
                current_span.set_attribute(
                    "traceloop.association.properties.session_id", session_id
                )
            if user_id is not None:
                current_span.set_attribute(
                    "traceloop.association.properties.user_id", user_id
                )
        Traceloop.set_association_properties(
            {
                "session_id": session_id,
                "user_id": user_id,
            }
        )

    def clear_session(self):
        props: dict = copy.copy(context.get_value("association_properties"))
        props.pop("session_id", None)
        props.pop("user_id", None)
        Traceloop.set_association_properties(props)

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.project_api_key,
            "Content-Type": "application/json",
        }
