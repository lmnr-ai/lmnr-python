from .tracing_types import Span, Trace
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import AttributeValue
from traceloop.sdk import Traceloop

from pydantic.alias_generators import to_snake
from typing import Any, Optional, Union

import datetime
import dotenv
import json
import os
import requests
import uuid

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
                Must usually be SpanContext.id()
                Defaults to None.
            trace_id (Optional[uuid.UUID], optional):
                trace id for the resulting trace.
                Must usually be TraceContext.id()
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
        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp())

        event = {
            "__lmnr_event_type": "default",
            "value": value,
        }

        get_current_span().add_event(name, event, timestamp)

    def evaluate_event(
        self,
        name: str,
        evaluator: str,
        data: dict[str, AttributeValue],
        env: dict[str, str] = {},
    ):
        event = {
            "__lmnr_event_type": "evaluate",
            "evaluator": evaluator,
            "data": json.dumps(data),
            "env": json.dumps(env),
        }

        get_current_span().add_event(name, event)

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.project_api_key,
            "Content-Type": "application/json",
        }
