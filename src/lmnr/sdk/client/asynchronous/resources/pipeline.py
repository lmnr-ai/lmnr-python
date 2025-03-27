"""Pipeline resource for running Laminar pipelines."""

import uuid
from typing import Optional
from opentelemetry import trace

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.types import (
    NodeInput,
    PipelineRunError,
    PipelineRunRequest,
    PipelineRunResponse,
)


class AsyncPipeline(BaseAsyncResource):
    """Resource for interacting with Laminar pipelines."""

    async def run(
        self,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        parent_span_id: Optional[uuid.UUID] = None,
        trace_id: Optional[uuid.UUID] = None,
    ) -> PipelineRunResponse:
        """Run a pipeline with the given inputs and environment variables.

        Args:
            pipeline (str): pipeline name
            inputs (dict[str, NodeInput]): input values for the pipeline
            env (dict[str, str], optional): environment variables for the pipeline
            metadata (dict[str, str], optional): metadata for the pipeline run
            parent_span_id (Optional[uuid.UUID], optional): parent span id for the pipeline
            trace_id (Optional[uuid.UUID], optional): trace id for the pipeline

        Raises:
            ValueError: if the project API key is not set
            PipelineRunError: if the pipeline run fails

        Returns:
            PipelineRunResponse: response from the pipeline run
        """
        if self._project_api_key is None:
            raise ValueError(
                "Please initialize the Laminar object with your project "
                "API key or set the LMNR_PROJECT_API_KEY environment variable"
            )

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
            env=env or {},
            metadata=metadata,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
        )

        response = await self._client.post(
            self._base_url + "/v1/pipeline/run",
            json=request.to_dict(),
            headers=self._headers(),
        )

        if response.status_code != 200:
            raise PipelineRunError(response)

        try:
            from pydantic.alias_generators import to_snake

            resp_json = response.json()
            keys = list(resp_json.keys())
            for key in keys:
                value = resp_json[key]
                del resp_json[key]
                resp_json[to_snake(key)] = value
            return PipelineRunResponse(**resp_json)
        except Exception:
            raise PipelineRunError(response)
