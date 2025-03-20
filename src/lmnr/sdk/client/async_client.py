"""
Laminar HTTP client. Used to send data to/from the Laminar API.
Initialized in `Laminar` singleton, but can be imported
in other classes.
"""

import httpx
import json
import gzip
from opentelemetry import trace
from pydantic.alias_generators import to_snake
import requests
from typing import Awaitable, Optional, Union
import urllib.parse
import uuid

from lmnr.sdk.types import (
    EvaluationResultDatapoint,
    GetDatapointsResponse,
    InitEvaluationResponse,
    NodeInput,
    PipelineRunError,
    PipelineRunRequest,
    PipelineRunResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from lmnr.version import SDK_VERSION


class AsyncLaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.AsyncClient = None

    def __init__(self, base_url: str, project_api_key: str):
        self.__base_url = base_url
        self.__project_api_key = project_api_key
        self.__client = httpx.AsyncClient(
            headers=self._headers(),
        )

    async def shutdown(self):
        await self.__client.aclose()

    async def run_pipeline(
        self,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        parent_span_id: Optional[uuid.UUID] = None,
        trace_id: Optional[uuid.UUID] = None,
    ) -> Union[PipelineRunResponse, Awaitable[PipelineRunResponse]]:
        if self.__project_api_key is None:
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
        response = await self.__client.post(
            self.__base_url + "/v1/pipeline/run",
            json=request.to_dict(),
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

    async def semantic_search(
        self,
        query: str,
        dataset_id: uuid.UUID,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResponse:
        request = SemanticSearchRequest(
            query=query,
            dataset_id=dataset_id,
            limit=limit,
            threshold=threshold,
        )
        response = await self.__client.post(
            self.__base_url + "/v1/semantic-search",
            json=request.to_dict(),
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error performing semantic search: [{response.status_code}] {response.text}"
            )
        try:
            resp_json = response.json()
            for result in resp_json["results"]:
                result["dataset_id"] = uuid.UUID(result["datasetId"])
            return SemanticSearchResponse(**resp_json)
        except Exception as e:
            raise ValueError(
                f"Error parsing semantic search response: status={response.status_code} error={e}"
            )

    async def init_eval(
        self, name: Optional[str] = None, group_name: Optional[str] = None
    ) -> InitEvaluationResponse:
        response = await self.__client.post(
            self.__base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        resp_json = response.json()
        return InitEvaluationResponse.model_validate(resp_json)

    async def save_eval_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint],
        groupName: Optional[str] = None,
    ):
        response = await self.__client.post(
            self.__base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": [datapoint.to_dict() for datapoint in datapoints],
                "groupName": groupName,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(f"Error saving evaluation datapoints: {response.text}")

    async def send_browser_events(
        self,
        session_id: str,
        trace_id: str,
        events: list[dict],
        source: str,
    ):
        url = self.__base_url + "/v1/browser-sessions/events"
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": source,
            "sdkVersion": SDK_VERSION,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.__client.post(
            url,
            content=compressed_payload,
            headers={
                **self._headers(),
                "Content-Encoding": "gzip",
            },
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to send events: [{response.status_code}] {response.text}"
            )

    async def get_datapoints(
        self,
        dataset_name: str,
        offset: int,
        limit: int,
    ) -> GetDatapointsResponse:
        params = {"name": dataset_name, "offset": offset, "limit": limit}
        url = (
            self.__base_url
            + "/v1/datasets/datapoints?"
            + urllib.parse.urlencode(params)
        )
        response = await self.__client.get(url, headers=self._headers())
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

    def _headers(self):
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
