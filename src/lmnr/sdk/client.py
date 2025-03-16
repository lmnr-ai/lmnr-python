"""
Laminar HTTP client. Used to send data to/from the Laminar API.
Initialized in `Laminar` singleton, but can be imported
in other classes.
"""

import asyncio
import json
import aiohttp
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


class LaminarClient:
    __base_url: str
    __project_api_key: str
    __session: aiohttp.ClientSession = None
    __sync_session: requests.Session = None

    @classmethod
    def initialize(cls, base_url: str, project_api_key: str):
        cls.__base_url = base_url
        cls.__project_api_key = project_api_key
        cls.__sync_session = requests.Session()
        loop = asyncio.get_event_loop()
        if loop.is_running():
            cls.__session = aiohttp.ClientSession()

    @classmethod
    def shutdown(cls):
        cls.__sync_session.close()
        if cls.__session is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    cls.__session.close()
                else:
                    asyncio.run(cls.__session.close())
            except Exception:
                asyncio.run(cls.__session.close())

    @classmethod
    async def shutdown_async(cls):
        if cls.__session is not None:
            await cls.__session.close()

    @classmethod
    def run_pipeline(
        cls,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        parent_span_id: Optional[uuid.UUID] = None,
        trace_id: Optional[uuid.UUID] = None,
    ) -> Union[PipelineRunResponse, Awaitable[PipelineRunResponse]]:
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
                env=env or {},
                metadata=metadata,
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            )
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_in_executor(None, cls.__run, request)
            else:
                return asyncio.run(cls.__run(request))
        except Exception as e:
            raise ValueError(f"Invalid request: {e}")

    @classmethod
    def semantic_search(
        cls,
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_in_executor(None, cls.__semantic_search, request)
        else:
            return asyncio.run(cls.__semantic_search(request))

    @classmethod
    async def init_eval(
        cls, name: Optional[str] = None, group_name: Optional[str] = None
    ) -> InitEvaluationResponse:
        session = await cls.__get_session()
        async with session.post(
            cls.__base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
            },
            headers=cls._headers(),
        ) as response:
            resp_json = await response.json()
            return InitEvaluationResponse.model_validate(resp_json)

    @classmethod
    async def save_eval_datapoints(
        cls,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint],
        groupName: Optional[str] = None,
    ):
        session = await cls.__get_session()
        async with session.post(
            cls.__base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": [datapoint.to_dict() for datapoint in datapoints],
                "groupName": groupName,
            },
            headers=cls._headers(),
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"Error saving evaluation datapoints: {await response.text()}"
                )

    @classmethod
    async def send_browser_events(
        cls,
        session_id: str,
        trace_id: str,
        events: list[dict],
        source: str,
    ):
        session = await cls.__get_session()
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": source,
            "sdkVersion": SDK_VERSION,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))

        async with session.post(
            cls.__base_url + "/v1/browser-sessions/events",
            data=compressed_payload,
            headers={
                **cls._headers(),
                "Content-Encoding": "gzip",
            },
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"Failed to send events: [{response.status}] {await response.text()}"
                )

    @classmethod
    def send_browser_events_sync(
        cls,
        session_id: str,
        trace_id: str,
        events: list[dict],
        source: str,
    ):
        url = cls.__base_url + "/v1/browser-sessions/events"
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": source,
            "sdkVersion": SDK_VERSION,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = cls.__sync_session.post(
            url,
            data=compressed_payload,
            headers={
                **cls._headers(),
                "Content-Encoding": "gzip",
            },
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to send events: [{response.status_code}] {response.text}"
            )

    @classmethod
    def get_datapoints(
        cls,
        dataset_name: str,
        offset: int,
        limit: int,
    ) -> GetDatapointsResponse:
        # TODO: Use aiohttp. Currently, this function is called from within
        # `LaminarDataset.__len__`, which is sync, but can be called from
        # both sync and async (primarily async). Python does not make it easy
        # to mix things this way, so we should probably refactor `LaminarDataset`.
        params = {"name": dataset_name, "offset": offset, "limit": limit}
        url = (
            cls.__base_url + "/v1/datasets/datapoints?" + urllib.parse.urlencode(params)
        )
        response = cls.__sync_session.get(url, headers=cls._headers())
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
    async def __run(
        cls,
        request: PipelineRunRequest,
    ) -> PipelineRunResponse:
        session = await cls.__get_session()
        async with session.post(
            cls.__base_url + "/v1/pipeline/run",
            data=json.dumps(request.to_dict()),
            headers=cls._headers(),
        ) as response:
            if response.status != 200:
                raise PipelineRunError(response)
            try:
                resp_json = await response.json()
                keys = list(resp_json.keys())
                for key in keys:
                    value = resp_json[key]
                    del resp_json[key]
                    resp_json[to_snake(key)] = value
                return PipelineRunResponse(**resp_json)
            except Exception:
                raise PipelineRunError(response)

    @classmethod
    async def __semantic_search(
        cls,
        request: SemanticSearchRequest,
    ) -> SemanticSearchResponse:
        session = await cls.__get_session()
        async with session.post(
            cls.__base_url + "/v1/semantic-search",
            data=json.dumps(request.to_dict()),
            headers=cls._headers(),
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"Error performing semantic search: [{response.status}] {await response.text()}"
                )
            try:
                resp_json = await response.json()
                for result in resp_json["results"]:
                    result["dataset_id"] = uuid.UUID(result["datasetId"])
                return SemanticSearchResponse(**resp_json)
            except Exception as e:
                raise ValueError(
                    f"Error parsing semantic search response: status={response.status} error={e}"
                )

    @classmethod
    def _headers(cls):
        assert cls.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + cls.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @classmethod
    async def __get_session(cls):
        if cls.__session is None:
            cls.__session = aiohttp.ClientSession()
        return cls.__session
