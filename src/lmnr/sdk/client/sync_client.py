"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import re
import httpx
import json
import gzip
from opentelemetry import trace
from pydantic.alias_generators import to_snake
from typing import Generator, Literal, Optional, TypeVar, Union
from typing_extensions import overload
from types import TracebackType
import urllib.parse
import uuid

from lmnr.sdk.types import (
    AgentOutput,
    AgentState,
    EvaluationResultDatapoint,
    GetDatapointsResponse,
    InitEvaluationResponse,
    LaminarSpanContext,
    ModelProvider,
    NodeInput,
    PipelineRunError,
    PipelineRunRequest,
    PipelineRunResponse,
    RunAgentRequest,
    RunAgentResponseChunk,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from lmnr.version import PYTHON_VERSION, __version__

_T = TypeVar("_T", bound="LaminarClient")


class LaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.Client = None

    def __init__(
        self,
        base_url: str,
        project_api_key: str,
        port: Optional[int] = None,
    ):
        """Initializer for the Laminar HTTP client.

        Args:
            base_url (str): base URL of the Laminar API. If you include a port,
                the `port` argument will be ignored.
            project_api_key (str): Laminar project API key
            port (Optional[int], optional): port of the Laminar API HTTP server.
                Defaults to None. If none is provided, the default port (443) will
                be used.
        """
        # If port is already in the base URL, use it as is
        if re.search(r":\d{1,5}$", base_url):
            self.__base_url = base_url
        else:
            self.__base_url = f"{base_url}:{port or 443}"
        # Remove trailing slash from base URL
        self.__base_url = re.sub(r"/$", "", self.__base_url)
        self.__project_api_key = project_api_key
        self.__client = httpx.Client(
            headers=self._headers(),
            timeout=350,
        )

    def shutdown(self):
        self.__client.close()

    def run_pipeline(
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
        response = self.__client.post(
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

    def semantic_search(
        self,
        query: str,
        dataset_id: uuid.UUID,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResponse:
        """Perform a semantic search on the given dataset.

        Args:
            query (str): query to search for
            dataset_id (uuid.UUID): dataset ID created in the UI
            limit (Optional[int], optional): maximum number of results to return
            threshold (Optional[float], optional): lowest similarity score to return

        Raises:
            ValueError: if an error happens while performing the semantic search

        Returns:
            SemanticSearchResponse: response from the semantic search
        """
        request = SemanticSearchRequest(
            query=query,
            dataset_id=dataset_id,
            limit=limit,
            threshold=threshold,
        )
        response = self.__client.post(
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

    def _init_eval(
        self, name: Optional[str] = None, group_name: Optional[str] = None
    ) -> InitEvaluationResponse:
        response = self.__client.post(
            self.__base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        resp_json = response.json()
        return InitEvaluationResponse.model_validate(resp_json)

    def _save_eval_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint],
        groupName: Optional[str] = None,
    ):
        response = self.__client.post(
            self.__base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": [datapoint.to_dict() for datapoint in datapoints],
                "groupName": groupName,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(f"Error saving evaluation datapoints: {response.text}")

    def _send_browser_events(
        self,
        session_id: str,
        trace_id: str,
        events: list[dict],
    ):
        url = self.__base_url + "/v1/browser-sessions/events"
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": f"python@{PYTHON_VERSION}",
            "sdkVersion": __version__,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.__client.post(
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

    def _get_datapoints(
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
        response = self.__client.get(url, headers=self._headers())
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {json.dumps(resp_json)}"
                )
            except Exception:
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {response.text}"
                )
        return GetDatapointsResponse.model_validate(response.json())

    @overload
    def run_agent(
        self,
        prompt: str,
        state: Optional[AgentState] = None,
        span_context: Optional[LaminarSpanContext] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: Literal[True] = True,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> Generator[RunAgentResponseChunk, None, None]: ...

    @overload
    def run_agent(
        self,
        prompt: str,
        state: Optional[AgentState] = None,
        span_context: Optional[LaminarSpanContext] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: Literal[False] = False,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> AgentOutput: ...

    def run_agent(
        self,
        prompt: str,
        state: Optional[AgentState] = None,
        span_context: Optional[LaminarSpanContext] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: bool = False,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            state (Optional[AgentState], optional): state as returned by the previous agent run
            span_context (Optional[LaminarSpanContext], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            stream (bool, optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            cdp_url (Optional[str], optional): CDP URL to connect to an existing browser session.

        Returns:
            Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]: agent output or a generator of response chunks
        """
        request = RunAgentRequest(
            prompt=prompt,
            state=state,
            span_context=span_context,
            model_provider=model_provider,
            model=model,
            # We always connect to stream, because our TLS listeners on AWS
            # Network load balancers have a hard fixed idle timeout of 350 seconds.
            # This means that if we don't stream, the connection will be closed.
            # For now, we just return the content of the final chunk if `stream` is
            # `False`.
            # https://aws.amazon.com/blogs/networking-and-content-delivery/introducing-nlb-tcp-configurable-idle-timeout/
            stream=True,
            enable_thinking=enable_thinking,
            cdp_url=cdp_url,
        )

        # For streaming case, use a generator function
        if stream:
            return self._run_agent_streaming(request)
        else:
            # For non-streaming case, process all chunks and return the final result
            return self._run_agent_non_streaming(request)

    def _run_agent_streaming(
        self, request: RunAgentRequest
    ) -> Generator[RunAgentResponseChunk, None, None]:
        with self.__client.stream(
            "POST",
            self.__base_url + "/v1/agent/run",
            json=request.to_dict(),
            headers=self._headers(),
        ) as response:
            for line in response.iter_lines():
                line = str(line)
                if line.startswith("[DONE]"):
                    break
                if not line.startswith("data: "):
                    continue
                line = line[6:]
                if line:
                    chunk = RunAgentResponseChunk.model_validate_json(line)
                    yield chunk.root

    def _run_agent_non_streaming(self, request: RunAgentRequest) -> AgentOutput:
        final_chunk = None

        with self.__client.stream(
            "POST",
            self.__base_url + "/v1/agent/run",
            json=request.to_dict(),
            headers=self._headers(),
        ) as response:
            for line in response.iter_lines():
                line = str(line)
                if line.startswith("[DONE]"):
                    break
                if not line.startswith("data: "):
                    continue
                line = line[6:]
                if line:
                    chunk = RunAgentResponseChunk.model_validate_json(line)
                    if chunk.root.chunkType == "finalOutput":
                        final_chunk = chunk.root

        return final_chunk.content if final_chunk is not None else AgentOutput()

    def is_closed(self) -> bool:
        return self.__client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "_client"):
            self.__client.close()

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
