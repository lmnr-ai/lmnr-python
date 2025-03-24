"""Agent resource for interacting with Laminar agents."""

import gzip
import json
from typing import Generator, Literal, Optional, Union, overload

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.types import (
    AgentOutput,
    LaminarSpanContext,
    ModelProvider,
    RunAgentRequest,
    RunAgentResponseChunk,
)
from lmnr.version import PYTHON_VERSION, __version__


class Agent(BaseResource):
    """Resource for interacting with Laminar agents."""

    @overload
    def run(
        self,
        prompt: str,
        stream: Literal[True],
        span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> Generator[RunAgentResponseChunk, None, None]:
        """Run Laminar index agent in streaming mode.

        Args:
            prompt (str): prompt for the agent
            stream (Literal[True]): whether to stream the agent's response
            span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            cdp_url (Optional[str], optional): CDP URL to connect to an existing browser session.

        Returns:
            Generator[RunAgentResponseChunk, None, None]: a generator of response chunks
        """
        pass

    @overload
    def run(
        self,
        prompt: str,
        span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            cdp_url (Optional[str], optional): CDP URL to connect to an existing browser session.

        Returns:
            AgentOutput: agent output
        """
        pass

    @overload
    def run(
        self,
        prompt: str,
        span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: Literal[False] = False,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            stream (Literal[False], optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            cdp_url (Optional[str], optional): CDP URL to connect to an existing browser session.

        Returns:
            AgentOutput: agent output
        """
        pass

    def run(
        self,
        prompt: str,
        span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: bool = False,
        enable_thinking: bool = True,
        cdp_url: Optional[str] = None,
    ) -> Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            stream (bool, optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            cdp_url (Optional[str], optional): CDP URL to connect to an existing browser session.

        Returns:
            Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]: agent output or a generator of response chunks
        """
        if span_context is not None and isinstance(span_context, LaminarSpanContext):
            span_context = str(span_context)
        request = RunAgentRequest(
            prompt=prompt,
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
            return self.__run_streaming(request)
        else:
            # For non-streaming case, process all chunks and return the final result
            return self.__run_non_streaming(request)

    def __run_streaming(
        self, request: RunAgentRequest
    ) -> Generator[RunAgentResponseChunk, None, None]:
        """Run agent in streaming mode.

        Args:
            request (RunAgentRequest): The request to run the agent with.

        Yields:
            RunAgentResponseChunk: Chunks of the agent's response.
        """
        with self._client.stream(
            "POST",
            self._base_url + "/v1/agent/run",
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

    def __run_non_streaming(self, request: RunAgentRequest) -> AgentOutput:
        """Run agent in non-streaming mode.

        Args:
            request (RunAgentRequest): The request to run the agent with.

        Returns:
            AgentOutput: The agent's output.
        """
        final_chunk = None

        with self._client.stream(
            "POST",
            self._base_url + "/v1/agent/run",
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

    def _send_browser_events(
        self,
        session_id: str,
        trace_id: str,
        events: list[dict],
    ):
        url = self._base_url + "/v1/browser-sessions/events"
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": f"python@{PYTHON_VERSION}",
            "sdkVersion": __version__,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self._client.post(
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
