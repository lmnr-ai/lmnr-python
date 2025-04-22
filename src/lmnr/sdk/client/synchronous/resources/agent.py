"""Agent resource for interacting with Laminar agents."""

from typing import Generator, Literal, Optional, Union, overload
import uuid

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from opentelemetry import trace

from lmnr.sdk.types import (
    AgentOutput,
    LaminarSpanContext,
    ModelProvider,
    RunAgentRequest,
    RunAgentResponseChunk,
)


class Agent(BaseResource):
    """Resource for interacting with Laminar agents."""

    @overload
    def run(
        self,
        prompt: str,
        stream: Literal[True],
        parent_span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        enable_thinking: bool = True,
        agent_state: Optional[str] = None,
        storage_state: Optional[str] = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        timeout: Optional[int] = None,
        cdp_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        thinking_token_budget: Optional[int] = None,
        start_url: Optional[str] = None,
    ) -> Generator[RunAgentResponseChunk, None, None]:
        """Run Laminar index agent in streaming mode.

        Args:
            prompt (str): prompt for the agent
            stream (Literal[True]): whether to stream the agent's response
            parent_span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (Optional[str], optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (Optional[str], optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            timeout (Optional[int], optional): timeout seconds for the agent's response. Default to None.
            cdp_url (Optional[str], optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (Optional[int], optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (Optional[int], optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (Optional[str], optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
        Returns:
            Generator[RunAgentResponseChunk, None, None]: a generator of response chunks
        """
        pass

    @overload
    def run(
        self,
        prompt: str,
        parent_span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        enable_thinking: bool = True,
        agent_state: Optional[str] = None,
        storage_state: Optional[str] = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        timeout: Optional[int] = None,
        cdp_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        thinking_token_budget: Optional[int] = None,
        start_url: Optional[str] = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (Optional[str], optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (Optional[str], optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            timeout (Optional[int], optional): timeout seconds for the agent's response. Default to None.
            cdp_url (Optional[str], optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (Optional[int], optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (Optional[int], optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (Optional[str], optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
        Returns:
            AgentOutput: agent output
        """
        pass

    @overload
    def run(
        self,
        prompt: str,
        parent_span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: Literal[False] = False,
        enable_thinking: bool = True,
        agent_state: Optional[str] = None,
        storage_state: Optional[str] = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        timeout: Optional[int] = None,
        cdp_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        thinking_token_budget: Optional[int] = None,
        start_url: Optional[str] = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            stream (Literal[False], optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (Optional[str], optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (Optional[str], optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            timeout (Optional[int], optional): timeout seconds for the agent's response. Default to None.
            cdp_url (Optional[str], optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (Optional[int], optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (Optional[int], optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (Optional[str], optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
        Returns:
            AgentOutput: agent output
        """
        pass

    def run(
        self,
        prompt: str,
        parent_span_context: Optional[Union[LaminarSpanContext, str]] = None,
        model_provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        stream: bool = False,
        enable_thinking: bool = True,
        agent_state: Optional[str] = None,
        storage_state: Optional[str] = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        timeout: Optional[int] = None,
        cdp_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        thinking_token_budget: Optional[int] = None,
        start_url: Optional[str] = None,
    ) -> Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (Optional[Union[LaminarSpanContext, str]], optional): span context if the agent is part of a trace
            model_provider (Optional[ModelProvider], optional): LLM model provider
            model (Optional[str], optional): LLM model name
            stream (bool, optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (Optional[str], optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (Optional[str], optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            timeout (Optional[int], optional): timeout seconds for the agent's response. Default to None.
            cdp_url (Optional[str], optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (Optional[int], optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (Optional[int], optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (Optional[str], optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.

        Returns:
            Union[AgentOutput, Generator[RunAgentResponseChunk, None, None]]: agent output or a generator of response chunks
        """
        if parent_span_context is None:
            span = trace.get_current_span()
            if span != trace.INVALID_SPAN:
                parent_span_context = LaminarSpanContext(
                    trace_id=uuid.UUID(int=span.get_span_context().trace_id),
                    span_id=uuid.UUID(int=span.get_span_context().span_id),
                    is_remote=span.get_span_context().is_remote,
                )
        if parent_span_context is not None and isinstance(
            parent_span_context, LaminarSpanContext
        ):
            parent_span_context = str(parent_span_context)
        request = RunAgentRequest(
            prompt=prompt,
            parent_span_context=parent_span_context,
            model_provider=model_provider,
            model=model,
            agent_state=agent_state,
            storage_state=storage_state,
            # We always connect to stream, because our network configuration
            # has a hard fixed idle timeout of 350 seconds.
            # This means that if we don't stream, the connection will be closed.
            # For now, we just return the content of the final chunk if `stream` is
            # `False`.
            stream=True,
            enable_thinking=enable_thinking,
            return_screenshots=return_screenshots,
            return_agent_state=return_agent_state,
            return_storage_state=return_storage_state,
            timeout=timeout,
            cdp_url=cdp_url,
            max_steps=max_steps,
            thinking_token_budget=thinking_token_budget,
            start_url=start_url,
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
            json=request.model_dump(by_alias=True),
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
                    if chunk.root.chunk_type in ["finalOutput", "error"]:
                        break

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
            json=request.model_dump(by_alias=True),
            headers=self._headers(),
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(response.read())
            for line in response.iter_lines():
                line = str(line)
                if line.startswith("[DONE]"):
                    break
                if not line.startswith("data: "):
                    continue
                line = line[6:]
                if line:
                    chunk = RunAgentResponseChunk.model_validate_json(line)
                    if chunk.root.chunk_type == "finalOutput":
                        final_chunk = chunk.root
                    elif chunk.root.chunk_type == "error":
                        raise RuntimeError(chunk.root.error)
                    elif chunk.root.chunk_type == "timeout":
                        raise TimeoutError("Agent timed out")

        return final_chunk.content if final_chunk is not None else AgentOutput()
