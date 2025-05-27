"""Agent resource for interacting with Laminar agents."""

from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Literal,
    overload,
)
import uuid

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.types import (
    AgentOutput,
    LaminarSpanContext,
    ModelProvider,
    RunAgentRequest,
    RunAgentResponseChunk,
)

from opentelemetry import trace


class AsyncAgent(BaseAsyncResource):
    """Resource for interacting with Laminar agents."""

    @overload
    async def run(
        self,
        prompt: str,
        stream: Literal[True],
        parent_span_context: LaminarSpanContext | str | None = None,
        model_provider: ModelProvider | None = None,
        model: str | None = None,
        enable_thinking: bool = True,
        agent_state: str | None = None,
        storage_state: str | None = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        disable_give_control: bool = False,
        timeout: int | None = None,
        cdp_url: str | None = None,
        max_steps: int | None = None,
        thinking_token_budget: int | None = None,
        start_url: str | None = None,
        user_agent: str | None = None,
    ) -> AsyncIterator[RunAgentResponseChunk]:
        """Run Laminar index agent in streaming mode.

        Args:
            prompt (str): prompt for the agent
            stream (Literal[True]): whether to stream the agent's response
            parent_span_context (LaminarSpanContext | str | None, optional): span context if the agent is part of a trace
            model_provider (ModelProvider | None, optional): LLM model provider
            model (str | None, optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (str | None, optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (str | None, optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state in the final chunk. Default to False.
            return_storage_state (bool, optional): whether to return the storage state in the final chunk. Default to False.
            disable_give_control (bool, optional): whether to NOT give the agent additional direction to give control to the user for tasks such as login. Default to False.
            timeout (int | None, optional): timeout seconds for the agent's response. Default to None.
            cdp_url (str | None, optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (int | None, optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (int | None, optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (str | None, optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
            user_agent (str | None, optional): the user to be sent to the browser. If not specified, Laminar uses the default user agent. Default to None.

        Returns:
            AsyncIterator[RunAgentResponseChunk]: a generator of response chunks
        """
        pass

    @overload
    async def run(
        self,
        prompt: str,
        parent_span_context: LaminarSpanContext | str | None = None,
        model_provider: ModelProvider | None = None,
        model: str | None = None,
        enable_thinking: bool = True,
        agent_state: str | None = None,
        storage_state: str | None = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        disable_give_control: bool = False,
        timeout: int | None = None,
        cdp_url: str | None = None,
        max_steps: int | None = None,
        thinking_token_budget: int | None = None,
        start_url: str | None = None,
        user_agent: str | None = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (LaminarSpanContext | str | None, optional): span context if the agent is part of a trace
            model_provider (ModelProvider | None, optional): LLM model provider
            model (str | None, optional): LLM model name
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (str | None, optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (str | None, optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            disable_give_control (bool, optional): whether to NOT give the agent additional direction to give control to the user for tasks such as login. Default to False.
            timeout (int | None, optional): timeout seconds for the agent's response. Default to None.
            cdp_url (str | None, optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (int | None, optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (int | None, optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (str | None, optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
            user_agent (str | None, optional): the user to be sent to the browser. If not specified, Laminar uses the default user agent. Default to None.

        Returns:
            AgentOutput: agent output
        """
        pass

    @overload
    async def run(
        self,
        prompt: str,
        parent_span_context: LaminarSpanContext | str | None = None,
        model_provider: ModelProvider | None = None,
        model: str | None = None,
        stream: Literal[False] = False,
        enable_thinking: bool = True,
        agent_state: str | None = None,
        storage_state: str | None = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        disable_give_control: bool = False,
        timeout: int | None = None,
        max_steps: int | None = None,
        thinking_token_budget: int | None = None,
        start_url: str | None = None,
        user_agent: str | None = None,
    ) -> AgentOutput:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (LaminarSpanContext | str | None, optional): span context if the agent is part of a trace
            model_provider (ModelProvider | None, optional): LLM model provider
            model (str | None, optional): LLM model name
            stream (Literal[False], optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (str | None, optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (str | None, optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            disable_give_control (bool, optional): whether to NOT give the agent additional direction to give control to the user for tasks such as login. Default to False.
            timeout (int | None, optional): timeout seconds for the agent's response. Default to None.
            cdp_url (str | None, optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (int | None, optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (int | None, optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (str | None, optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.
            user_agent (str | None, optional): the user to be sent to the browser. If not specified, Laminar uses the default user agent. Default to None.

        Returns:
            AgentOutput: agent output
        """
        pass

    async def run(
        self,
        prompt: str,
        parent_span_context: LaminarSpanContext | str | None = None,
        model_provider: ModelProvider | None = None,
        model: str | None = None,
        stream: bool = False,
        enable_thinking: bool = True,
        agent_state: str | None = None,
        storage_state: str | None = None,
        return_screenshots: bool = False,
        return_agent_state: bool = False,
        return_storage_state: bool = False,
        disable_give_control: bool = False,
        timeout: int | None = None,
        cdp_url: str | None = None,
        max_steps: int | None = None,
        thinking_token_budget: int | None = None,
        start_url: str | None = None,
        user_agent: str | None = None,
    ) -> AgentOutput | Awaitable[AsyncIterator[RunAgentResponseChunk]]:
        """Run Laminar index agent.

        Args:
            prompt (str): prompt for the agent
            parent_span_context (LaminarSpanContext | str | None, optional): span context if the agent is part of a trace
            model_provider (ModelProvider | None, optional): LLM model provider
            model (str | None, optional): LLM model name
            stream (bool, optional): whether to stream the agent's response
            enable_thinking (bool, optional): whether to enable thinking on the underlying LLM. Default to True.
            agent_state (str | None, optional): the agent's state as returned by the previous agent run. Default to None.
            storage_state (str | None, optional): the browser's storage state as returned by the previous agent run. Default to None.
            return_screenshots (bool, optional): whether to return screenshots of the agent's states at every step. Default to False.
            return_agent_state (bool, optional): whether to return the agent's state. Default to False.
            return_storage_state (bool, optional): whether to return the storage state. Default to False.
            disable_give_control (bool, optional): whether to NOT give the agent additional direction to give control to the user for tasks such as login. Default to False.
            timeout (int | None, optional): timeout seconds for the agent's response. Default to None.
            cdp_url (str | None, optional): Chrome DevTools Protocol URL of an existing browser session. Default to None.
            max_steps (int | None, optional): maximum number of steps the agent can take. If not set, the backend will use a default value (currently 100). Default to None.
            thinking_token_budget (int | None, optional): maximum number of tokens the underlying LLM can spend on thinking in each step, if supported by the model. Default to None.
            start_url (str | None, optional): the URL to start the agent on. Must be a valid URL - refer to https://playwright.dev/docs/api/class-page#page-goto. If not specified, the agent infers this from the prompt. Default to None.


        Returns:
            AgentOutput | AsyncIterator[RunAgentResponseChunk]: agent output or a generator of response chunks
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
            disable_give_control=disable_give_control,
            user_agent=user_agent,
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
            return await self.__run_non_streaming(request)

    async def __run_streaming(
        self, request: RunAgentRequest
    ) -> AsyncGenerator[RunAgentResponseChunk, None]:
        """Run agent in streaming mode.

        Args:
            request (RunAgentRequest): The request to run the agent with.

        Yields:
            RunAgentResponseChunk: Chunks of the agent's response.
        """
        async with self._client.stream(
            "POST",
            self._base_url + "/v1/agent/run",
            json=request.model_dump(by_alias=True),
            headers=self._headers(),
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(await response.read())
            async for line in response.aiter_lines():
                line = str(line)
                if line.startswith("[DONE]"):
                    break
                if not line.startswith("data: "):
                    continue
                line = line[6:]
                if line:
                    chunk = RunAgentResponseChunk.model_validate_json(line)
                    yield chunk.root
                    if chunk.root.chunk_type in ["finalOutput", "error", "timeout"]:
                        break

    async def __run_non_streaming(self, request: RunAgentRequest) -> AgentOutput:
        """Run agent in non-streaming mode.

        Args:
            request (RunAgentRequest): The request to run the agent with.

        Returns:
            AgentOutput: The agent's output.
        """
        final_chunk = None

        async with self._client.stream(
            "POST",
            self._base_url + "/v1/agent/run",
            json=request.model_dump(by_alias=True),
            headers=self._headers(),
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(await response.read())
            async for line in response.aiter_lines():
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
