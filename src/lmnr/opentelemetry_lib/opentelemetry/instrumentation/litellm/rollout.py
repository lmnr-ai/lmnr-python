"""
Rollout wrapper for LiteLLM instrumentation.

This module adds caching and override capabilities to LiteLLM instrumentation
during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout_control import is_rollout_mode

logger = get_default_logger(__name__)


class DualIteratorWrapper:
    """
    Wrapper that supports both sync and async iteration.
    This allows cached streaming responses to work in both contexts.
    """

    def __init__(self, sync_iterator, cached_response=None):
        """
        Args:
            sync_iterator: A sync iterator/generator to wrap
            cached_response: Optional cached response object for setting span attributes
        """
        self._sync_iterator = sync_iterator
        self._items = None  # Cache items for potential reuse
        self._cached_response = cached_response

    def set_span_attributes(self, span, record_raw_response=False):
        """
        Set span attributes from the cached response without consuming the iterator.

        Args:
            span: The span to set attributes on
            record_raw_response: Whether to record raw response
        """
        if not self._cached_response:
            return

        from lmnr.sdk.utils import json_dumps
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
            set_span_attribute,
            to_dict,
        )

        response_dict = to_dict(self._cached_response)

        # Set common attributes based on response type
        set_span_attribute(span, "gen_ai.response.id", response_dict.get("id"))
        set_span_attribute(span, "gen_ai.response.model", response_dict.get("model"))

        # For completion responses
        if "choices" in response_dict:
            set_span_attribute(
                span,
                "gen_ai.response.system_fingerprint",
                response_dict.get("system_fingerprint"),
            )
            messages = [
                choice.get("message", {}) for choice in response_dict.get("choices", [])
            ]
            span.set_attribute("gen_ai.output.messages", json_dumps(messages))

        # For responses API
        elif "output" in response_dict:
            final_items = []
            if reasoning := response_dict.get("reasoning"):
                reasoning_dict = to_dict(reasoning)
                if reasoning_dict.get("summary") or reasoning_dict.get("effort"):
                    final_items.append(reasoning_dict)
            if isinstance(response_dict.get("output"), list):
                for item in response_dict.get("output"):
                    final_items.append(to_dict(item))
            span.set_attribute("gen_ai.output.messages", json_dumps(final_items))

        # Record raw response in rollout mode
        if record_raw_response:
            set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(response_dict))

    def __iter__(self):
        """Sync iteration support."""
        if self._items is not None:
            # If we already consumed it, replay from cache
            for item in self._items:
                yield item
            return

        # First time iteration - consume and cache
        self._items = []
        for item in self._sync_iterator:
            self._items.append(item)
            yield item

    async def __aiter__(self):
        """Async iteration support."""
        if self._items is not None:
            # If we already consumed it, replay from cache
            for item in self._items:
                yield item
        else:
            # First time iteration - consume and cache
            self._items = []
            for item in self._sync_iterator:
                self._items.append(item)
                yield item


def _import_litellm_types():
    """Lazy import litellm types to avoid import errors."""
    types = {
        "ModelResponse": None,
        "ResponsesAPIResponse": None,
        "Delta": None,
        "ResponseCompletedEvent": None,
    }

    try:
        from litellm import ModelResponse

        types["ModelResponse"] = ModelResponse
    except ImportError:
        pass

    try:
        from litellm.responses.main import ResponsesAPIResponse

        types["ResponsesAPIResponse"] = ResponsesAPIResponse
    except ImportError:
        pass

    try:
        from litellm.types.utils import Delta

        types["Delta"] = Delta
    except ImportError:
        pass

    try:
        from litellm.types.llms.openai import ResponseCompletedEvent

        types["ResponseCompletedEvent"] = ResponseCompletedEvent
    except ImportError:
        pass

    return types


class LiteLLMRolloutWrapper(RolloutInstrumentationWrapper):
    """
    Rollout wrapper specific to LiteLLM instrumentation.

    Handles:
    - Converting cached responses to LiteLLM format (both completion and responses)
    - Applying overrides to system prompts and tool definitions
    - Setting rollout-specific span attributes
    """

    def cached_response_to_completion(self, cached_span: dict[str, Any]) -> Any:
        """
        Convert cached span data to LiteLLM ModelResponse format.

        Args:
            cached_span: Cached span data from cache server

        Returns:
            ModelResponse object, dict, or None
        """
        types = _import_litellm_types()
        ModelResponse = types["ModelResponse"]

        # Try to parse from raw response first
        if raw_response := cached_span.get("attributes", {}).get(
            "lmnr.sdk.raw.response"
        ):
            try:
                response_dict = None
                if isinstance(raw_response, dict):
                    response_dict = raw_response
                elif isinstance(raw_response, str):
                    response_dict = json.loads(raw_response)

                if response_dict and ModelResponse:
                    try:
                        return ModelResponse.model_validate(response_dict)
                    except Exception as e:
                        logger.debug(
                            f"Failed to validate ModelResponse, returning dict: {e}"
                        )
                        return response_dict
                elif response_dict:
                    return response_dict
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")
                pass  # fallback to the legacy parsing path

        try:
            attributes = cached_span.get("attributes", {})
            output_str = cached_span.get("output", "")

            if not output_str:
                logger.warning("Cached span has no output")
                return None

            # Parse the output JSON - should be array of messages
            messages = json.loads(output_str)
            if not isinstance(messages, list):
                logger.warning(f"Unexpected output format: {type(messages)}")
                return None

            # Reconstruct ModelResponse dict
            choices = []
            for i, message in enumerate(messages):
                choices.append(
                    {
                        "index": i,
                        "message": message,
                        "finish_reason": "stop",  # Default for cached responses
                    }
                )

            response_dict = {
                "id": attributes.get("gen_ai.response.id", "cached"),
                "model": attributes.get("gen_ai.response.model", "unknown"),
                "choices": choices,
                "created": int(cached_span.get("start_time", 0) / 1_000_000_000),
                "object": "chat.completion",
                "system_fingerprint": attributes.get(
                    "gen_ai.response.system_fingerprint"
                ),
                "usage": None,  # Cached responses don't track tokens
            }

            # Try to construct ModelResponse object
            if ModelResponse:
                try:
                    return ModelResponse.model_validate(response_dict)
                except Exception as e:
                    logger.debug(
                        f"Failed to validate ModelResponse, returning dict: {e}"
                    )
                    return response_dict

            return response_dict

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to completion format: {e}",
                exc_info=True,
            )
            return None

    def cached_response_to_responses(self, cached_span: dict[str, Any]) -> Any:
        """
        Convert cached span data to LiteLLM ResponsesAPIResponse format.

        Args:
            cached_span: Cached span data from cache server

        Returns:
            ResponsesAPIResponse object, dict, or None
        """
        types = _import_litellm_types()
        ResponsesAPIResponse = types["ResponsesAPIResponse"]

        # Try to parse from raw response first
        if raw_response := cached_span.get("attributes", {}).get(
            "lmnr.sdk.raw.response"
        ):
            try:
                response_dict = None
                if isinstance(raw_response, dict):
                    response_dict = raw_response
                elif isinstance(raw_response, str):
                    response_dict = json.loads(raw_response)

                if response_dict and ResponsesAPIResponse:
                    try:
                        return ResponsesAPIResponse.model_validate(response_dict)
                    except Exception as e:
                        logger.debug(
                            f"Failed to validate ResponsesAPIResponse, returning dict: {e}"
                        )
                        return response_dict
                elif response_dict:
                    return response_dict
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")
                pass  # fallback to the legacy parsing path

        try:
            attributes = cached_span.get("attributes", {})
            output_str = cached_span.get("output", "")

            if not output_str:
                logger.warning("Cached span has no output")
                return None

            # Parse the output JSON - may contain reasoning + output items
            items = json.loads(output_str)
            if not isinstance(items, list):
                logger.warning(f"Unexpected output format: {type(items)}")
                return None

            # Separate reasoning from output items
            reasoning = None
            output_items = []

            for item in items:
                # Check if it's a reasoning object (has summary or effort)
                if isinstance(item, dict) and (
                    item.get("summary") or item.get("effort")
                ):
                    reasoning = item
                else:
                    output_items.append(item)

            # Reconstruct ResponsesAPIResponse dict
            response_dict = {
                "id": attributes.get("gen_ai.response.id", "cached"),
                "model": attributes.get("gen_ai.response.model", "unknown"),
                "output": output_items,
                "usage": None,  # Cached responses don't track tokens
                "created_at": int(cached_span.get("end_time", 0) / 1_000_000_000),
            }

            if reasoning:
                response_dict["reasoning"] = reasoning

            # Try to construct ResponsesAPIResponse object
            if ResponsesAPIResponse:
                try:
                    return ResponsesAPIResponse.model_validate(response_dict)
                except Exception as e:
                    logger.debug(
                        f"Failed to validate ResponsesAPIResponse, returning dict: {e}"
                    )
                    return response_dict

            return response_dict

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to responses format: {e}",
                exc_info=True,
            )
            return None

    def _create_cached_completion_stream(
        self, response: Any
    ) -> Generator[Any, None, None]:
        """
        Create a generator that yields a cached completion response as a single chunk.

        Args:
            response: Cached response to yield (ModelResponse or dict)

        Yields:
            Chunk object or dict: The cached response formatted as a chunk
        """
        types = _import_litellm_types()
        ModelResponse = types["ModelResponse"]
        Delta = types["Delta"]

        # Convert response to dict if needed
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = response

        # Convert the full response to a streaming chunk format
        chunk_dict = {
            "id": response_dict.get("id"),
            "model": response_dict.get("model"),
            "object": "chat.completion.chunk",
            "created": response_dict.get("created"),
            "choices": [],
        }

        for choice in response_dict.get("choices", []):
            chunk_dict["choices"].append(
                {
                    "index": choice.get("index", 0),
                    "delta": choice.get("message", {}),
                    "finish_reason": choice.get("finish_reason"),
                }
            )

        # Try to validate as ModelResponse if available
        if ModelResponse:
            try:
                chunk = ModelResponse.model_validate(chunk_dict)
                if Delta:
                    for choice in chunk.choices:
                        if hasattr(choice, "delta"):
                            choice.delta = Delta.model_validate(choice.delta)
                yield chunk
                return
            except Exception:
                logger.debug("Failed to validate ModelResponse", exc_info=True)
                pass

        yield chunk_dict

    async def _create_async_cached_completion_stream(
        self, response: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Create an async generator that yields a cached completion response as a single chunk.

        Args:
            response: Cached response to yield (ModelResponse or dict)

        Yields:
            Chunk object or dict: The cached response formatted as a chunk
        """
        types = _import_litellm_types()
        ModelResponse = types["ModelResponse"]
        Delta = types["Delta"]

        # Convert response to dict if needed
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = response

        # Convert the full response to a streaming chunk format
        chunk_dict = {
            "id": response_dict.get("id"),
            "model": response_dict.get("model"),
            "object": "chat.completion.chunk",
            "created": response_dict.get("created"),
            "choices": [],
        }

        for choice in response_dict.get("choices", []):
            chunk_dict["choices"].append(
                {
                    "index": choice.get("index", 0),
                    "delta": choice.get("message", {}),
                    "finish_reason": choice.get("finish_reason"),
                }
            )

        # Try to validate as ModelResponse if available
        if ModelResponse:
            try:
                chunk = ModelResponse.model_validate(chunk_dict)
                if Delta:
                    for choice in chunk.choices:
                        if hasattr(choice, "delta"):
                            choice.delta = Delta.model_validate(choice.delta)
                yield chunk
                return
            except Exception:
                pass

        yield chunk_dict

    def _create_cached_responses_stream(
        self, response: Any
    ) -> Generator[Any, None, None]:
        """
        Create a generator that yields a cached responses response as a completed event.

        Args:
            response: Cached response to yield (ResponsesAPIResponse or dict)

        Yields:
            ResponseCompletedEvent or dict: The cached response formatted as a response.completed event
        """
        types = _import_litellm_types()
        ResponseCompletedEvent = types["ResponseCompletedEvent"]

        event_dict = {
            "type": "response.completed",
            "response": response,
        }

        # Try to validate as ResponseCompletedEvent if available
        if ResponseCompletedEvent:
            try:
                event = ResponseCompletedEvent.model_validate(event_dict)
                yield event
                return
            except Exception:
                logger.debug("Failed to validate ResponseCompletedEvent", exc_info=True)
                pass

        yield event_dict

    async def _create_async_cached_responses_stream(
        self, response: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Create an async generator that yields a cached responses response as a completed event.

        Args:
            response: Cached response to yield (ResponsesAPIResponse or dict)

        Yields:
            ResponseCompletedEvent or dict: The cached response formatted as a response.completed event
        """
        types = _import_litellm_types()
        ResponseCompletedEvent = types["ResponseCompletedEvent"]

        event_dict = {
            "type": "response.completed",
            "response": response,
        }

        # Try to validate as ResponseCompletedEvent if available
        if ResponseCompletedEvent:
            try:
                event = ResponseCompletedEvent.model_validate(event_dict)
                yield event
                return
            except Exception:
                logger.debug("Failed to validate ResponseCompletedEvent", exc_info=True)
                pass

        yield event_dict

    def wrap_completion(
        self,
        wrapped,
        args,
        kwargs,
        is_streaming: bool = False,
    ) -> Any:
        """
        Wrap completion with rollout logic.

        Args:
            wrapped: Original function
            instance: Instance object
            args: Positional arguments
            kwargs: Keyword arguments
            is_streaming: Whether this is a streaming call
            is_async: Whether this is an async call

        Returns:
            Response (cached or live), or Generator/AsyncGenerator
        """
        # Check if rollout mode is active
        if not self.should_use_rollout():
            return wrapped(*args, **kwargs)

        # Get span path
        span_path = self.get_span_path()
        if not span_path:
            return wrapped(*args, **kwargs)

        # Get current call index
        current_index = self.get_current_index_for_path(span_path)

        logger.debug(f"LiteLLM completion call at {span_path}:{current_index}")
        span = Laminar.get_current_span()

        # Check cache
        if self.should_use_cache(span_path, current_index):
            cached_span = self.get_cached_response(span_path, current_index)
            if cached_span:
                logger.debug(
                    f"Using cached response for LiteLLM completion at {span_path}:{current_index}"
                )

                # Convert cached data to completion response
                response = self.cached_response_to_completion(cached_span)
                if response:
                    # Mark span as cached
                    try:
                        if span and span.is_recording():
                            span.set_attributes(
                                {
                                    "lmnr.span.type": "CACHED",
                                    "lmnr.rollout.cache_index": current_index,
                                    "lmnr.span.original_type": "LLM",
                                }
                            )
                    except Exception:
                        pass

                    # For streaming, return a dual iterator that works in both sync and async contexts
                    if is_streaming:
                        sync_gen = self._create_cached_completion_stream(response)
                        return DualIteratorWrapper(sync_gen, cached_response=response)

                    return response

        # Cache miss or conversion failed - execute live
        logger.debug(
            f"Executing live LiteLLM completion call for {span_path}:{current_index}"
        )
        return wrapped(*args, **kwargs)

    def wrap_responses(
        self,
        wrapped,
        args,
        kwargs,
        is_streaming: bool = False,
    ) -> Any:
        """
        Wrap responses with rollout logic.

        Args:
            wrapped: Original function
            args: Positional arguments
            kwargs: Keyword arguments
            is_streaming: Whether this is a streaming call

        Returns:
            Response (cached or live), or Generator/AsyncGenerator
        """
        # Check if rollout mode is active
        if not self.should_use_rollout():
            return wrapped(*args, **kwargs)

        # Get span path
        span_path = self.get_span_path()
        if not span_path:
            return wrapped(*args, **kwargs)

        # Get current call index
        current_index = self.get_current_index_for_path(span_path)

        logger.debug(f"LiteLLM responses call at {span_path}:{current_index}")
        span = Laminar.get_current_span()

        # Check cache
        if self.should_use_cache(span_path, current_index):
            cached_span = self.get_cached_response(span_path, current_index)
            if cached_span:
                logger.debug(
                    f"Using cached response for LiteLLM responses at {span_path}:{current_index}"
                )

                # Convert cached data to responses response
                response = self.cached_response_to_responses(cached_span)
                if response:
                    # Mark span as cached
                    try:
                        if span and span.is_recording():
                            span.set_attributes(
                                {
                                    "lmnr.span.type": "CACHED",
                                    "lmnr.rollout.cache_index": current_index,
                                    "lmnr.span.original_type": "LLM",
                                }
                            )
                    except Exception:
                        pass

                    # For streaming, return a dual iterator that works in both sync and async contexts
                    if is_streaming:
                        sync_gen = self._create_cached_responses_stream(response)
                        return DualIteratorWrapper(sync_gen, cached_response=response)

                    return response

        # Cache miss or conversion failed - execute live
        logger.debug(
            f"Executing live LiteLLM responses call for {span_path}:{current_index}"
        )
        return wrapped(*args, **kwargs)


# Singleton instance
_litellm_rollout_wrapper: LiteLLMRolloutWrapper | None = None


def get_litellm_rollout_wrapper() -> LiteLLMRolloutWrapper | None:
    """
    Get or create the LiteLLM rollout wrapper singleton.

    Returns:
        Optional[LiteLLMRolloutWrapper]: Wrapper instance or None if not in rollout mode
    """
    global _litellm_rollout_wrapper

    if not is_rollout_mode():
        return None

    if _litellm_rollout_wrapper is None:
        try:
            _litellm_rollout_wrapper = LiteLLMRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create LiteLLM rollout wrapper: {e}")
            return None

    return _litellm_rollout_wrapper
