"""
Debug replay wrapper for LiteLLM instrumentation.

On a debug run with replay configured, each live LLM call asks the server-side
cache (shared spec §7) what to do: on a HIT it reconstructs a LiteLLM
`ModelResponse` / `ResponsesAPIResponse` from the cached span and serves it; on
MISS / LIVE it runs the call live. The decision is made by `cache_outcome_for`.

LiteLLM has no async cache helper: its instrumentation dispatches both sync and
async calls through these sync wrappers and only checks `inspect.iscoroutine` on
the result, so the cache lookup is a blocking HTTP call even on the async path
(a known v1 limitation carried forward — the lookup is small and one-shot).
"""

import json
from typing import Any, AsyncGenerator, Generator

from lmnr.sdk.debug.replay import (
    cache_outcome_for,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

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


class LiteLLMRolloutWrapper:
    """Serves cached LiteLLM responses on a debug run; runs live otherwise."""

    def cached_response_to_completion(self, cached_span: dict[str, Any]) -> Any:
        """Convert cached span envelope to LiteLLM ModelResponse format."""
        types_dict = _import_litellm_types()
        ModelResponse = types_dict["ModelResponse"]

        envelope_type = cached_span.get("type")
        if envelope_type not in ("raw", "genAi"):
            logger.warning(f"Unknown cached span type: {envelope_type!r}")
            return None

        if envelope_type == "raw":
            try:
                raw = cached_span.get("response")
                if not raw:
                    logger.warning("Cached span type='raw' has no response field")
                    return None
                response_dict = raw if isinstance(raw, dict) else json.loads(raw)
                if ModelResponse:
                    try:
                        return ModelResponse.model_validate(response_dict)
                    except Exception as e:
                        logger.debug(f"Failed to validate ModelResponse, returning dict: {e}")
                        return response_dict
                return response_dict
            except Exception as e:
                logger.debug(f"Failed to parse raw LiteLLM completion response: {e}", exc_info=True)
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")
            finish_reasons = cached_span.get("finishReasons", [])

            choices = []
            for i, message in enumerate(messages):
                finish_reason = finish_reasons[i] if i < len(finish_reasons) else "stop"
                choices.append({"index": i, "message": message, "finish_reason": finish_reason})

            response_dict = {
                "id": "cached",
                "model": model,
                "choices": choices,
                "created": 0,
                "object": "chat.completion",
                "usage": None,
            }

            if ModelResponse:
                try:
                    return ModelResponse.model_validate(response_dict)
                except Exception as e:
                    logger.debug(f"Failed to validate ModelResponse, returning dict: {e}")
                    return response_dict
            return response_dict

        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to LiteLLM completion format: {e}",
                exc_info=True,
            )
            return None

    def cached_response_to_responses(self, cached_span: dict[str, Any]) -> Any:
        """Convert cached span envelope to LiteLLM ResponsesAPIResponse format."""
        types_dict = _import_litellm_types()
        ResponsesAPIResponse = types_dict["ResponsesAPIResponse"]

        envelope_type = cached_span.get("type")
        if envelope_type not in ("raw", "genAi"):
            logger.warning(f"Unknown cached span type: {envelope_type!r}")
            return None

        if envelope_type == "raw":
            try:
                raw = cached_span.get("response")
                if not raw:
                    logger.warning("Cached span type='raw' has no response field")
                    return None
                response_dict = raw if isinstance(raw, dict) else json.loads(raw)
                if ResponsesAPIResponse:
                    try:
                        return ResponsesAPIResponse.model_validate(response_dict)
                    except Exception as e:
                        logger.debug(f"Failed to validate ResponsesAPIResponse, returning dict: {e}")
                        return response_dict
                return response_dict
            except Exception as e:
                logger.debug(f"Failed to parse raw LiteLLM responses response: {e}", exc_info=True)
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")

            reasoning = None
            output_items = []
            for item in messages:
                if isinstance(item, dict) and (item.get("summary") or item.get("effort")):
                    reasoning = item
                else:
                    output_items.append(item)

            response_dict = {
                "id": "cached",
                "model": model,
                "output": output_items,
                "usage": None,
                "created_at": 0,
            }
            if reasoning:
                response_dict["reasoning"] = reasoning

            if ResponsesAPIResponse:
                try:
                    return ResponsesAPIResponse.model_validate(response_dict)
                except Exception as e:
                    logger.debug(f"Failed to validate ResponsesAPIResponse, returning dict: {e}")
                    return response_dict
            return response_dict

        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to LiteLLM responses format: {e}",
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
        """Serve a cached completion response if available, else run live."""
        span = Laminar.get_current_span()
        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_completion(outcome.cached)
            if response is not None:
                logger.debug("Serving cached LiteLLM completion from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    sync_gen = self._create_cached_completion_stream(response)
                    return DualIteratorWrapper(sync_gen, cached_response=response)
                return response

        return wrapped(*args, **kwargs)

    def wrap_responses(
        self,
        wrapped,
        args,
        kwargs,
        is_streaming: bool = False,
    ) -> Any:
        """Serve a cached responses-API response if available, else run live."""
        span = Laminar.get_current_span()
        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_responses(outcome.cached)
            if response is not None:
                logger.debug("Serving cached LiteLLM responses from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    sync_gen = self._create_cached_responses_stream(response)
                    return DualIteratorWrapper(sync_gen, cached_response=response)
                return response

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

    if not replay_enabled():
        return None

    if _litellm_rollout_wrapper is None:
        try:
            _litellm_rollout_wrapper = LiteLLMRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create LiteLLM debugger wrapper: {e}")
            return None

    return _litellm_rollout_wrapper
