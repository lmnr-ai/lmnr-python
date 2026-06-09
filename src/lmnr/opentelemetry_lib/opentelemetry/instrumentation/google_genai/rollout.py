"""
Debug replay wrapper for Google GenAI instrumentation.

On a debug run with replay configured, each live LLM call asks the server-side
cache (shared spec §7) what to do: on a HIT it reconstructs a
`GenerateContentResponse` from the cached span and serves it; on MISS / LIVE it
runs the call live. The decision is made by `cache_outcome_for` (sync) /
`acache_outcome_for` (async); there is no in-process cache anymore.
"""

import json
from typing import Any, AsyncGenerator, Generator

from google.genai import types

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    is_model_valid,
    to_dict,
)
from lmnr.sdk.debug.replay import (
    acache_outcome_for,
    cache_outcome_for,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class GoogleGenAIRolloutWrapper:
    """Serves cached Google GenAI responses on a debug run; runs live otherwise."""

    def _parse_output_json(self, output_str: str) -> Any:
        """
        Parse output JSON, handling potential double-stringification.

        Args:
            output_str: JSON string from span output

        Returns:
            Parsed JSON object
        """
        try:
            parsed = json.loads(output_str)
            # Try parsing again if it's still a string (double-stringified)
            while isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse output JSON: {e}")
            return None

    def _add_parsed_to_response(
        self,
        response: types.GenerateContentResponse,
        parts: list[dict[str, Any]],
        config: types.GenerateContentConfig | dict[str, Any] | None = None,
    ) -> types.GenerateContentResponse:
        # Handle structured output (parsed field)
        if config:
            response_schema = None
            if isinstance(config, dict):
                response_schema = config.get("response_schema")
            elif hasattr(config, "response_schema"):
                response_schema = config.response_schema

            if response_schema is not None:
                # Try to populate the parsed field
                parsed_value = self._parse_structured_output(parts, response_schema)
                if parsed_value is not None:
                    # Set the parsed field on the response
                    # Note: This is a bit hacky but necessary for cached responses
                    response.parsed = parsed_value

    def cached_response_to_google_genai(
        self, cached_span: dict[str, Any], config: Any | None = None
    ) -> types.GenerateContentResponse | None:
        """Convert cached span envelope to Google GenAI GenerateContentResponse."""

        def is_raw_genai_candidate_like(candidate: dict[str, Any]) -> bool:
            if not isinstance(candidate, dict):
                return False
            if content := candidate.get("content"):
                return (
                    isinstance(content, dict)
                    and isinstance(content.get("role"), str)
                    and isinstance(content.get("parts"), list)
                    and all(
                        is_model_valid(part, types.Part)
                        for part in content.get("parts", [])
                    )
                )
            return False

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
                if isinstance(raw, dict):
                    response = types.GenerateContentResponse.model_validate(raw)
                elif isinstance(raw, str):
                    response = types.GenerateContentResponse.model_validate_json(raw)
                else:
                    logger.warning(f"Unexpected raw response type: {type(raw)}")
                    return None
                if response:
                    self._add_parsed_to_response(
                        response,
                        response.candidates[0].content.parts,
                        config,
                    )
                    return response
                return None
            except Exception as e:
                logger.debug(
                    f"Failed to parse raw Google GenAI response: {e}", exc_info=True
                )
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list) or not messages:
                logger.warning("Cached span type='genAi' has no messages")
                return None
            finish_reasons = cached_span.get("finishReasons", [])
            finish_reason = finish_reasons[0] if finish_reasons else "STOP"

            if all(is_raw_genai_candidate_like(candidate) for candidate in messages):
                response = types.GenerateContentResponse.model_validate(
                    {"candidates": messages}
                )
                self._add_parsed_to_response(
                    response,
                    messages[0]["content"]["parts"],
                    config,
                )
                return response

            message = messages[0]
            content_blocks = message.get("content", [])

            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(types.Part(text=block.get("text", "")))
                elif block_type == "tool_call":
                    function_call = types.FunctionCall(
                        name=block.get("name", ""),
                        args=block.get("arguments", {}),
                    )
                    parts.append(types.Part(function_call=function_call))
                else:
                    logger.debug(f"Unknown content block type: {block_type}")

            content = types.Content(parts=parts, role="model")
            candidate = types.Candidate(
                content=content,
                finish_reason=finish_reason,
            )
            response = types.GenerateContentResponse(
                candidates=[candidate],
                usage_metadata=None,
                model_version=None,
            )
            self._add_parsed_to_response(response, content_blocks, config)
            return response

        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to Google GenAI format: {e}",
                exc_info=True,
            )
            return None

    def _parse_structured_output(
        self, content_blocks: list[dict[str, Any]], response_schema: Any
    ) -> Any | None:
        """
        Parse structured output from content blocks.

        Args:
            content_blocks: Content blocks from cached span
            response_schema: Response schema (pydantic model, Enum, or dict)

        Returns:
            Parsed value or None
        """
        try:
            # Extract text content
            text_content = ""
            for block in content_blocks:
                if (
                    to_dict(block).get("type") == "text"
                    or to_dict(block).get("text") is not None
                ):
                    text_content += to_dict(block).get("text", "")

            if not text_content:
                return None

            # Try to parse as JSON first
            try:
                json_data = json.loads(text_content)
            except json.JSONDecodeError:
                return None

            # If response_schema is a pydantic model, try to validate
            if hasattr(response_schema, "model_validate"):
                try:
                    return response_schema.model_validate(json_data)
                except Exception as e:
                    logger.debug(f"Failed to validate with pydantic model: {e}")
                    # Fall back to returning the dict
                    return json_data

            # If it's an Enum, try to construct it
            if hasattr(response_schema, "__members__"):
                try:
                    return response_schema(json_data)
                except Exception as e:
                    logger.debug(f"Failed to construct Enum: {e}")
                    return json_data

            # Otherwise, return the parsed JSON dict/list
            return json_data

        except Exception as e:
            logger.debug(f"Failed to parse structured output: {e}")
            return None

    def wrap_generate_content(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        is_streaming: bool = False,
        is_async: bool = False,
    ) -> Any:
        """
        Wrap generate_content with rollout logic.

        Args:
            wrapped: Original function
            instance: Instance object
            args: Positional arguments
            kwargs: Keyword arguments
            is_streaming: Whether this is a streaming method
            is_async: Whether this is an async method

        Returns:
            GenerateContentResponse (cached or live), or Generator/AsyncGenerator
        """
        # Capture the span synchronously: the caller's `Laminar.use_span(span)`
        # only spans the sync call to this method, so on the async path the
        # current-span context is already gone by the time the coroutine runs.
        span = Laminar.get_current_span()
        if is_async:
            return self._awrap_generate_content(
                wrapped, args, kwargs, span, is_streaming
            )

        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            config = kwargs.get("config")
            response = self.cached_response_to_google_genai(outcome.cached, config)
            if response is not None:
                logger.debug("Serving cached Google GenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

    async def _awrap_generate_content(
        self, wrapped, args, kwargs, span, is_streaming
    ) -> Any:
        """Async cache lookup; on a HIT serve cached, else run the call live."""
        outcome = await acache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            config = kwargs.get("config")
            response = self.cached_response_to_google_genai(outcome.cached, config)
            if response is not None:
                logger.debug("Serving cached Google GenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_async_cached_stream(response)
                return response

        return await wrapped(*args, **kwargs)

    def _create_cached_stream(
        self, response: types.GenerateContentResponse
    ) -> Generator[types.GenerateContentResponse, None, None]:
        """
        Create a generator that yields a cached response as a single chunk.

        Args:
            response: Cached response to yield

        Yields:
            types.GenerateContentResponse: The cached response
        """
        yield response

    async def _create_async_cached_stream(
        self, response: types.GenerateContentResponse
    ) -> AsyncGenerator[types.GenerateContentResponse, None]:
        """
        Create an async generator that yields a cached response as a single chunk.

        Args:
            response: Cached response to yield

        Yields:
            types.GenerateContentResponse: The cached response
        """
        yield response


# Singleton instance
_google_genai_rollout_wrapper: GoogleGenAIRolloutWrapper | None = None


def get_google_genai_rollout_wrapper() -> GoogleGenAIRolloutWrapper | None:
    """
    Get or create the Google GenAI rollout wrapper singleton.

    Returns:
        Optional[GoogleGenAIRolloutWrapper]: Wrapper instance or None if not in rollout mode
    """
    global _google_genai_rollout_wrapper

    if not replay_enabled():
        return None

    if _google_genai_rollout_wrapper is None:
        try:
            _google_genai_rollout_wrapper = GoogleGenAIRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create Google GenAI replay wrapper: {e}")
            return None

    return _google_genai_rollout_wrapper


def wrap_google_genai_for_rollout():
    """
    Apply the replay wrapper to Google GenAI instrumentation.

    This should be called at the end of the instrumentation process
    if a debug replay run is active.
    """
    if not replay_enabled():
        return

    wrapper = get_google_genai_rollout_wrapper()
    if not wrapper:
        logger.warning("Debug replay active but failed to create wrapper")
        return

    logger.debug("Google GenAI replay wrapper initialized")
