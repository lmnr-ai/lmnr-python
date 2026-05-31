"""
Debug replay wrapper for Google GenAI instrumentation.

On a debug run with replay enabled, serves the first N spine LLM calls from the
in-process `ReplayCache` (reconstructing a `GenerateContentResponse` from cached
attributes) and runs the rest live. Overrides + HTTP cache server are gone (§H).
"""

import json
from typing import Any, AsyncGenerator, Generator

from google.genai import types

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    is_model_valid,
    to_dict,
)
from lmnr.sdk.debug.replay import (
    cached_payload_for,
    mark_span_cached,
    replay_enabled,
    span_path_from_span,
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
        """
        Convert cached span data to Google GenAI GenerateContentResponse.

        Args:
            cached_span: Cached span data from cache server
            config: Optional config that may contain response_schema for structured output

        Returns:
            Optional[types.GenerateContentResponse]: Reconstructed response or None
        """

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

        if raw_response := cached_span.get("attributes", {}).get(
            "lmnr.sdk.raw.response"
        ):
            try:
                if isinstance(raw_response, dict):
                    response = types.GenerateContentResponse.model_validate(
                        raw_response
                    )
                elif isinstance(raw_response, str):
                    response = types.GenerateContentResponse.model_validate_json(
                        raw_response
                    )
                if response:
                    self._add_parsed_to_response(
                        response,
                        response.candidates[0].content.parts,
                        config,
                    )
                    return response
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")
                pass  # fallback to the legacy parsing path

        try:
            output_str = cached_span.get("output", "")
            if not output_str:
                logger.warning("Cached span has no output")
                return None

            # Parse the output JSON
            parsed = self._parse_output_json(output_str)
            if not parsed:
                return None

            # Expected format: [{"role":"model","content":[...]}]
            if not isinstance(parsed, list) or len(parsed) == 0:
                logger.warning(f"Unexpected output format: {type(parsed)}")
                return None

            if all(is_raw_genai_candidate_like(candidate) for candidate in parsed):
                response = types.GenerateContentResponse.model_validate(
                    {"candidates": parsed}
                )
                self._add_parsed_to_response(
                    response,
                    parsed[0]["content"]["parts"],
                    config,
                )
                return response
            message = parsed[0]
            content_blocks = message.get("content", [])

            # Convert content blocks to Google GenAI Parts
            parts = []
            for block in content_blocks:
                block_type = block.get("type")

                if block_type == "text":
                    parts.append(types.Part(text=block.get("text", "")))

                elif block_type == "tool_call":
                    # Convert to function_call
                    function_call = types.FunctionCall(
                        name=block.get("name", ""),
                        args=block.get("arguments", {}),
                    )
                    parts.append(types.Part(function_call=function_call))

                else:
                    logger.debug(f"Unknown content block type: {block_type}")

            # Build the response
            content = types.Content(parts=parts, role="model")

            # Get finish reason from attributes
            attributes = cached_span.get("attributes", {})
            finish_reason = attributes.get("ai.response.finishReason", "stop")

            candidate = types.Candidate(
                content=content,
                finish_reason=finish_reason,
            )

            response = types.GenerateContentResponse(
                candidates=[candidate],
                usage_metadata=None,  # Cached responses don't track tokens
                model_version=None,
            )
            self._add_parsed_to_response(response, content_blocks, config)

            return response

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to Google GenAI format: {e}",
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
        span = Laminar.get_current_span()
        span_path = span_path_from_span(span)
        cached = cached_payload_for(span_path)
        if cached is not None:
            config = kwargs.get("config")
            response = self.cached_response_to_google_genai(cached, config)
            if response is not None:
                logger.debug("Replaying cached Google GenAI response at %s", span_path)
                mark_span_cached(span)
                if is_streaming:
                    if is_async:
                        return self._create_async_cached_stream(response)
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

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
