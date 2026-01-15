"""
Rollout wrapper for Google GenAI instrumentation.

This module adds caching and override capabilities to Google GenAI instrumentation
during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator

from google.genai import types
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    is_model_valid,
    to_dict,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout_control import is_rollout_mode

logger = get_default_logger(__name__)


class GoogleGenAIRolloutWrapper(RolloutInstrumentationWrapper):
    """
    Rollout wrapper specific to Google GenAI instrumentation.

    Handles:
    - Converting cached responses to Google GenAI format
    - Applying overrides to system prompts and tool definitions
    - Setting rollout-specific span attributes
    """

    def apply_google_genai_overrides(
        self, kwargs: dict[str, Any], path: str, span: Span | None = None
    ) -> dict[str, Any]:
        """
        Apply overrides to Google GenAI specific parameters.

        Override mapping:
        - system -> config.system_instruction
        - tools -> config.tools or kwargs.tools

        Args:
            kwargs: Original kwargs for generate_content
            path: Current span path for getting path-specific overrides

        Returns:
            dict[str, Any]: Modified kwargs
        """
        overrides = self.get_overrides(path)
        if not overrides:
            return kwargs

        modified_kwargs = kwargs.copy()

        # Apply system instruction override
        if "system" in overrides:
            system_override = overrides["system"]
            logger.debug(f"Applying system override for {path}")

            if span and span.is_recording():
                if span.attributes.get("gen_ai.prompt.0.role") == "system":
                    span.set_attribute(
                        "gen_ai.prompt.0.content",
                        system_override,
                    )

            # Ensure config exists
            if "config" not in modified_kwargs or modified_kwargs["config"] is None:
                modified_kwargs["config"] = {}

            # Handle config as dict or types.GenerateContentConfig
            if isinstance(modified_kwargs["config"], dict):
                modified_kwargs["config"]["system_instruction"] = system_override
            else:
                # It's a GenerateContentConfig object - convert to dict, modify, and back
                config_dict = (
                    modified_kwargs["config"].__dict__.copy()
                    if hasattr(modified_kwargs["config"], "__dict__")
                    else {}
                )
                config_dict["system_instruction"] = system_override
                modified_kwargs["config"] = config_dict

        # Apply tool overrides
        if "tools" in overrides:
            tool_overrides = overrides["tools"]
            logger.debug(
                f"Applying tool overrides for {path}: {len(tool_overrides)} tools"
            )

            # Get existing tools from config or kwargs
            existing_tools = None
            if "config" in modified_kwargs and modified_kwargs["config"] is not None:
                if isinstance(modified_kwargs["config"], dict):
                    existing_tools = modified_kwargs["config"].get("tools")
                else:
                    existing_tools = getattr(modified_kwargs["config"], "tools", None)

            if existing_tools is None:
                existing_tools = modified_kwargs.get("tools")

            # Apply overrides to tools
            updated_tools = self._apply_tool_overrides(existing_tools, tool_overrides)

            # Set updated tools back
            if updated_tools is not None:
                # Prefer setting in config
                if "config" not in modified_kwargs or modified_kwargs["config"] is None:
                    modified_kwargs["config"] = {}

                if isinstance(modified_kwargs["config"], dict):
                    modified_kwargs["config"]["tools"] = updated_tools
                else:
                    config_dict = (
                        modified_kwargs["config"].__dict__.copy()
                        if hasattr(modified_kwargs["config"], "__dict__")
                        else {}
                    )
                    config_dict["tools"] = updated_tools
                    modified_kwargs["config"] = config_dict

        return modified_kwargs

    def _apply_tool_overrides(
        self,
        existing_tools: list[Any] | None,
        tool_overrides: list[dict[str, Any]],
    ) -> list[types.Tool] | None:
        """
        Apply tool overrides to existing tools.

        Args:
            existing_tools: Existing tools (list of types.Tool or dicts)
            tool_overrides: Override definitions with name, description, parameters

        Returns:
            Optional[List[types.Tool]]: Updated tools list
        """
        if not tool_overrides:
            return existing_tools

        # Convert existing tools to a workable format
        tools_by_name: dict[str, types.FunctionDeclaration] = {}

        if existing_tools:
            for tool in existing_tools:
                if isinstance(tool, types.Tool):
                    for func_decl in tool.function_declarations or []:
                        tools_by_name[func_decl.name] = func_decl
                elif isinstance(tool, dict) and "function_declarations" in tool:
                    for func_decl in tool["function_declarations"]:
                        if isinstance(func_decl, types.FunctionDeclaration):
                            tools_by_name[func_decl.name] = func_decl
                        elif isinstance(func_decl, dict):
                            tools_by_name[func_decl["name"]] = (
                                types.FunctionDeclaration(**func_decl)
                            )

        # Apply overrides
        for override in tool_overrides:
            name = override.get("name")
            if not name:
                continue

            if name in tools_by_name:
                # Update existing tool
                existing_decl = tools_by_name[name]
                updated_decl = types.FunctionDeclaration(
                    name=name,
                    description=override.get("description", existing_decl.description),
                    parameters=override.get("parameters", existing_decl.parameters),
                )
                tools_by_name[name] = updated_decl
            else:
                # Add new tool if it has parameters
                if "parameters" in override:
                    new_decl = types.FunctionDeclaration(
                        name=name,
                        description=override.get("description"),
                        parameters=override["parameters"],
                    )
                    tools_by_name[name] = new_decl

        # Convert back to list of Tool objects
        if tools_by_name:
            return [types.Tool(function_declarations=list(tools_by_name.values()))]

        return None

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
        # Check if rollout mode is active
        if not self.should_use_rollout():
            if is_async:
                return wrapped(*args, **kwargs)
            else:
                return wrapped(*args, **kwargs)

        # Get span path
        span_path = self.get_span_path()
        if not span_path:
            if is_async:
                return wrapped(*args, **kwargs)
            else:
                return wrapped(*args, **kwargs)

        # Get current call index
        current_index = self.get_current_index_for_path(span_path)

        logger.debug(f"Google GenAI call at {span_path}:{current_index}")
        span = Laminar.get_current_span()

        # Check cache
        if self.should_use_cache(span_path, current_index):
            cached_span = self.get_cached_response(span_path, current_index)
            if cached_span:
                logger.debug(
                    f"Using cached response for Google GenAI at {span_path}:{current_index}"
                )

                # Get config for structured output handling
                config = kwargs.get("config")

                # Convert cached data to Google GenAI response
                response = self.cached_response_to_google_genai(cached_span, config)
                if response:
                    # Mark span as cached (if span is available)
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

                    # For streaming methods, wrap response in a generator
                    if is_streaming:
                        if is_async:
                            return self._create_async_cached_stream(response)
                        else:
                            return self._create_cached_stream(response)

                    return response

        # Cache miss or conversion failed - apply overrides and execute
        modified_kwargs = self.apply_google_genai_overrides(kwargs, span_path, span)

        logger.debug(
            f"Executing live Google GenAI call for {span_path}:{current_index}"
        )
        return wrapped(*args, **modified_kwargs)

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

    if not is_rollout_mode():
        return None

    if _google_genai_rollout_wrapper is None:
        try:
            _google_genai_rollout_wrapper = GoogleGenAIRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create Google GenAI rollout wrapper: {e}")
            return None

    return _google_genai_rollout_wrapper


def wrap_google_genai_for_rollout():
    """
    Apply rollout wrapper to Google GenAI instrumentation.

    This should be called at the end of the instrumentation process
    if rollout mode is active.
    """
    if not is_rollout_mode():
        return

    wrapper = get_google_genai_rollout_wrapper()
    if not wrapper:
        logger.warning("Rollout mode active but failed to create wrapper")
        return

    logger.info("Google GenAI rollout wrapper initialized")
