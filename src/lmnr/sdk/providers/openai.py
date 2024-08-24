from .base import Provider
from .utils import parse_or_dump_to_dict

from collections import defaultdict
from typing import Any, Optional, Union
import logging
import pydantic

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class OpenAI(Provider):
    logger = logging.getLogger("lmnr.sdk.tracing.providers.openai")

    def display_name(self) -> str:
        return "OpenAI"

    def stream_list_to_dict(
        self, response: list[Union[ChatCompletionChunk, str]]
    ) -> dict[str, Any]:
        model = None
        finish_reasons = []
        output_tokens = 0
        outputs = defaultdict(lambda: defaultdict(str))
        try:
            for chunk in response:
                chunk = parse_or_dump_to_dict(chunk)
                if model is None:
                    model = chunk["model"]
                finish_reasons = [
                    choice.get("finish_reason") for choice in chunk.get("choices", [])
                ]
                for i, choice in enumerate(chunk.get("choices", [])):
                    if choice["delta"] and isinstance(choice["delta"], dict):
                        for key in choice["delta"]:
                            if choice["delta"][key] is None:
                                if key not in outputs[i]:
                                    outputs[i][key] = None
                                continue
                            outputs[i][key] += choice["delta"][key]
                        output_tokens += 1
        except Exception as e:
            self.logger.error(f"Error parsing streamming response: {e}")
            pass

        output_key_values = [
            self._message_to_key_and_output(dict(outputs[i]))
            for i in range(len(outputs))
        ]
        return {
            "model": model,
            "prompt_tokens": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": output_tokens,
                "total_tokens": output_tokens,
                "finish_reason": self._from_singleton_list(finish_reasons),
            },
            "choices": [
                (
                    {
                        "message": {
                            output[0]: output[1],
                            "role": "assistant",
                        }
                    }
                    if output
                    else None
                )
                for output in output_key_values
            ],
        }

    def extract_llm_attributes_from_response(
        self, response: Union[str, dict[str, Any], pydantic.BaseModel]
    ) -> dict[str, Any]:
        obj = parse_or_dump_to_dict(response)

        choices = obj.get("choices", [])
        decisions = []
        for choice in choices:
            # choice = parse_or_dump_to_dict(choice)
            if choice.get("content"):
                decisions.append("completion")
            elif choice.get("refusal"):
                decisions.append("refusal")
            elif choice.get("tool_calls"):
                decisions.append("tool_calls")
            else:
                decisions.append(None)

        return {
            "response_model": obj.get("model"),
            "input_token_count": obj.get("usage", {}).get("prompt_tokens"),
            "output_token_count": obj.get("usage", {}).get("completion_tokens"),
            "total_token_count": obj.get("usage", {}).get("total_tokens"),
            "finish_reason": obj.get("finish_reason"),
            "decision": self._from_singleton_list(decisions),
        }

    def extract_llm_output(
        self, result: Union[str, dict[str, Any], ChatCompletion]
    ) -> Any:
        result = parse_or_dump_to_dict(result)
        choices = result.get("choices")
        if not choices:
            return None
        outputs = [choice["message"] for choice in choices]

        return self._from_singleton_list(outputs)

    def extract_llm_attributes_from_args(
        self, func_args: list[Any], func_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "request_model": func_kwargs.get("model"),
            "temperature": func_kwargs.get("temperature"),
            "top_p": func_kwargs.get("top_p"),
            "stream": func_kwargs.get("stream", False),
        }

    def _message_to_key_and_output(
        self, message: Union[dict[str, Any], ChatCompletionMessage]
    ) -> Optional[tuple[str, str]]:
        message = parse_or_dump_to_dict(message)

        for key in ["content", "refusal", "tool_calls"]:
            if message.get(key) is not None:
                return (key, message[key])
        return None

    def _from_singleton_list(self, obj: Any) -> Any:
        # OpenAI returns list of choices. This will have more than one item
        # only if request parameter `n` is specified and is grate than 1.
        # That's a rare case, so we return the [contents of the] choice alone if there is just one.
        if isinstance(obj, list) and len(obj) == 1:
            return obj[0]
        return obj
