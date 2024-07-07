import requests
import json

from lmnr_engine.engine.action import NodeRunError, RunOutput
from lmnr_engine.types import ChatMessage, NodeInput


{% for task in cookiecutter._tasks.values() %}
{% if task.node_type == "LLM" %}
def {{task.function_name}}({{ task.handle_args }}, _env: dict[str, str]) -> RunOutput:
    {% set chat_messages_found = false %}
    {% for input_handle_name in task.input_handle_names %}
    {% if input_handle_name == 'chat_messages' %}
    {% set chat_messages_found = true %}
    {% endif %}
    {% endfor %}

    {% if chat_messages_found %}
    input_chat_messages = chat_messages
    {% else %}
    input_chat_messages = []
    {% endif %}

    rendered_prompt = """{{task.config.prompt}}"""
    {% set prompt_variables = task.input_handle_names|reject("equalto", "chat_messages") %}
    {% for prompt_variable in prompt_variables %}
    # TODO: Fix this. Using double curly braces in quotes because normal double curly braces
    # get replaced during rendering by Cookiecutter. This is a hacky solution.
    rendered_prompt = rendered_prompt.replace("{{'{{'}}{{prompt_variable}}{{'}}'}}", {{prompt_variable}})  # type: ignore
    {% endfor %}

    {% if task.config.model_params == none %}
    params = {}
    {% else %}
    params = json.loads(
        """{{task.config.model_params}}"""
    )
    {% endif %}

    messages = [ChatMessage(role="system", content=rendered_prompt)]
    messages.extend(input_chat_messages)

    {% if task.config.provider == "openai" %}
    message_jsons = [
        {"role": message.role, "content": message.content} for message in messages
    ]

    data = {
        "model": "{{task.config.model}}",
        "messages": message_jsons,
    }
    data.update(params)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_env['OPENAI_API_KEY']}",
    }
    res = requests.post(
        "https://api.openai.com/v1/chat/completions", json=data, headers=headers
    )

    if res.status_code != 200:
        res_json = res.json()
        raise NodeRunError(f'OpenAI completions request failed: {res_json["error"]["message"]}')

    chat_completion = res.json()

    completion_message = chat_completion["choices"][0]["message"]["content"]

    meta_log = {}
    meta_log["node_chunk_id"] = None  # TODO: Add node chunk id
    meta_log["model"] = "{{task.config.model}}"
    meta_log["prompt"] = rendered_prompt
    meta_log["input_message_count"] = len(messages)
    meta_log["input_token_count"] = chat_completion["usage"]["prompt_tokens"]
    meta_log["output_token_count"] = chat_completion["usage"]["completion_tokens"]
    meta_log["total_token_count"] = (
        chat_completion["usage"]["prompt_tokens"] + chat_completion["usage"]["completion_tokens"]
    )
    meta_log["approximate_cost"] = None  # TODO: Add approximate cost
    {% elif task.config.provider == "anthropic" %}
    data = {
        "model": "{{task.config.model}}",
        "max_tokens": 4096,
    }
    data.update(params)

    # TODO: Generate appropriate code based on this if-else block
    if len(messages) == 1 and messages[0].role == "system":
        messages[0].role = "user"
        message_jsons = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        data["messages"] = message_jsons
    else:
        data["system"] = messages[0].content
        message_jsons = [
            {"role": message.role, "content": message.content} for message in messages[1:]
        ]
        data["messages"] = message_jsons

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": _env['ANTHROPIC_API_KEY'],
        "Anthropic-Version": "2023-06-01",
    }
    res = requests.post(
        "https://api.anthropic.com/v1/messages", json=data, headers=headers
    )

    if res.status_code != 200:
        raise NodeRunError(f"Anthropic message request failed: {res.text}")

    chat_completion = res.json()

    completion_message = chat_completion["content"][0]["text"]

    meta_log = {}
    meta_log["node_chunk_id"] = None  # TODO: Add node chunk id
    meta_log["model"] = "{{task.config.model}}"
    meta_log["prompt"] = rendered_prompt
    meta_log["input_message_count"] = len(messages)
    meta_log["input_token_count"] = chat_completion["usage"]["input_tokens"]
    meta_log["output_token_count"] = chat_completion["usage"]["output_tokens"]
    meta_log["total_token_count"] = (
        chat_completion["usage"]["input_tokens"] + chat_completion["usage"]["output_tokens"]
    )
    meta_log["approximate_cost"] = None  # TODO: Add approximate cost
    {% else %}
    {% endif %}

    return RunOutput(status="Success", output=completion_message)


{% elif task.node_type == "Output" %}
def {{task.function_name}}(output: NodeInput, _env: dict[str, str]) -> RunOutput:
    return RunOutput(status="Success", output=output)


{% elif task.node_type == "Input" %}
{# Do nothing for Input tasks #}
{% else %}
def {{task.function_name}}(output: NodeInput, _env: dict[str, str]) -> RunOutput:
    return RunOutput(status="Success", output=output)


{% endif %}
{% endfor %}
# Other functions can be added here
