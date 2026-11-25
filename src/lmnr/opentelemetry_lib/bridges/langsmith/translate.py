"""Pure conversions from LangSmith run payloads to Laminar span attributes.

A run's fields are what LangSmith's client is about to send (`create_run` /
`update_run` after sampling and redaction), already reduced to JSON data. No
LangSmith or LangChain import here; `runs.py` calls into these.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.bridges.langfuse.attributes import (
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_COST,
    GEN_AI_USAGE_INPUT_COST,
    GEN_AI_USAGE_OUTPUT_COST,
    LLM_USAGE_TOTAL_TOKENS,
    LS_PROVIDER_ALIASES,
)
from lmnr.opentelemetry_lib.bridges.langfuse.translate import (
    model_parameter_attributes,
    normalize_messages,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.chain_state import small_state
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    USER_ID,
)
from lmnr.sdk.utils import json_dumps

# LangSmith run types recorded as LLM / TOOL spans; the rest are DEFAULT.
LLM_RUN_TYPES = frozenset({"llm", "embedding"})
TOOL_RUN_TYPES = frozenset({"tool"})

# Metadata keys LangSmith reads as a thread id, in the order its UI does.
SESSION_METADATA_KEYS = ("session_id", "thread_id", "conversation_id")
USER_METADATA_KEY = "user_id"

# Metadata that LangSmith, LangChain and LangGraph add themselves. Not the
# user's trace metadata, so it is left out of Laminar's.
_INTERNAL_METADATA_PREFIXES = ("ls_", "langgraph_", "LANGSMITH_", "LANGCHAIN_", "__")
_INTERNAL_METADATA_KEYS = frozenset(
    {
        "revision_id",
        "lc_versions",
        "_type",
        "usage_metadata",
        "checkpoint_ns",
        "checkpoint_id",
        *SESSION_METADATA_KEYS,
        USER_METADATA_KEY,
    }
)

# Run tags LangChain and LangGraph add to mark steps (`seq:step:1`,
# `graph:step:2`, `map:key:...`) and LangSmith's own `langsmith:*` markers.
_INTERNAL_TAG = re.compile(r"^(seq:step:\d+|graph:step:\d+|map:key:.*|langsmith:.*)$")
HIDDEN_TAG = "langsmith:hidden"

# Nested LangChain runs that only glue steps together; the LangChain
# instrumentation records no span for them either (see `langchain/AGENTS.md`).
GLUE_RUN_NAMES = frozenset({"RunnableSequence", "RunnableLambda"})

_LC_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "developer": "developer",
    "tool": "tool",
    "function": "tool",
}
_LC_ROLE_BY_CLASS = {
    "HumanMessage": "user",
    "HumanMessageChunk": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "FunctionMessage": "tool",
    "FunctionMessageChunk": "tool",
}

# Request parameters LangSmith's wrappers and LangChain record as metadata.
_LS_REQUEST_PARAMETERS = (
    ("ls_temperature", GEN_AI_REQUEST_TEMPERATURE),
    ("ls_max_tokens", GEN_AI_REQUEST_MAX_TOKENS),
)


def _dict(value: Any) -> dict[str, Any] | None:
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _list(value: Any) -> list[Any] | None:
    return cast(list[Any], value) if isinstance(value, list) else None


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def is_langchain_run(extra: Mapping[str, Any] | None) -> bool:
    """Whether LangChain's tracer recorded the run (LangChain, LangGraph):
    it stamps its own runtime over LangSmith's."""
    runtime = _dict((extra or {}).get("runtime"))
    library = runtime.get("library") if runtime else None
    return isinstance(library, str) and library.startswith("langchain")


def span_type(run_type: str | None) -> str:
    if run_type in LLM_RUN_TYPES:
        return "LLM"
    if run_type in TOOL_RUN_TYPES:
        return "TOOL"
    return "DEFAULT"


# --- messages ----------------------------------------------------------------


def lc_message(message: Any) -> dict[str, Any] | None:
    """A serialized LangChain message as a `{role, content, ...}` dict, or
    None if `message` isn't one.

    LangChain's tracer sends `dumpd` output (`{lc: 1, type: "constructor",
    id: [..., "AIMessage"], kwargs: {...}}`); `messages_to_dict` gives
    `{type: "ai", data: {...}}`.
    """
    msg = _dict(message)
    if msg is None:
        return None
    kwargs = _dict(msg.get("kwargs"))
    data = _dict(msg.get("data"))
    if msg.get("lc") == 1 and kwargs is not None:
        ids = _list(msg.get("id")) or []
        class_name = ids[-1] if ids and isinstance(ids[-1], str) else ""
        role = _LC_ROLE_BY_CLASS.get(class_name) or _LC_ROLE_BY_TYPE.get(
            cast(str, kwargs.get("type") or "")
        )
        fields = kwargs
    elif data is not None and isinstance(msg.get("type"), str):
        role = _LC_ROLE_BY_TYPE.get(cast(str, msg["type"]))
        fields = data
    else:
        return None
    if role is None:
        # `ChatMessage` carries its own role.
        role = fields.get("role") if isinstance(fields.get("role"), str) else "user"
    out: dict[str, Any] = {"role": role, "content": fields.get("content")}
    tool_calls = _list(fields.get("tool_calls"))
    additional = _dict(fields.get("additional_kwargs")) or {}
    if not tool_calls:
        # Older LangChain kept OpenAI's tool calls only here.
        tool_calls = _list(additional.get("tool_calls"))
    if tool_calls:
        out["tool_calls"] = tool_calls
    if not tool_calls and isinstance(additional.get("function_call"), dict):
        out["function_call"] = additional["function_call"]
    if fields.get("tool_call_id") is not None:
        out["tool_call_id"] = fields["tool_call_id"]
    return out


def _as_messages(items: list[Any]) -> list[Any]:
    """Messages with LangChain's serialized ones unpacked; others as given."""
    return [lc_message(item) or item for item in items]


def _strip_system(messages: list[Any], system: Any) -> list[Any]:
    if isinstance(system, str) and system:
        return [{"role": "system", "content": system}, *messages]
    if isinstance(system, list) and system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def llm_input(inputs: Any, invocation_params: Any = None) -> tuple[list[Any], Any] | None:
    """Split an LLM run's inputs into (messages, tool definitions).

    Shapes:
    - LangChain chat models: `{messages: [[<serialized message>, ...]]}`
      (one list per prompt in the batch; the first is used), tools in the
      run's `invocation_params`;
    - LangChain completion models: `{prompts: ["..."]}`;
    - `wrap_openai` / `wrap_anthropic` / hand-written `@traceable` runs: the
      request kwargs, `{messages, tools, model, ...}`, Anthropic's `system`
      already folded into the messages by `wrap_anthropic`;
    - the OpenAI Responses API through `wrap_openai`: `{input, instructions}`;
    - a bare `{prompt: "..."}` or string.
    Returns None for anything else; the caller keeps the raw inputs.
    """
    if isinstance(inputs, str):
        return ([{"role": "user", "content": inputs}], None) if inputs else None
    data = _dict(inputs)
    if data is None:
        return None
    params = _dict(invocation_params) or {}
    tools = data.get("tools") or data.get("functions") or params.get("tools")
    tools = tools or params.get("functions")

    messages = _list(data.get("messages"))
    if messages is not None:
        if messages and all(isinstance(m, list) for m in messages):
            messages = cast(list[Any], messages[0])
        return _strip_system(_as_messages(messages), data.get("system")), tools
    prompts = _list(data.get("prompts"))
    if prompts and isinstance(prompts[0], str):
        return [{"role": "user", "content": prompts[0]}], tools
    if "input" in data:
        inp = data["input"]
        if isinstance(inp, str):
            items: list[Any] = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            items = _as_messages(cast(list[Any], inp))
        else:
            return None
        instructions = data.get("instructions")
        if isinstance(instructions, str) and instructions:
            items.insert(0, {"role": "system", "content": instructions})
        return items, tools
    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt:
        return [{"role": "user", "content": prompt}], tools
    return None


def _first_generation(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
    generations = _list(outputs.get("generations"))
    if not generations:
        return None
    first = generations[0]
    if isinstance(first, list):
        first = cast(list[Any], first)[0] if first else None
    return _dict(first)


def llm_output(outputs: Any) -> list[Any] | None:
    """An LLM run's outputs as a message list, before `normalize_messages`.

    Shapes: LangChain's `{generations: [[{message | text}]]}`; an OpenAI chat
    completion (`choices[].message`, also what `wrap_openai` reduces a stream
    to); a Responses API response (`output` items); a message dict
    (`wrap_anthropic`, hand-written runs); `{messages: [...]}`; or a string
    (`{output: "..."}` from a `@traceable` that returned one).
    """
    if isinstance(outputs, str):
        return [{"role": "assistant", "content": outputs}] if outputs else None
    data = _dict(outputs)
    if data is None:
        return None
    if "generations" in data:
        generations = _list(data["generations"]) or []
        first: Any = generations[0] if generations else []
        batch = cast(list[Any], first) if isinstance(first, list) else [first]
        messages: list[Any] = []
        for generation in batch:
            gen = _dict(generation) or {}
            message = lc_message(gen.get("message"))
            if message is not None:
                messages.append(message)
            elif isinstance(gen.get("text"), str):
                messages.append({"role": "assistant", "content": gen["text"]})
        return messages or None
    choices = _list(data.get("choices"))
    if choices is not None:
        messages = []
        for choice in choices:
            c = _dict(choice) or {}
            message = _dict(c.get("message"))
            if message is not None:
                messages.append(message)
            elif isinstance(c.get("text"), str):
                messages.append({"role": "assistant", "content": c["text"]})
        return messages or None
    output = data.get("output")
    if isinstance(output, list) and data.get("object") == "response":
        return _as_messages(cast(list[Any], output))
    if "role" in data or ("content" in data and "tool_calls" in data):
        return [data]
    listed = _list(data.get("messages"))
    if listed is not None:
        return _as_messages(listed)
    if isinstance(data.get("content"), (str, list)):
        return [{"role": "assistant", "content": data["content"]}]
    if isinstance(output, str) and output:
        return [{"role": "assistant", "content": output}]
    message = lc_message(output)
    if message is not None:
        return [message]
    return None


def input_attributes(
    run_type: str | None,
    inputs: Any,
    invocation_params: Any,
    is_langchain: bool,
) -> dict[str, AttributeValue]:
    """`gen_ai.input.messages` / `gen_ai.tool.definitions` for an LLM run
    with a recognized shape, `lmnr.span.input` otherwise.

    LangChain chain runs (graphs, nodes, sequences) carry whole graph state
    as input and output, so only its known-small keys are kept, as the
    LangChain instrumentation does.
    """
    if inputs is None or inputs == {}:
        return {}
    if run_type in LLM_RUN_TYPES:
        split = llm_input(inputs, invocation_params)
        if split is not None:
            messages, tools = split
            out: dict[str, AttributeValue] = {
                GEN_AI_INPUT_MESSAGES: json_dumps(normalize_messages(messages, False))
            }
            if tools:
                out[GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)
            return out
    if is_langchain and run_type == "chain":
        kept = small_state(inputs)
        return {SPAN_INPUT: json_dumps(kept)} if kept else {}
    return {SPAN_INPUT: json_dumps(inputs)}


def output_attributes(
    run_type: str | None, outputs: Any, is_langchain: bool
) -> dict[str, AttributeValue]:
    """`gen_ai.output.messages` for an LLM run, `lmnr.span.output` otherwise."""
    if outputs is None or outputs == {}:
        return {}
    if run_type in LLM_RUN_TYPES:
        messages = llm_output(outputs)
        if messages is not None:
            return {GEN_AI_OUTPUT_MESSAGES: json_dumps(normalize_messages(messages, True))}
    if is_langchain and run_type == "chain":
        kept = small_state(outputs)
        return {SPAN_OUTPUT: json_dumps(kept)} if kept else {}
    return {SPAN_OUTPUT: json_dumps(outputs)}


# --- model, provider, usage --------------------------------------------------


def _usage_metadata(outputs: Any, metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """LangSmith's `usage_metadata`, where its backend looks for it: in the
    outputs, in the run's metadata (streamed / integration runs), or on
    LangChain's AI message."""
    data = _dict(outputs) or {}
    usage = _dict(data.get("usage_metadata")) or _dict(metadata.get("usage_metadata"))
    if usage is not None:
        return usage
    generation = _first_generation(data)
    if generation is not None:
        message = _dict(generation.get("message")) or {}
        fields = _dict(message.get("kwargs")) or _dict(message.get("data")) or {}
        return _dict(fields.get("usage_metadata"))
    return None


def _provider_usage(outputs: Any) -> dict[str, Any] | None:
    """A provider's raw usage object as LangSmith's `usage_metadata`, for runs
    whose outputs carry one but no `usage_metadata`: OpenAI's `usage`,
    Anthropic's `usage` (input excludes cache tokens), LangChain's
    `llm_output.token_usage`."""
    data = _dict(outputs) or {}
    usage = _dict(data.get("usage"))
    if usage is None:
        llm_output_ = _dict(data.get("llm_output")) or {}
        usage = _dict(llm_output_.get("token_usage"))
    if usage is None:
        return None
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt = _dict(usage.get("prompt_tokens_details")) or {}
        completion = _dict(usage.get("completion_tokens_details")) or {}
        return {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "input_token_details": {"cache_read": prompt.get("cached_tokens")},
            "output_token_details": {"reasoning": completion.get("reasoning_tokens")},
        }
    if "input_tokens" in usage:
        cache_read = _number(usage.get("cache_read_input_tokens"))
        cache_write = _number(usage.get("cache_creation_input_tokens"))
        input_tokens = _number(usage.get("input_tokens"))
        if input_tokens is not None and (cache_read or cache_write):
            input_tokens += (cache_read or 0) + (cache_write or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": usage.get("output_tokens"),
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
        }
    return None


def usage_attributes(outputs: Any, metadata: Mapping[str, Any]) -> dict[str, AttributeValue]:
    """Token counts and costs from LangSmith's `usage_metadata`.

    Unlike Langfuse's usage details, LangSmith's are inclusive like Laminar's:
    `input_tokens` already counts cached tokens and `output_tokens` reasoning
    ones, so they map one to one.
    """
    usage = _usage_metadata(outputs, metadata) or _provider_usage(outputs)
    if usage is None:
        return {}
    input_details = _dict(usage.get("input_token_details")) or {}
    output_details = _dict(usage.get("output_token_details")) or {}
    input_tokens = _number(usage.get("input_tokens"))
    output_tokens = _number(usage.get("output_tokens"))
    total_tokens = _number(usage.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    out: dict[str, AttributeValue] = {}
    for key, value in (
        (GEN_AI_USAGE_INPUT_TOKENS, input_tokens),
        (GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens),
        (LLM_USAGE_TOTAL_TOKENS, total_tokens),
        (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, _number(input_details.get("cache_read"))),
        (GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, _number(input_details.get("cache_creation"))),
        (GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, _number(output_details.get("reasoning"))),
    ):
        if value is not None:
            out[key] = int(value)
    # Costs a run reports itself (`usage_metadata.{input,output,total}_cost`).
    input_cost = _number(usage.get("input_cost"))
    output_cost = _number(usage.get("output_cost"))
    total_cost = _number(usage.get("total_cost"))
    if total_cost is None and (input_cost is not None or output_cost is not None):
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)
    for key, value in (
        (GEN_AI_USAGE_INPUT_COST, input_cost),
        (GEN_AI_USAGE_OUTPUT_COST, output_cost),
        (GEN_AI_USAGE_COST, total_cost),
    ):
        if value is not None:
            out[key] = float(value)
    return out


def request_attributes(
    inputs: Any, metadata: Mapping[str, Any], invocation_params: Any
) -> dict[str, AttributeValue]:
    """Model, provider and request parameters of an LLM run, from LangSmith's
    `ls_*` metadata (what its own cost tracking reads), then the request."""
    out: dict[str, AttributeValue] = {}
    data = _dict(inputs) or {}
    params = _dict(invocation_params) or {}
    model = metadata.get("ls_model_name") or data.get("model") or params.get("model")
    model = model or params.get("model_name")
    if isinstance(model, str) and model:
        out[GEN_AI_REQUEST_MODEL] = model
    provider = metadata.get("ls_provider")
    if isinstance(provider, str) and provider:
        out[GEN_AI_SYSTEM] = LS_PROVIDER_ALIASES.get(provider, provider)
    # The request's own parameters: LangChain's invocation params, or the
    # kwargs a wrapper recorded as inputs.
    mapped, _rest = model_parameter_attributes(params or data)
    out.update(mapped)
    for key, attr in _LS_REQUEST_PARAMETERS:
        value = _number(metadata.get(key))
        if value is not None and attr not in out:
            out[attr] = value
    stop = metadata.get("ls_stop")
    if (
        GEN_AI_REQUEST_STOP_SEQUENCES not in out
        and isinstance(stop, list)
        and stop
        and all(isinstance(s, str) for s in cast(list[Any], stop))
    ):
        out[GEN_AI_REQUEST_STOP_SEQUENCES] = cast(list[str], stop)
    return out


def response_attributes(outputs: Any) -> dict[str, AttributeValue]:
    """Response model and id: OpenAI / Anthropic response fields, or the
    `response_metadata` on LangChain's AI message."""
    data = _dict(outputs) or {}
    model: Any = data.get("model")
    response_id: Any = data.get("id")
    generation = _first_generation(data)
    if generation is not None:
        message = _dict(generation.get("message")) or {}
        fields = _dict(message.get("kwargs")) or _dict(message.get("data")) or {}
        response_metadata = _dict(fields.get("response_metadata")) or {}
        model = response_metadata.get("model_name") or response_metadata.get("model")
        response_id = response_metadata.get("id")
        if not model:
            llm_output_ = _dict(data.get("llm_output")) or {}
            model = llm_output_.get("model_name")
    out: dict[str, AttributeValue] = {}
    if isinstance(model, str) and model:
        out[GEN_AI_RESPONSE_MODEL] = model
    if isinstance(response_id, str) and response_id:
        out[GEN_AI_RESPONSE_ID] = response_id
    return out


# --- session, user, tags, metadata -------------------------------------------


def _text(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value:
        return value
    return None


def association_attributes(
    metadata: Mapping[str, Any],
    tags: Any,
    is_root: bool,
    invocation_params: Any = None,
) -> dict[str, AttributeValue]:
    """Session, user, tags and (on the root run) trace metadata.

    LangSmith groups runs into threads by the `session_id`, `thread_id` or
    `conversation_id` metadata key (LangGraph sets `thread_id`), which
    becomes Laminar's session. Metadata is inherited by child runs in
    LangSmith, so only the root run's becomes Laminar trace metadata,
    without the keys LangSmith, LangChain and LangGraph add themselves.
    """
    out: dict[str, AttributeValue] = {}
    for key in SESSION_METADATA_KEYS:
        if session_id := _text(metadata.get(key)):
            out[f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"] = session_id
            break
    if user_id := _text(metadata.get(USER_METADATA_KEY)):
        out[f"{ASSOCIATION_PROPERTIES}.{USER_ID}"] = user_id

    if isinstance(tags, list):
        kept = [
            t for t in cast(list[Any], tags) if isinstance(t, str) and not _INTERNAL_TAG.match(t)
        ]
        if kept := list(dict.fromkeys(kept)):
            out[f"{ASSOCIATION_PROPERTIES}.tags"] = kept

    if is_root:
        # LangChain copies a chat model's invocation params into its metadata.
        params = _dict(invocation_params) or {}
        for key, value in metadata.items():
            if (
                key in _INTERNAL_METADATA_KEYS
                or key.startswith(_INTERNAL_METADATA_PREFIXES)
                or (key in params and params[key] == value)
                or value is None
            ):
                continue
            out[f"{ASSOCIATION_PROPERTIES}.metadata.{key}"] = (
                value if isinstance(value, (str, bool, int, float)) else json_dumps(value)
            )
    return out


def is_glue_run(name: str | None, tags: Any, is_langchain: bool) -> bool:
    """Runs left out of Laminar's tree when nested, their children moving up
    to the nearest recorded ancestor: LangChain's unnamed sequences and
    lambdas (as the LangChain instrumentation does) and runs LangSmith's UI
    hides (`langsmith:hidden`)."""
    if isinstance(tags, list) and HIDDEN_TAG in cast(list[Any], tags):
        return True
    return is_langchain and name in GLUE_RUN_NAMES


# --- errors -------------------------------------------------------------------

_EXCEPTION_REPR = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$", re.DOTALL)


def exception_attributes(error: str) -> dict[str, AttributeValue]:
    """An `exception` event's attributes from LangSmith's error string:
    `repr(exc)`, a blank line, then the traceback (Python), or the message
    (JS)."""
    head, _, stacktrace = error.partition("\n\n")
    head = head.strip()
    exc_type = "Error"
    message = head
    if match := _EXCEPTION_REPR.match(head):
        exc_type = match.group(1)
        message = match.group(2).strip()
        if len(message) >= 2 and message[0] == message[-1] and message[0] in "'\"":
            message = message[1:-1]
    elif ":" in head and re.match(r"^[A-Za-z_][\w.]*:", head):
        exc_type, _, message = head.partition(":")
        message = message.strip()
    out: dict[str, AttributeValue] = {
        "exception.type": exc_type,
        "exception.message": message,
    }
    if stacktrace.strip():
        out["exception.stacktrace"] = stacktrace.strip()
    return out
