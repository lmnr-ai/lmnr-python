# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportReturnType=false
"""Record what LangSmith's client hands to `Client._create_run` / `_update_run`
(the bridge's tap) for each integration, against mocked provider APIs.

    uv run python tests/test_langsmith/fixtures/record.py \
        tests/test_langsmith/fixtures/python_sdk_payloads.json

LangSmith's own sends go to an unreachable endpoint and are retried, so a run
takes a few minutes; point `LANGSMITH_ENDPOINT` at any local HTTP server to
make it fast.
"""

import datetime
import json
import os
import sys
import uuid
from typing import Any

os.environ.update(
    LANGSMITH_TRACING="true",
    LANGSMITH_API_KEY="ls-test",
    LANGSMITH_ENDPOINT=os.environ.get("LANGSMITH_ENDPOINT", "http://127.0.0.1:9"),
    LANGSMITH_PROJECT="bridge-fixtures",
)

import httpx
import langsmith.client as lc
from langsmith import traceable
from langsmith.run_trees import get_cached_client

RECORDS: dict[str, list[dict[str, Any]]] = {}
CURRENT = ["none"]


def to_json(v):
    if isinstance(v, dict):
        return {str(k): to_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [to_json(x) for x in v]
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "model_dump"):
        return to_json(v.model_dump())
    return repr(v)


orig_create, orig_update = lc.Client._create_run, lc.Client._update_run


def _create_run(self, run_create, **kw):
    RECORDS.setdefault(CURRENT[0], []).append({"op": "create", "run": to_json(run_create)})
    return orig_create(self, run_create, **kw)


def _update_run(self, run_update, **kw):
    RECORDS.setdefault(CURRENT[0], []).append({"op": "update", "run": to_json(run_update)})
    return orig_update(self, run_update, **kw)


lc.Client._create_run = _create_run
lc.Client._update_run = _update_run

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def chat_completion(body):
    has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))
    if has_tool_result or not body.get("tools"):
        message = {"role": "assistant", "content": "It is sunny in Paris."}
        finish = "stop"
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
        finish = "tool_calls"
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def openai_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi!", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )
    if body.get("stream"):
        base = {
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
        }
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
            {**base, "choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                **base,
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            },
        ]
        return httpx.Response(200, text=sse(chunks), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=chat_completion(body))


def anthropic_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    has_result = any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
        for m in body.get("messages", [])
    )
    content = (
        [{"type": "text", "text": "Sunny."}]
        if has_result
        else [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ]
    )
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": content,
            "stop_reason": "end_turn" if has_result else "tool_use",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 10,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
            },
        },
    )


def scenario(name):
    def deco(fn):
        CURRENT[0] = name
        try:
            fn()
        except Exception as e:
            print(name, "raised", repr(e))
        get_cached_client().flush()
        CURRENT[0] = "none"
        return fn

    return deco


@scenario("traceable")
def _():
    @traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o"})
    def llm(messages):
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

    @traceable(run_type="tool")
    def search(query: str) -> list[str]:
        return ["a", "b"]

    @traceable(
        name="agent", tags=["t1"], metadata={"session_id": "s-1", "user_id": "u-1", "env": "test"}
    )
    def agent(q):
        search(q)
        return llm([{"role": "system", "content": "be brief"}, {"role": "user", "content": q}])

    agent("hello")


@scenario("traceable_error")
def _():
    @traceable
    def boom(x):
        raise ValueError("bad input")

    boom(1)


@scenario("wrap_openai_tools")
def _():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    client = wrap_openai(
        OpenAI(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)))
    )

    @traceable(name="tool_loop")
    def loop():
        messages = [{"role": "user", "content": "Weather in Paris?"}]
        r1 = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=[TOOL], temperature=0.2
        )
        msg = r1.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "sunny"})
        client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=[TOOL])

    loop()


@scenario("wrap_openai_stream")
def _():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    client = wrap_openai(
        OpenAI(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)))
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for _chunk in stream:
        pass


@scenario("wrap_openai_responses")
def _():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    client = wrap_openai(
        OpenAI(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)))
    )
    client.responses.create(model="gpt-4.1-mini", input="Hi", instructions="Be nice")


@scenario("wrap_anthropic_tools")
def _():
    import httpx2
    from anthropic import Anthropic
    from langsmith.wrappers import wrap_anthropic

    def handler(request):
        r = anthropic_handler(
            httpx.Request(request.method, str(request.url), content=request.content)
        )
        return httpx2.Response(
            r.status_code, content=r.content, headers={"content-type": "application/json"}
        )

    client = wrap_anthropic(
        Anthropic(api_key="x", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    r1 = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=100, system="Be brief", messages=messages, tools=tools
    )
    messages.append({"role": "assistant", "content": [b.model_dump() for b in r1.content]})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}],
        }
    )
    client.messages.create(
        model="claude-sonnet-4-5", max_tokens=100, messages=messages, tools=tools
    )


@scenario("langchain_openai_tools")
def _():
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(city: str) -> str:
        """Weather in a city"""
        return "sunny"

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="x",
        temperature=0.1,
        http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)),
    ).bind_tools([get_weather])

    def run(question: str) -> str:
        messages = [HumanMessage(question)]
        ai = llm.invoke(messages)
        messages.append(ai)
        for call in ai.tool_calls:
            messages.append(ToolMessage(get_weather.invoke(call["args"]), tool_call_id=call["id"]))
        return llm.invoke(messages).content

    chain = RunnableLambda(run, name="weather_agent") | RunnableLambda(lambda s: s.upper())
    chain.invoke(
        "Weather in Paris?", config={"metadata": {"thread_id": "th-1"}, "tags": ["lc-tag"]}
    )


@scenario("langgraph")
def _():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        messages: list[str]
        steps_taken: int

    def plan(state: State):
        return {
            "messages": state["messages"] + ["planned"],
            "steps_taken": state["steps_taken"] + 1,
        }

    def act(state: State):
        return {"messages": state["messages"] + ["acted"], "steps_taken": state["steps_taken"] + 1}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", END)
    app = graph.compile()
    app.invoke(
        {"messages": ["hi"], "steps_taken": 0}, config={"configurable": {"thread_id": "thread-42"}}
    )


with open(sys.argv[1], "w") as out:
    json.dump(RECORDS, out, indent=1, sort_keys=True)
print({k: len(v) for k, v in RECORDS.items()})
